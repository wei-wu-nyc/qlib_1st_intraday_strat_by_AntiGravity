"""
Opening Range Breakout (ORB) Strategy for Intraday Trading.

A classic intraday strategy that:
1. Defines the opening range (first 30 min highs/lows)
2. Enters long when price breaks above opening range high
3. Uses opening range low as stop
4. Targets 1-2x the range as profit

Special considerations:
- Only enters after first 30 minutes (6 bars)
- Stops trading after 14:00 (24 bars before close)  
- Forces close at 15:55
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_strategy import RuleBasedStrategy


class OpeningRangeBreakoutStrategy(RuleBasedStrategy):
    """
    Opening Range Breakout strategy.
    
    Parameters:
    - opening_bars: Number of bars to define opening range (default: 6 = 30 min)
    - breakout_threshold: % above range high for breakout (default: 0.001 = 0.1%)
    - stop_buffer: % below range low for stop (default: 0.001)
    - profit_target_multiple: Target as multiple of range (default: 1.5)
    - last_entry_hour: Latest hour to enter (default: 14 = 2pm)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        self.opening_bars = self.config.get('opening_bars', 6)  # 30 minutes
        self.breakout_threshold = self.config.get('breakout_threshold', 0.001)
        self.stop_buffer = self.config.get('stop_buffer', 0.001)
        self.profit_target_multiple = self.config.get('profit_target_multiple', 1.5)
        self.last_entry_hour = self.config.get('last_entry_hour', 14)
    
    def get_name(self) -> str:
        return "OpeningRangeBreakout"
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_position: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate ORB signals.
        
        Requires grouping by date to calculate opening range.
        """
        result = features.copy()
        
        # Extract date from index
        if isinstance(result.index, pd.DatetimeIndex):
            result['_date'] = result.index.date
            result['_time'] = result.index.time
        elif isinstance(result.index, pd.MultiIndex):
            result['_date'] = result.index.get_level_values(0).date
            result['_time'] = result.index.get_level_values(0).time
        
        # Calculate opening range for each date
        result = self._calc_opening_range(result)
        
        # Entry conditions
        # 1. Past the opening period
        past_opening = result['bar_index'] > self.opening_bars
        
        # 2. Price breaks above opening high
        breakout_up = result['close'] > result['_or_high'] * (1 + self.breakout_threshold)
        
        # 3. Volume confirmation (optional)
        vol_col = None
        for period in [6, 12]:
            if f'vol_ratio_{period}' in result.columns:
                vol_col = f'vol_ratio_{period}'
                break
        
        if vol_col is not None:
            vol_ok = result[vol_col] > 1.0  # Above average
        else:
            vol_ok = True
        
        # 4. Not too late in the day
        if 'bar_index' in result.columns:
            # Last entry at 2pm (bar ~54 for 14:00)
            last_entry_bar = (self.last_entry_hour - 9) * 12 + 6  # Approximate
            time_ok = result['bar_index'] <= last_entry_bar
        else:
            time_ok = True
        
        # Combine entry conditions
        entry_signal = past_opening & breakout_up & vol_ok & time_ok
        entry_signal = self.apply_entry_rules(result, entry_signal)
        
        # Exit conditions
        # 1. Stop loss: price below opening low
        stop_hit = result['close'] < result['_or_low'] * (1 - self.stop_buffer)
        
        # 2. Profit target: price above opening high by X * range
        or_range = result['_or_high'] - result['_or_low']
        profit_target = result['_or_high'] + or_range * self.profit_target_multiple
        target_hit = result['close'] >= profit_target
        
        exit_signal = stop_hit | target_hit
        
        # Signal strength (based on breakout strength)
        breakout_pct = (result['close'] / result['_or_high'] - 1).clip(0, 0.02) / 0.02
        range_quality = (or_range / result['close']).clip(0, 0.02) / 0.02  # Larger range = better
        
        signal_strength = (breakout_pct * 0.6 + range_quality * 0.4)
        signal_strength = signal_strength.where(entry_signal, 0)
        
        # Output
        result['signal'] = entry_signal.astype(int)
        result['signal_strength'] = signal_strength
        result['exit_signal'] = exit_signal.astype(int)
        result['or_high'] = result['_or_high']
        result['or_low'] = result['_or_low']
        result['or_range'] = or_range
        
        # Clean up temp columns
        temp_cols = [c for c in result.columns if c.startswith('_')]
        result = result.drop(columns=temp_cols)
        
        return result
    
    def _calc_opening_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate opening range (high/low of first N bars) for each date."""
        result = df.copy()
        
        # Get opening bars only
        opening_mask = df['bar_index'] <= self.opening_bars
        
        # Calculate high/low of opening period per date
        opening_highs = df[opening_mask].groupby('_date')['high'].max() if 'high' in df.columns else df[opening_mask].groupby('_date')['close'].max()
        opening_lows = df[opening_mask].groupby('_date')['low'].min() if 'low' in df.columns else df[opening_mask].groupby('_date')['close'].min()
        
        # Map back to all rows
        result['_or_high'] = result['_date'].map(opening_highs)
        result['_or_low'] = result['_date'].map(opening_lows)
        
        return result


if __name__ == "__main__":
    print("Testing OpeningRangeBreakoutStrategy...")
    
    np.random.seed(42)
    
    # Create sample data for a full trading day
    dates = pd.date_range('2023-12-20 09:35', periods=78, freq='5min')
    
    # Simulate trending up after open
    prices = [100]
    for i in range(77):
        if i < 6:  # Opening range - tight
            prices.append(prices[-1] + np.random.randn() * 0.05)
        elif i < 20:  # Morning - breakout up
            prices.append(prices[-1] + 0.1 + np.random.randn() * 0.05)
        else:  # Rest of day
            prices.append(prices[-1] + np.random.randn() * 0.08)
    
    sample_df = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.randn()) * 0.1 for p in prices],
        'low': [p - abs(np.random.randn()) * 0.1 for p in prices],
        'close': prices,
        'vol_ratio_6': 1 + np.abs(np.random.randn(78) * 0.5),
        'bar_index': range(1, 79),
    }, index=dates)
    
    strategy = OpeningRangeBreakoutStrategy()
    signals = strategy.generate_signals(sample_df)
    
    print(f"\nSignals generated:")
    print(f"  Entry signals: {signals['signal'].sum()}")
    print(f"  Exit signals: {signals['exit_signal'].sum()}")
    print(f"  Opening range high: {signals['or_high'].iloc[0]:.2f}")
    print(f"  Opening range low: {signals['or_low'].iloc[0]:.2f}")
    
    signal_bars = signals[signals['signal'] == 1]
    if len(signal_bars) > 0:
        print(f"\n  First entry signal at bar {signal_bars['bar_index'].iloc[0]}")
    
    print("\nORB test completed!")
