"""
VWAP Scalping Strategy for Intraday Trading.

A short-term scalping strategy that:
1. Looks for quick mean-reversion to VWAP
2. Uses tick imbalance for entry confirmation
3. Targets very short holding periods (2-6 bars = 10-30 min)

Good for capturing quick reversions when price deviates from VWAP.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_strategy import RuleBasedStrategy


class VWAPScalpingStrategy(RuleBasedStrategy):
    """
    VWAP Scalping strategy for quick mean-reversion trades.
    
    Entry when:
    - Price deviates significantly from VWAP (oversold relative to VWAP)
    - Tick ratio shows buying pressure (reversal signal)
    - Volume is elevated (confirms activity)
    
    Exit when:
    - Price returns to VWAP
    - Target bars reached (default: 4 bars = 20 min)
    - Stop loss hit
    
    Parameters:
    - vwap_dev_threshold: Min % deviation from VWAP for entry (default: 0.3%)
    - tick_ratio_threshold: Min tick ratio for entry (default: 0.55)
    - target_bars: Target holding period (default: 4)
    - stop_pct: Stop loss percentage (default: 0.2%)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        self.vwap_dev_threshold = self.config.get('vwap_dev_threshold', 0.003)  # 0.3%
        self.tick_ratio_threshold = self.config.get('tick_ratio_threshold', 0.55)
        self.target_bars = self.config.get('target_bars', 4)  # 20 minutes
        self.stop_pct = self.config.get('stop_pct', 0.002)  # 0.2%
        self.vol_threshold = self.config.get('vol_threshold', 1.2)  # Above avg volume
        
        # For scalping, we can trade later in the day
        self.min_bars_to_close = 4  # Only need 20 min before close
    
    def get_name(self) -> str:
        return "VWAPScalping"
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_position: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate VWAP scalping signals."""
        result = features.copy()
        
        # Get VWAP column
        vwap_col = None
        for period in [12, 6, 24]:
            if f'vwap_{period}' in result.columns:
                vwap_col = f'vwap_{period}'
                break
        
        if vwap_col is None:
            # Calculate simple VWAP approximation if not available
            if all(c in result.columns for c in ['high', 'low', 'close', 'volume']):
                typical_price = (result['high'] + result['low'] + result['close']) / 3
                result['_vwap'] = (typical_price * result['volume']).rolling(12).sum() / result['volume'].rolling(12).sum()
                vwap_col = '_vwap'
            else:
                # Can't compute VWAP, no signals
                result['signal'] = 0
                result['signal_strength'] = 0
                result['exit_signal'] = 0
                return result
        
        # Calculate deviation from VWAP
        vwap_dev = (result['close'] - result[vwap_col]) / result[vwap_col]
        result['vwap_deviation'] = vwap_dev
        
        # Entry conditions for LONG (price below VWAP, expecting reversion up)
        # 1. Price significantly below VWAP
        below_vwap = vwap_dev < -self.vwap_dev_threshold
        
        # 2. Tick ratio shows buying pressure (if available)
        if 'tick_ratio' in result.columns:
            tick_confirm = result['tick_ratio'] > self.tick_ratio_threshold
        elif 'tick_ratio_ma_6' in result.columns:
            tick_confirm = result['tick_ratio_ma_6'] > self.tick_ratio_threshold
        else:
            tick_confirm = True  # No tick data, skip this filter
        
        # 3. Volume elevated
        vol_col = None
        for period in [6, 12]:
            if f'vol_ratio_{period}' in result.columns:
                vol_col = f'vol_ratio_{period}'
                break
        
        if vol_col is not None:
            vol_confirm = result[vol_col] > self.vol_threshold
        else:
            vol_confirm = True
        
        # 4. Not too close to VWAP (avoid noise)
        not_too_close = abs(vwap_dev) > self.vwap_dev_threshold * 0.5
        
        # Combine entry conditions
        entry_signal = below_vwap & tick_confirm & vol_confirm & not_too_close
        entry_signal = self.apply_entry_rules(result, entry_signal)
        
        # Exit conditions
        # 1. Price returned to VWAP (or above)
        at_vwap = vwap_dev >= 0
        
        # 2. Price moved against us (stop loss)
        if 'close' in result.columns:
            price_drop = result['close'].pct_change(self.target_bars) < -self.stop_pct
        else:
            price_drop = False
        
        exit_signal = at_vwap | price_drop
        
        # Signal strength based on:
        # - Deviation magnitude (larger = stronger signal)
        # - Tick ratio strength
        # - Volume strength
        dev_strength = (abs(vwap_dev) / (self.vwap_dev_threshold * 2)).clip(0, 1)
        
        if 'tick_ratio' in result.columns:
            tick_strength = ((result['tick_ratio'] - 0.5) / 0.3).clip(0, 1)
        else:
            tick_strength = 0.5
        
        if vol_col is not None:
            vol_strength = ((result[vol_col] - 1) / 1).clip(0, 1)
        else:
            vol_strength = 0.5
        
        signal_strength = (dev_strength * 0.5 + tick_strength * 0.3 + vol_strength * 0.2)
        signal_strength = signal_strength.where(entry_signal, 0)
        
        # Output
        result['signal'] = entry_signal.astype(int)
        result['signal_strength'] = signal_strength
        result['exit_signal'] = exit_signal.astype(int)
        
        # Clean up temp columns
        if '_vwap' in result.columns:
            result = result.drop(columns=['_vwap'])
        
        return result


class TickImbalanceScalping(RuleBasedStrategy):
    """
    Tick Imbalance Scalping strategy.
    
    Uses upticks/downticks imbalance to detect short-term momentum
    and scalp quick moves.
    
    Entry when:
    - Strong tick imbalance (ratio > threshold)
    - Recent price move confirms direction
    - Volume above average
    
    Very short holding: 2-4 bars (10-20 min)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        self.tick_imbalance_threshold = self.config.get('tick_imbalance_threshold', 0.65)
        self.price_confirm_bars = self.config.get('price_confirm_bars', 2)
        self.target_bars = self.config.get('target_bars', 3)
        self.min_bars_to_close = 3
    
    def get_name(self) -> str:
        return "TickImbalanceScalping"
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_position: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate tick imbalance signals."""
        result = features.copy()
        
        # Need tick ratio
        if 'tick_ratio' not in result.columns and 'tick_ratio_ma_6' not in result.columns:
            result['signal'] = 0
            result['signal_strength'] = 0
            result['exit_signal'] = 0
            return result
        
        tick_col = 'tick_ratio' if 'tick_ratio' in result.columns else 'tick_ratio_ma_6'
        
        # Entry: Strong buying pressure (high tick ratio)
        strong_buying = result[tick_col] > self.tick_imbalance_threshold
        
        # Confirm with price movement
        if 'return_1bar' in result.columns:
            price_up = result['return_1bar'].rolling(self.price_confirm_bars).sum() > 0
        else:
            price_up = result['close'].pct_change(self.price_confirm_bars) > 0
        
        entry_signal = strong_buying & price_up
        entry_signal = self.apply_entry_rules(result, entry_signal)
        
        # Exit: Tick ratio normalizes or reverses
        tick_normal = result[tick_col] < 0.5
        
        exit_signal = tick_normal
        
        # Signal strength
        signal_strength = ((result[tick_col] - 0.5) / 0.3).clip(0, 1)
        signal_strength = signal_strength.where(entry_signal, 0)
        
        result['signal'] = entry_signal.astype(int)
        result['signal_strength'] = signal_strength
        result['exit_signal'] = exit_signal.astype(int)
        
        return result


if __name__ == "__main__":
    print("Testing VWAP Scalping Strategy...")
    
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2023-12-20 09:35', periods=n, freq='5min')
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    
    # Create VWAP that price oscillates around
    vwap = pd.Series(close).rolling(20).mean()
    
    sample_df = pd.DataFrame({
        'open': close - np.random.rand(n) * 0.05,
        'high': close + np.abs(np.random.randn(n) * 0.1),
        'low': close - np.abs(np.random.randn(n) * 0.1),
        'close': close,
        'volume': np.random.randint(1000, 100000, n),
        'vwap_12': vwap,
        'tick_ratio': 0.5 + np.random.randn(n) * 0.1,
        'vol_ratio_6': 1 + np.abs(np.random.randn(n) * 0.5),
        'bar_index': [(i % 78) + 1 for i in range(n)],
    }, index=dates)
    
    # Make tick_ratio more extreme at some points
    sample_df.loc[sample_df.index[20:25], 'tick_ratio'] = 0.7
    sample_df.loc[sample_df.index[20:25], 'close'] = sample_df['vwap_12'].iloc[20:25] * 0.995
    
    strategy = VWAPScalpingStrategy()
    signals = strategy.generate_signals(sample_df)
    
    print(f"\nVWAP Scalping signals:")
    print(f"  Entry signals: {signals['signal'].sum()}")
    print(f"  Exit signals: {signals['exit_signal'].sum()}")
    
    # Test Tick Imbalance
    strategy2 = TickImbalanceScalping()
    signals2 = strategy2.generate_signals(sample_df)
    
    print(f"\nTick Imbalance signals:")
    print(f"  Entry signals: {signals2['signal'].sum()}")
    
    print("\nScalping strategy test completed!")
