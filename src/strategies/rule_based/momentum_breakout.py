"""
Momentum Breakout Strategy for Intraday Trading.

A trend-following strategy that enters long positions when:
- Price breaks above Bollinger Bands (upper band breakout)
- Volume confirms the move (above average)
- RSI shows momentum but not extreme overbought

Exits when:
- Price falls back to middle Bollinger Band
- RSI becomes extreme overbought (>70)
- Time limit reached
- Forced close at 15:55
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_strategy import RuleBasedStrategy, Signal


class MomentumBreakoutStrategy(RuleBasedStrategy):
    """
    Trend-following momentum breakout strategy.
    
    Parameters (configurable):
    - bb_threshold: How far above upper BB to trigger (default: 0.0)
    - volume_multiplier: Volume must be > MA * multiplier (default: 1.2)
    - rsi_low: Minimum RSI for entry (default: 40)
    - rsi_high: Maximum RSI for entry (default: 70)
    - exit_rsi: Exit when RSI exceeds this (default: 75)
    - atr_filter: Minimum ATR for volatility filter (default: None)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # Strategy parameters with defaults
        self.bb_threshold = self.config.get('bb_threshold', 0.0)
        self.volume_multiplier = self.config.get('volume_multiplier', 1.2)
        self.rsi_low = self.config.get('rsi_low', 40)
        self.rsi_high = self.config.get('rsi_high', 70)
        self.exit_rsi = self.config.get('exit_rsi', 75)
        self.atr_filter = self.config.get('atr_filter', None)
        
        # Holding period config
        self.max_holding_bars = self.config.get('max_holding_bars', 24)  # 2 hours max
    
    def get_name(self) -> str:
        return "MomentumBreakout"
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_position: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate momentum breakout signals.
        
        Args:
            features: DataFrame with required indicators:
                - close, bb_upper, bb_middle, bb_position
                - rsi_12 (or similar RSI)
                - vol_ratio_12 (or similar volume ratio)
                - bar_index (for time filtering)
            current_position: Currently held instrument
            
        Returns:
            DataFrame with signal columns
        """
        result = features.copy()
        
        # Check for required columns
        required = ['close', 'bb_upper', 'bb_middle', 'bb_position']
        missing = [c for c in required if c not in result.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Get RSI column (try different periods)
        rsi_col = None
        for period in [12, 6, 24]:
            if f'rsi_{period}' in result.columns:
                rsi_col = f'rsi_{period}'
                break
        
        if rsi_col is None:
            # Create a simple RSI if not available
            result['_rsi'] = 50  # Neutral
            rsi_col = '_rsi'
        
        # Get volume ratio column
        vol_col = None
        for period in [12, 6, 24]:
            if f'vol_ratio_{period}' in result.columns:
                vol_col = f'vol_ratio_{period}'
                break
        
        if vol_col is None:
            result['_vol_ratio'] = 1.0
            vol_col = '_vol_ratio'
        
        # Entry conditions
        # 1. Price above upper Bollinger Band
        bb_breakout = result['close'] > result['bb_upper'] * (1 + self.bb_threshold)
        
        # 2. Volume confirmation
        volume_confirm = result[vol_col] > self.volume_multiplier
        
        # 3. RSI in acceptable range (not extreme)
        rsi_ok = (result[rsi_col] > self.rsi_low) & (result[rsi_col] < self.rsi_high)
        
        # 4. Optional ATR filter
        if self.atr_filter is not None and 'atr_12' in result.columns:
            atr_ok = result['atr_12'] > self.atr_filter
        else:
            atr_ok = True
        
        # Combine entry conditions
        entry_signal = bb_breakout & volume_confirm & rsi_ok & atr_ok
        
        # Apply time filter (no entries too close to EOD)
        entry_signal = self.apply_entry_rules(result, entry_signal)
        
        # Exit conditions (for tracking, backtest will handle actual exits)
        # 1. Price falls to middle BB
        bb_exit = result['close'] < result['bb_middle']
        
        # 2. RSI extreme
        rsi_exit = result[rsi_col] > self.exit_rsi
        
        # Combine exit conditions
        exit_signal = bb_exit | rsi_exit
        
        # Calculate signal strength (0-1)
        # Based on how far above BB and volume strength
        bb_distance = (result['close'] / result['bb_upper'] - 1).clip(0, 0.05) / 0.05
        vol_strength = ((result[vol_col] - 1) / 2).clip(0, 1)
        rsi_strength = ((result[rsi_col] - self.rsi_low) / (self.rsi_high - self.rsi_low)).clip(0, 1)
        
        signal_strength = (bb_distance * 0.4 + vol_strength * 0.3 + rsi_strength * 0.3)
        signal_strength = signal_strength.where(entry_signal, 0)
        
        # Output columns
        result['signal'] = entry_signal.astype(int)
        result['signal_strength'] = signal_strength
        result['exit_signal'] = exit_signal.astype(int)
        
        # Clean up temp columns
        if '_rsi' in result.columns:
            result = result.drop(columns=['_rsi'])
        if '_vol_ratio' in result.columns:
            result = result.drop(columns=['_vol_ratio'])
        
        return result
    
    def generate_signals_multi_instrument(
        self,
        features_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate signals for multiple instruments and select best.
        
        Args:
            features_dict: Dict of instrument -> features DataFrame
            
        Returns:
            DataFrame with selected instrument and signal
        """
        signals = {}
        strengths = {}
        
        for instrument, features in features_dict.items():
            result = self.generate_signals(features)
            signals[instrument] = result['signal']
            strengths[instrument] = result['signal_strength']
        
        # Combine into single DataFrame
        signal_df = pd.DataFrame(signals)
        strength_df = pd.DataFrame(strengths)
        
        # Select best instrument at each time
        best_instrument = strength_df.idxmax(axis=1)
        best_strength = strength_df.max(axis=1)
        best_signal = signal_df.max(axis=1)  # 1 if any instrument has signal
        
        # If no signal, select CASH
        best_instrument = best_instrument.where(best_signal > 0, 'CASH')
        best_strength = best_strength.where(best_signal > 0, 0)
        
        result = pd.DataFrame({
            'selected_instrument': best_instrument,
            'signal': best_signal,
            'signal_strength': best_strength,
        }, index=signal_df.index)
        
        return result


if __name__ == "__main__":
    # Test the momentum breakout strategy
    print("Testing MomentumBreakoutStrategy...")
    
    # Create sample data
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2023-12-20 09:35', periods=n, freq='5min')
    close = 100 + np.cumsum(np.random.randn(n) * 0.2)
    
    # Create Bollinger Bands
    sma = pd.Series(close).rolling(20).mean()
    std = pd.Series(close).rolling(20).std()
    
    sample_df = pd.DataFrame({
        'close': close,
        'bb_upper': sma + 2 * std,
        'bb_middle': sma,
        'bb_lower': sma - 2 * std,
        'bb_position': (close - (sma - 2 * std)) / (4 * std),
        'rsi_12': 50 + np.random.randn(n) * 15,
        'vol_ratio_12': 1 + np.abs(np.random.randn(n) * 0.5),
        'bar_index': [(i % 78) + 1 for i in range(n)],
    }, index=dates)
    
    print(f"\nSample data shape: {sample_df.shape}")
    
    # Test strategy
    strategy = MomentumBreakoutStrategy()
    signals = strategy.generate_signals(sample_df)
    
    print(f"\nSignals generated:")
    print(f"  Total bars: {len(signals)}")
    print(f"  Entry signals: {signals['signal'].sum()}")
    print(f"  Exit signals: {signals['exit_signal'].sum()}")
    
    # Show bars with signals
    signal_bars = signals[signals['signal'] == 1]
    if len(signal_bars) > 0:
        print(f"\n  Bars with entry signals:")
        print(signal_bars[['close', 'bb_upper', 'rsi_12', 'signal_strength']].head())
    else:
        print("\n  No entry signals in sample (tighten parameters or check data)")
    
    print("\nMomentum breakout test completed!")
