"""
Mean Reversion Strategy for Intraday Trading.

A counter-trend strategy that enters long positions when:
- Price is oversold (RSI low, price at lower Bollinger Band)
- Expects reversion to mean

Exits when:
- Price reaches middle Bollinger Band
- RSI normalizes (>50)
- Time limit reached
- Forced close at 15:55
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_strategy import RuleBasedStrategy


class MeanReversionStrategy(RuleBasedStrategy):
    """
    Mean reversion strategy for oversold conditions.
    
    Parameters:
    - rsi_oversold: RSI threshold for oversold (default: 30)
    - rsi_exit: RSI threshold for exit (default: 50)
    - bb_threshold: How far below lower BB for entry (default: 0.0)
    - min_bars_for_entry: Don't enter in last N bars before close
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_exit = self.config.get('rsi_exit', 50)
        self.bb_threshold = self.config.get('bb_threshold', 0.0)
        
        # For mean reversion, require more time to close
        self.min_bars_to_close = self.config.get('min_bars_to_close', 12)  # 1 hour
    
    def get_name(self) -> str:
        return "MeanReversion"
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_position: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate mean reversion signals."""
        result = features.copy()
        
        # Get RSI column
        rsi_col = None
        for period in [12, 6, 24]:
            if f'rsi_{period}' in result.columns:
                rsi_col = f'rsi_{period}'
                break
        
        if rsi_col is None:
            result['_rsi'] = 50
            rsi_col = '_rsi'
        
        # Entry conditions
        # 1. RSI oversold
        rsi_oversold = result[rsi_col] < self.rsi_oversold
        
        # 2. Price at or below lower Bollinger Band
        if 'bb_lower' in result.columns:
            bb_oversold = result['close'] <= result['bb_lower'] * (1 + self.bb_threshold)
        elif 'bb_position' in result.columns:
            bb_oversold = result['bb_position'] < 0.1  # Bottom 10% of band
        else:
            bb_oversold = True  # No BB filter
        
        # 3. Volume not extremely low (some activity)
        vol_col = None
        for period in [12, 6, 24]:
            if f'vol_ratio_{period}' in result.columns:
                vol_col = f'vol_ratio_{period}'
                break
        
        if vol_col is not None:
            vol_ok = result[vol_col] > 0.5  # At least half of average
        else:
            vol_ok = True
        
        # Combine entry conditions
        entry_signal = rsi_oversold & bb_oversold & vol_ok
        entry_signal = self.apply_entry_rules(result, entry_signal)
        
        # Exit conditions
        # 1. RSI normalized
        rsi_exit = result[rsi_col] > self.rsi_exit
        
        # 2. Price reached middle BB
        if 'bb_middle' in result.columns:
            bb_exit = result['close'] >= result['bb_middle']
        elif 'bb_position' in result.columns:
            bb_exit = result['bb_position'] >= 0.5
        else:
            bb_exit = False
        
        exit_signal = rsi_exit | bb_exit
        
        # Signal strength (how oversold)
        rsi_strength = ((self.rsi_oversold - result[rsi_col]) / self.rsi_oversold).clip(0, 1)
        if 'bb_position' in result.columns:
            bb_strength = (0.2 - result['bb_position'].clip(0, 0.2)) / 0.2
        else:
            bb_strength = 0.5
        
        signal_strength = (rsi_strength * 0.6 + bb_strength * 0.4)
        signal_strength = signal_strength.where(entry_signal, 0)
        
        # Output
        result['signal'] = entry_signal.astype(int)
        result['signal_strength'] = signal_strength
        result['exit_signal'] = exit_signal.astype(int)
        
        if '_rsi' in result.columns:
            result = result.drop(columns=['_rsi'])
        
        return result


if __name__ == "__main__":
    print("Testing MeanReversionStrategy...")
    
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2023-12-20 09:35', periods=n, freq='5min')
    close = 100 + np.cumsum(np.random.randn(n) * 0.2)
    
    sma = pd.Series(close).rolling(20).mean()
    std = pd.Series(close).rolling(20).std()
    
    sample_df = pd.DataFrame({
        'close': close,
        'bb_upper': sma + 2 * std,
        'bb_middle': sma,
        'bb_lower': sma - 2 * std,
        'bb_position': (close - (sma - 2 * std)) / (4 * std),
        'rsi_12': 30 + np.random.randn(n) * 20,  # Lower RSI for testing
        'vol_ratio_12': 1 + np.random.randn(n) * 0.3,
        'bar_index': [(i % 78) + 1 for i in range(n)],
    }, index=dates)
    
    strategy = MeanReversionStrategy()
    signals = strategy.generate_signals(sample_df)
    
    print(f"\nSignals generated:")
    print(f"  Entry signals: {signals['signal'].sum()}")
    print(f"  Exit signals: {signals['exit_signal'].sum()}")
    
    print("\nMean reversion test completed!")
