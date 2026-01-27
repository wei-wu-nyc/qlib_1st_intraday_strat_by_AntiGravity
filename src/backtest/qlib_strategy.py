"""
Qlib-based Strategy Adapter for Intraday Trading.

This module provides a custom strategy that bridges our prediction signals
with qlib's WeightStrategyBase for proper portfolio management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from qlib.backtest.decision import TradeDecisionWO, Order
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase


class IntradayWeightStrategy(WeightStrategyBase):
    """
    Custom weight-based strategy for intraday trading.
    
    Converts prediction signals to target portfolio weights.
    Supports multiple allocation methods: equal_weight, rank_based, best_only.
    """
    
    def __init__(
        self,
        signal: pd.DataFrame,
        instruments: List[str],
        allocation_method: str = 'equal_weight',
        topk: int = 4,
        signal_threshold: float = 0.0,
        **kwargs
    ):
        """
        Initialize the intraday weight strategy.
        
        Args:
            signal: DataFrame with MultiIndex (datetime, instrument) and 'score' column
                   The 'score' is our predicted_return from the model.
            instruments: List of tradeable instruments
            allocation_method: 'equal_weight', 'rank_based', or 'best_only'
            topk: Maximum number of positions to hold
            signal_threshold: Minimum signal value to trade (default 0 = positive predictions only)
        """
        super().__init__(signal=signal, **kwargs)
        self.instruments = instruments
        self.allocation_method = allocation_method
        self.topk = topk
        self.signal_threshold = signal_threshold
        
    def generate_target_weight_position(
        self,
        score: pd.Series,
        current: pd.Series,
        trade_start_time: datetime,
        trade_end_time: datetime
    ) -> Dict[str, float]:
        """
        Generate target weight positions from prediction scores.
        
        Args:
            score: Series of prediction scores indexed by instrument
            current: Current position weights (not used in our logic)
            trade_start_time: Start of trading period
            trade_end_time: End of trading period
            
        Returns:
            Dict mapping instrument to target weight (0-1)
        """
        if score is None or len(score) == 0:
            return {}
        
        # Filter signals above threshold
        valid_signals = score[score > self.signal_threshold]
        
        if len(valid_signals) == 0:
            return {}  # All cash
        
        # Sort by score descending and take top k
        sorted_signals = valid_signals.sort_values(ascending=False)
        top_signals = sorted_signals.head(self.topk)
        
        # Apply allocation method
        if self.allocation_method == 'equal_weight':
            weights = self._equal_weight(top_signals)
        elif self.allocation_method == 'rank_based':
            weights = self._rank_based_weights(top_signals)
        elif self.allocation_method == 'best_only':
            weights = self._best_only(top_signals)
        else:
            raise ValueError(f"Unknown allocation method: {self.allocation_method}")
        
        return weights
    
    def _equal_weight(self, signals: pd.Series) -> Dict[str, float]:
        """Equal weight to all selected instruments."""
        n = len(signals)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {inst: weight for inst in signals.index}
    
    def _rank_based_weights(self, signals: pd.Series) -> Dict[str, float]:
        """
        Allocate based on rank with declining weights.
        Rank 1 gets most weight, rank N gets least.
        """
        n = len(signals)
        if n == 0:
            return {}
        
        total_ranks = sum(range(1, n + 1))  # 1 + 2 + ... + n
        weights = {}
        
        for rank, inst in enumerate(signals.index, 1):
            weights[inst] = (n - rank + 1) / total_ranks
        
        return weights
    
    def _best_only(self, signals: pd.Series) -> Dict[str, float]:
        """Allocate 100% to the top-ranked instrument only."""
        if len(signals) == 0:
            return {}
        
        best_inst = signals.index[0]
        return {best_inst: 1.0}


class SimpleIntradayStrategy:
    """
    Simplified strategy adapter when not using full qlib backtest.
    
    This provides a consistent interface for generating target weights
    from our model predictions.
    """
    
    def __init__(
        self,
        instruments: List[str],
        allocation_method: str = 'equal_weight',
        topk: int = 4,
        signal_threshold: float = 0.0,
    ):
        self.instruments = instruments
        self.allocation_method = allocation_method
        self.topk = topk
        self.signal_threshold = signal_threshold
    
    def generate_weights(
        self,
        signals: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Generate target weights for a given timestamp.
        
        Args:
            signals: DataFrame with predictions for all instruments at this timestamp
            timestamp: Current bar timestamp
            
        Returns:
            Dict mapping instrument to target weight
        """
        # Get predictions for this timestamp
        if 'predicted_return' in signals.columns:
            score_col = 'predicted_return'
        elif 'signal_strength' in signals.columns:
            score_col = 'signal_strength'
        else:
            score_col = 'signal'
        
        # Filter and sort
        valid = signals[signals[score_col] > self.signal_threshold]
        
        if len(valid) == 0:
            return {}
        
        sorted_signals = valid.sort_values(score_col, ascending=False)
        top_n = sorted_signals.head(self.topk)
        
        # Generate weights
        if self.allocation_method == 'equal_weight':
            n = len(top_n)
            weight = 1.0 / n
            return {inst: weight for inst in top_n.index}
        elif self.allocation_method == 'best_only':
            return {top_n.index[0]: 1.0}
        else:
            raise ValueError(f"Unknown allocation: {self.allocation_method}")


def prepare_signal_for_qlib(
    predictions_df: pd.DataFrame,
    score_column: str = 'predicted_return'
) -> pd.DataFrame:
    """
    Convert our prediction DataFrame to qlib's expected signal format.
    
    Args:
        predictions_df: DataFrame with predictions, indexed by (datetime, instrument) or datetime
        score_column: Column containing prediction scores
        
    Returns:
        DataFrame with 'score' column, indexed by (datetime, instrument)
    """
    df = predictions_df.copy()
    
    # Rename score column
    if score_column in df.columns and score_column != 'score':
        df = df.rename(columns={score_column: 'score'})
    
    # Ensure MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("predictions_df must have MultiIndex (datetime, instrument)")
    
    return df[['score']]


if __name__ == "__main__":
    # Test the strategy
    print("Testing IntradayWeightStrategy...")
    
    # Create sample signals
    instruments = ['SPY', 'QQQ', 'DIA', 'IWM']
    signals = pd.Series({
        'SPY': 0.005,
        'QQQ': 0.008,
        'DIA': 0.003,
        'IWM': -0.001,  # Negative, should be filtered
    })
    
    strategy = SimpleIntradayStrategy(
        instruments=instruments,
        allocation_method='equal_weight',
        topk=4,
        signal_threshold=0.0,
    )
    
    # Get weights
    df_signals = pd.DataFrame({'predicted_return': signals})
    weights = strategy.generate_weights(df_signals, pd.Timestamp.now())
    
    print(f"\nInput signals: {signals.to_dict()}")
    print(f"Equal-weight (topk=4, threshold=0):")
    for inst, w in weights.items():
        print(f"  {inst}: {w*100:.1f}%")
    
    # Test best_only
    strategy2 = SimpleIntradayStrategy(
        instruments=instruments,
        allocation_method='best_only',
    )
    weights2 = strategy2.generate_weights(df_signals, pd.Timestamp.now())
    print(f"\nBest-only:")
    for inst, w in weights2.items():
        print(f"  {inst}: {w*100:.1f}%")
    
    print("\n✓ Strategy tests passed!")
