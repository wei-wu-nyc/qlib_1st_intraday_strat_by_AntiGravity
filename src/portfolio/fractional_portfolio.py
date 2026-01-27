"""
Fractional Portfolio Allocation for Intraday Trading.

Allocates capital across multiple ETFs based on signal strength/ranking
instead of all-in on a single ETF.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioWeights:
    """Container for portfolio weights at a point in time."""
    timestamp: pd.Timestamp
    weights: Dict[str, float]  # instrument -> weight (0-1)
    cash_weight: float
    
    def to_dict(self) -> Dict[str, float]:
        result = self.weights.copy()
        result['CASH'] = self.cash_weight
        return result


class FractionalPortfolio:
    """
    Portfolio manager that allocates fractionally across multiple ETFs.
    
    Allocation methods:
    - rank_based: Allocate based on signal rank
    - signal_weighted: Allocate proportionally to signal strength
    - equal_weight: Equal weight to all positive signals
    """
    
    def __init__(
        self,
        instruments: List[str],
        allocation_method: str = 'rank_based',
        max_positions: int = 4,
        min_weight: float = 0.0,
        signal_threshold: float = 0.0,
    ):
        """
        Initialize fractional portfolio.
        
        Args:
            instruments: List of tradeable instruments
            allocation_method: 'rank_based', 'signal_weighted', or 'equal_weight'
            max_positions: Maximum number of positions (excluding cash)
            min_weight: Minimum weight per position (0-1)
            signal_threshold: Minimum signal to be considered
        """
        self.instruments = instruments
        self.allocation_method = allocation_method
        self.max_positions = max_positions
        self.min_weight = min_weight
        self.signal_threshold = signal_threshold
    
    def calculate_weights(
        self,
        signals: Dict[str, float],
    ) -> PortfolioWeights:
        """
        Calculate portfolio weights from signals.
        
        Args:
            signals: Dict of instrument -> signal strength
            
        Returns:
            PortfolioWeights with allocation
        """
        # Filter signals above threshold
        valid_signals = {
            inst: sig for inst, sig in signals.items()
            if sig > self.signal_threshold and inst in self.instruments
        }
        
        if not valid_signals:
            # All cash if no valid signals
            return PortfolioWeights(
                timestamp=pd.Timestamp.now(),
                weights={inst: 0.0 for inst in self.instruments},
                cash_weight=1.0
            )
        
        # Select top N instruments
        sorted_signals = sorted(valid_signals.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_signals[:self.max_positions]
        
        if self.allocation_method == 'rank_based':
            weights = self._rank_based_weights(top_n)
        elif self.allocation_method == 'signal_weighted':
            weights = self._signal_weighted(top_n)
        elif self.allocation_method == 'equal_weight':
            weights = self._equal_weight(top_n)
        elif self.allocation_method == 'best_only':
            weights = self._best_only(top_n)
        else:
            raise ValueError(f"Unknown allocation method: {self.allocation_method}")
        
        # Apply minimum weight constraint
        weights = self._apply_min_weight(weights)
        
        # Calculate cash weight
        total_weight = sum(weights.values())
        cash_weight = max(0, 1 - total_weight)
        
        return PortfolioWeights(
            timestamp=pd.Timestamp.now(),
            weights=weights,
            cash_weight=cash_weight
        )
    
    def _rank_based_weights(
        self,
        ranked_signals: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        """
        Allocate based on rank with declining weights.
        
        Rank 1 gets most weight, rank N gets least.
        Uses formula: weight = (N - rank + 1) / sum(1..N)
        """
        n = len(ranked_signals)
        if n == 0:
            return {}
        
        total_ranks = sum(range(1, n + 1))  # 1 + 2 + ... + n
        weights = {}
        
        for rank, (inst, sig) in enumerate(ranked_signals, 1):
            # Higher rank (lower number) gets more weight
            weights[inst] = (n - rank + 1) / total_ranks
        
        return weights
    
    def _signal_weighted(
        self,
        ranked_signals: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        """Allocate proportionally to signal strength."""
        if not ranked_signals:
            return {}
        
        total_signal = sum(sig for _, sig in ranked_signals)
        if total_signal <= 0:
            return {}
        
        return {inst: sig / total_signal for inst, sig in ranked_signals}
    
    def _equal_weight(
        self,
        ranked_signals: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        """Equal weight to all selected instruments."""
        if not ranked_signals:
            return {}
        
        n = len(ranked_signals)
        weight = 1.0 / n
        return {inst: weight for inst, _ in ranked_signals}
    
    def _best_only(
        self,
        ranked_signals: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        """Allocate 100% to the top-ranked instrument only."""
        if not ranked_signals:
            return {}
        
        # Take only the first (highest) signal
        best_inst, _ = ranked_signals[0]
        return {best_inst: 1.0}
    
    def _apply_min_weight(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum weight constraint, redistributing as needed."""
        if self.min_weight <= 0:
            return weights
        
        # Filter out positions below minimum
        valid = {inst: w for inst, w in weights.items() if w >= self.min_weight}
        
        if not valid:
            return {}
        
        # Renormalize
        total = sum(valid.values())
        if total > 0:
            return {inst: w / total for inst, w in valid.items()}
        return valid
    
    def generate_weights_for_dataframe(
        self,
        signals_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate weights for all timestamps in a signals DataFrame.
        
        Args:
            signals_df: DataFrame with columns for each instrument's signal
                        e.g., 'signal_SPY', 'signal_DIA', etc.
                        
        Returns:
            DataFrame with weight columns: 'weight_SPY', 'weight_DIA', etc.
        """
        # Identify signal columns
        signal_cols = [c for c in signals_df.columns if c.startswith('signal_')]
        
        weights_data = []
        
        for idx, row in signals_df.iterrows():
            signals = {
                col.replace('signal_', ''): row[col]
                for col in signal_cols
            }
            
            pw = self.calculate_weights(signals)
            
            row_weights = {'timestamp': idx}
            for inst in self.instruments:
                row_weights[f'weight_{inst}'] = pw.weights.get(inst, 0.0)
            row_weights['weight_CASH'] = pw.cash_weight
            
            weights_data.append(row_weights)
        
        return pd.DataFrame(weights_data).set_index('timestamp')


def combine_multi_instrument_signals(
    signals_dict: Dict[str, pd.DataFrame],
    signal_col: str = 'signal_strength',
) -> pd.DataFrame:
    """
    Combine signals from multiple instruments into a single DataFrame.
    
    Args:
        signals_dict: Dict of instrument -> signals DataFrame
        signal_col: Column name containing signal strength
        
    Returns:
        DataFrame with signal columns for each instrument
    """
    combined = None
    
    for inst, df in signals_dict.items():
        if signal_col in df.columns:
            sig_series = df[[signal_col]].rename(columns={signal_col: f'signal_{inst}'})
            
            if combined is None:
                combined = sig_series
            else:
                combined = combined.join(sig_series, how='outer')
    
    return combined.fillna(0)


if __name__ == "__main__":
    print("Testing FractionalPortfolio...")
    
    instruments = ['SPY', 'DIA', 'QQQ', 'IWM']
    
    # Test signals
    signals = {
        'SPY': 0.8,
        'DIA': 0.6,
        'QQQ': 0.9,
        'IWM': 0.4,
    }
    
    print(f"\nInput signals: {signals}")
    
    # Test rank-based allocation
    portfolio = FractionalPortfolio(
        instruments=instruments,
        allocation_method='rank_based',
        max_positions=4,
    )
    
    weights = portfolio.calculate_weights(signals)
    print(f"\nRank-based weights:")
    for inst, w in weights.weights.items():
        print(f"  {inst}: {w*100:.1f}%")
    print(f"  CASH: {weights.cash_weight*100:.1f}%")
    
    # Test signal-weighted
    portfolio2 = FractionalPortfolio(
        instruments=instruments,
        allocation_method='signal_weighted',
        max_positions=4,
    )
    
    weights2 = portfolio2.calculate_weights(signals)
    print(f"\nSignal-weighted:")
    for inst, w in weights2.weights.items():
        print(f"  {inst}: {w*100:.1f}%")
    
    # Test equal weight
    portfolio3 = FractionalPortfolio(
        instruments=instruments,
        allocation_method='equal_weight',
        max_positions=2,  # Only top 2
    )
    
    weights3 = portfolio3.calculate_weights(signals)
    print(f"\nEqual weight (top 2):")
    for inst, w in weights3.weights.items():
        if w > 0:
            print(f"  {inst}: {w*100:.1f}%")
    print(f"  CASH: {weights3.cash_weight*100:.1f}%")
    
    print("\nFractional portfolio test completed!")
