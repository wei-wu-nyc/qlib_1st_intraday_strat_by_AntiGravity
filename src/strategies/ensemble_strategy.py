import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from .base_strategy import BaseStrategy

class EnsembleStrategy(BaseStrategy):
    """
    Ensemble Strategy that averages predictions from multiple base strategies.
    
    It assumes:
    1. All base strategies are already loaded/fitted.
    2. All base strategies implement `generate_signals(df)`.
    3. The `generate_signals` output contains a `predicted_return` column.
    """
    
    def __init__(self, strategies: List[BaseStrategy], weights: Optional[List[float]] = None, config: Optional[Dict] = None):
        # Set strategies first because get_name() (called by super) needs them
        self.strategies = strategies
        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError("Weights must match number of strategies")
            self.weights = weights
            
        super().__init__(config)
            
    def get_name(self) -> str:
        names = [s.get_name() for s in self.strategies]
        return f"Ensemble({'+'.join(names)})"

    def fit(self, *args, **kwargs):
        raise NotImplementedError("EnsembleStrategy does not support fitting. Train base models individually.")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        predictions = []
        
        # Collect predictions from each strategy
        for i, strategy in enumerate(self.strategies):
            signals = strategy.generate_signals(df)
            
            if 'predicted_return' not in signals.columns:
                raise ValueError(f"Strategy {strategy.get_name()} did not return 'predicted_return' column")
            
            # Use pandas concat to align indices properly
            pred = signals['predicted_return'] * self.weights[i]
            predictions.append(pred)
            
        # Sum weighted predictions
        avg_pred = pd.concat(predictions, axis=1).sum(axis=1)
        
        # Construct result dataframe
        result = df.copy()
        result['predicted_return'] = avg_pred
        
        # Minimum return threshold
        min_ret = self.config.get('min_pred_return', 0.0)
        
        result['signal'] = (avg_pred > min_ret).astype(int)
        
        # Signal strength
        result['signal_strength'] = np.abs(avg_pred).clip(0, 0.02) / 0.02
        result['signal_strength'] = result['signal_strength'].where(result['signal'] == 1, 0)
        
        # Add metadata needed for backtest
        result['bar_index'] = df['bar_index'] if 'bar_index' in df else 0
        
        return result
