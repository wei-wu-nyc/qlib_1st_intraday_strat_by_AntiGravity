import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import json
from datetime import time

from src.strategies.base_strategy import BaseStrategy

class MoEIntradayStrategy(BaseStrategy):
    """
    Mixture of Experts (MoE) Strategy.
    Switches between different models based on time of day.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models: Dict[time, Any] = {}
        self.manifest: Dict[str, str] = {}
        self.feature_names = []
        
    def load_model(self, model_dir: str):
        """
        Load MoE models from directory using manifest.json.
        """
        path = Path(model_dir)
        manifest_path = path / 'manifest.json'
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found in {model_dir}")
            
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        print(f"Loading {len(self.manifest)} MoE models...")
        
        first_model = None
        for time_str, filename in self.manifest.items():
            hour, minute = map(int, time_str.split(':'))
            key = time(hour, minute)
            
            model_path = path / filename
            model = joblib.load(model_path)
            self.models[key] = model
            
            if first_model is None:
                first_model = model
        
        # Try to identify feature names
        if hasattr(first_model, 'feature_name_'):
            self.feature_names = first_model.feature_name_
        elif hasattr(first_model, 'feature_names_in_'):
            self.feature_names = first_model.feature_names_in_
            
    def get_name(self) -> str:
        return "MoEIntradayStrategy"
            
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using the appropriate model for each timestamp.
        """
        if not self.models:
            raise ValueError("Models not loaded! Call load_model() first.")
            
        signals = df.copy()
        signals['predicted_return'] = np.nan
        
        # Sort keys for efficient lookup
        sorted_times = sorted(self.models.keys())
        
        # Extract times
        times = df.index.get_level_values(0).time
        unique_times = sorted(list(set(times)))
        
        print("Predicting with MoE models...")
        
        # Define ignore columns for fallback
        ignore_cols = [
            'open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 
            'bar_index', 'datetime', 'instrument', 'date', 'time',
            'ret_24bar', 'ret_12bar', 'ret_6bar', 'ret_eod' # labels
        ]
        
        for t in unique_times:
            # Find the active model: largest key <= t
            active_key = None
            for key in reversed(sorted_times):
                if t >= key:
                    active_key = key
                    break
            
            if active_key is None:
                continue
                
            model = self.models[active_key]
            
            # Select rows for this time
            mask = (times == t)
            if not mask.any():
                continue
                
            batch_df = df[mask]
            
            
            # Prepare features
            if len(self.feature_names) > 0:
                # Use explicit features if available
                # Handle potential mismatch if some columns missing?
                available_feats = [f for f in self.feature_names if f in batch_df.columns]
                # If significant mismatch, might warn...
                X_batch = batch_df[available_feats] 
                # Note: If XGBoost expects exact columns, passing subset might fail if mismatch is large
                # But typically feature_names comes from the model itself
            else:
                # Fallback: Drop known metadata/label columns to leave only features
                # This ensures we don't pass 'date'/'time' objects to XGBoost
                feature_cols = [c for c in batch_df.columns if c not in ignore_cols and not c.startswith('ret_')]
                # Also ensure numeric and CLEAN data (remove Inf/NaN)
                X_batch = batch_df[feature_cols].select_dtypes(include=[np.number])
            
            # Critical: Clean Infs/NaNs for XGB/RF
            X_batch = X_batch.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Predict
            try:
                # Some models (XGB) might need .values or strictly matching cols
                pred = model.predict(X_batch)
                signals.loc[mask, 'predicted_return'] = pred
            except Exception as e:
                print(f"Error predicting at {t}: {e}")
                # Try passing values if dataframe failed (sometimes fixes mismatch name issues)
                try:
                    pred = model.predict(X_batch.values)
                    signals.loc[mask, 'predicted_return'] = pred
                except:
                    pass

        # Generate binary signal
        signals['signal'] = (signals['predicted_return'] > 0).astype(int)
        
        # Add metadata needed for backtest
        signals['bar_index'] = df['bar_index'] if 'bar_index' in df else 0
        
        # Fill NaN
        signals['predicted_return'] = signals['predicted_return'].fillna(0)
        signals['signal'] = signals['signal'].fillna(0)
        
        return signals
