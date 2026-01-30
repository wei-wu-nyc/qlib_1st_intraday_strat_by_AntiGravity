"""
Random Forest Strategy for Intraday Trading.

ML-based strategy using Random Forest for return prediction.
Robust to overfitting and handles non-linear relationships well.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_strategy import MLStrategy

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit


class RandomForestIntradayStrategy(MLStrategy):
    """
    Random Forest-based intraday trading strategy.
    
    Parameters:
    - n_estimators: Number of trees (default: 100)
    - max_depth: Maximum tree depth (default: 12)
    - min_samples_leaf: Min samples per leaf (default: 20)
    - n_jobs: Number of parallel jobs (default: -1)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 12)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 20)
        self.random_state = self.config.get('random_state', 42)
        self.n_jobs = self.config.get('n_jobs', -1)
        
        self.min_pred_return = self.config.get('min_pred_return', 0.0)
        
        self.model = None
        self.feature_names = None
    
    def get_name(self) -> str:
        return "RandomForestIntraday"
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        sample_weight: Optional[pd.Series] = None,
        **kwargs
    ) -> 'RandomForestIntradayStrategy':
        """Fit Random Forest model."""
        self.feature_names = list(X.columns)
        
        # Handle missing and infinite values
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_clean = y.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Align weights if provided
        w_clean = None
        if sample_weight is not None:
            w_clean = sample_weight.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Filter out any remaining problematic rows
        valid_mask = np.isfinite(y_clean) & X_clean.apply(np.isfinite).all(axis=1)
        if w_clean is not None:
             valid_mask &= np.isfinite(w_clean)
        
        X_clean = X_clean[valid_mask]
        y_clean = y_clean[valid_mask]
        if w_clean is not None:
            w_clean = w_clean[valid_mask]
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0,
            **kwargs
        )
        
        # RF doesn't support eval_set/early_stopping natively in sklearn like LGB/XGB
        # so we just fit on train data
        self.model.fit(X_clean, y_clean, sample_weight=w_clean)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate return predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Clean data similar to fit
        if X.columns.duplicated().any():
            X = X.loc[:, ~X.columns.duplicated()].copy()
            
        X_cols = set(X.columns.tolist())
        X_clean = pd.DataFrame(0.0, index=X.index, columns=self.feature_names)
        
        for col in self.feature_names:
            if col in X_cols:
                X_clean[col] = X[col].values
        
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return self.model.predict(X_clean.values)
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_position: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate signals."""
        if features.columns.duplicated().any():
            features = features.loc[:, ~features.columns.duplicated()].copy()
            
        result = features.copy()
        
        feature_cols = [c for c in self.feature_names if c in features.columns]
        preds = self.predict(features[feature_cols])
        result['predicted_return'] = preds
        
        result['signal'] = (preds > self.min_pred_return).astype(int)
        result['signal_strength'] = np.abs(preds).clip(0, 0.02) / 0.02
        result['signal_strength'] = result['signal_strength'].where(result['signal'] == 1, 0)
        
        # Time filter
        if 'bar_index' in result.columns:
            last_entry_bar = 67 # 15:00
            bar_idx = result['bar_index']
            # Unwrap if dataframe
            if hasattr(bar_idx, 'iloc') and len(bar_idx.shape) > 1:
                bar_idx = bar_idx.iloc[:, 0]
            too_late = bar_idx > last_entry_bar
            result.loc[too_late, 'signal'] = 0
            result.loc[too_late, 'signal_strength'] = 0
            
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
    def save_model(self, path: str) -> None:
        """Save trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        import joblib
        from pathlib import Path as P
        
        save_path = P(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_path = str(save_path) + '_model.joblib'
        joblib.dump(self.model, model_path)
        
        metadata = {
            'feature_names': self.feature_names,
            'config': self.config
        }
        meta_path = str(save_path) + '_meta.joblib'
        joblib.dump(metadata, meta_path)
        print(f"Model saved to {model_path}")

    def load_model(self, path: str) -> 'RandomForestIntradayStrategy':
        """Load trained model."""
        import joblib
        
        model_path = str(path) + '_model.joblib'
        meta_path = str(path) + '_meta.joblib'
        
        self.model = joblib.load(model_path)
        metadata = joblib.load(meta_path)
        
        self.feature_names = metadata['feature_names']
        self.config = metadata.get('config', {})
        self.is_fitted = True
        
        print(f"Model loaded from {model_path}")
        return self

if __name__ == "__main__":
    print("Testing RandomForestIntradayStrategy...")
    np.random.seed(42)
    n = 100
    features = pd.DataFrame(np.random.randn(n, 5), columns=['f1','f2','f3','f4','f5'])
    target = features['f1'] * 0.5 + np.random.randn(n) * 0.1
    
    rf = RandomForestIntradayStrategy({'n_estimators': 10})
    rf.fit(features, target)
    print("Fit complete")
    preds = rf.predict(features)
    print(f"Preds shape: {preds.shape}")
