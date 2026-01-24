"""
XGBoost Strategy for Intraday Trading.

ML-based strategy using XGBoost for return prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_strategy import MLStrategy

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed")

from sklearn.model_selection import TimeSeriesSplit


class XGBoostIntradayStrategy(MLStrategy):
    """
    XGBoost-based intraday trading strategy.
    
    Uses XGBoost regressor to predict forward returns,
    then selects the best ETF based on predicted return.
    
    Parameters:
    - n_estimators: Number of boosting rounds (default: 500)
    - max_depth: Maximum tree depth (default: 6)
    - learning_rate: Learning rate (default: 0.05)
    - early_stopping_rounds: Early stopping (default: 50)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        if not HAS_XGB:
            raise ImportError("xgboost is required for XGBoostIntradayStrategy")
        
        self.n_estimators = self.config.get('n_estimators', 500)
        self.max_depth = self.config.get('max_depth', 6)
        self.learning_rate = self.config.get('learning_rate', 0.05)
        self.early_stopping_rounds = self.config.get('early_stopping_rounds', 50)
        self.subsample = self.config.get('subsample', 0.8)
        self.colsample_bytree = self.config.get('colsample_bytree', 0.8)
        self.random_state = self.config.get('random_state', 42)
        
        # Minimum predicted return to enter position
        self.min_pred_return = self.config.get('min_pred_return', 0.0)
        
        self.model = None
        self.feature_names = None
    
    def get_name(self) -> str:
        return "XGBoostIntraday"
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'XGBoostIntradayStrategy':
        """
        Fit XGBoost model.
        
        Args:
            X: Feature DataFrame
            y: Target Series (forward returns)
            eval_set: Optional validation set tuple
            **kwargs: Additional XGBoost parameters
        """
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle missing and infinite values
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_clean = y.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Filter out any remaining problematic rows
        valid_mask = np.isfinite(y_clean) & X_clean.apply(np.isfinite).all(axis=1)
        X_clean = X_clean[valid_mask]
        y_clean = y_clean[valid_mask]
        
        # Initialize model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            objective='reg:squarederror',
            n_jobs=-1,
            **kwargs
        )
        
        # Fit with optional early stopping
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_clean = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
            y_val_clean = y_val.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            self.model.fit(
                X_clean, y_clean,
                eval_set=[(X_val_clean, y_val_clean)],
                verbose=False
            )
        else:
            self.model.fit(X_clean, y_clean, verbose=False)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate return predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # First, remove any duplicate columns from input DataFrame
        # This is critical because pandas allows duplicate column names
        # but XGBoost cannot handle them
        if X.columns.duplicated().any():
            X = X.loc[:, ~X.columns.duplicated()].copy()
        
        # Get the set of available column names
        X_cols = set(X.columns.tolist())
        
        # Build a clean DataFrame with exactly the features needed
        # Start with zeros for all features
        X_clean = pd.DataFrame(0.0, index=X.index, columns=self.feature_names)
        
        # Copy over available features
        for col in self.feature_names:
            if col in X_cols:
                X_clean[col] = X[col].values
        
        # Replace inf values and NaN
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Convert to numpy array to avoid any pandas indexing issues
        return self.model.predict(X_clean.values)
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_position: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals from predictions.
        
        Args:
            features: DataFrame with features
            current_position: Currently held position
            
        Returns:
            DataFrame with signal columns
        """
        # First, deduplicate columns if any duplicates exist
        if features.columns.duplicated().any():
            features = features.loc[:, ~features.columns.duplicated()].copy()
        
        result = features.copy()
        
        # Get predictions - only pass feature columns
        feature_cols = [c for c in self.feature_names if c in features.columns]
        preds = self.predict(features[feature_cols])
        result['predicted_return'] = preds
        
        # Signal based on predicted return
        result['signal'] = (preds > self.min_pred_return).astype(int)
        
        # Signal strength based on predicted return magnitude
        result['signal_strength'] = np.abs(preds).clip(0, 0.02) / 0.02
        result['signal_strength'] = result['signal_strength'].where(result['signal'] == 1, 0)
        
        # Filter by time (no entries too close to close)
        if 'bar_index' in result.columns:
            last_entry_bar = self.position_close_bar - self.min_bars_to_close
            bar_idx = result['bar_index']
            # Handle case where bar_index might be a DataFrame (shouldn't happen now)
            if hasattr(bar_idx, 'iloc') and len(bar_idx.shape) > 1:
                bar_idx = bar_idx.iloc[:, 0]
            too_late = bar_idx > last_entry_bar
            result.loc[too_late, 'signal'] = 0
            result.loc[too_late, 'signal_strength'] = 0
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of CV splits
            
        Returns:
            Dictionary of metric lists
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metrics = {
            'train_rmse': [],
            'val_rmse': [],
            'train_corr': [],
            'val_corr': [],
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit on this fold
            self.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            # Predict
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)
            
            # Calculate metrics
            train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            train_corr = np.corrcoef(train_pred, y_train)[0, 1]
            val_corr = np.corrcoef(val_pred, y_val)[0, 1]
            
            metrics['train_rmse'].append(train_rmse)
            metrics['val_rmse'].append(val_rmse)
            metrics['train_corr'].append(train_corr)
            metrics['val_corr'].append(val_corr)
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Path to save model (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        import joblib
        from pathlib import Path as P
        
        save_path = P(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = str(save_path) + '_model.joblib'
        joblib.dump(self.model, model_path)
        
        # Save metadata (feature names, config)
        metadata = {
            'feature_names': self.feature_names,
            'config': self.config,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
        }
        meta_path = str(save_path) + '_meta.joblib'
        joblib.dump(metadata, meta_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, path: str) -> 'XGBoostIntradayStrategy':
        """
        Load trained model from disk.
        
        Args:
            path: Path to model (without extension)
            
        Returns:
            Self with loaded model
        """
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
    print("Testing XGBoostIntradayStrategy...")
    
    if not HAS_XGB:
        print("XGBoost not installed, skipping test")
        exit()
    
    np.random.seed(42)
    n = 1000
    
    # Create synthetic features
    features = pd.DataFrame({
        'return_1bar': np.random.randn(n) * 0.01,
        'rsi_12': 50 + np.random.randn(n) * 15,
        'bb_position': np.random.uniform(0, 1, n),
        'vol_ratio_12': 1 + np.random.randn(n) * 0.3,
        'time_of_day': np.tile(np.linspace(0, 1, 78), n // 78 + 1)[:n],
        'bar_index': np.tile(range(1, 79), n // 78 + 1)[:n],
    })
    
    # Create synthetic target (with some signal)
    target = (features['return_1bar'] * 0.3 + 
              (features['rsi_12'] - 50) / 500 +
              features['bb_position'] * 0.002 - 0.001 +
              np.random.randn(n) * 0.005)
    
    print(f"\nData shape: {features.shape}")
    
    # Train/test split
    split = int(n * 0.8)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = target[:split], target[split:]
    
    # Fit model
    strategy = XGBoostIntradayStrategy({'n_estimators': 100})
    feature_cols = ['return_1bar', 'rsi_12', 'bb_position', 'vol_ratio_12', 'time_of_day']
    
    strategy.fit(X_train[feature_cols], y_train)
    
    # Generate signals
    signals = strategy.generate_signals(features[feature_cols].join(features['bar_index']))
    
    print(f"\nSignals generated:")
    print(f"  Total signals: {signals['signal'].sum()}")
    print(f"  Signal rate: {signals['signal'].mean()*100:.1f}%")
    
    # Feature importance
    print(f"\nTop features:")
    importance = strategy.get_feature_importance()
    print(importance.head())
    
    print("\nXGBoost test completed!")
