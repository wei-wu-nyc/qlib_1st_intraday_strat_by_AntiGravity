
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import IntradayDataLoader
from src.strategies.ml_models.xgboost_intraday import XGBoostIntradayStrategy
# from config.config_loader import load_config # config handled by loader

def main():
    print("Debugging XGBoost Signals...")
    
    # Load data loader (handles config)
    loader = IntradayDataLoader('config/intraday_config.yaml')
    config = loader.config
    
    print("Loading test data...")
    # Load just one instrument to be fast
    # Using get_period_data for 'test' but filtering to short range
    # Or just use get_data_with_time_features
    test_data = loader.get_data_with_time_features(
        symbols=['SPY'],
        start_date='2022-01-01',
        end_date='2022-06-30' # Just 6 months
    )
    
    # Add alpha features
    from src.features.intraday_alpha import IntradayAlphaFeatures
    alpha_gen = IntradayAlphaFeatures('config/intraday_config.yaml')
    test_data = alpha_gen.generate_all_features(test_data)
    
    # Add seasonality
    from src.features.seasonality_features import SeasonalityFeatures
    season_gen = SeasonalityFeatures('config/intraday_config.yaml')
    test_data = season_gen.generate_all_features(test_data)
    
    print(f"Data shape: {test_data.shape}")
    
    # Load model
    model_path = project_root / 'results' / 'models' / 'xgboostintraday'
    print(f"Loading model from {model_path}")
    
    strategy = XGBoostIntradayStrategy(config)
    try:
        strategy.load_model(str(model_path))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Check feature mismatch
    missing_cols = [c for c in strategy.feature_names if c not in test_data.columns]
    if missing_cols:
        print(f"WARNING: {len(missing_cols)} features missing from data: {missing_cols[:5]}...")
    else:
        print("All features present.")

    # Generate signals
    print("Generating signals...")
    # Fix for duplicate columns if any (same logic as in strategy)
    if test_data.columns.duplicated().any():
        test_data = test_data.loc[:, ~test_data.columns.duplicated()]
        
    signals = strategy.generate_signals(test_data)
    
    # Analysis
    print("\n--- Analysis ---")
    
    if 'predicted_return' in signals.columns:
        preds = signals['predicted_return']
        print(f"Predicted Return Stats:")
        print(f"  Min:  {preds.min():.6f}")
        print(f"  Max:  {preds.max():.6f}")
        print(f"  Mean: {preds.mean():.6f}")
        print(f"  Std:  {preds.std():.6f}")
        print(f"  Zeros: {(preds == 0).sum()}")
    else:
        print("No 'predicted_return' column found!")

    if 'signal' in signals.columns:
        n_signals = signals['signal'].sum()
        print(f"\nTotal Signals (signal==1): {n_signals}")
        print(f"Min pred return threshold: {strategy.min_pred_return}")
        
        if n_signals > 0:
            print("\nSample Signals:")
            print(signals[signals['signal'] == 1][['close', 'predicted_return', 'bar_index', 'signal']].head(10))
            
            # Check timestamps
            sig_times = signals[signals['signal'] == 1].index
            print(f"\nFirst signal: {sig_times[0]}")
            print(f"Last signal:  {sig_times[-1]}")
    else:
        print("No 'signal' column found!")
        
    # Check bar_index
    if 'bar_index' in signals.columns:
        print(f"\nBar Index range: {signals['bar_index'].min()} - {signals['bar_index'].max()}")
        print(f"Last entry bar configured: {strategy.last_entry_bar}")
    else:
        print("No 'bar_index' column found!")

if __name__ == "__main__":
    main()
