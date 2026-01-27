
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import IntradayDataLoader
from src.backtest.intraday_backtest import IntradayBacktest
from src.strategies.ml_models.xgboost_intraday import XGBoostIntradayStrategy
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures

def main():
    print("Debugging Integration (Model + Backtest)...")
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    
    print("Loading test data (1 month)...")
    data = loader.get_data_with_time_features(
        symbols=['DIA', 'SPY', 'QQQ', 'IWM'],
        start_date='2022-01-01',
        end_date='2022-01-31' 
    )
    
    # Generate features needed for model
    print("Generating features...")
    alpha_gen = IntradayAlphaFeatures('config/intraday_config.yaml')
    data = alpha_gen.generate_all_features(data)
    season_gen = SeasonalityFeatures('config/intraday_config.yaml')
    data = season_gen.generate_all_features(data)
    
    # Load model
    print("Loading model...")
    model_path = project_root / 'results' / 'models' / 'xgboostintraday'
    strategy = XGBoostIntradayStrategy(loader.config)
    strategy.load_model(str(model_path))
    
    # Signal generator wrapper
    def signal_gen(df):
        # Fix for duplicate columns if any
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()
        
        return strategy.generate_signals(df)
    
    # Run backtest
    bt = IntradayBacktest()
    print("\nRunning backtest...")
    results = bt.run(data, signal_gen)
    
    print("\nBacktest complete.")
    print(f"Num Trades: {results['num_trades']}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    
    if len(results['trades']) > 0:
        print("\nTrades:")
        print(results['trades'][['entry_time', 'exit_time', 'exit_reason']].to_string())

if __name__ == "__main__":
    main()
