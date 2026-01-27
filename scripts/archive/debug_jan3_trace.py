#!/usr/bin/env python3
"""
Debug script to trace LightGBM trades and equity on 2022-01-03.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from data.data_loader import IntradayDataLoader
from features.intraday_alpha import IntradayAlphaFeatures
from features.seasonality_features import SeasonalityFeatures
from strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy

def debug_run():
    """Trace LightGBM on 2022-01-03."""
    
    print("=" * 60)
    print("DEBUG: LightGBM Trade Trace for 2022-01-03")
    print("=" * 60)
    
    # Load model
    model = LightGBMIntradayStrategy()
    model_path = project_root / 'results' / 'models' / 'lightgbmintraday'
    model_file = Path(str(model_path) + '_model.joblib')
    if model_file.exists():
        model.load_model(str(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print("ERROR: Model file not found!")
        return
    
    # Load data for just Jan 3, 2022
    loader = IntradayDataLoader()
    
    # Get test period data, filter to Jan 3
    print("Loading test data...")
    raw_data = loader.get_period_data('test')
    
    raw_data = raw_data.reset_index()
    raw_data['date'] = raw_data['datetime'].dt.date
    jan3 = raw_data[raw_data['date'] == pd.Timestamp('2022-01-03').date()].copy()
    jan3 = jan3.set_index(['datetime', 'instrument'])
    
    print(f"Loaded {len(jan3)} bars for 2022-01-03")
    
    # Generate features
    print("Generating features...")
    alpha_gen = IntradayAlphaFeatures()
    alpha = alpha_gen.generate_all_features(jan3)
    
    seasonality_gen = SeasonalityFeatures()
    data = seasonality_gen.generate_all_features(alpha)
    
    # Get model feature columns
    feature_cols = model.feature_names
    print(f"Using {len(feature_cols)} features")
    
    # Prepare data - just SPY
    spy_data = data.xs('SPY', level='instrument').copy()
    spy_data = spy_data.sort_index()
    
    # Fill NaN and get predictions
    X = spy_data[feature_cols].fillna(0)
    predictions = model.model.predict(X)
    spy_data['prediction'] = predictions
    spy_data['signal'] = (predictions > 0).astype(int)
    
    # Now manually trace the backtest
    print("\n" + "=" * 80)
    print("Bar-by-bar trace:")
    print("=" * 80)
    
    initial_capital = 1000000
    capital = initial_capital
    position = None  # (shares, entry_price, entry_time)
    transaction_cost_bps = 2
    
    print(f"\nInitial Capital: ${capital:,.2f}")
    print(f"\n{'Time':20} {'Bar':4} {'Price':10} {'Signal':6} {'Shares':12} {'Equity':15} {'Action'}")
    print("-" * 110)
    
    for idx, row in spy_data.iterrows():
        timestamp = idx
        bar_index = int(row['bar_index'])
        price = row['close']
        signal = row['signal']
        
        # Calculate current equity
        if position is not None:
            current_equity = position[0] * price
        else:
            current_equity = capital
        
        action = ""
        
        # Forced EOD close at bar 77
        if bar_index >= 77 and position is not None:
            # Close position
            shares = position[0]
            exit_value = shares * price
            exit_value -= exit_value * transaction_cost_bps / 10000
            pnl = exit_value - position[0] * position[1]
            capital = exit_value
            action = f"EOD CLOSE: Sold {shares:.2f} @ ${price:.2f}, PnL: ${pnl:,.2f}"
            position = None
            current_equity = capital
        
        # Entry
        elif position is None and signal == 1 and bar_index <= 67:
            cost_mult = 1 + transaction_cost_bps / 10000
            shares = capital / (price * cost_mult)
            entry_cost = capital * transaction_cost_bps / 10000
            capital -= entry_cost
            position = (shares, price, timestamp)
            action = f"ENTRY: Bought {shares:.2f} shares @ ${price:.2f}"
            current_equity = shares * price
        
        shares_held = position[0] if position else 0
        print(f"{str(timestamp):20} {bar_index:4} ${price:9.2f} {signal:6} {shares_held:12.2f} ${current_equity:14,.2f} {action}")
    
    print("-" * 110)
    print(f"\nFinal Capital: ${capital:,.2f}")
    print(f"Day Return: {(capital/initial_capital - 1)*100:.2f}%")

if __name__ == "__main__":
    debug_run()
