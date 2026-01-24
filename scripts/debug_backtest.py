
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import IntradayDataLoader
from src.backtest.intraday_backtest import IntradayBacktest

def main():
    print("Debugging IntradayBacktest...")
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    
    print("Loading test data...")
    # Load 1 month of data
    data = loader.get_data_with_time_features(
        symbols=['SPY'],
        start_date='2022-01-01',
        end_date='2022-01-31' 
    )
    
    print(f"Data loaded: {len(data)} rows")
    
    # Create simple signal generator: Enter every day at bar 10
    def signal_gen(df):
        res = df.copy()
        res['signal'] = 0
        res['exit_signal'] = 0
        
        # Signal at bar 10
        if 'bar_index' in res.columns:
            res.loc[res['bar_index'] == 10, 'signal'] = 1
        
        return res
    
    # Run backtest
    bt = IntradayBacktest()
    print("\nRunning backtest...")
    results = bt.run(data, signal_gen)
    
    print("\nBacktest complete.")
    print(f"Num Trades: {results['num_trades']}")
    
    if len(results['trades']) > 0:
        print("\nFirst 5 trades:")
        print(results['trades'][['entry_time', 'exit_time', 'exit_reason']].head().to_string())
    else:
        print("No trades generated!")

if __name__ == "__main__":
    main()
