
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import yaml

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import IntradayDataLoader

def calculate_annualized_return(df):
    if len(df) == 0:
        return 0.0
    
    # Intraday 5-min data
    # Annualized return = (Final / Initial) ^ (252 / Years) - 1
    # Approximation: Daily mean return * 252
    
    # Resample to daily close
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index().set_index('datetime')
    elif 'datetime' in df.columns:
        df = df.set_index('datetime')
        
    if 'close' not in df.columns:
        return 0.0
        
    daily_closes = df['close'].resample('D').last().dropna()
    if len(daily_closes) < 2:
        return 0.0
        
    total_return = daily_closes.iloc[-1] / daily_closes.iloc[0] - 1
    days = (daily_closes.index[-1] - daily_closes.index[0]).days
    if days <= 0:
        return 0.0
        
    ann_return = (1 + total_return) ** (365 / days) - 1
    return ann_return

def main():
    print("Calculating Benchmark (SPY) Returns...")
    
    config_path = project_root / 'config' / 'intraday_config.yaml'
    loader = IntradayDataLoader(str(config_path))
    
    periods = ['train', 'valid', 'test']
    results = {}
    
    for period in periods:
        print(f"Loading {period} data...")
        df = loader.get_period_data(period, symbols=['SPY'])
        
        # Calculate return
        ann_ret = calculate_annualized_return(df)
        results[period] = ann_ret
        print(f"  {period.capitalize()} Ann. Return: {ann_ret*100:.2f}%")
        
    # Save to simplistic file or just print for manual usage (agent will parse)
    print("\nRESULTS_DICT = " + str(results))

if __name__ == "__main__":
    main()
