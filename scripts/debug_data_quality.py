
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import IntradayDataLoader

def main():
    print("Checking Data Quality for NaN/Zero/Inf values...")
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    
    for period in ['train', 'valid', 'test']:
        print(f"\nChecking {period} data...")
        df = loader.get_period_data(period)
        
        # Check close price
        if 'close' in df.columns:
            zeros = (df['close'] == 0).sum()
            nans = df['close'].isna().sum()
            infs = np.isinf(df['close']).sum()
            
            print(f"  Rows: {len(df)}")
            print(f"  Zeros in close: {zeros}")
            print(f"  NaNs in close: {nans}")
            print(f"  Infs in close: {infs}")
            
            if zeros > 0 or nans > 0 or infs > 0:
                print("  !!! Data Issues Found !!!")
                # Show sample
                if zeros > 0:
                    print("  Sample Zeros:")
                    print(df[df['close'] == 0].head())
                if nans > 0:
                    print("  Sample NaNs:")
                    print(df[df['close'].isna()].head())
        else:
            print("  'close' column missing!")

if __name__ == "__main__":
    main()
