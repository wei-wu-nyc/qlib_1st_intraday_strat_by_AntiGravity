
import pandas as pd
from pathlib import Path

def main():
    root = Path(__file__).parent.parent
    input_path = root / 'results' / 'metrics' / 'equity_curves.csv'
    
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort to ensure correct order
    df = df.sort_values(['Strategy', 'Date'])
    
    strategies = df['Strategy'].unique()
    fixed_dfs = []
    
    for strategy in strategies:
        strat_df = df[df['Strategy'] == strategy].copy()
        
        # Split by period
        train_df = strat_df[strat_df['Period'] == 'train'].copy()
        valid_df = strat_df[strat_df['Period'] == 'valid'].copy()
        test_df = strat_df[strat_df['Period'] == 'test'].copy()
        
        # Chain them
        # Train stays as is (starts at 1.0)
        
        # Valid starts where Train ended
        if not train_df.empty and not valid_df.empty:
            train_end = train_df['Equity'].iloc[-1]
            valid_df['Equity'] = valid_df['Equity'] * train_end
            
        # Test starts where Valid ended
        if not valid_df.empty and not test_df.empty:
            valid_end = valid_df['Equity'].iloc[-1]
            test_df['Equity'] = test_df['Equity'] * valid_end
        elif not train_df.empty and not test_df.empty: # Fallback if no valid
            train_end = train_df['Equity'].iloc[-1]
            test_df['Equity'] = test_df['Equity'] * train_end
            
        # Combine
        fixed_strat = pd.concat([train_df, valid_df, test_df])
        fixed_dfs.append(fixed_strat)
        
        print(f"Fixed {strategy}: Train End={train_df['Equity'].iloc[-1]:.2f}, Valid End={valid_df['Equity'].iloc[-1]:.2f}")

    final_df = pd.concat(fixed_dfs)
    final_df.to_csv(input_path, index=False)
    print(f"Saved chained equity curves to {input_path}")

if __name__ == "__main__":
    main()
