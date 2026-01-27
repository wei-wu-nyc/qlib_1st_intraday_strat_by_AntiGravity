
import pandas as pd
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.reporting.dashboard import generate_dashboard

def main():
    metrics_path = project_root / 'results' / 'metrics' / 'summary_metrics.csv'
    output_path = project_root / 'results' / 'dashboard.html'
    
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        return
    
    print(f"Loading metrics from {metrics_path}...")
    df = pd.read_csv(metrics_path)
    
    # Transform to dictionary structure: results[Strategy][Period][Metric] = Value
    results = {}
    
    for _, row in df.iterrows():
        strategy = row['Strategy']
        metric = row['Metric']
        
        if strategy not in results:
            results[strategy] = {'train': {}, 'valid': {}, 'test': {}}
            
        results[strategy]['train'][metric] = row['Train']
        results[strategy]['valid'][metric] = row['Valid']
        results[strategy]['test'][metric] = row['Test']
        
    print(f"Generating dashboard for {len(results)} strategies...")
    
    # Load experiment metrics if available
    experiment_metrics_path = project_root / 'results' / 'metrics' / 'exit_experiments_summary.csv'
    if experiment_metrics_path.exists():
        print(f"Loading experiment metrics from {experiment_metrics_path}...")
        exp_df = pd.read_csv(experiment_metrics_path)
        for _, row in exp_df.iterrows():
            strategy = row['Strategy']
            metric = row['Metric']
            if strategy not in results:
                results[strategy] = {'train': {}, 'valid': {}, 'test': {}}
            if 'Train' in row: results[strategy]['train'][metric] = row['Train']
            if 'Valid' in row: results[strategy]['valid'][metric] = row['Valid']
            if 'Test' in row: results[strategy]['test'][metric] = row['Test']
        print(f"Total strategies after merging experiments: {len(results)}")
    
    # Load equity curves if available
    equity_curves_path = project_root / 'results' / 'metrics' / 'equity_curves.csv'
    equity_curves = {}
    
    def load_curves(path, target_dict):
        if path.exists():
            print(f"Loading equity curves from {path}...")
            eq_df = pd.read_csv(path)
            eq_df['Date'] = pd.to_datetime(eq_df['Date'])
            for strategy in eq_df['Strategy'].unique():
                strat_df = eq_df[eq_df['Strategy'] == strategy].copy()
                strat_df = strat_df.set_index('Date')
                strat_df = strat_df.rename(columns={'Equity': 'equity'})
                target_dict[strategy] = strat_df
                print(f"  Loaded {strategy}: {len(strat_df)} points")

    load_curves(equity_curves_path, equity_curves)
    
    # Load experiment equity curves
    exp_equity_path = project_root / 'results' / 'metrics' / 'exit_experiments_equity.csv'
    load_curves(exp_equity_path, equity_curves)
            
    print(f"Total strategies with equity curves: {len(equity_curves)}")
    
    # Calculate benchmark metrics from equity curve if available
    bench_returns = {
        'train': {'Ann. Return': '2.85%'},
        'valid': {'Ann. Return': '23.96%'},
        'test': {'Ann. Return': '9.52%'}
    }
    
    # Calculate Sharpe and MaxDD for benchmark from equity curve
    if 'Benchmark (SPY)' in equity_curves:
        bench_eq = equity_curves['Benchmark (SPY)']['equity']
        
        for period, (start, end) in [
            ('train', ('2000-07-01', '2018-12-31')),
            ('valid', ('2019-01-01', '2021-12-31')),
            ('test', ('2022-01-01', '2025-12-31'))
        ]:
            try:
                period_eq = bench_eq[(bench_eq.index >= start) & (bench_eq.index <= end)]
                if len(period_eq) > 0:
                    # Calculate daily returns
                    daily_eq = period_eq.resample('D').last().dropna()
                    daily_returns = daily_eq.pct_change().dropna()
                    
                    # Sharpe Ratio (annualized)
                    if len(daily_returns) > 0 and daily_returns.std() > 0:
                        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
                        bench_returns[period]['Sharpe Ratio'] = f'{sharpe:.2f}'
                    
                    # Max Drawdown
                    cummax = daily_eq.cummax()
                    drawdown = (daily_eq - cummax) / cummax
                    max_dd = drawdown.min() * -100
                    bench_returns[period]['Max Drawdown'] = f'{max_dd:.2f}%'
            except Exception as e:
                print(f"Warning: Could not calculate benchmark metrics for {period}: {e}")
    
    dashboard_path = generate_dashboard(
        results, 
        equity_curves=equity_curves,
        output_path=str(output_path), 
        bench_returns=bench_returns
    )
    print(f"Dashboard saved to: {dashboard_path}")

if __name__ == "__main__":
    main()
