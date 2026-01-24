"""
Results Report Generator.

Generates formatted performance tables and charts for all strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def generate_summary_table(
    results: Dict[str, Dict[str, Dict]],
    periods: List[str] = ['train', 'valid', 'test'],
) -> pd.DataFrame:
    """
    Generate summary table for all strategies across periods.
    
    Args:
        results: Dict of {strategy: {period: metrics_dict}}
        periods: List of period names
        
    Returns:
        Formatted DataFrame
    """
    rows = []
    
    # Key metrics to show
    key_metrics = [
        'Ann. Return', 'Sharpe Ratio', 'Max Drawdown', 
        'Win Rate', 'Num Trades', 'Time in Market',
        'Avg Holding (bars)'
    ]
    
    for strategy_name, period_results in results.items():
        for metric in key_metrics:
            row = {'Strategy': strategy_name, 'Metric': metric}
            
            for period in periods:
                if period in period_results:
                    metrics = period_results[period]
                    if hasattr(metrics, 'to_dict'):
                        metric_dict = metrics.to_dict()
                        row[period.capitalize()] = metric_dict.get(metric, 'N/A')
                    elif isinstance(metrics, dict):
                        row[period.capitalize()] = metrics.get(metric, 'N/A')
                    else:
                        row[period.capitalize()] = 'N/A'
                else:
                    row[period.capitalize()] = 'N/A'
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def generate_comparison_table(
    results: Dict[str, Dict[str, Dict]],
    metric: str = 'Sharpe Ratio',
    periods: List[str] = ['train', 'valid', 'test'],
) -> pd.DataFrame:
    """
    Generate comparison table for a single metric across strategies.
    
    Args:
        results: Results dictionary
        metric: Metric name to compare
        periods: Periods to include
        
    Returns:
        DataFrame with strategies as rows and periods as columns
    """
    rows = []
    
    for strategy_name, period_results in results.items():
        row = {'Strategy': strategy_name}
        
        for period in periods:
            if period in period_results:
                metrics = period_results[period]
                if hasattr(metrics, 'to_dict'):
                    row[period.capitalize()] = metrics.to_dict().get(metric, 'N/A')
                elif isinstance(metrics, dict):
                    row[period.capitalize()] = metrics.get(metric, 'N/A')
            else:
                row[period.capitalize()] = 'N/A'
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_results_to_csv(
    results: Dict[str, Dict[str, Dict]],
    output_dir: str,
    filename: str = 'summary_metrics.csv',
) -> None:
    """
    Save all results to CSV files.
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
        filename: Output filename (default: summary_metrics.csv)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate and save summary table
    summary_df = generate_summary_table(results)
    summary_df.to_csv(output_path / filename, index=False)
    
    # Generate comparison tables for key metrics
    for metric in ['Sharpe Ratio', 'Win Rate', 'Ann. Return', 'Max Drawdown']:
        comparison = generate_comparison_table(results, metric)
        filename = metric.lower().replace(' ', '_').replace('.', '') + '_comparison.csv'
        comparison.to_csv(output_path / filename, index=False)
    
    print(f"Results saved to {output_path}")


def save_equity_curves(
    results: Dict[str, Dict[str, Dict]],
    output_dir: str,
    filename: str = 'equity_curves.csv',
) -> None:
    """
    Save equity curves to a single CSV file for dashboard plotting.
    Chains periods (Train -> Valid -> Test) to create cumulative curves.
    Also calculates and adds a Benchmark (SPY) strategy.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        filename: Output filename (default: equity_curves.csv)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_equity = []
    periods_order = ['train', 'valid', 'test']
    
    # 1. Process Strategy Curves
    for strategy_name, period_results in results.items():
        cumulative_scaler = 1.0
        
        for period_name in periods_order:
            if period_name in period_results:
                metrics = period_results[period_name]
                
                # Extract equity DataFrame
                equity_df = None
                if hasattr(metrics, 'equity_curve'):
                    equity_df = metrics.equity_curve
                elif isinstance(metrics, dict) and 'equity_curve' in metrics:
                    equity_df = metrics['equity_curve']
                
                if equity_df is not None and not equity_df.empty:
                    # FIX: Deduplicate to one row per timestamp
                    # The backtest produces one row per (timestamp, instrument) pair
                    # We need to aggregate to one row per timestamp for consistent chaining
                    # Use the FIRST value at each timestamp (consistent instrument reference)
                    if isinstance(equity_df.index, pd.MultiIndex):
                        # MultiIndex (datetime, instrument) - group by datetime
                        dedup_equity = equity_df.groupby(level=0)['equity'].first()
                    else:
                        # Single index - might still have duplicates from iteration order
                        dedup_equity = equity_df.groupby(equity_df.index)['equity'].first()
                    
                    # Normalize this period to start at 1.0
                    period_equity = dedup_equity / dedup_equity.iloc[0]
                    
                    # Scale by cumulative factor from previous period
                    scaled_equity = period_equity * cumulative_scaler
                    
                    # Update scaler for next period (end value of current)
                    cumulative_scaler = scaled_equity.iloc[-1]
                    
                    # Create DataFrame part
                    part_df = pd.DataFrame({
                        'Strategy': strategy_name,
                        'Period': period_name,
                        'Date': scaled_equity.index,
                        'Equity': scaled_equity.values
                    })
                    all_equity.append(part_df)
    
    # 2. Process Benchmark (SPY) Curve
    # We need to reconstruct it from the same data source to match timestamps/periods
    try:
        from data.data_loader import IntradayDataLoader
        loader = IntradayDataLoader()
        
        cumulative_scaler = 1.0
        for period_name in periods_order:
            # Get raw benchmark data for period
            # Note: We need to use the config from the loader to get dates
            # But the loader might not be exactly aware of the *backtest* split dates if they were modified at runtime
            # We will rely on the configured dates in the loader
            
            df = loader.get_period_data(period_name, symbols=['SPY'])
            if not df.empty:
                # Calculate cumulative return
                # Simple close-to-close return for equity curve approximation
                # Or better: use the pct_change of close price
                # Ensure sorted
                df = df.sort_index()
                
                # Assume $1 start for this period, compounded returns
                # We can just normalize the Close price trajectory
                # This is equivalent to buying 1 share (or fractional) at start and holding
                start_price = df['close'].iloc[0]
                period_equity = df['close'] / start_price
                
                # Scale
                scaled_equity = period_equity * cumulative_scaler
                cumulative_scaler = scaled_equity.iloc[-1]
                
                # Create DataFrame
                part_df = pd.DataFrame({
                    'Strategy': 'Benchmark (SPY)',
                    'Period': period_name,
                    'Date': df.index.get_level_values('datetime'), # MultiIndex
                    'Equity': scaled_equity.values
                })
                all_equity.append(part_df)
                print(f"Added Benchmark for {period_name}")
                
    except Exception as e:
        print(f"Warning: Could not generate Benchmark equity curve: {e}")

    if all_equity:
        final_df = pd.concat(all_equity)
        final_df.to_csv(output_path / filename, index=False)
        print(f"Equity curves saved to {output_path / filename}")


def plot_equity_curves(
    equity_curves: Dict[str, pd.DataFrame],
    benchmark_equity: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot equity curves for all strategies.
    
    Args:
        equity_curves: Dict of {strategy: equity_df}
        benchmark_equity: Optional benchmark equity curve
        output_path: Optional path to save figure
    """
    if not HAS_PLOTTING:
        print("Matplotlib not installed, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for strategy_name, equity_df in equity_curves.items():
        # Normalize to start at 1
        equity = equity_df['equity'].values
        normalized = equity / equity[0]
        ax.plot(equity_df.index, normalized, label=strategy_name)
    
    if benchmark_equity is not None:
        equity = benchmark_equity['equity'].values
        normalized = equity / equity[0]
        ax.plot(benchmark_equity.index, normalized, 
                label='Benchmark (SPY)', linestyle='--', color='gray')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Equity')
    ax.set_title('Strategy Equity Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Equity curve plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_drawdown(
    equity_curve: pd.DataFrame,
    strategy_name: str = 'Strategy',
    output_path: Optional[str] = None,
) -> None:
    """Plot drawdown chart for a single strategy."""
    if not HAS_PLOTTING:
        return
    
    equity = equity_curve['equity'].values
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100  # As percentage
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(equity_curve.index, drawdown, 0, alpha=0.5, color='red')
    ax.plot(equity_curve.index, drawdown, color='red', linewidth=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(f'{strategy_name} Drawdown')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
    
    plt.close()


def generate_report_markdown(
    results: Dict[str, Dict[str, Dict]],
    output_path: str,
) -> None:
    """
    Generate a markdown report with all results.
    
    Args:
        results: Results dictionary
        output_path: Path to save markdown file
    """
    lines = []
    lines.append("# Intraday Trading Strategy Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Summary table
    lines.append("## Performance Summary\n\n")
    
    summary = generate_summary_table(results)
    lines.append(summary.to_markdown(index=False))
    lines.append("\n\n")
    
    # Detailed results per strategy
    lines.append("## Strategy Details\n\n")
    
    for strategy_name, period_results in results.items():
        lines.append(f"### {strategy_name}\n\n")
        
        for period, metrics in period_results.items():
            lines.append(f"**{period.capitalize()}**\n\n")
            
            if hasattr(metrics, 'to_dict'):
                for name, value in metrics.to_dict().items():
                    lines.append(f"- {name}: {value}\n")
            elif isinstance(metrics, dict):
                for name, value in metrics.items():
                    lines.append(f"- {name}: {value}\n")
            
            lines.append("\n")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    print("Testing Report Generator...")
    
    # Create mock results
    from dataclasses import dataclass
    
    @dataclass
    class MockMetrics:
        def to_dict(self):
            return {
                'Ann. Return': '12.5%',
                'Sharpe Ratio': '1.45',
                'Max Drawdown': '-15.2%',
                'Win Rate': '54.2%',
                'Num Trades': 125,
                'Time in Market': '68.5%',
            }
    
    mock_results = {
        'MomentumBreakout': {
            'train': MockMetrics(),
            'valid': MockMetrics(),
            'test': MockMetrics(),
        },
        'MeanReversion': {
            'train': MockMetrics(),
            'valid': MockMetrics(),
            'test': MockMetrics(),
        },
    }
    
    # Generate summary
    summary = generate_summary_table(mock_results)
    print("\nSummary Table:")
    print(summary)
    
    # Generate comparison
    comparison = generate_comparison_table(mock_results, 'Sharpe Ratio')
    print("\nSharpe Ratio Comparison:")
    print(comparison)
    
    print("\nReport generation test completed!")
