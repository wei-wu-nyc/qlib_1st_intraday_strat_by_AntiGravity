#!/usr/bin/env python3
"""
Run All Intraday Trading Strategies.

Main script to:
1. Load data for all ETFs
2. Generate features and labels
3. Run all strategies (rule-based and ML)
4. Calculate metrics for Train/Valid/Test
5. Generate performance reports
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from data.data_loader import IntradayDataLoader
from features.intraday_alpha import IntradayAlphaFeatures
from features.seasonality_features import SeasonalityFeatures
from labels.intraday_labels import IntradayLabels
from backtest.intraday_backtest import IntradayBacktest
from metrics.performance_metrics import calculate_metrics, PerformanceMetrics
from reporting.results_report import (
    generate_summary_table, save_results_to_csv, generate_report_markdown,
    save_equity_curves
)

# Import strategies
from strategies.rule_based.momentum_breakout import MomentumBreakoutStrategy
from strategies.rule_based.mean_reversion import MeanReversionStrategy
from strategies.rule_based.opening_range_breakout import OpeningRangeBreakoutStrategy

try:
    from strategies.ml_models.xgboost_intraday import XGBoostIntradayStrategy
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

try:
    from strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available")


def load_config(config_path: str) -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(loader: IntradayDataLoader, period: str) -> pd.DataFrame:
    """
    Load and prepare data for a specific period.
    
    Args:
        loader: Data loader instance
        period: 'train', 'valid', or 'test'
        
    Returns:
        DataFrame with features and labels
    """
    print(f"  Loading {period} data...")
    
    # Get raw data for period
    df = loader.get_period_data(period)
    
    print(f"    Loaded {len(df)} rows")
    
    # Generate alpha features
    print(f"    Generating alpha features...")
    alpha_gen = IntradayAlphaFeatures()
    df = alpha_gen.generate_all_features(df)
    
    # Generate seasonality features
    print(f"    Generating seasonality features...")
    season_gen = SeasonalityFeatures()
    df = season_gen.generate_all_features(df)
    
    # Generate labels
    print(f"    Generating labels...")
    config_path = project_root / 'config' / 'intraday_config.yaml'
    label_gen = IntradayLabels(str(config_path))
    df = label_gen.generate_all_labels(df)
    
    # Drop rows with NaN in key columns
    # initial_rows = len(df)
    # df = df.dropna(subset=['ret_8bar'])
    # print(f"    Dropped {initial_rows - len(df)} rows with NaN labels")
    
    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df


def run_rule_based_strategy(
    strategy,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: dict,
) -> Dict[str, Dict]:
    """Run a rule-based strategy and calculate metrics."""
    strategy_name = strategy.get_name()
    print(f"\n  Running {strategy_name}...")
    
    results = {}
    
    for period_name, data in [
        ('train', train_data),
        ('valid', valid_data),
        ('test', test_data)
    ]:
        print(f"    {period_name}...")
        
        # Create backtest
        bt = IntradayBacktest()
        
        # Signal generator function
        def signal_gen(df):
            return strategy.generate_signals(df)
        
        # Run backtest
        bt_results = bt.run(data, signal_gen)
        
        # Calculate metrics
        metrics = calculate_metrics(bt_results)
        results[period_name] = metrics
        
        print(f"      Trades: {metrics.num_trades}, Return: {metrics.total_return*100:.2f}%")
    
    return results


def run_ml_strategy(
    strategy_class,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: dict,
    feature_cols: List[str],
    horizon = 8,  # Can be int or 'eod'
) -> Dict[str, Dict]:
    """Run an ML strategy and calculate metrics."""
    # Handle EOD horizon specially
    if horizon == 'eod':
        label_col = 'ret_to_close'
        horizon_name = 'eod'
    else:
        label_col = f'ret_{horizon}bar'
        horizon_name = f'{horizon}bar'
    
    # Initialize strategy
    strategy = strategy_class(config.get('models', {}).get(
        strategy_class.__name__.lower().replace('intradaystrategy', ''), {}
    ))
    base_name = strategy.get_name()
    strategy_name = f"{base_name}_{horizon_name}"
    print(f"\n  Running {strategy_name}...")
    
    # Filter feature columns to those that exist (and deduplicate)
    available_features = list(dict.fromkeys([c for c in feature_cols if c in train_data.columns]))
    print(f"    Using {len(available_features)} features, label: {label_col}")
    
    # Prepare training data - Drop NaNs ONLY for training
    # Use label_col to filter valid rows for training
    train_valid = train_data.dropna(subset=[label_col] + available_features)
    
    X_train = train_valid[available_features].copy()
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    y_train = train_valid[label_col]
    
    # Prepare validation data - Drop NaNs for validation set evaluation
    valid_valid = valid_data.dropna(subset=[label_col] + available_features)
    
    X_valid = valid_valid[available_features].copy()
    X_valid = X_valid.loc[:, ~X_valid.columns.duplicated()]
    y_valid = valid_valid[label_col]
    
    # Fit model
    print(f"    Training model...")
    strategy.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    
    # Store feature names for signal generation
    strategy.feature_names = available_features
    
    # Save model with horizon suffix
    model_dir = project_root / 'results' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_filename = strategy_class.__name__.lower().replace('strategy', '') + f'_{horizon_name}'
    strategy.save_model(str(model_dir / model_filename))
    
    results = {}
    
    for period_name, data in [
        ('train', train_data),
        ('valid', valid_data),
        ('test', test_data)
    ]:
        print(f"    {period_name}...")
        
        bt = IntradayBacktest()
        
        def signal_gen(df):
            # Remove duplicate columns from input DataFrame first
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()].copy()
            # Pass full DataFrame to generate_signals - it will preserve 'close' and other columns
            return strategy.generate_signals(df)
        
        bt_results = bt.run(data, signal_gen)
        metrics = calculate_metrics(bt_results)
        results[period_name] = metrics
        
        print(f"      Trades: {metrics.num_trades}, Return: {metrics.total_return*100:.2f}%")
    
    return results




def main(dry_run: bool = False):
    """Main execution function."""
    print("=" * 60)
    print("Intraday Trading Strategy Runner")
    print("=" * 60)
    
    # Load config
    config_path = project_root / 'config' / 'intraday_config.yaml'
    config = load_config(config_path)
    
    print(f"\nConfiguration:")
    print(f"  ETFs: {config['instruments']['symbols']}")
    print(f"  Benchmark: {config['instruments']['benchmark']}")
    print(f"  Train: {config['periods']['train']['start']} to {config['periods']['train']['end']}")
    print(f"  Valid: {config['periods']['valid']['start']} to {config['periods']['valid']['end']}")
    print(f"  Test: {config['periods']['test']['start']} to {config['periods']['test']['end']}")
    
    # Initialize data loader
    print("\nInitializing data loader...")
    loader = IntradayDataLoader(str(config_path))
    
    # Prepare data for each period
    print("\nPreparing data...")
    train_data = prepare_data(loader, 'train')
    valid_data = prepare_data(loader, 'valid')
    test_data = prepare_data(loader, 'test')
    
    if dry_run:
        print("\n[DRY RUN] Data preparation complete. Exiting.")
        return
    
    # Get feature columns for ML
    alpha_gen = IntradayAlphaFeatures()
    season_gen = SeasonalityFeatures()
    feature_cols = alpha_gen.get_feature_names() + season_gen.get_feature_names()
    
    # Store all results
    all_results = {}
    
    # Run Rule-Based Strategies
    print("\n" + "=" * 60)
    print("Rule-Based Strategies")
    print("=" * 60)
    
    # Momentum Breakout
    momentum_strategy = MomentumBreakoutStrategy()
    all_results['MomentumBreakout'] = run_rule_based_strategy(
        momentum_strategy, train_data, valid_data, test_data, config
    )
    
    # Mean Reversion
    mr_strategy = MeanReversionStrategy()
    all_results['MeanReversion'] = run_rule_based_strategy(
        mr_strategy, train_data, valid_data, test_data, config
    )
    
    # Opening Range Breakout
    orb_strategy = OpeningRangeBreakoutStrategy()
    all_results['ORB'] = run_rule_based_strategy(
        orb_strategy, train_data, valid_data, test_data, config
    )
    
    # Run ML Strategies
    print("\n" + "=" * 60)
    print("ML-Based Strategies")
    print("=" * 60)
    
    # Get all horizons from config
    primary_horizon = config.get('trading', {}).get('primary_label_horizon', 24)
    alt_horizons = config.get('trading', {}).get('alternative_horizons', [36])
    all_horizons = [primary_horizon] + alt_horizons
    
    # Add EOD as special case
    all_horizons.append('eod')
    
    for horizon in all_horizons:
        if horizon == 'eod':
            print(f"\n  --- Horizon: EOD (return to close) ---")
            label_col = 'ret_to_close'
            horizon_name = 'eod'
        else:
            print(f"\n  --- Horizon: {horizon} bars ({horizon * 5} minutes) ---")
            label_col = f'ret_{horizon}bar'
            horizon_name = f'{horizon}bar'
        
        if HAS_XGBOOST:
            all_results[f'XGBoost_{horizon_name}'] = run_ml_strategy(
                XGBoostIntradayStrategy, train_data, valid_data, test_data,
                config, feature_cols, horizon=horizon
            )
        
        if HAS_LIGHTGBM:
            all_results[f'LightGBM_{horizon_name}'] = run_ml_strategy(
                LightGBMIntradayStrategy, train_data, valid_data, test_data,
                config, feature_cols, horizon=horizon
            )


    
    # Generate Reports
    print("\n" + "=" * 60)
    print("Generating Reports")
    print("=" * 60)
    
    output_dir = project_root / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Save CSV results
    save_results_to_csv(all_results, str(output_dir / 'metrics'))
    
    # Save equity curves for dashboard
    save_equity_curves(all_results, str(output_dir / 'metrics'))
    
    # Generate markdown report
    generate_report_markdown(all_results, str(output_dir / 'strategy_report.md'))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    summary = generate_summary_table(all_results)
    print(summary.to_string())
    
    print(f"\n\nResults saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run intraday trading strategies')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Only load data, do not run strategies')
    
    args = parser.parse_args()
    
    main(dry_run=args.dry_run)
