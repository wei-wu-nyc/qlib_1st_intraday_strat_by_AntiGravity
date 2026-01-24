#!/usr/bin/env python3
"""
Run Exit Rule Experiments on Champion Model.

Focues on LightGBM_24bar and tests various exit configurations:
- Stop Loss
- Trailing Stop
- Profit Target
- Fixed Horizon
- Negative Return Gate
"""

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.labels.intraday_labels import IntradayLabels
from src.strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy
from src.backtest.intraday_backtest import IntradayBacktest
from src.backtest.exit_rules import (
    ExitRule, FixedHorizonExit, StopLossExit, TrailingStopExit, 
    ProfitTargetExit, NegativeReturnGateExit, CompositeExitRule
)
from src.reporting.results_report import save_results_to_csv, save_equity_curves
from src.metrics.performance_metrics import calculate_metrics

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(
    backtest_engine: IntradayBacktest,
    signals: pd.DataFrame,
    exit_rules: List[ExitRule],
    strategy_name: str
) -> Dict[str, Dict]:
    """Run a single backtest experiment."""
    print(f"  Running {strategy_name}...")
    
    # Set exit rules on the engine
    backtest_engine.exit_rules = exit_rules
    
    # Run backtest (engine resets internally)
    # We pass a lambda for signal generator that just returns our pre-computed signals
    # to avoid re-running prediction logic if possible, but the engine expects a generator.
    # Actually, the engine calls signal_generator(data). 
    # Since we already have signals, we can wrap a simple function.
    
    def simple_pass_through(data):
        # We assume 'signals' matches 'data' index. 
        # In this script we'll align them beforehand.
        return signals.reindex(data.index).fillna(0)
    
    return backtest_engine.run(signals, simple_pass_through)

def main():
    config_path = project_root / 'config' / 'intraday_config.yaml'
    config = load_config(str(config_path))
    
    # 1. Load Data & Generate Initial Signals (The heavy lifting)
    print("=" * 60)
    print("Loading Data and Champion Model (LightGBM_24bar)")
    print("=" * 60)
    
    loader = IntradayDataLoader(str(config_path))
    
    # Load model
    # Note: load_model appends _model.joblib, so we provide the base path without extension
    model_base_name = 'lightgbmintraday_24bar'
    model_path = project_root / 'results' / 'models' / model_base_name
    
    if not (model_path.parent / (model_base_name + '_model.joblib')).exists():
        print(f"Error: Model file not found at {model_path}_model.joblib")
        return
        
    strategy = LightGBMIntradayStrategy(config.get('models', {}).get('lightgbm', {}))
    strategy.load_model(str(model_path))
    
    # Prepare data dict
    datasets = {}
    signal_sets = {}
    
    # We only care about Test period for experiments mostly, but we'll do Valid+Test
    # to be safe and consistent.
    for period in ['valid', 'test']:
        print(f"\nProcessing {period} data...")
        df = loader.get_period_data(period)
        
        # Features
        print("  Generating features...")
        alpha = IntradayAlphaFeatures()
        df = alpha.generate_all_features(df)
        season = SeasonalityFeatures()
        df = season.generate_all_features(df)
        
        # Labels (needed for 24bar prediction mainly to ensure columns align, though not strictly for prediction)
        # Actually we just need feature columns.
        
        # Generate Signals using the Model
        print(f"  Generating base signals...")
        # strategy.predict expects dataframe with features
        # It adds 'pred' and 'signal' columns
        signals = strategy.generate_signals(df)
        
        # Add necessary columns for backtest if missing
        if 'bar_index' not in signals.columns:
            # Re-add bar index helper
            times = pd.to_datetime(signals.index)
            minutes_from_open = (times.hour - 9) * 60 + times.minute - 35
            signals['bar_index'] = (minutes_from_open / 5).astype(int) + 1
            
        signal_sets[period] = signals
        datasets[period] = df
        
    print("\nData preparation complete.")
    
    # 2. Define Experiments
    experiments = [
        {
            'suffix': 'Base',
            'rules': [] # Default behavior only
        },
        # Fix Horizon
        {
            'suffix': 'Fix12',
            'rules': [FixedHorizonExit(12)]
        },
        {
            'suffix': 'Fix18',
            'rules': [FixedHorizonExit(18)]
        },
        # Stop Loss
        {
            'suffix': 'SL0.5%',
            'rules': [StopLossExit(-0.005)]
        },
        {
            'suffix': 'SL1.0%',
            'rules': [StopLossExit(-0.01)]
        },
        # Trailing Stop
        {
            'suffix': 'Trail0.3%',
            'rules': [TrailingStopExit(-0.003)]
        },
        {
            'suffix': 'Trail0.5%',
            'rules': [TrailingStopExit(-0.005)]
        },
        {
            'suffix': 'Trail0.75%',
            'rules': [TrailingStopExit(-0.0075)]
        },
        # Profit Target
        {
            'suffix': 'Target0.5%',
            'rules': [ProfitTargetExit(0.005)]
        },
        {
            'suffix': 'Target1.0%',
            'rules': [ProfitTargetExit(0.01)]
        },
        {
            'suffix': 'Target1.5%',
            'rules': [ProfitTargetExit(0.015)]
        },
        {
            'suffix': 'Target2.0%',
            'rules': [ProfitTargetExit(0.02)]
        },
        # Time-based negative stop
        {
            'suffix': 'NegGate12',
            'rules': [NegativeReturnGateExit(12)]
        },
        # Combinations
        {
            'suffix': 'Combo_SL0.5_Target1.0',
            'rules': [StopLossExit(-0.005), ProfitTargetExit(0.01)]
        },
        {
            'suffix': 'Combo_Trail0.5_Target2.0',
            'rules': [TrailingStopExit(-0.005), ProfitTargetExit(0.02)]
        }
    ]
    
    # 3. Running Experiments
    print("\n" + "=" * 60)
    print(f"Running {len(experiments)} Exit Rule Experiments")
    print("=" * 60)
    
    backtest = IntradayBacktest(str(config_path))
    backtest.transaction_cost_bps = 1.0 # Set specific cost (1bp)
    all_results = {}
    
    for exp in experiments:
        suffix = exp['suffix']
        rules = exp['rules']
        strat_name = f"LightGBM_24bar_{suffix}"
        
        print(f"\nExperiment: {strat_name}")
        exp_results = {}
        
        for period in ['valid', 'test']:
            print(f"  Period: {period}")
            signals = signal_sets[period]
            
            # Hack: The run_experiment helper needs the engine and rules
            # We set rules directly here
            backtest.exit_rules = rules
            
            # Use 'simple_pass_through' equivalent logic
            res = backtest.run(signals, lambda x: x)
            
            # Calculate metrics explicitly
            metrics = calculate_metrics(res)
            
            # Attach equity curve to metrics object for later saving
            metrics.equity_curve = res['equity_curve']
            
            exp_results[period] = metrics
            
            # Store equity curve in memory for saving later?
            # actually backtest.run returns a dict with 'equity_curve'
            # We need to construct the full result object expected by save_results functions
            # which is {strategy: {period: metrics_obj}}
            # But duplicate equity curve logic is needed.
            
            # Let's attach equity curve to the metrics object or handle it separately
            # The metrics object from IntradayBacktest is a PerformanceMetrics object
            # It has .equity_curve field!
            
        all_results[strat_name] = exp_results

    # 4. Save Results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    output_dir = project_root / 'results' 
    
    # Save CSV results
    save_results_to_csv(all_results, str(output_dir / 'metrics'), filename='exit_experiments_summary.csv')
    
    # Save equity curves
    # Note: save_equity_curves expects {strategy: {period: metrics}} structure
    # and the metrics object must have .equity_curve attribute.
    # IntradayBacktest returns {'metrics': PerformanceMetrics(...)}
    save_equity_curves(all_results, str(output_dir / 'metrics'), filename='exit_experiments_equity.csv')

    print("\nDone! Now run generate_dashboard_from_csv.py to visualize.")

if __name__ == "__main__":
    main()
