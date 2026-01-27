#!/usr/bin/env python
"""
Verification Test for Rewritten Qlib-based Backtest Engine.

Compares new backtest against old backtest to ensure they match.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy
from src.backtest.qlib_backtest import QlibIntradayBacktest, BacktestConfig
from src.backtest.intraday_backtest import IntradayBacktest as OldBacktest


def main():
    print("=" * 60)
    print("BACKTEST VERIFICATION: OLD vs NEW")
    print("=" * 60)
    
    # Load data
    print("\nLoading test data...")
    loader = IntradayDataLoader('config/intraday_config.yaml')
    df_all = loader.get_period_data('test')
    
    alpha = IntradayAlphaFeatures()
    df_all = alpha.generate_all_features(df_all)
    season = SeasonalityFeatures()
    df_all = season.generate_all_features(df_all)
    
    spy_df = df_all.xs('SPY', level='instrument').copy()
    qqq_df = df_all.xs('QQQ', level='instrument').copy()
    
    print(f"SPY: {len(spy_df)} bars")
    print(f"QQQ: {len(qqq_df)} bars")
    
    # Load model
    strategy = LightGBMIntradayStrategy({})
    strategy.load_model(str(Path.cwd() / 'results' / 'models' / 'lightgbmintraday_24bar'))
    
    # ============================================================
    # TEST 1: SPY with OLD backtest
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: SPY with OLD backtest (0bp cost)")
    print("=" * 60)
    
    old_bt = OldBacktest()
    old_bt.transaction_cost_bps = 0
    old_results = old_bt.run(spy_df, strategy.generate_signals)
    
    print(f"OLD Backtest SPY:")
    print(f"  Total Return: {old_results['total_return']*100:.2f}%")
    print(f"  Trades: {old_results['num_trades']}")
    
    # ============================================================
    # TEST 2: SPY with NEW backtest
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: SPY with NEW backtest (0bp cost)")
    print("=" * 60)
    
    config = BacktestConfig(transaction_cost_bps=0)
    new_bt = QlibIntradayBacktest(config)
    new_results = new_bt.run(spy_df, strategy.generate_signals, instruments=['SPY'])
    
    print(f"NEW Backtest SPY:")
    print(f"  Total Return: {new_results.total_return*100:.2f}%")
    print(f"  Trades: {new_results.num_trades}")
    
    # ============================================================
    # COMPARISON
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    old_ret = old_results['total_return'] * 100
    new_ret = new_results.total_return * 100
    diff = abs(old_ret - new_ret)
    
    print(f"\n{'Metric':<20} {'OLD':<15} {'NEW':<15} {'Diff':<10}")
    print("-" * 60)
    print(f"{'Total Return':<20} {old_ret:.2f}%{'':<10} {new_ret:.2f}%{'':<10} {diff:.2f}%")
    print(f"{'Trades':<20} {old_results['num_trades']:<15} {new_results.num_trades:<15} {abs(old_results['num_trades']-new_results.num_trades)}")
    
    if diff < 0.5 and abs(old_results['num_trades'] - new_results.num_trades) <= 5:
        print("\n✓ PASS: Old and New backtests produce matching results!")
    else:
        print("\n✗ FAIL: Results differ significantly. Need investigation.")
    
    # ============================================================
    # TEST 3: QQQ comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: QQQ comparison")
    print("=" * 60)
    
    old_bt2 = OldBacktest()
    old_bt2.transaction_cost_bps = 0
    old_qqq = old_bt2.run(qqq_df, strategy.generate_signals)
    
    new_bt2 = QlibIntradayBacktest(BacktestConfig(transaction_cost_bps=0))
    new_qqq = new_bt2.run(qqq_df, strategy.generate_signals, instruments=['QQQ'])
    
    print(f"OLD QQQ: {old_qqq['total_return']*100:.2f}%, {old_qqq['num_trades']} trades")
    print(f"NEW QQQ: {new_qqq.total_return*100:.2f}%, {new_qqq.num_trades} trades")


if __name__ == "__main__":
    main()
