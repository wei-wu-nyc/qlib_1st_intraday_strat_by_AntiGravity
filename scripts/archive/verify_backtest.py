#!/usr/bin/env python
"""
Backtest Verification Script

Purpose: Verify multi-asset backtest logic is correct by comparing:
1. Buy-and-hold returns for QQQ and SPY
2. Single-asset strategy returns for QQQ and SPY  
3. Equal-weight strategy for QQQ+SPY

Expected results:
- Single-asset strategy should be in ballpark of buy-and-hold
- Equal-weight should be approximately average of single-asset returns
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
from src.backtest.intraday_backtest import IntradayBacktest
from src.metrics.performance_metrics import calculate_metrics


def calculate_buy_and_hold(df: pd.DataFrame) -> dict:
    """Calculate simple buy-and-hold return."""
    first_close = df['close'].iloc[0]
    last_close = df['close'].iloc[-1]
    total_return = last_close / first_close - 1
    
    # Annualize
    first_date = df.index[0]
    last_date = df.index[-1]
    years = (last_date - first_date).days / 365.25
    annual_return = (1 + total_return) ** (1/years) - 1
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'first_price': first_close,
        'last_price': last_close,
        'years': years
    }


def run_single_asset_strategy(df: pd.DataFrame, strategy) -> dict:
    """Run strategy on single asset."""
    bt = IntradayBacktest()
    bt.transaction_cost_bps = 0  # No transaction cost
    
    results = bt.run(df, strategy.generate_signals)
    metrics = calculate_metrics(results)
    
    return {
        'total_return': results['total_return'],
        'annual_return': metrics.annualized_return,
        'sharpe': metrics.sharpe_ratio,
        'trades': results['num_trades']
    }


def main():
    print("=" * 60)
    print("BACKTEST VERIFICATION TEST")
    print("=" * 60)
    
    # Load data
    print("\nLoading test data...")
    loader = IntradayDataLoader('config/intraday_config.yaml')
    df_all = loader.get_period_data('test')
    
    # Generate features
    alpha = IntradayAlphaFeatures()
    df_all = alpha.generate_all_features(df_all)
    season = SeasonalityFeatures()
    df_all = season.generate_all_features(df_all)
    
    # Extract single instruments
    spy_df = df_all.xs('SPY', level='instrument').copy()
    qqq_df = df_all.xs('QQQ', level='instrument').copy()
    
    print(f"SPY: {len(spy_df)} bars, {spy_df.index[0].date()} to {spy_df.index[-1].date()}")
    print(f"QQQ: {len(qqq_df)} bars, {qqq_df.index[0].date()} to {qqq_df.index[-1].date()}")
    
    # Load model
    strategy = LightGBMIntradayStrategy({})
    strategy.load_model(str(Path.cwd() / 'results' / 'models' / 'lightgbmintraday_24bar'))
    
    # ============================================================
    # TEST 1: Buy-and-Hold
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Buy-and-Hold Returns")
    print("=" * 60)
    
    spy_bh = calculate_buy_and_hold(spy_df)
    qqq_bh = calculate_buy_and_hold(qqq_df)
    
    print(f"\nSPY Buy-and-Hold:")
    print(f"  First: ${spy_bh['first_price']:.2f} -> Last: ${spy_bh['last_price']:.2f}")
    print(f"  Total Return: {spy_bh['total_return']*100:.2f}%")
    print(f"  Annual Return: {spy_bh['annual_return']*100:.2f}%")
    
    print(f"\nQQQ Buy-and-Hold:")
    print(f"  First: ${qqq_bh['first_price']:.2f} -> Last: ${qqq_bh['last_price']:.2f}")
    print(f"  Total Return: {qqq_bh['total_return']*100:.2f}%")
    print(f"  Annual Return: {qqq_bh['annual_return']*100:.2f}%")
    
    avg_bh = (spy_bh['annual_return'] + qqq_bh['annual_return']) / 2
    print(f"\nAverage B&H: {avg_bh*100:.2f}%")
    
    # ============================================================
    # TEST 2: Single-Asset Strategy
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Single-Asset Strategy (0bp cost)")
    print("=" * 60)
    
    spy_strat = run_single_asset_strategy(spy_df, strategy)
    qqq_strat = run_single_asset_strategy(qqq_df, strategy)
    
    print(f"\nSPY Strategy:")
    print(f"  Total Return: {spy_strat['total_return']*100:.2f}%")
    print(f"  Annual Return: {spy_strat['annual_return']*100:.2f}%")
    print(f"  Sharpe: {spy_strat['sharpe']:.2f}")
    print(f"  Trades: {spy_strat['trades']}")
    
    print(f"\nQQQ Strategy:")
    print(f"  Total Return: {qqq_strat['total_return']*100:.2f}%")
    print(f"  Annual Return: {qqq_strat['annual_return']*100:.2f}%")
    print(f"  Sharpe: {qqq_strat['sharpe']:.2f}")
    print(f"  Trades: {qqq_strat['trades']}")
    
    avg_strat = (spy_strat['annual_return'] + qqq_strat['annual_return']) / 2
    print(f"\nAverage Strategy: {avg_strat*100:.2f}%")
    
    # ============================================================
    # TEST 3: Equal-Weight Strategy (QQQ + SPY)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Equal-Weight Strategy (QQQ + SPY)")
    print("=" * 60)
    
    # Run both strategies and collect daily returns
    bt_spy = IntradayBacktest()
    bt_spy.transaction_cost_bps = 0
    res_spy = bt_spy.run(spy_df, strategy.generate_signals)
    
    bt_qqq = IntradayBacktest()
    bt_qqq.transaction_cost_bps = 0
    res_qqq = bt_qqq.run(qqq_df, strategy.generate_signals)
    
    # Get daily returns from both
    spy_daily = np.array(res_spy['daily_returns'])
    qqq_daily = np.array(res_qqq['daily_returns'])
    
    # Ensure same length (should be)
    min_len = min(len(spy_daily), len(qqq_daily))
    spy_daily = spy_daily[:min_len]
    qqq_daily = qqq_daily[:min_len]
    
    # Equal weight: average of daily returns
    equal_weight_daily = (spy_daily + qqq_daily) / 2
    
    # Calculate metrics for equal-weight
    equal_total_return = np.prod(1 + equal_weight_daily) - 1
    trading_days = len(equal_weight_daily)
    years = trading_days / 252.0
    equal_annual_return = (1 + equal_total_return) ** (1/years) - 1
    
    # Sharpe calculation
    mean_daily = np.mean(equal_weight_daily)
    std_daily = np.std(equal_weight_daily)
    equal_sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    
    print(f"\nEqual-Weight Strategy (50% SPY + 50% QQQ):")
    print(f"  Total Return: {equal_total_return*100:.2f}%")
    print(f"  Annual Return: {equal_annual_return*100:.2f}%")
    print(f"  Sharpe: {equal_sharpe:.2f}")
    print(f"  Trading Days: {trading_days}")
    
    print(f"\nExpected (avg of single-asset): {avg_strat*100:.2f}%")
    print(f"Actual Equal-Weight: {equal_annual_return*100:.2f}%")
    diff_from_expected = abs(equal_annual_return - avg_strat)
    print(f"Difference: {diff_from_expected*100:.2f}%")
    
    if diff_from_expected < 0.02:  # Within 2%
        print("\n✓ PASS: Equal-weight matches expected average")
    else:
        print("\n✗ WARNING: Equal-weight differs from expected average")
    
    # ============================================================
    # VALIDATION
    # ============================================================
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    
    # Check 1: Strategy should be in ballpark of buy-and-hold
    spy_diff = abs(spy_strat['annual_return'] - spy_bh['annual_return'])
    qqq_diff = abs(qqq_strat['annual_return'] - qqq_bh['annual_return'])
    
    print(f"\nSPY: Strategy vs B&H difference: {spy_diff*100:.2f}%")
    print(f"QQQ: Strategy vs B&H difference: {qqq_diff*100:.2f}%")
    
    # Reasonable threshold: strategy within 20% of B&H (since we have signals)
    if spy_diff < 0.20 and qqq_diff < 0.20:
        print("\n✓ PASS: Single-asset strategies are in reasonable range of B&H")
    else:
        print("\n✗ FAIL: Single-asset strategies deviate too much from B&H")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'SPY':<15} {'QQQ':<15}")
    print("-" * 55)
    print(f"{'Buy-and-Hold Annual':<25} {spy_bh['annual_return']*100:.2f}%{'':<10} {qqq_bh['annual_return']*100:.2f}%")
    print(f"{'Strategy Annual':<25} {spy_strat['annual_return']*100:.2f}%{'':<10} {qqq_strat['annual_return']*100:.2f}%")
    print(f"{'Difference':<25} {(spy_strat['annual_return']-spy_bh['annual_return'])*100:.2f}%{'':<10} {(qqq_strat['annual_return']-qqq_bh['annual_return'])*100:.2f}%")


if __name__ == "__main__":
    main()
