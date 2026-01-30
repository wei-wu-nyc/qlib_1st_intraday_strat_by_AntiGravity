import pandas as pd
import sys
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import numpy as np
import yaml

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.training.moe_trainer import train_moe_models
from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.strategies.moe_strategy import MoEIntradayStrategy
from src.strategies.ensemble_strategy import EnsembleStrategy
from src.backtest.strategies.multi_trade import MultiTradeStrategy
from src.backtest.engine import BacktestConfig

def run_quarterly_validation():
    print("🚀 Starting Quarterly Validation (2013-2025)...")
    print("Strategy: AE200 (6-Slot, 2.0x, Short-Day Fix)")
    
    # 1. Config
    years = range(2013, 2026)
    train_start_fixed = "2008-01-01"
    
    # Base dirs
    output_dir = Path.cwd() / 'results' / 'quarterly_validation'
    trades_dir = output_dir / 'trades'
    models_dir = output_dir / 'models' 
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Existing Annual Models (from Extended History Run)
    annual_models_base = Path.cwd() / 'results' / 'extended_history' / 'models'

    # Accumulators
    all_trades = []
    daily_returns = [] 
    
    etfs = ['QQQ', 'SPY', 'IWM', 'DIA']
    
    bt_config = BacktestConfig(
        transaction_cost_bps=1.0, 
        initial_capital=1_000_000.0,
        position_close_bar=77,
        borrow_rate_annual=0.0
    )
    entries = [3, 6, 14, 28, 30, 34]
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    
    for year in years:
        for q in [1, 2, 3, 4]:
            q_name = f"{year}_Q{q}"
            print(f"\n{'='*60}\n📅 Processing {q_name}\n{'='*60}")
            
            # Define Quarter Dates
            if q == 1:
                train_end = f"{year-1}-12-31"
                test_start = f"{year}-01-01"
                test_end = f"{year}-03-31"
            elif q == 2:
                train_end = f"{year}-03-31"
                test_start = f"{year}-04-01"
                test_end = f"{year}-06-30"
            elif q == 3:
                train_end = f"{year}-06-30"
                test_start = f"{year}-07-01"
                test_end = f"{year}-09-30"
            elif q == 4:
                train_end = f"{year}-09-30"
                test_start = f"{year}-10-01"
                test_end = f"{year}-12-31"

            # 2. Model Management
            use_path = None
            
            # A. Check if Q1 reuse is possible (Annual Model)
            if q == 1:
                # Check extended_history folder
                annual_path = annual_models_base / f"models_{year}"
                if annual_path.exists() and (annual_path / "moe_lgb" / "manifest.json").exists():
                    print(f"  ✅ Reusing Annual Q1 Model for {year}...")
                    use_path = annual_path

            # B. Check Local Cache (Already trained Q2-Q4 or missing Q1)
            if use_path is None:
                local_model_path = models_dir / f"models_{q_name}"
                if local_model_path.exists() and (local_model_path / "moe_lgb" / "manifest.json").exists():
                     print(f"  ✅ Found locally trained models for {q_name}")
                     use_path = local_model_path
                else:
                    # TRAIN NEW MODEL
                    print(f"  ⚡️ Training new models for {q_name} (Train: {train_start_fixed} to {train_end})...")
                    
                    train_moe_models(
                        train_start_date=train_start_fixed,
                        train_end_date=train_end,
                        output_dir=local_model_path,
                        config_path='config/intraday_config.yaml',
                        model_types=['xgb', 'rf', 'lgb']
                    )
                    use_path = local_model_path
            
            # Load Models
            lgb = MoEIntradayStrategy({}); lgb.load_model(str(use_path / 'moe_lgb'))
            xgb = MoEIntradayStrategy({}); xgb.load_model(str(use_path / 'moe_xgb'))
            rf = MoEIntradayStrategy({}); rf.load_model(str(use_path / 'moe_rf'))
            ensemble = EnsembleStrategy([lgb, xgb, rf], [1/3, 1/3, 1/3])
            
            # 3. Load Data & Predict
            print(f"  Loading data for {q_name}...")
            
            # Load appropriate period based on Year
            load_period = 'train'
            if year >= 2022: load_period = 'test'
            elif year >= 2019: load_period = 'valid'
            
            # Optimization: Pull overlapping period data once? 
            # Given memory constraints, we just load fresh.
            df_q = loader.get_period_data(load_period, symbols=etfs)
            
            # Filter strict dates
            mask = (df_q.index.get_level_values(0) >= test_start) & \
                   (df_q.index.get_level_values(0) <= test_end)
            df_q = df_q[mask]
            
            if df_q.empty:
                 # Fallback logic (e.g. 2018 is in Train, 2019 in Valid)
                 # Try other periods if empty
                for fallback in ['train', 'valid', 'test']:
                    if fallback == load_period: continue
                    print(f"  ⚠️ Empty slice in {load_period}, checking {fallback}...")
                    df_fallback = loader.get_period_data(fallback, symbols=etfs)
                    mask = (df_fallback.index.get_level_values(0) >= test_start) & \
                           (df_fallback.index.get_level_values(0) <= test_end)
                    df_q = df_fallback[mask]
                    if not df_q.empty: break
            
            if df_q.empty:
                print(f"❌ No data found for {q_name}. Skipping.")
                continue
                
            print(f"  Generating signals ({len(df_q)} bars)...")
            alpha = IntradayAlphaFeatures()
            df_q = alpha.generate_all_features(df_q)
            season = SeasonalityFeatures()
            df_q = season.generate_all_features(df_q)
            
            signals = ensemble.generate_signals(df_q)
            
            # 4. Run Strategy (AE200)
            print(f"  Running AE200 Strategy for {q_name}...")
            strat = MultiTradeStrategy(bt_config, exit_bars=36, allowed_entry_bars=entries, fixed_pos_pct=0.3333)
            res = strat.run(df_q, lambda d: signals, etfs)
            
            print(f"  Result: {res.total_return*100:.1f}% Return, {res.sharpe_ratio:.2f} Sharpe")
            
            # Store Daily Returns (Raw from Strategy Result)
            # Need to align dates
            # (Same logic as extended_history: derive from equity curve resampling)
            res_eq_df = res.equity_curve
            if not res_eq_df.empty:
                daily_vals = res_eq_df['equity'].resample('D').last().dropna()
                d_rets = daily_vals.pct_change().fillna(0)
                for d, r in d_rets.items():
                    daily_returns.append({'date': d, 'return': r})

            # Save Trades
            trade_df = res.trades
            if not trade_df.empty:
                trade_df['year'] = year
                trade_df['quarter'] = q
                trade_path = trades_dir / f"trades_{q_name}.csv"
                trade_df.to_csv(trade_path, index=False)
                all_trades.append(trade_df)
                print(f"  Saved {len(trade_df)} trades to {trade_path}")
            
    # 6. Final Aggregate
    if not all_trades:
        print("No trades generated!")
        return

    full_trade_df = pd.concat(all_trades)
    full_trade_path = output_dir / "all_trades_quarterly_2013_2025.csv"
    full_trade_df.to_csv(full_trade_path, index=False)
    print(f"\n💾 Saved All Trades to {full_trade_path}")
    
    # 7. Generate Dashboard
    # We call the dashboard generation script (need to create/import it)
    # For now, let's just save the CSV. The dashboard script will be separate.
    print("Dashboards should be generated separately using `scripts/regenerate_comparison_dashboard.py` (after update).")

if __name__ == "__main__":
    run_quarterly_validation()
