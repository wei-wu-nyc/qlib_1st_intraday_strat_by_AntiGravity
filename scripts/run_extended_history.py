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

def run_extended_history():
    print("🚀 Starting Extended History Validation (2013-2025)...")
    print("Strategy: AE200 (6-Slot, 2.0x, Short-Day Fix)")
    
    # 1. Config
    years = range(2013, 2026)
    train_start_fixed = "2008-01-01"
    
    # Base dirs
    output_dir = Path.cwd() / 'results' / 'extended_history'
    trades_dir = output_dir / 'trades'
    models_dir = output_dir / 'models' 
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Existing rolling models (reuse if possible) - for 2019-2025
    rolling_models_base = Path.cwd() / 'results' / 'rolling_validation'

    # Accumulators
    all_trades = []
    daily_returns = [] # (date, ret)
    equity_curve = [1.0] # starts at 1.0
    
    etfs = ['QQQ', 'SPY', 'IWM', 'DIA']
    
    # Strategy Config
    bt_config = BacktestConfig(
        transaction_cost_bps=1.0, 
        initial_capital=1_000_000.0,
        position_close_bar=77,
        borrow_rate_annual=0.0
    )
    # 9:45, 10:00, 10:40, 11:50, 12:00, 12:20 (Indices: 3, 6, 14, 28, 30, 34)
    entries = [3, 6, 14, 28, 30, 34]
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    
    for year in years:
        print(f"\n{'='*60}\n📅 Processing Year: {year}\n{'='*60}")
        
        # 2. Model Management
        # Check standard rolling dir first (for 2019+)
        current_model_path = rolling_models_base / f"models_{year}"
        use_path = None
        
        if current_model_path.exists() and (current_model_path / "moe_lgb" / "manifest.json").exists():
            print(f"  ✅ Found existing pre-trained models for {year}")
            use_path = current_model_path
        else:
            # Check if we already trained it in this run's local dir
            local_model_path = models_dir / f"models_{year}"
            if local_model_path.exists() and (local_model_path / "moe_lgb" / "manifest.json").exists():
                 print(f"  ✅ Found locally trained models for {year}")
                 use_path = local_model_path
            else:
                # TRAIN NEW MODEL
                print(f"  ⚡️ Training new models for {year} (Train: {train_start_fixed} to {year-1}-12-31)...")
                train_end = f"{year-1}-12-31"
                
                # We reuse the train_moe_models function
                # Note: train_moe_models assumes it loads data via loader config logic. 
                # We might need to trick it or pass filtered data?
                # Actually train_moe_models loads EVERYTHING available in loader.
                # So we just run it and let it filter by date.
                
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
        # We need data for just this year
        # Loader 'test' period starts 2022. 'valid' starts 2019. 'train' ends 2018.
        # So for 2013, we must pull from 'train'.
        print(f"  Loading data for {year}...")
        
        # Heuristic: Load 'train' if < 2019, 'valid' if < 2022, 'test' if >= 2022
        # Actually safer to load ALL and filter.
        # But allow specific Period loading to save RAM.
        load_period = 'train'
        if year >= 2022: load_period = 'test'
        elif year >= 2019: load_period = 'valid'
        
        df_year = loader.get_period_data(load_period, symbols=etfs)
        
        # Filter strict year
        mask = (df_year.index.get_level_values(0) >= f"{year}-01-01") & \
               (df_year.index.get_level_values(0) <= f"{year}-12-31")
        df_year = df_year[mask]
        
        if df_year.empty:
            print(f"❌ No data found for {year} in period {load_period}. Trying fallback...")
            # Fallback: maybe it's in a different period?
            # Try loading train + valid
            df_year = loader.get_period_data('train', symbols=etfs)
            mask = (df_year.index.get_level_values(0) >= f"{year}-01-01") & \
                   (df_year.index.get_level_values(0) <= f"{year}-12-31")
            df_year = df_year[mask]
        
        if df_year.empty:
            print("❌ Still no data. Skipping year.")
            continue
            
        print(f"  Generating signals ({len(df_year)} bars)...")
        alpha = IntradayAlphaFeatures()
        df_year = alpha.generate_all_features(df_year)
        season = SeasonalityFeatures()
        df_year = season.generate_all_features(df_year)
        
        signals = ensemble.generate_signals(df_year)
        
        # 4. Run Strategy (AE200)
        print("  Running AE200 Strategy...")
        # fixed_pos_pct = 2.0 / 6 = 0.3333
        strat = MultiTradeStrategy(bt_config, exit_bars=36, allowed_entry_bars=entries, fixed_pos_pct=0.3333)
        res = strat.run(df_year, lambda d: signals, etfs)
        
        # 5. Save & Accumulate
        print(f"  Result: {res.total_return*100:.1f}% Return, {res.sharpe_ratio:.2f} Sharpe")
        
        # Calculate Equity Curve Continuation
        # We stitch them together.
        # Current ec starts at 1.0 (relative to year start).
        # Convert to absolute relative to 2013 start.
        last_eq = equity_curve[-1]
        
        year_returns = res.daily_returns
        for r in year_returns:
            last_eq *= (1 + r)
            equity_curve.append(last_eq)
            
        # Store Daily Returns with Dates
        # Need dates from equity curve
        # res.equity_curve index is timestamp.
        # Align returns.
        # Simplest: use res.daily_returns assuming sequential days
        # Better: Extract dates from res.equity_curve (it has one entry per day if strategy logic holds? No, equity curve is Intraday? 
        # Wait, MultiTradeStrategy.equity_curve is list of (ts, val) per BAR). 
        # MultiTradeStrategy.daily_returns is list of floats per DAY.
        # We need dates. BacktestEngine doesn't expose list of dates directly in daily_returns structure. 
        # We can reconstruct from trades or index.
        # Let's extract from res.equity_curve (take last val of each day) to get dates.
        
        res_eq_df = res.equity_curve # This is dataframe in Results, list in Engine. run() returns Results -> DataFrame
        # Results.equity_curve is DataFrame index=ts, col=equity
        daily_vals = res_eq_df['equity'].resample('D').last().dropna()
        # Calculate daily returns from this exact series to match dates
        # Note: res.daily_returns might include overnight if modeled.
        # Let's trust res.daily_returns for value, but need dates.
        # Re-derive dates from df_year index unique dates
        trading_dates = sorted(list(set(df_year.index.get_level_values(0).date)))
        if len(trading_dates) != len(res.daily_returns):
            # Mismatch? likely short days or 0 return days.
            # Use daily_vals pct_change
             d_rets = daily_vals.pct_change().fillna(0)
             for d, r in d_rets.items():
                 daily_returns.append({'date': d, 'return': r})
        else:
            for d, r in zip(trading_dates, res.daily_returns):
                daily_returns.append({'date': d, 'return': r})

        # Save Trades
        trade_df = res.trades
        if not trade_df.empty:
            trade_df['year'] = year
            trade_path = trades_dir / f"trades_{year}.csv"
            trade_df.to_csv(trade_path, index=False)
            all_trades.append(trade_df)
            print(f"  Saved {len(trade_df)} trades to {trade_path}")
            
    # 6. Generate Dashboard
    if not all_trades:
        print("No trades generated!")
        return

    full_trade_df = pd.concat(all_trades)
    full_trade_path = output_dir / "all_trades_2013_2025.csv"
    full_trade_df.to_csv(full_trade_path, index=False)
    print(f"\n💾 Saved All Trades to {full_trade_path}")
    
    # Generate HTML Dashboard
    generate_comparison_dashboard(daily_returns, full_trade_df, output_dir)

def generate_comparison_dashboard(daily_returns_list, trades_df, output_dir):
    print("\n📊 Generating Comparison Dashboard...")
    
    df_ret = pd.DataFrame(daily_returns_list)
    df_ret['date'] = pd.to_datetime(df_ret['date'])
    df_ret.set_index('date', inplace=True)
    
    periods = {
        '2013-2015': ('2013-01-01', '2015-12-31'),
        '2016-2018': ('2016-01-01', '2018-12-31'),
        '2019-2022': ('2019-01-01', '2022-12-31'),
        '2023-2025': ('2023-01-01', '2025-12-31')
    }
    
    stats_cards = []
    
    for name, (start, end) in periods.items():
        mask = (df_ret.index >= start) & (df_ret.index <= end)
        sub = df_ret[mask]
        
        # Trade Analysis
        t_mask = (trades_df['entry_time'] >= start) & (trades_df['entry_time'] <= end)
        t_sub = trades_df[t_mask].copy()
        
        if sub.empty:
            stats_cards.append({'Period': name, 'Ret': '-', 'SR': '-', 'DD': '-', 'Count': 0})
            continue
            
        # Stats
        r = sub['return'].values
        tot = np.prod(1+r) - 1
        ann = (1+tot)**(252/len(r)) - 1
        vol = np.std(r)*np.sqrt(252)
        sr = (np.mean(r)/np.std(r))*np.sqrt(252) if np.std(r)>0 else 0
        eq = np.cumprod(1+r)
        dd = np.min((eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq))
        
        # Invested % (Approx from trades duration? Hard without exposure logs. Skip or estimate)
        # We will focus on detailed trade stats
        
        # Time of Day Stats
        t_sub['time'] = pd.to_datetime(t_sub['entry_time']).dt.strftime('%H:%M')
        tod_grp = t_sub.groupby('time')['pnl'].sum().to_dict()
        best_time = max(tod_grp, key=tod_grp.get) if tod_grp else "-"
        
        stats_cards.append({
            'Period': name,
            'Total Return': f"{tot*100:.1f}%",
            'Annualized': f"{ann*100:.1f}%",
            'Sharpe': f"{sr:.2f}",
            'Max DD': f"{dd*100:.1f}%",
            'Trade Count': len(t_sub),
            'Best Slot': best_time,
            'Win Rate': f"{(t_sub['return_pct']>0).mean()*100:.1f}%"
        })

    # Prepare HTML with specific layout
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Extended History Validation (2013-2025)</title>
    <style>
        body {{ font-family: system-ui; background: #0f172a; color: #fff; padding: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 8px; }}
        .card h2 {{ color: #38bdf8; margin-top: 0; }}
        .metric {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        .label {{ color: #94a3b8; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        td {{ padding: 5px 0; border-bottom: 1px solid #334155; }}
    </style>
</head>
<body>
    <h1>🕰 Extended History Validation (2013-2025)</h1>
    <p>Strategy: AE200 (2.0x Leverage, 6-Slots)</p>
    
    <div class="grid">
        { "".join([f'''
        <div class="card">
            <h2>{c['Period']}</h2>
            <div class="label">Total Return</div>
            <div class="metric" style="color: { 'lime' if '-' not in c['Total Return'] and float(c['Total Return'][:-1])>0 else 'white' }">{c['Total Return']}</div>
            
            <table>
                <tr><td>Annualized</td><td>{c['Annualized']}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{c['Sharpe']}</td></tr>
                <tr><td>Max Drawdown</td><td style="color: #f87171">{c['Max DD']}</td></tr>
                <tr><td>Win Rate</td><td>{c['Win Rate']}</td></tr>
                <tr><td>Trades</td><td>{c['Trade Count']}</td></tr>
                <tr><td>Best Slot</td><td style="color: #fbbf24">{c['Best Slot']}</td></tr>
            </table>
        </div>
        ''' for c in stats_cards]) }
    </div>
</body>
</html>
    """
    
    with open(output_dir / "comparison_dashboard.html", "w") as f:
        f.write(html)
    print(f"✅ Dashboard Saved: {output_dir / 'comparison_dashboard.html'}")

if __name__ == "__main__":
    run_extended_history()
