import pandas as pd
import sys
import json
import warnings
from pathlib import Path
from datetime import time

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy
from src.strategies.ml_models.xgboost_intraday import XGBoostIntradayStrategy
from src.strategies.ml_models.random_forest_intraday import RandomForestIntradayStrategy
from src.strategies.moe_strategy import MoEIntradayStrategy
from src.strategies.ensemble_strategy import EnsembleStrategy
from src.backtest.strategies.once_per_day import OncePerDayStrategy
from src.backtest.engine import BacktestConfig

def run_horizon_analysis():
    print("🚀 Starting Horizon Sensitivity Analysis (Ensemble MoE)...")
    
    # --- Load Ensemble Components ---
    print("Loading Models...")
    # LGB
    lgb_m = MoEIntradayStrategy({})
    lgb_m.load_model(str(Path.cwd() / 'results' / 'models' / 'moe'))
    
    # XGB
    xgb_m = MoEIntradayStrategy({})
    xgb_m.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_xgb'))
    
    # RF
    rf_m = MoEIntradayStrategy({})
    rf_m.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_rf'))
    
    # Ensemble
    ensemble_moe = EnsembleStrategy(
        strategies=[lgb_m, xgb_m, rf_m],
        weights=[1/3, 1/3, 1/3]
    )
    
    # --- Run for Validation Period ---
    run_period_sweep(ensemble_moe, 'valid')

def run_period_sweep(ensemble_moe, period):
    loader = IntradayDataLoader('config/intraday_config.yaml')
    print(f"Loading {period.capitalize()} Data...")
    df = loader.get_period_data(period)
    
    print("Generating features...")
    alpha = IntradayAlphaFeatures()
    df = alpha.generate_all_features(df)
    season = SeasonalityFeatures()
    df = season.generate_all_features(df)
    
    etfs = ['SPY', 'QQQ', 'DIA', 'IWM']
    horizons = [12, 18, 24, 30, 36]
    
    timings = [
        (2, '09:40 Open+10m'),
        (6, '10:00 Morning'),
        (12, '10:30 Late Morning'),
        (30, '12:00 Noon'),
        (42, '13:00 Afternoon'),
        (54, '14:00 Late Afternoon'),
        (66, ' Market Close'),
    ]
    
    # --- Analysis Loop ---
    results = {} # horizon -> {time -> metrics}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for h in horizons:
            print(f"\n⏳ Testing Horizon: {h} bars ({(h*5)/60:.1f} hours)")
            h_results = []
            
            config = BacktestConfig(transaction_cost_bps=1.0)
            
            for entry_bar, time_name in timings:
                print(f"  Time: {time_name}")
                
                # Run Backtest with specific horizon
                bt = OncePerDayStrategy(config, entry_bar=entry_bar, exit_bars=h)
                res = bt.run(df, ensemble_moe.generate_signals, etfs)
                
                metrics = {
                    'ret': res.total_return * 100,
                    'bp': res.trades['return_pct'].mean() * 10000 if len(res.trades) > 0 else 0,
                    'wr': res.win_rate * 100,
                    'trd': res.num_trades
                }
                h_results.append({'time': time_name, 'metrics': metrics})
                
            results[h] = h_results
            
    generate_dashboard(results, horizons, period)

def generate_dashboard(results, horizons, period):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Horizon Sensitivity ({period.capitalize()}): Ensemble MoE</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #e8e8e8; padding: 20px; }}
        h1 {{ text-align: center; color: #00d9ff; font-size: 28px; margin-bottom: 20px; }}
        .heatmap-table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.05); font-size: 14px; margin-bottom: 40px; }}
        th, td {{ padding: 12px; text-align: right; border: 1px solid rgba(255,255,255,0.1); }}
        th {{ background: rgba(0,0,0,0.3); text-align: center; }}
        th.row-header {{ text-align: left; min-width: 150px; }}
        
        .best {{ background: rgba(0, 255, 136, 0.15); color: #00ff88; font-weight: bold; }}
        .worst {{ background: rgba(255, 102, 102, 0.15); color: #ff6666; }}
        
        .metric-title {{ color: #aaa; margin-top: 30px; border-bottom: 1px solid #444; }}
    </style>
</head>
<body>
    <h1>⏳ Ensemble MoE: Horizon Sensitivity Analysis</h1>
    <p style="text-align:center; color:#888;">{period.capitalize()} Period | Metrics: Basis Points per Trade</p>
    """
    
    # Pivot results for easy table generation
    # Rows: Time, Cols: Horizon
    times = [r['time'] for r in results[horizons[0]]]
    
    # 1. Basis Points Table
    html += "<h2 class='metric-title'>Basis Points per Trade (BP)</h2>"
    html += "<table class='heatmap-table'><thead><tr><th class='row-header'>Entry Time</th>"
    for h in horizons:
        html += f"<th>{h} Bars ({(h*5)/60:.1f}h)</th>"
    html += "</tr></thead><tbody>"
    
    for i, t in enumerate(times):
        html += f"<tr><td class='row-header'>{t}</td>"
        vals = []
        for h in horizons:
            vals.append(results[h][i]['metrics']['bp'])
        
        max_val = max(vals)
        min_val = min(vals)
        
        for val in vals:
            cls = ""
            if val == max_val: cls = "best"
            elif val == min_val: cls = "worst"
            html += f"<td class='{cls}'>{val:.2f}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    
    # 2. Total Return Table
    html += "<h2 class='metric-title'>Total Return (%)</h2>"
    html += "<table class='heatmap-table'><thead><tr><th class='row-header'>Entry Time</th>"
    for h in horizons:
        html += f"<th>{h} Bars</th>"
    html += "</tr></thead><tbody>"
    
    for i, t in enumerate(times):
        html += f"<tr><td class='row-header'>{t}</td>"
        vals = []
        for h in horizons:
            vals.append(results[h][i]['metrics']['ret'])
        
        max_val = max(vals)
        
        for val in vals:
            cls = "best" if val == max_val else ""
            html += f"<td class='{cls}'>{val:.2f}%</td>"
        html += "</tr>"
    html += "</tbody></table>"
    
    # 3. Win Rate Table
    html += "<h2 class='metric-title'>Win Rate (%)</h2>"
    html += "<table class='heatmap-table'><thead><tr><th class='row-header'>Entry Time</th>"
    for h in horizons:
        html += f"<th>{h} Bars</th>"
    html += "</tr></thead><tbody>"
    
    for i, t in enumerate(times):
        html += f"<tr><td class='row-header'>{t}</td>"
        vals = []
        for h in horizons:
            vals.append(results[h][i]['metrics']['wr'])
        
        for val in vals:
            html += f"<td>{val:.1f}%</td>"
        html += "</tr>"
    html += "</tbody></table>"
    
    html += "</body></html>"
    
    filename = f'results/active/horizon_sensitivity_{period}_dashboard.html'
    with open(filename, 'w') as f:
        f.write(html)
    print(f"✅ Dashboard updated: {filename}")

if __name__ == "__main__":
    run_horizon_analysis()
