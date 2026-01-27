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

def run_ensemble_comparison():
    print("🚀 Starting Ensemble Model Comparison (Include Avg of 3 Models)...")
    
    # --- Load All Models ---
    print("Loading Models...")
    models = {}
    
    # LightGBM
    lgb_g = LightGBMIntradayStrategy({})
    lgb_g.load_model(str(Path.cwd() / 'results' / 'models' / 'lightgbmintraday_24bar'))
    
    lgb_m = MoEIntradayStrategy({})
    lgb_m.load_model(str(Path.cwd() / 'results' / 'models' / 'moe'))
    
    models['LGB'] = {'Global': lgb_g, 'MoE': lgb_m}
    
    # XGBoost
    xgb_g = XGBoostIntradayStrategy({})
    xgb_g.load_model(str(Path.cwd() / 'results' / 'models' / 'xgboostintraday_24bar'))
    
    xgb_m = MoEIntradayStrategy({})
    xgb_m.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_xgb'))
    
    models['XGB'] = {'Global': xgb_g, 'MoE': xgb_m}
    
    # Random Forest
    rf_g = RandomForestIntradayStrategy({})
    rf_g.load_model(str(Path.cwd() / 'results' / 'models' / 'randomforest_global'))
    
    rf_m = MoEIntradayStrategy({})
    rf_m.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_rf'))
    
    models['RF'] = {'Global': rf_g, 'MoE': rf_m}
    
    # --- Define Ensemble Models ---
    # Global Ensemble = Avg(LGB Global, XGB Global, RF Global)
    ensemble_global = EnsembleStrategy(
        strategies=[lgb_g, xgb_g, rf_g],
        weights=[1/3, 1/3, 1/3]
    )
    
    # MoE Ensemble = Avg(LGB MoE, XGB MoE, RF MoE)
    ensemble_moe = EnsembleStrategy(
        strategies=[lgb_m, xgb_m, rf_m],
        weights=[1/3, 1/3, 1/3]
    )
    
    models['Ensemble'] = {'Global': ensemble_global, 'MoE': ensemble_moe}
    
    # --- Comparison Loop ---
    periods = ['test', 'valid', 'train']
    all_results = {}
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    
    timings = [
        (2, '09:40 Open+10m'),
        (6, '10:00 Morning'),
        (12, '10:30 Late Morning'),
        (30, '12:00 Noon'),
        (42, '13:00 Afternoon'),
        (54, '14:00 Late Afternoon'),
        (66, '15:00 Market Close'),
    ]
    
    config = BacktestConfig(transaction_cost_bps=1.0)
    etfs = ['SPY', 'QQQ', 'DIA', 'IWM']
    
    for period in periods:
        print(f"\n{'='*50}")
        print(f"Analyzing Period: {period.upper()}")
        print(f"{'='*50}")
        
        df = loader.get_period_data(period)
        
        print("Generating features...")
        alpha = IntradayAlphaFeatures()
        df = alpha.generate_all_features(df)
        season = SeasonalityFeatures()
        df = season.generate_all_features(df)
        
        period_data = [] # List of rows
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for entry_bar, time_name in timings:
                print(f"  Time: {time_name}")
                row = {'time': time_name}
                
                # Order: LGB, XGB, RF, Ensemble
                fam_order = ['LGB', 'XGB', 'RF', 'Ensemble']
                
                for fam_name in fam_order:
                    fam_models = models[fam_name]
                    
                    # Global
                    bt_g = OncePerDayStrategy(config, entry_bar=entry_bar, exit_bars=24)
                    res_g = bt_g.run(df, fam_models['Global'].generate_signals, etfs)
                    
                    # MoE
                    bt_m = OncePerDayStrategy(config, entry_bar=entry_bar, exit_bars=24)
                    res_m = bt_m.run(df, fam_models['MoE'].generate_signals, etfs)
                    
                    def extract(res):
                        return {
                            'ret': res.total_return * 100,
                            'bp': res.trades['return_pct'].mean() * 10000 if len(res.trades) > 0 else 0,
                            'wr': res.win_rate * 100,
                            'trd': res.num_trades
                        }

                    row[f'{fam_name}_Global'] = extract(res_g)
                    row[f'{fam_name}_MoE'] = extract(res_m)
                
                period_data.append(row)
                
        all_results[period] = period_data
        generate_dashboard(all_results) # Save incrementally

def generate_dashboard(all_results):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Intraday Model Comparison: Ensemble Included</title>
    <style>
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #e8e8e8; padding: 20px; }
        h1 { text-align: center; color: #00d9ff; font-size: 28px; margin-bottom: 20px; }
        .period-section { margin-bottom: 60px; }
        h2 { color: #ccc; border-bottom: 1px solid #444; padding-bottom: 10px; margin-top: 40px; }
        
        /* Tabs */
        .tabs { display: flex; justify-content: center; margin-bottom: 20px; gap: 10px; }
        .tab-btn { background: #333; color: #aaa; border: none; padding: 10px 20px; cursor: pointer; border-radius: 4px; font-size: 16px; transition: 0.2s; }
        .tab-btn:hover { background: #444; }
        .tab-btn.active { background: #00d9ff; color: #000; font-weight: bold; }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        table { width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.05); font-size: 13px; margin-bottom: 20px; }
        th, td { padding: 10px; text-align: right; border: 1px solid rgba(255,255,255,0.1); }
        th { background: rgba(0,0,0,0.3); text-align: center; }
        th.row-header { text-align: left; min-width: 120px; }
        
        .pos { color: #00ff88; }
        .neg { color: #ff6666; }
        .neu { color: #888; }
        
        .fam-header { font-size: 16px; font-weight: bold; }
        .lgb-h { color: #ff99cc; background: rgba(255,153,204,0.1); }
        .xgb-h { color: #99ccff; background: rgba(153,204,255,0.1); }
        .rf-h  { color: #ccff99; background: rgba(204,255,153,0.1); }
        .ens-h { color: #ffd700; background: rgba(255, 215, 0, 0.1); }
        
    </style>
    <script>
        function showTab(period, fam) {
            // Hide all contents for this period
            document.querySelectorAll(`.content-${period}`).forEach(el => el.classList.remove('active'));
            document.querySelectorAll(`.btn-${period}`).forEach(el => el.classList.remove('active'));
            
            // Show selected
            document.getElementById(`tab-${period}-${fam}`).classList.add('active');
            document.getElementById(`btn-${period}-${fam}`).classList.add('active');
        }
    </script>
</head>
<body>
    <h1>🤖 Comparative Analysis: Ensemble vs Individual Models</h1>
    <p style="text-align:center; color:#888;">Adding Ensemble (Average of LGB + XGB + RF)</p>
    """
    
    periods_map = {'test': 'Test Period (2022-2025)', 'valid': 'Validation (2019-2021)', 'train': 'Training (2000-2018)'}
    model_fams = ['LGB', 'XGB', 'RF', 'Ensemble']
    colors = {'LGB': 'lgb-h', 'XGB': 'xgb-h', 'RF': 'rf-h', 'Ensemble': 'ens-h'}
    
    for pid in ['test', 'valid', 'train']:
        if pid not in all_results: continue
        rows = all_results[pid]
        pname = periods_map[pid]
        
        html += f"""
        <div class="period-section">
            <h2>{pname}</h2>
            
            <div class="tabs">
                <button id="btn-{pid}-Ensemble" class="tab-btn btn-{pid} active" onclick="showTab('{pid}', 'Ensemble')">⭐ Ensemble</button>
                <button id="btn-{pid}-LGB" class="tab-btn btn-{pid}" onclick="showTab('{pid}', 'LGB')">LightGBM</button>
                <button id="btn-{pid}-XGB" class="tab-btn btn-{pid}" onclick="showTab('{pid}', 'XGB')">XGBoost</button>
                <button id="btn-{pid}-RF" class="tab-btn btn-{pid}" onclick="showTab('{pid}', 'RF')">Random Forest</button>
            </div>
        """
        
        for fam in model_fams:
            active = 'active' if fam == 'Ensemble' else ''
            
            html += f"""
            <div id="tab-{pid}-{fam}" class="tab-content content-{pid} {active}">
                <table>
                    <thead>
                        <tr>
                            <th rowspan="2" class="row-header">Time of Day</th>
                            <th colspan="4" class="fam-header {colors[fam]}">Global {fam}</th>
                            <th colspan="4" class="fam-header {colors[fam]}">MoE {fam}</th>
                            <th rowspan="2">Edge (bp)</th>
                        </tr>
                        <tr>
                            <th>Ret</th><th>BP</th><th>Win%</th><th>Trds</th>
                            <th>Ret</th><th>BP</th><th>Win%</th><th>Trds</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for r in rows:
                g = r[f'{fam}_Global']
                m = r[f'{fam}_MoE']
                edge = m['bp'] - g['bp']
                edge_cls = 'pos' if edge > 0 else 'neg'
                
                def fmt(val, is_pct=False):
                    cls = 'pos' if val > 0 else 'neg'
                    suffix = '%' if is_pct else ''
                    return f'<td class="{cls}">{val:.2f}{suffix}</td>'
                
                html += f"""
                <tr>
                    <td class="row-header">{r['time']}</td>
                    {fmt(g['ret'], True)} <td>{g['bp']:.2f}</td> <td>{g['wr']:.1f}%</td> <td>{g['trd']}</td>
                    {fmt(m['ret'], True)} <td>{m['bp']:.2f}</td> <td>{m['wr']:.1f}%</td> <td>{m['trd']}</td>
                    <td class="{edge_cls}">{edge:+.2f}</td>
                </tr>
                """
            
            html += "</tbody></table></div>"
            
        html += "</div>"
        
    html += "</body></html>"
    
    with open('results/active/ensemble_comparison_dashboard.html', 'w') as f:
        f.write(html)
    print("✅ Dashboard updated: results/active/ensemble_comparison_dashboard.html")

if __name__ == "__main__":
    run_ensemble_comparison()
