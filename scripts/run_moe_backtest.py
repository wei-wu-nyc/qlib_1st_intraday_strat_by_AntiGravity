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
from src.strategies.moe_strategy import MoEIntradayStrategy
from src.backtest.strategies.once_per_day import OncePerDayStrategy
from src.backtest.engine import BacktestConfig

def run_moe_comparison():
    print("🚀 Starting MoE vs Global Comparison (Full History)...")
    
    # Load Models Once
    print("Loading Models...")
    global_strategy = LightGBMIntradayStrategy({})
    global_strategy.load_model(str(Path.cwd() / 'results' / 'models' / 'lightgbmintraday_24bar'))
    
    moe_strategy = MoEIntradayStrategy({})
    moe_strategy.load_model(str(Path.cwd() / 'results' / 'models' / 'moe'))
    
    periods = ['train', 'valid', 'test']
    all_results = {}
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    
    for period in periods:
        print(f"\n{'='*50}")
        print(f"Analyzing Period: {period.upper()}")
        print(f"{'='*50}")
        
        # 1. Load Data
        df = loader.get_period_data(period)
        
        # 2. Features
        print("Generating features...")
        alpha = IntradayAlphaFeatures()
        df = alpha.generate_all_features(df)
        season = SeasonalityFeatures()
        df = season.generate_all_features(df)
        
        # 3. Compare by Time Block
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
        
        period_results = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for entry_bar, name in timings:
                print(f"  Time: {name}")
                
                # Run Global
                bt_g = OncePerDayStrategy(config, entry_bar=entry_bar, exit_bars=36)
                res_g = bt_g.run(df, global_strategy.generate_signals, etfs)
                
                # Run MoE
                bt_m = OncePerDayStrategy(config, entry_bar=entry_bar, exit_bars=36)
                res_m = bt_m.run(df, moe_strategy.generate_signals, etfs)
                
                # Metrics
                g_ret = res_g.total_return * 100
                m_ret = res_m.total_return * 100
                g_bp = res_g.trades['return_pct'].mean() * 10000 if len(res_g.trades) > 0 else 0
                m_bp = res_m.trades['return_pct'].mean() * 10000 if len(res_m.trades) > 0 else 0
                
                period_results.append({
                    'name': name,
                    'global_ret': g_ret, 'global_bp': g_bp, 'global_trd': res_g.num_trades,
                    'moe_ret': m_ret, 'moe_bp': m_bp, 'moe_trd': res_m.num_trades
                })
        
        all_results[period] = period_results

    # 4. Generate Dashboard
    generate_comparison_html(all_results)

def generate_comparison_html(all_results: dict):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MoE vs Global Strategy Comparison</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #e8e8e8; padding: 40px; }}
        h1 {{ text-align: center; color: #00d9ff; font-size: 32px; margin-bottom: 10px; }}
        p {{ font-size: 18px; margin-bottom: 40px; }}
        h2 {{ color: #ccc; border-bottom: 2px solid #333; padding-bottom: 15px; margin-top: 60px; font-size: 24px; }}
        table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.05); border-radius: 12px; margin-top: 20px; font-size: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        th, td {{ padding: 16px 24px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.08); }}
        th {{ background: rgba(0,217,255,0.15); color: #00d9ff; text-align: center; font-weight: 600; font-size: 18px; }}
        th:first-child, td:first-child {{ text-align: left; font-weight: 500; }}
        .positive {{ color: #00ff88; font-weight: bold; }}
        .negative {{ color: #ff6666; font-weight: bold; }}
        .winner {{ background: rgba(0, 255, 136, 0.08); }}
        .group-header {{ background: rgba(255,255,255,0.1); font-size: 18px; letter-spacing: 1px; color: #fff; }}
    </style>
</head>
<body>
    <h1>🧠 MoE vs Global Model Comparison</h1>
    <p style="text-align: center; color: #888">Comparison across Training (In-Sample), Validation, and Test (Out-of-Sample) Periods</p>
    """
    
    # Order: Train, Valid, Test
    names = {'train': 'Training Period (2000-2018)', 'valid': 'Validation Period (2019-2021)', 'test': 'Test Period (2022-2025)'}
    
    for period in ['train', 'valid', 'test']:
        results = all_results.get(period, [])
        if not results: continue
        
        html += f"""
        <h2>{names[period]}</h2>
        <table>
            <thead>
                <tr>
                    <th rowspan="2">Time Period</th>
                    <th colspan="3" class="group-header">Global Model</th>
                    <th colspan="3" class="group-header">MoE Model</th>
                    <th rowspan="2">Improvement (bp)</th>
                </tr>
                <tr>
                    <th>Total Ret</th>
                    <th>Avg/Trd</th>
                    <th>Trades</th>
                    <th>Total Ret</th>
                    <th>Avg/Trd</th>
                    <th>Trades</th>
                </tr>
            </thead>
            <tbody>"""
            
        for r in results:
            diff_bp = r['moe_bp'] - r['global_bp']
            is_moe_better = diff_bp > 0.5
            imp_class = "positive" if diff_bp > 0 else "negative"
            row_class = "winner" if is_moe_better else ""
            
            html += f"""
                <tr class="{row_class}">
                    <td>{r['name']}</td>
                    <td class="{ 'positive' if r['global_ret']>0 else 'negative' }">{r['global_ret']:.2f}%</td>
                    <td>{r['global_bp']:.2f} bp</td>
                    <td>{r['global_trd']}</td>
                    
                    <td class="{ 'positive' if r['moe_ret']>0 else 'negative' }">{r['moe_ret']:.2f}%</td>
                    <td>{r['moe_bp']:.2f} bp</td>
                    <td>{r['moe_trd']}</td>
                    
                    <td class="{imp_class}">{diff_bp:+.2f} bp</td>
                </tr>"""
                
        html += "</tbody></table>"

    html += """</body></html>"""
    
    output_path = Path.cwd() / 'results' / 'active' / 'moe_full_dashboard.html'
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\n✅ Full Comparison Dashboard: {output_path}")

if __name__ == "__main__":
    run_moe_comparison()
