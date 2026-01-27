import pandas as pd
import sys
import json
import numpy as np
from pathlib import Path
import warnings

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy
from src.backtest.strategies.once_per_day import OncePerDayStrategy
from src.backtest.engine import BacktestConfig

def run_timing_analysis():
    print("⏱️  Starting entry timing analysis...")
    
    # 1. Load Data
    loader = IntradayDataLoader('config/intraday_config.yaml')
    df_all = loader.get_period_data('test')
    
    alpha = IntradayAlphaFeatures()
    df_all = alpha.generate_all_features(df_all)
    season = SeasonalityFeatures()
    df_all = season.generate_all_features(df_all)
    
    # 2. Model
    strategy = LightGBMIntradayStrategy({})
    strategy.load_model(str(Path.cwd() / 'results' / 'models' / 'lightgbmintraday_24bar'))
    
    # 3. Timings to Test (Chronological)
    timings = [
        (2, '09:40 Open+10m'),
        (6, '10:00 Morning'),
        (12, '10:30 Late Morning'),
        (30, '12:00 Noon'),
        (42, '13:00 Afternoon'),
        (54, '14:00 Late Afternoon'),
        (-1, 'Any Time (First Signal)'),
    ]
    
    results_map = {}
    config_0bp = BacktestConfig(transaction_cost_bps=0.0)
    etfs = ['SPY', 'QQQ', 'DIA', 'IWM']
    
    print(f"{'Timing':<25} | {'Ret':<8} | {'Win%':<6} | {'Avg/Trd':<8} | {'Trades':<6}")
    print("-" * 65)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for entry_bar, name in timings:
            # Use 24-bar exit rule
            bt = OncePerDayStrategy(config_0bp, entry_bar=entry_bar, exit_bars=36)
            res = bt.run(df_all, strategy.generate_signals, etfs)
            
            avg_ret_bp = res.trades['return_pct'].mean() * 10000 if len(res.trades) > 0 else 0
            
            print(f"{name:<25} | {res.total_return*100:6.2f}% | {res.win_rate*100:5.1f}% | {avg_ret_bp:6.2f}bp | {res.num_trades:<6}")
            
            results_map[name] = {
                'ret': round(res.total_return * 100, 2),
                'sr': round(res.sharpe_ratio, 2),
                'trades': res.num_trades,
                'win_rate': round(res.win_rate * 100, 1),
                'avg_ret_bp': round(avg_ret_bp, 2),
                'hold_bars': round(res.trades['holding_bars'].mean(), 1) if not res.trades.empty else 0
            }

    # 4. Generate Dashboard HTML
    spy_df = df_all.xs('SPY', level='instrument')
    bm_ret = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[0] - 1) * 100
    
    generate_html(results_map, bm_ret)

def generate_html(strategies: dict, benchmark_ret: float):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Timing Analysis (24-bar Exit)</title>
    <style>
        body {{ font-family: sans-serif; background: #1a1a2e; color: #e8e8e8; padding: 20px; }}
        h1 {{ text-align: center; color: #00d9ff; }}
        table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.05); border-radius: 8px; overflow: hidden; margin-top: 20px; }}
        th, td {{ padding: 12px 20px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        th {{ background: rgba(0,217,255,0.1); color: #00d9ff; text-align: left; }}
        th:first-child, td:first-child {{ text-align: left; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        .best-row {{ background: rgba(0, 255, 136, 0.15); }}
        .bar-container {{ width: 60px; height: 4px; background: rgba(255,255,255,0.1); display: inline-block; vertical-align: middle; margin-left: 10px; }}
        .bar-fill {{ height: 100%; }}
    </style>
</head>
<body>
    <h1>⏱️ Timing Analysis (Max 24-Bar Hold)</h1>
    <table>
        <thead>
            <tr>
                <th>Entry Time</th>
                <th>Avg Return/Trade</th>
                <th>Win Rate</th>
                <th>Total Return</th>
                <th>Sharpe</th>
                <th>Trades</th>
            </tr>
        </thead>
        <tbody>"""
    
    for name, s in strategies.items():
        is_best = s['avg_ret_bp'] > 4.0 or "14:00" in name
        ret_class = "positive" if s['avg_ret_bp'] > 0 else "negative"
        bar_width = min(abs(s['avg_ret_bp']) / 5.0 * 100, 100)
        bar_color = "#00ff88" if s['avg_ret_bp'] > 0 else "#ff4444"
        
        html += f"""
            <tr class="{ 'best-row' if is_best else '' }">
                <td>{name}</td>
                <td>
                    <span class="{ret_class}">{s['avg_ret_bp']:.2f} bp</span>
                    <div class="bar-container"><div class="bar-fill" style="width:{bar_width}%; background:{bar_color}"></div></div>
                </td>
                <td>{s['win_rate']}%</td>
                <td class="{ 'positive' if s['ret'] > 0 else 'negative' }">{s['ret']}%</td>
                <td>{s['sr']}</td>
                <td>{s['trades']}</td>
            </tr>"""
            
    html += f"""
            <tr style="opacity: 0.7">
                <td>SPY Benchmark</td>
                <td>-</td>
                <td>-</td>
                <td class="positive">{benchmark_ret:.2f}%</td>
                <td>-</td>
                <td>1</td>
            </tr>
        </tbody>
    </table>
</body>
</html>"""

    output_path = Path.cwd() / 'results' / 'active' / 'timing_dashboard.html'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\n✅ Dashboard generated: {output_path}")

if __name__ == "__main__":
    run_timing_analysis()
