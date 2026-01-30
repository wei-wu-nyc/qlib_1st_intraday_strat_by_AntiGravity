import pandas as pd
import sys
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta, time
import numpy as np

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.strategies.moe_strategy import MoEIntradayStrategy
from src.strategies.ensemble_strategy import EnsembleStrategy
from src.backtest.strategies.multi_trade import MultiTradeStrategy
from src.backtest.engine import BacktestConfig

def run_experiment():
    print("🚀 Starting Single Bar Entry Experiment (Exhaustive)...")
    
    # --- Configuration ---
    years = range(2019, 2026) 
    base_results_dir = Path.cwd() / 'results' / 'rolling_validation'
    
    # Storage: {bar_time_str: {returns: [], exposures: []}}
    bar_results = {} 
    
    bt_config = BacktestConfig(
        transaction_cost_bps=1.0, 
        initial_capital=1_000_000.0,
        position_close_bar=77,
        borrow_rate_annual=0.0
    )
    etfs = ['QQQ', 'SPY', 'IWM', 'DIA']
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    
    # We will identify all unique bars dynamically from data
    all_bar_times = set()
    
    # Helper to map bar index to time string
    def get_time_str(bar_idx):
        # Bar 1 = 9:35, Bar 2 = 9:40 ...
        # min_from_open = (bar_idx - 1) * 5 + 5 = bar_idx * 5
        # market open 9:30. 
        # Time = 9:30 + bar_idx * 5 mins
        total_min = 9*60 + 30 + (bar_idx * 5)
        h = total_min // 60
        m = total_min % 60
        return f"{h:02d}:{m:02d}"

    for year in years:
        print(f"\n{'='*60}\n📅 Year: {year}\n{'='*60}")
        
        valid_start = f"{year}-01-01"
        valid_end = f"{year}-12-31"
        
        # 1. Load Models
        model_dir = base_results_dir / f"models_{year}"
        if not (model_dir.exists() and (model_dir / "moe_lgb" / "manifest.json").exists()):
            print(f"⚠️ Models for {year} not found! Skipping...")
            continue
            
        lgb_moe = MoEIntradayStrategy({})
        lgb_moe.load_model(str(model_dir / 'moe_lgb'))
        xgb_moe = MoEIntradayStrategy({})
        xgb_moe.load_model(str(model_dir / 'moe_xgb'))
        rf_moe = MoEIntradayStrategy({})
        rf_moe.load_model(str(model_dir / 'moe_rf'))
        
        ensemble = EnsembleStrategy(strategies=[lgb_moe, xgb_moe, rf_moe], weights=[1/3, 1/3, 1/3])
        
        # 2. Data
        target_period = 'test' if year >= 2022 else 'valid'
        df_year = loader.get_period_data(target_period, symbols=etfs)
        mask = (df_year.index.get_level_values(0) >= pd.Timestamp(valid_start)) & \
               (df_year.index.get_level_values(0) <= pd.Timestamp(valid_end))
        df_year = df_year[mask]
        
        if df_year.empty: continue
            
        alpha = IntradayAlphaFeatures()
        df_year = alpha.generate_all_features(df_year)
        season = SeasonalityFeatures()
        df_year = season.generate_all_features(df_year)
        
        # 3. Signals
        signals = ensemble.generate_signals(df_year)
        cols_needed = ['close', 'bar_index', 'signal', 'predicted_return']
        avail_cols = [c for c in cols_needed if c in signals.columns]
        signals_lean = signals[avail_cols].copy()
        
        # Identify available bars in this year
        available_bars = sorted(df_year['bar_index'].unique().astype(int))
        print(f"  Available Bar Indices: {min(available_bars)} to {max(available_bars)}")
        
        # 4. Run Backtest for EACH BAR
        for bar_idx in available_bars:
            # Skip EOD bars where we can't trade (min bars close logic might filter them anyway, but lets be thorough)
            # last_entry_bar is usually 15:00 (bar 66) in config, but user wants ALL bars.
            # We override config.last_entry_bar dynamically?
            # Actually MultiTradeStrategy uses self.config.last_entry_bar.
            # We should update the config passed to it.
            
            # Constraint: Can't enter if forced closed immediately. 
            # position_close_bar = 77.
            # If we enter at 77, we exit at 77. 0 holding time.
            # Let's run up to 76 (15:50).
            if bar_idx >= 77: continue
            
            time_str = get_time_str(bar_idx)
            
            # Create strategy for this specific SINGLE BAR
            strat = MultiTradeStrategy(
                bt_config, 
                exit_bars=36, 
                fixed_pos_pct=0.40, 
                allowed_entry_bars=[bar_idx]
            )
            # Run
            res = strat.run(df_year, lambda d: signals_lean, etfs)
            
            if time_str not in bar_results:
                bar_results[time_str] = {}
            all_bar_times.add(time_str)

            # Store returns by year
            if year not in bar_results[time_str]:
                bar_results[time_str][year] = []
            bar_results[time_str][year].extend(res.daily_returns)
            
    # --- Generate Statistics ---
    print("\nComputing Statistics...")
    
    def calc_stats_for_period(years_list):
        stats = []
        for t in sorted(list(all_bar_times)):
            merged_rets = []
            for y in years_list:
                if y in bar_results[t]:
                    merged_rets.extend(bar_results[t][y])
            
            if not merged_rets: continue
            
            arr = np.array(merged_rets)
            tot_ret = np.prod(1+arr) - 1
            n = len(arr)
            ann_ret = (1+tot_ret)**(252/n) - 1 if n > 0 else 0
            std = np.std(arr)
            sr = (np.mean(arr)/std)*np.sqrt(252) if std > 0 else 0
            
            stats.append({
                'Time': t, 'Annualized': ann_ret, 'TotalReturn': tot_ret, 'Sharpe': sr
            })
        return stats

    stats_full = calc_stats_for_period(range(2019, 2026))
    stats_early = calc_stats_for_period(range(2019, 2023)) # 2019-2022
    stats_late = calc_stats_for_period(range(2023, 2026))  # 2023-2025
        
    generate_dashboard(stats_full, stats_early, stats_late)

def generate_dashboard(stats_full, stats_early, stats_late):
    import json
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    def extract_data(stats):
        return {
            'times': [s['Time'] for s in stats],
            'rets': [s['Annualized'] * 100 for s in stats],
            'sharpes': [s['Sharpe'] for s in stats]
        }

    data_full = extract_data(stats_full)
    data_early = extract_data(stats_early)
    data_late = extract_data(stats_late)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Single Bar Entry Analysis (Sub-Periods)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', system-ui; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .charts {{ display: grid; grid-template-rows: 1fr 1fr; gap: 30px; height: 1200px; }}
        .chart-container {{ background: #1e293b; border-radius: 10px; padding: 20px; }}
        .section-title {{ color: #facc15; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-top: 40px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        th, td {{ padding: 8px; text-align: right; border-bottom: 1px solid #334155; }}
        th {{ text-align: right; color: #94a3b8; }}
        th:first-child, td:first-child {{ text-align: left; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⏱ Single Bar Entry: Period Comparison</h1>
        <p style="color: #64748b">Comparison: Full (2019-2025) vs Early (2019-2022) vs Late (2023-2025)</p>
    </div>

    <div class="charts">
        <div id="retChart" class="chart-container"></div>
        <div id="srChart" class="chart-container"></div>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
        <div>
            <h3 class="section-title">Full Period (2019-2025)</h3>
            <table>
                <thead><tr><th>Time</th><th>Ann %</th><th>SR</th></tr></thead>
                <tbody>
                    { "".join([f"<tr><td>{s['Time']}</td><td>{s['Annualized']*100:.1f}</td><td>{s['Sharpe']:.2f}</td></tr>" for s in stats_full]) }
                </tbody>
            </table>
        </div>
        <div>
            <h3 class="section-title">Early (2019-2022)</h3>
            <table>
                <thead><tr><th>Time</th><th>Ann %</th><th>SR</th></tr></thead>
                <tbody>
                    { "".join([f"<tr><td>{s['Time']}</td><td>{s['Annualized']*100:.1f}</td><td>{s['Sharpe']:.2f}</td></tr>" for s in stats_early]) }
                </tbody>
            </table>
        </div>
        <div>
            <h3 class="section-title">Late (2023-2025)</h3>
            <table>
                <thead><tr><th>Time</th><th>Ann %</th><th>SR</th></tr></thead>
                <tbody>
                    { "".join([f"<tr><td>{s['Time']}</td><td>{s['Annualized']*100:.1f}</td><td>{s['Sharpe']:.2f}</td></tr>" for s in stats_late]) }
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const full = {json.dumps(data_full, cls=NpEncoder)};
        const early = {json.dumps(data_early, cls=NpEncoder)};
        const late = {json.dumps(data_late, cls=NpEncoder)};
        
        const layout = {{
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            xaxis: {{ tickangle: -45, gridcolor: '#334155' }},
            yaxis: {{ gridcolor: '#334155' }},
            barmode: 'group'
        }};

        // Return Chart
        Plotly.newPlot('retChart', [
            {{ x: full.times, y: full.rets, name: 'Full (19-25)', type: 'bar', marker: {{ color: '#facc15' }} }},
            {{ x: early.times, y: early.rets, name: 'Early (19-22)', type: 'bar', marker: {{ color: '#60a5fa' }} }},
            {{ x: late.times, y: late.rets, name: 'Late (23-25)', type: 'bar', marker: {{ color: '#f472b6' }} }}
        ], {{ ...layout, title: 'Annualized Return by Period' }}, {{responsive: true}});
        
        // Sharpe Chart
        Plotly.newPlot('srChart', [
            {{ x: full.times, y: full.sharpes, name: 'Full (19-25)', type: 'bar', marker: {{ color: '#facc15' }} }},
            {{ x: early.times, y: early.sharpes, name: 'Early (19-22)', type: 'bar', marker: {{ color: '#60a5fa' }} }},
            {{ x: late.times, y: late.sharpes, name: 'Late (23-25)', type: 'bar', marker: {{ color: '#f472b6' }} }}
        ], {{ ...layout, title: 'Sharpe Ratio by Period' }}, {{responsive: true}});
    </script>
</body>
</html>
    """
    
    with open('results/active/single_bar_entry_dashboard.html', 'w') as f:
        f.write(html)
    print("✅ Dashboard generated: results/active/single_bar_entry_dashboard.html")

if __name__ == "__main__":
    run_experiment()
