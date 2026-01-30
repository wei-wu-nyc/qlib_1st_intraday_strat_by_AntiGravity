import pandas as pd
import sys
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.strategies.moe_strategy import MoEIntradayStrategy
from src.strategies.ensemble_strategy import EnsembleStrategy
from src.backtest.strategies.multi_trade import MultiTradeStrategy
from src.backtest.strategies.relaxed_entry_strategy import RelaxedMultiTradeStrategy
from src.backtest.engine import BacktestConfig

def run_experiment():
    print("🚀 Starting Relaxed Entry Experiment (Fixed vs Relaxed)...")
    
    # --- Configuration ---
    years = range(2019, 2026) 
    base_results_dir = Path.cwd() / 'results' / 'rolling_validation'
    
    accumulated_results = {
        'dates': [],
        'bench_returns': [],
        'Fixed_returns': [], 'Fixed_exposures': [], 'Fixed_max_exposures': [],
        'Relaxed_returns': [], 'Relaxed_exposures': [], 'Relaxed_max_exposures': [],
        'Relaxed_trades': [] 
    }
    
    annual_stats = []
    
    bt_config = BacktestConfig(
        transaction_cost_bps=1.0, 
        initial_capital=1_000_000.0,
        position_close_bar=77,
        borrow_rate_annual=0.0
    )
    etfs = ['QQQ', 'SPY', 'IWM', 'DIA']
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    
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
        
        # 4. Run Variants
        # Variant 1: Fixed AE200 (Default)
        print("  Running Fixed Entry...")
        strat_fixed = MultiTradeStrategy(bt_config, exit_bars=36, fixed_pos_pct=0.40)
        res_fixed = strat_fixed.run(df_year, lambda d: signals_lean, etfs)
        
        # Variant 2: Relaxed AE200
        print("  Running Relaxed Entry...")
        strat_relaxed = RelaxedMultiTradeStrategy(bt_config, exit_bars=36, fixed_pos_pct=0.40)
        res_relaxed = strat_relaxed.run(df_year, lambda d: signals_lean, etfs)
        
        # Store
        accumulated_results['Fixed_exposures'].append(res_fixed.daily_exposure)
        accumulated_results['Fixed_max_exposures'].append(res_fixed.max_exposure)
        
        accumulated_results['Relaxed_exposures'].append(res_relaxed.daily_exposure)
        accumulated_results['Relaxed_max_exposures'].append(res_relaxed.max_exposure)
        accumulated_results['Relaxed_trades'].extend(res_relaxed.trades.to_dict('records'))

        # Benchmark
        bench_ret = []
        if 'QQQ' in df_year.index.get_level_values(1).unique():
            spy_df = df_year.xs('QQQ', level=1)['close']
            spy_daily = spy_df.groupby(spy_df.index.date).last()
            bench_ret = spy_daily.pct_change().dropna()
        
        # Align
        dates = sorted(list(set(pd.to_datetime(res_fixed.equity_curve.index).date)))
        accumulated_results['dates'].extend([d.strftime('%Y-%m-%d') for d in dates])
        
        def get_aligned_ret(res_obj):
            r = res_obj.daily_returns
            needed = len(dates)
            if len(r) < needed: return [0.0]*(needed-len(r)) + r
            return r[-needed:]

        accumulated_results['bench_returns'].extend([bench_ret.loc[d] if d in bench_ret.index else 0.0 for d in dates])
        accumulated_results['Fixed_returns'].extend(get_aligned_ret(res_fixed))
        accumulated_results['Relaxed_returns'].extend(get_aligned_ret(res_relaxed))
            
        # Annual Stats
        def calc_s(r):
            if not r: return 0.0, 0.0
            r = np.array(r)
            return np.prod(1+r)-1, (np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r)>0 else 0.0

        yr_stats = {'Year': year}
        rb, sb = calc_s([bench_ret.loc[d] if d in bench_ret.index else 0.0 for d in dates])
        yr_stats['Bench_Ret'] = f"{rb*100:.1f}%"
        
        r, s = calc_s(res_fixed.daily_returns)
        yr_stats['Fixed_Ret'] = f"{r*100:.1f}%"
        
        r, s = calc_s(res_relaxed.daily_returns)
        yr_stats['Relaxed_Ret'] = f"{r*100:.1f}%"
        
        annual_stats.append(yr_stats)
        
    generate_dashboard(accumulated_results, annual_stats)

def generate_dashboard(data, annual_stats):
    def calc_full_stats(returns, name, exposures=None, max_exposures=None):
        if not returns: return {}
        arr = np.array(returns)
        tot = np.prod(1+arr)-1
        ann = (1+tot)**(252/len(arr))-1
        vol = np.std(arr)*np.sqrt(252)
        sr = (np.mean(arr)/np.std(arr))*np.sqrt(252) if np.std(arr)>0 else 0
        eq = np.cumprod(1+arr)
        dd = np.min((eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq))
        
        inv = f"{np.mean(exposures)*100:.1f}%" if exposures else "-"
        max_inv = f"{np.max(max_exposures)*100:.1f}%" if max_exposures else "-"
        
        return {
            'Strategy': name, 'Total Return': f"{tot*100:.1f}%", 'Annualized': f"{ann*100:.1f}%",
            'Sharpe': f"{sr:.2f}", 'Max DD': f"{dd*100:.1f}%", '% Invested': inv, 'Max Invested': max_inv
        }
        
    stats_list = []
    stats_list.append(calc_full_stats(data['bench_returns'], 'Benchmark', [1.0], [1.0]))
    stats_list.append(calc_full_stats(data['Fixed_returns'], 'AE 200 (Fixed/Current)', data['Fixed_exposures'], data['Fixed_max_exposures']))
    stats_list.append(calc_full_stats(data['Relaxed_returns'], 'AE 200 (Relaxed Mode)', data['Relaxed_exposures'], data['Relaxed_max_exposures']))

    # Entry Time Hist
    if data['Relaxed_trades']:
        df = pd.DataFrame(data['Relaxed_trades'])
        df['time'] = df['entry_time'].dt.strftime('%H:%M')
        # We want to see if it varies from 9:40, 10:00, etc.
        # Just top 10 most common entry times
        counts = df['time'].value_counts().head(15).sort_index()
        hist_html = "<ul>" + "".join([f"<li>{t}: {c} trades</li>" for t, c in counts.items()]) + "</ul>"
    else:
        hist_html = "No trades"

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
            
    def make_eq(rets):
        eq = [1.0]
        for v in rets: eq.append(eq[-1]*(1+v))
        return eq
    
    chart_data = {'dates': data['dates']}
    chart_data['bench'] = make_eq(data['bench_returns'])[1:]
    chart_data['Fixed'] = make_eq(data['Fixed_returns'])[1:]
    chart_data['Relaxed'] = make_eq(data['Relaxed_returns'])[1:]

    json_data = json.dumps(chart_data, cls=NpEncoder)
    
    def getColor(name): 
        return {'Benchmark':'#64748b', 'AE 200 (Fixed/Current)':'#facc15', 'AE 200 (Relaxed Mode)':'#4ade80'}.get(name,'#fff')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Relaxed Entry Experiment</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', system-ui; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .card {{ background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 10px; border: 1px solid #334155; text-align: center; }}
        .card h3 {{ color: #94a3b8; font-size: 14px; margin: 0 0 10px 0; }}
        .card .val {{ font-size: 18px; font-weight: bold; color: #f8fafc; }}
        .card .sub {{ font-size: 12px; color: #64748b; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; margin-bottom: 30px; }}
        th, td {{ padding: 12px; text-align: right; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 600; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧘 Relaxed Entry Experiment</h1>
        <p style="color: #64748b">Fixed (1st Bar Only) vs Relaxed (Any Bar in Block, Single Entry)</p>
    </div>
    
    <div class="card-grid">
        { "".join([f'''
        <div class="card" style="border-top: 3px solid {getColor(s['Strategy'])}">
            <h3>{s['Strategy']}</h3>
            <div class="val">{s['Total Return']}</div>
            <div class="sub">Ann: {s['Annualized']} | Invest: {s['% Invested']} (Max {s['Max Invested']})</div>
            <div class="sub">SR: {s['Sharpe']} | DD: {s['Max DD']}</div>
        </div>
        ''' for s in stats_list]) }
    </div>

    <div id="chart" style="height: 600px; background: #1e293b; border-radius: 10px; margin-bottom: 30px;"></div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
            <h3>📅 Annual Returns</h3>
            <table>
                <thead>
                    <tr><th>Year</th><th>Benchmark</th><th>Fixed</th><th>Relaxed</th></tr>
                </thead>
                <tbody>
                    { "".join([f'''<tr><td>{r['Year']}</td><td>{r['Bench_Ret']}</td><td>{r['Fixed_Ret']}</td><td>{r['Relaxed_Ret']}</td></tr>''' for r in annual_stats]) }
                </tbody>
            </table>
        </div>
        <div>
            <h3>⏰ Entry Distribution (Relaxed)</h3>
            <div style="background: #1e293b; padding: 15px; border-radius: 8px;">
                {hist_html}
            </div>
        </div>
    </div>

    <script>
        const data = {json_data};
        const traces = [
            {{ x: data.dates, y: data.bench, name: 'Benchmark', line: {{ color: '#64748b', dash: 'dot' }} }},
            {{ x: data.dates, y: data.Fixed, name: 'Fixed', line: {{ color: '#facc15' }} }},
            {{ x: data.dates, y: data.Relaxed, name: 'Relaxed', line: {{ color: '#4ade80', width: 3 }} }}
        ];
        Plotly.newPlot('chart', traces, {{
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }}, xaxis: {{ gridcolor: '#334155' }},
            yaxis: {{ gridcolor: '#334155', type: 'log' }},
            margin: {{ t: 20, l: 40, r: 20, b: 40 }},
            legend: {{ orientation: 'h', y: 1.05 }}
        }}, {{responsive: true}});
    </script>
</body>
</html>
    """
    
    with open('results/active/relaxed_entry_dashboard.html', 'w') as f:
        f.write(html)
    print("✅ Dashboard generated: results/active/relaxed_entry_dashboard.html")

if __name__ == "__main__":
    run_experiment()
