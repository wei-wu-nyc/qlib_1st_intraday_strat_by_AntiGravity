import pandas as pd
import sys
import json
import warnings
from pathlib import Path
from datetime import time
import numpy as np
import joblib

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.strategies.moe_strategy import MoEIntradayStrategy
from src.strategies.ensemble_strategy import EnsembleStrategy
from src.backtest.strategies.multi_trade import MultiTradeStrategy
from src.backtest.strategies.rebalance import RebalanceStrategy
from src.backtest.engine import BacktestConfig

def run_start_time_experiment():
    print("🚀 Starting Start Time Experiment (9:40 vs 9:30)...")
    
    # --- Load ORIGINAL Models (Full Universe) ---
    print("Loading Original Models...")
    lgb_model = MoEIntradayStrategy({})
    lgb_model.load_model(str(Path.cwd() / 'results' / 'models' / 'moe'))
    xgb_model = MoEIntradayStrategy({})
    xgb_model.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_xgb'))
    rf_model = MoEIntradayStrategy({})
    rf_model.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_rf'))
    ensemble = EnsembleStrategy(strategies=[lgb_model, xgb_model, rf_model], weights=[1/3, 1/3, 1/3])
    
    # --- Config ---
    periods = ['valid', 'test'] 
    loader = IntradayDataLoader('config/intraday_config.yaml')
    etfs = ['QQQ', 'SPY', 'IWM', 'DIA'] # Full Universe
    
    config = BacktestConfig(
        transaction_cost_bps=1.0, 
        initial_capital=1_000_000.0,
        position_close_bar=77 # 15:55
    )
    
    # ENTRY CONFIGS
    # 9:40 Start (Baseline): 2, 6, 12, 30, 54
    # 9:30 Start (Experiment): 0, 2, 6, 12, 30, 54
    entry_bars_940 = [2, 6, 12, 30, 54]
    entry_bars_930 = [0, 2, 6, 12, 30, 54]
    
    period_results = {}
    
    for period in periods:
        print(f"\n{'='*50}\nAnalyzing Period: {period.upper()}\n{'='*50}")
        
        # Load Data
        df = loader.get_period_data(period, symbols=etfs)
        
        print("Generating features...")
        alpha = IntradayAlphaFeatures()
        df = alpha.generate_all_features(df)
        season = SeasonalityFeatures()
        df = season.generate_all_features(df)
        
        # --- Generate Signals ---
        print("Generating Signals...")
        signals = ensemble.generate_signals(df)
        cols_needed = ['close', 'bar_index', 'signal', 'predicted_return']
        avail_cols = [c for c in cols_needed if c in signals.columns]
        signals_lean = signals[avail_cols].copy()
        
        # --- Strategies Breakdown ---
        
        # 1. Baseline: 9:40 Start (Max 2)
        print("Running Baseline: 9:40 Start (Max 2)...")
        strat_940_conc = MultiTradeStrategy(config, max_positions=2, exit_bars=36, allowed_entry_bars=entry_bars_940)
        res_940_conc = strat_940_conc.run(df, lambda d: signals_lean, etfs)
        
        # 2. Experiment: 9:30 Start (Max 2)
        print("Running Experiment: 9:30 Start (Max 2)...")
        strat_930_conc = MultiTradeStrategy(config, max_positions=2, exit_bars=36, allowed_entry_bars=entry_bars_930)
        res_930_conc = strat_930_conc.run(df, lambda d: signals_lean, etfs)
        
        # 3. Baseline: 9:40 Start (Dynamic)
        print("Running Baseline: 9:40 Start (Dynamic)...")
        strat_940_reb = RebalanceStrategy(config, max_positions=99, exit_bars=36, allowed_entry_bars=entry_bars_940)
        res_940_reb = strat_940_reb.run(df, lambda d: signals_lean, etfs)

        # 4. Experiment: 9:30 Start (Dynamic)
        print("Running Experiment: 9:30 Start (Dynamic)...")
        strat_930_reb = RebalanceStrategy(config, max_positions=99, exit_bars=36, allowed_entry_bars=entry_bars_930)
        res_930_reb = strat_930_reb.run(df, lambda d: signals_lean, etfs)
        
        # --- Benchmark (QQQ) ---
        if 'QQQ' in df.index.get_level_values(1).unique():
            spy_df = df.xs('QQQ', level=1)['close']
            spy_daily = spy_df.groupby(spy_df.index.date).last()
            bench_ret = spy_daily.pct_change().dropna()
        else:
            bench_ret = pd.Series([])

        period_results[period] = {
            'res_940_conc': res_940_conc,
            'res_930_conc': res_930_conc,
            'res_940_reb': res_940_reb,
            'res_930_reb': res_930_reb, 
            'benchmark': bench_ret
        }

    # --- Aggregation & Dashboard ---
    print("Aggregating Results...")
    final_data = {}
    
    def merge_periods(p_list, name):
        full_dates = []
        full_940_conc = []
        full_930_conc = []
        full_940_reb = []
        full_930_reb = []
        full_bench = []
        
        for p in p_list:
            r = period_results[p]
            
            # Helper to get aligned returns
            dates = sorted(list(set(pd.to_datetime(r['res_940_conc'].equity_curve.index).date)))
            
            # Align Returns
            def get_ret(res_obj, d_len):
                ret = res_obj.daily_returns[-min(len(res_obj.daily_returns), d_len):]
                if len(ret) < d_len:
                    ret = [0.0]*(d_len-len(ret)) + list(ret)
                return ret
                
            # Benchmark alignment
            p_bench = []
            for d in dates:
                if d in r['benchmark'].index:
                    p_bench.append(r['benchmark'].loc[d])
                else:
                    p_bench.append(0.0)
            
            full_dates.extend([d.strftime('%Y-%m-%d') for d in dates])
            full_940_conc.extend(get_ret(r['res_940_conc'], len(dates)))
            full_930_conc.extend(get_ret(r['res_930_conc'], len(dates)))
            full_940_reb.extend(get_ret(r['res_940_reb'], len(dates)))
            full_930_reb.extend(get_ret(r['res_930_reb'], len(dates)))
            full_bench.extend(p_bench)
            
        def make_eq(rets):
            eq = [1.0]
            for val in rets: eq.append(eq[-1] * (1 + val))
            return eq
            
        def calc_stats(returns):
            if not returns: return {}
            arr = np.array(returns)
            tot_ret = np.prod(1 + arr) - 1
            n = len(arr)
            ann_ret = (1 + tot_ret) ** (252/n) - 1 if n > 0 else 0
            vol = np.std(arr) * np.sqrt(252)
            sr = (np.mean(arr) / np.std(arr)) * np.sqrt(252) if np.std(arr) > 0 else 0
            
            eq = np.cumprod(1 + arr)
            cummax = np.maximum.accumulate(eq)
            dd = np.min((eq - cummax) / cummax)
            
            return {
                'Total Return': f"{tot_ret*100:.2f}%",
                'Annualized': f"{ann_ret*100:.2f}%",
                'Sharpe': f"{sr:.2f}",
                'Max DD': f"{dd*100:.2f}%",
                'Vol': f"{vol*100:.2f}%"
            }

        return {
            'dates': full_dates,
            'eq_940_conc': make_eq(full_940_conc)[1:],
            'eq_930_conc': make_eq(full_930_conc)[1:],
            'eq_940_reb': make_eq(full_940_reb)[1:],
            'eq_930_reb': make_eq(full_930_reb)[1:],
            'eq_bench': make_eq(full_bench)[1:],
            'stats_940_conc': calc_stats(full_940_conc),
            'stats_930_conc': calc_stats(full_930_conc),
            'stats_940_reb': calc_stats(full_940_reb),
            'stats_930_reb': calc_stats(full_930_reb),
            'stats_bench': calc_stats(full_bench)
        }

    final_data['valid_test'] = merge_periods(['valid', 'test'], 'Valid + Test')
    final_data['test'] = merge_periods(['test'], 'Test Period')
    
    generate_comparison_dashboard(final_data)

def generate_comparison_dashboard(final_data):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NpEncoder, self).default(obj)

    data_json = json.dumps(final_data, cls=NpEncoder)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Start Time Experiment Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        h1 {{ text-align: center; color: #38bdf8; font-size: 28px; margin-bottom: 20px; }}
        .stats-container {{ display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 30px; }}
        .stats-box {{ flex: 1; min-width: 200px; background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.1); }}
        .stats-title {{ border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 15px; font-size: 16px; font-weight: 600; color: #94a3b8; }}
        .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }}
        .controls {{ display: flex; justify-content: center; gap: 10px; margin-bottom: 30px; }}
        .btn {{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 10px 20px; cursor: pointer; border-radius: 6px; transition: 0.2s; }}
        .btn:hover {{ background: #334155; color: #fff; }}
        .btn.active {{ background: #38bdf8; color: #0f172a; font-weight: bold; border-color: #38bdf8; }}
        .chart-container {{ width: 100%; height: 600px; background: rgba(15, 23, 42, 0.5); border-radius: 8px; margin-bottom: 30px; border: 1px solid rgba(148, 163, 184, 0.1); }}
    </style>
</head>
<body>
    <h1>🕘 Start Time Experiment (9:40 vs 9:30)</h1>
    <p style="text-align:center; color:#64748b; margin-bottom: 30px;">Evaluating the impact of trading the first 10 minutes (Market Open)</p>

    <div class="controls">
        <button class="btn active" onclick="render('valid_test')" id="btn-valid_test">Valid + Test</button>
        <button class="btn" onclick="render('test')" id="btn-test">Test Only</button>
    </div>

    <div class="stats-container">
        <div class="stats-box">
             <div class="stats-title" style="color: #94a3b8;">Benchmark (QQQ)</div>
             <div id="bench-stats"></div>
        </div>
        <div class="stats-box">
             <div class="stats-title" style="color: #60a5fa;">Start 9:40 (Max 2)</div>
             <div id="940-conc-stats"></div>
        </div>
        <div class="stats-box">
             <div class="stats-title" style="color: #22d3ee;">Start 9:30 (Max 2)</div>
             <div id="930-conc-stats"></div>
        </div>
        <div class="stats-box">
             <div class="stats-title" style="color: #a78bfa;">Start 9:40 (Dyn)</div>
             <div id="940-reb-stats"></div>
        </div>
        <div class="stats-box">
             <div class="stats-title" style="color: #e879f9;">Start 9:30 (Dyn)</div>
             <div id="930-reb-stats"></div>
        </div>
    </div>

    <div id="chart" class="chart-container"></div>
    
    <script>
        const data = {data_json};

        function render(key) {{
            const d = data[key];
            
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + key).classList.add('active');

            const metrics = ['Total Return', 'Sharpe', 'Max DD'];
            function renderStats(id, stats) {{
                let h = "";
                metrics.forEach(m => {{
                    h += `<div class='stat-row'><span>${{m}}</span><span style='font-weight:bold'>${{stats[m]||'-'}}</span></div>`;
                }});
                document.getElementById(id).innerHTML = h;
            }}
            
            renderStats('bench-stats', d.stats_bench);
            renderStats('940-conc-stats', d.stats_940_conc);
            renderStats('930-conc-stats', d.stats_930_conc);
            renderStats('940-reb-stats', d.stats_940_reb);
            renderStats('930-reb-stats', d.stats_930_reb);
            
            const traces = [
                {{
                    x: d.dates, y: d.eq_bench, type: 'scatter', mode: 'lines', name: 'Benchmark (QQQ)',
                    line: {{color: '#64748b', width: 1, dash: 'dot'}}
                }},
                {{
                    x: d.dates, y: d.eq_940_conc, type: 'scatter', mode: 'lines', name: 'Start 9:40 (Max 2)',
                    line: {{color: '#60a5fa', width: 2, dash: 'dash'}}
                }},
                {{
                    x: d.dates, y: d.eq_930_conc, type: 'scatter', mode: 'lines', name: 'Start 9:30 (Max 2)',
                    line: {{color: '#22d3ee', width: 2}}
                }},
                {{
                    x: d.dates, y: d.eq_940_reb, type: 'scatter', mode: 'lines', name: 'Start 9:40 (Dyn)',
                    line: {{color: '#a78bfa', width: 2, dash: 'dash'}}
                }},
                 {{
                    x: d.dates, y: d.eq_930_reb, type: 'scatter', mode: 'lines', name: 'Start 9:30 (Dyn)',
                    line: {{color: '#e879f9', width: 2}}
                }}
            ];
            
            const layout = {{
                title: 'Cumulative Returns (Normalized)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#e2e8f0' }},
                xaxis: {{ showgrid: false, gridcolor: '#334155' }},
                yaxis: {{ showgrid: true, gridcolor: '#334155' }},
                margin: {{ l: 50, r: 20, t: 40, b: 40 }},
                legend: {{ orientation: 'h', y: -0.1 }}
            }};
            
            Plotly.newPlot('chart', traces, layout, {{responsive: true}});
        }}
        render('valid_test');
    </script>
</body>
</html>
    """
    
    out_path = Path('results/active/start_time_experiment_dashboard.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"✅ Dashboard generated: {out_path}")

if __name__ == "__main__":
    run_start_time_experiment()
