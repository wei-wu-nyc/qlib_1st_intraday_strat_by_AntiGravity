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
from src.strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy
from src.strategies.ml_models.xgboost_intraday import XGBoostIntradayStrategy
from src.strategies.ml_models.random_forest_intraday import RandomForestIntradayStrategy
from src.strategies.moe_strategy import MoEIntradayStrategy
from src.strategies.ensemble_strategy import EnsembleStrategy
from src.backtest.strategies.multi_trade import MultiTradeStrategy
from src.backtest.strategies.rebalance import RebalanceStrategy
from src.backtest.engine import BacktestConfig

def run_qqq_simulation():
    print("🚀 Starting QQQ-Only Simulation & Comparison...")
    
    # --- Load OLD Models (Full Universe) ---
    print("Loading OLD Models (for reference)...")
    # Note: These paths assume the standard 'results/models' structure exists
    lgb_old = MoEIntradayStrategy({})
    lgb_old.load_model(str(Path.cwd() / 'results' / 'models' / 'moe'))
    xgb_old = MoEIntradayStrategy({})
    xgb_old.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_xgb'))
    rf_old = MoEIntradayStrategy({})
    rf_old.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_rf'))
    ensemble_old = EnsembleStrategy(strategies=[lgb_old, xgb_old, rf_old], weights=[1/3, 1/3, 1/3])
    
    # --- Load NEW Models (QQQ Only) ---
    print("Loading NEW Models (QQQ Only)...")
    base_new = Path.cwd() / 'results' / 'models_qqq_only'
    lgb_new = MoEIntradayStrategy({})
    lgb_new.load_model(str(base_new / 'moe_lgb'))
    xgb_new = MoEIntradayStrategy({})
    xgb_new.load_model(str(base_new / 'moe_xgb'))
    rf_new = MoEIntradayStrategy({})
    rf_new.load_model(str(base_new / 'moe_rf'))
    ensemble_new = EnsembleStrategy(strategies=[lgb_new, xgb_new, rf_new], weights=[1/3, 1/3, 1/3])
    
    # --- Config ---
    periods = ['valid', 'test'] # Focus on out-of-sample for comparison
    loader = IntradayDataLoader('config/intraday_config.yaml')
    etfs_qqq = ['QQQ']
    
    config = BacktestConfig(
        transaction_cost_bps=1.0, 
        initial_capital=1_000_000.0,
        position_close_bar=77 # 15:55
    )
    
    period_results = {}
    
    for period in periods:
        print(f"\n{'='*50}\nAnalyzing Period: {period.upper()}\n{'='*50}")
        
        # Load Data (QQQ Only is sufficient for QQQ strategies)
        # Ideally we load same data for fairness
        df = loader.get_period_data(period, symbols=['QQQ', 'SPY']) # Load SPY for robustness/checks if needed
        # Filter to just QQQ for trading
        # Actually, MultiTradeStrategy takes an ETF list, so passing full DF is fine.
        
        print("Generating features...")
        alpha = IntradayAlphaFeatures()
        df = alpha.generate_all_features(df)
        season = SeasonalityFeatures()
        df = season.generate_all_features(df)
        
        # --- Generate Signals (OLD) ---
        print("Generating Signals (OLD Models)...")
        signals_old = ensemble_old.generate_signals(df)
        cols_needed = ['close', 'bar_index', 'signal', 'predicted_return']
        avail_cols = [c for c in cols_needed if c in signals_old.columns]
        signals_old_lean = signals_old[avail_cols].copy()
        
        # --- Generate Signals (NEW) ---
        print("Generating Signals (NEW Models)...")
        signals_new = ensemble_new.generate_signals(df)
        signals_new_lean = signals_new[avail_cols].copy()
        
        # --- Strategies Breakdown ---
        
        # 1. OLD: QQQ Only (Max 2)
        print("Running OLD Strategy (QQQ Only Max 2)...")
        strat_old_conc = MultiTradeStrategy(config, max_positions=2, exit_bars=36)
        res_old_conc = strat_old_conc.run(df, lambda d: signals_old_lean, etfs_qqq)
        
        # 2. OLD: QQQ Only (Dynamic)
        print("Running OLD Strategy (QQQ Only Dynamic)...")
        strat_old_reb = RebalanceStrategy(config, max_positions=99, exit_bars=36)
        res_old_reb = strat_old_reb.run(df, lambda d: signals_old_lean, etfs_qqq)
        
        # 3. NEW: QQQ Only (Max 2)
        print("Running NEW Strategy (QQQ Only Max 2)...")
        strat_new_conc = MultiTradeStrategy(config, max_positions=2, exit_bars=36)
        res_new_conc = strat_new_conc.run(df, lambda d: signals_new_lean, etfs_qqq)
        
        # 4. NEW: QQQ Only (Dynamic)
        print("Running NEW Strategy (QQQ Only Dynamic)...")
        strat_new_reb = RebalanceStrategy(config, max_positions=99, exit_bars=36)
        res_new_reb = strat_new_reb.run(df, lambda d: signals_new_lean, etfs_qqq)
        
        # --- Benchmark (QQQ) ---
        # Robust check if 'QQQ' in DF
        if 'QQQ' in df.index.get_level_values(1).unique():
            spy_df = df.xs('QQQ', level=1)['close']
            spy_daily = spy_df.groupby(spy_df.index.date).last()
            bench_ret = spy_daily.pct_change().dropna()
        else:
            bench_ret = pd.Series([])

        period_results[period] = {
            'old_conc': res_old_conc,
            'old_reb': res_old_reb,
            'new_conc': res_new_conc,
            'new_reb': res_new_reb,
            'benchmark': bench_ret,
            'rec_exposures_new_conc': strat_new_conc.daily_exposures,
            'rec_exposures_new_reb': strat_new_reb.daily_exposures
        }

    # --- Aggregation & Dashboard ---
    print("Aggregating Results...")
    final_data = {}
    
    def merge_periods(p_list, name):
        full_dates = []
        full_old_conc = []
        full_old_reb = []
        full_new_conc = []
        full_new_reb = []
        full_bench = []
        
        merged_exp_new_conc = []
        merged_exp_new_reb = []
        
        for p in p_list:
            r = period_results[p]
            
            # Helper to get aligned returns
            # We assume all strategies run on same dates since input DF is same
            dates = sorted(list(set(pd.to_datetime(r['old_conc'].equity_curve.index).date)))
            
            # Align Returns
            def get_ret(res_obj, d_len):
                ret = res_obj.daily_returns[-min(len(res_obj.daily_returns), d_len):]
                if len(ret) < d_len:
                    ret = [0.0]*(d_len-len(ret)) + list(ret)
                return ret
                
            def get_exp(e_list, d_len):
                ex = e_list[-min(len(e_list), d_len):]
                if len(ex) < d_len:
                     ex = [0.0]*(d_len-len(ex)) + list(ex)
                return ex

            # Benchmark alignment
            p_bench = []
            for d in dates:
                if d in r['benchmark'].index:
                    p_bench.append(r['benchmark'].loc[d])
                else:
                    p_bench.append(0.0)
            
            full_dates.extend([d.strftime('%Y-%m-%d') for d in dates])
            full_old_conc.extend(get_ret(r['old_conc'], len(dates)))
            full_old_reb.extend(get_ret(r['old_reb'], len(dates)))
            full_new_conc.extend(get_ret(r['new_conc'], len(dates)))
            full_new_reb.extend(get_ret(r['new_reb'], len(dates)))
            full_bench.extend(p_bench)
            
            merged_exp_new_conc.extend(get_exp(r['rec_exposures_new_conc'], len(dates)))
            merged_exp_new_reb.extend(get_exp(r['rec_exposures_new_reb'], len(dates)))
            
        def make_eq(rets):
            eq = [1.0]
            for val in rets: eq.append(eq[-1] * (1 + val))
            return eq
            
        def calc_stats(returns, exposures=None):
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
            
            stats = {
                'Total Return': f"{tot_ret*100:.2f}%",
                'Annualized': f"{ann_ret*100:.2f}%",
                'Sharpe': f"{sr:.2f}",
                'Max DD': f"{dd*100:.2f}%",
                'Vol': f"{vol*100:.2f}%"
            }
            if exposures is not None and len(exposures) > 0:
                 avg_inv = np.mean(exposures)
                 stats['Avg Invested'] = f"{avg_inv*100:.1f}%"
            return stats

        def calc_tod_stats(trades_list):
            if not trades_list: return "<div style='text-align:center'>No Trades</div>"
            df_trades = pd.concat(trades_list)
            if df_trades.empty or 'entry_time' not in df_trades.columns:
                 return "<div style='text-align:center'>No Trades</div>"
            
            df_trades['entry_time_str'] = df_trades['entry_time'].dt.strftime('%H:%M')
            order = ['09:40', '10:00', '10:30', '12:00', '14:00']
            
            html = "<table><thead><tr><th class='left'>Time</th><th>Total PnL</th><th>Win Rate</th><th>Avg Return</th><th>Trades</th></tr></thead><tbody>"
            grp = df_trades.groupby('entry_time_str')
            
            for t_slot in order:
                if t_slot in grp.groups:
                    g = grp.get_group(t_slot)
                    pnl = g['pnl'].sum()
                    wr = (g['return_pct'] > 0).mean() * 100
                    avg = g['return_pct'].mean() * 10000
                    cnt = len(g)
                    cls = 'pos' if pnl > 0 else 'neg'
                    html += f"<tr><td class='left'>{t_slot}</td><td class='{cls}'>${pnl:,.0f}</td><td>{wr:.1f}%</td><td class='{cls}'>{avg:.1f} bp</td><td>{cnt}</td></tr>"
            html += "</tbody></table>"
            return html

        return {
            'dates': full_dates,
            'old_conc_eq': make_eq(full_old_conc)[1:],
            'old_reb_eq': make_eq(full_old_reb)[1:],
            'new_conc_eq': make_eq(full_new_conc)[1:],
            'new_reb_eq': make_eq(full_new_reb)[1:],
            'bench_eq': make_eq(full_bench)[1:],
            'old_conc_stats': calc_stats(full_old_conc),
            'old_reb_stats': calc_stats(full_old_reb),
            'new_conc_stats': calc_stats(full_new_conc, merged_exp_new_conc),
            'new_reb_stats': calc_stats(full_new_reb, merged_exp_new_reb),
            'bench_stats': calc_stats(full_bench, [1.0]*len(full_bench)),
            'old_tod_html': calc_tod_stats([pd.DataFrame(t) for t in [r['old_conc'].trades for r in [period_results[p] for p in p_list] if not r['old_conc'].trades.empty]]),
            'new_tod_html': calc_tod_stats([pd.DataFrame(t) for t in [r['new_conc'].trades for r in [period_results[p] for p in p_list] if not r['new_conc'].trades.empty]])
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
    <title>QQQ-Only Experiment Dashboard</title>
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
        
        table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.05); font-size: 14px; }}
        th, td {{ padding: 12px; text-align: right; border: 1px solid rgba(255,255,255,0.1); }}
        th {{ background: rgba(0,0,0,0.3); text-align: center; }}
        th.left {{ text-align: left; }}
        .pos {{ color: #4ade80; }}
        .neg {{ color: #f87171; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>🧪 QQQ-Only Experiment Results</h1>
    <p style="text-align:center; color:#64748b; margin-bottom: 30px;">Comparing models trained on Full Universe vs Models trained on QQQ Only</p>

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
             <div class="stats-title" style="color: #60a5fa;">OLD: Max 2</div>
             <div id="old-conc-stats"></div>
        </div>
        <div class="stats-box">
             <div class="stats-title" style="color: #22d3ee;">NEW: Max 2 (QQQ-Only Models)</div>
             <div id="new-conc-stats"></div>
        </div>
        <div class="stats-box">
             <div class="stats-title" style="color: #a78bfa;">OLD: Dynamic</div>
             <div id="old-reb-stats"></div>
        </div>
        <div class="stats-box">
             <div class="stats-title" style="color: #e879f9;">NEW: Dyn (QQQ-Only Models)</div>
             <div id="new-reb-stats"></div>
        </div>
    </div>

    <div id="chart" class="chart-container"></div>
    
    <div class="grid-2">
        <div>
            <h3>⏱️ Time-of-Day Analysis (OLD: Max 2)</h3>
            <div id="old-tod"></div>
        </div>
        <div>
            <h3>⏱️ Time-of-Day Analysis (NEW: Max 2)</h3>
            <div id="new-tod"></div>
        </div>
    </div>
    
    <script>
        const data = {data_json};

        function render(key) {{
            const d = data[key];
            
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + key).classList.add('active');

            const metrics = ['Total Return', 'Sharpe', 'Max DD', 'Avg Invested'];
            function renderStats(id, stats) {{
                let h = "";
                metrics.forEach(m => {{
                    h += `<div class='stat-row'><span>${{m}}</span><span style='font-weight:bold'>${{stats[m]||'-'}}</span></div>`;
                }});
                document.getElementById(id).innerHTML = h;
            }}
            
            renderStats('bench-stats', d.bench_stats);
            renderStats('old-conc-stats', d.old_conc_stats);
            renderStats('new-conc-stats', d.new_conc_stats);
            renderStats('old-reb-stats', d.old_reb_stats);
            renderStats('new-reb-stats', d.new_reb_stats);
            
            // Render ToD Tables
            document.getElementById('old-tod').innerHTML = d.old_tod_html;
            document.getElementById('new-tod').innerHTML = d.new_tod_html;
            
            const traces = [
                {{
                    x: d.dates, y: d.bench_eq, type: 'scatter', mode: 'lines', name: 'Benchmark (QQQ)',
                    line: {{color: '#64748b', width: 1, dash: 'dot'}}
                }},
                {{
                    x: d.dates, y: d.old_conc_eq, type: 'scatter', mode: 'lines', name: 'OLD: Max 2',
                    line: {{color: '#60a5fa', width: 2, dash: 'dash'}}
                }},
                {{
                    x: d.dates, y: d.new_conc_eq, type: 'scatter', mode: 'lines', name: 'NEW: Max 2',
                    line: {{color: '#22d3ee', width: 2}}
                }},
                {{
                    x: d.dates, y: d.old_reb_eq, type: 'scatter', mode: 'lines', name: 'OLD: Dyn',
                    line: {{color: '#a78bfa', width: 2, dash: 'dash'}}
                }},
                 {{
                    x: d.dates, y: d.new_reb_eq, type: 'scatter', mode: 'lines', name: 'NEW: Dyn',
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
    
    out_path = Path('results/active/qqq_experiment_dashboard.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"✅ Dashboard generated: {out_path}")

if __name__ == "__main__":
    run_qqq_simulation()
