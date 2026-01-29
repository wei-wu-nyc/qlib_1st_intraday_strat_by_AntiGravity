import pandas as pd
import sys
import json
import warnings
from pathlib import Path
from datetime import time
import numpy as np

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

def run_full_day_strategy():
    print("🚀 Starting Full Day Multi-Trade Strategy Analysis (v2)...")
    
    # --- Load All Models for Ensemble ---
    print("Loading Models...")
    # LGB MoE
    lgb_m = MoEIntradayStrategy({})
    lgb_m.load_model(str(Path.cwd() / 'results' / 'models' / 'moe'))
    # XGB MoE
    xgb_m = MoEIntradayStrategy({})
    xgb_m.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_xgb'))
    # RF MoE
    rf_m = MoEIntradayStrategy({})
    rf_m.load_model(str(Path.cwd() / 'results' / 'models' / 'moe_rf'))
    # Ensemble MoE
    ensemble_moe = EnsembleStrategy(strategies=[lgb_m, xgb_m, rf_m], weights=[1/3, 1/3, 1/3])
    
    # --- Config ---
    periods = ['train', 'valid', 'test'] 
    loader = IntradayDataLoader('config/intraday_config.yaml')
    etfs = ['SPY', 'QQQ', 'DIA', 'IWM']
    
    config = BacktestConfig(
        transaction_cost_bps=1.0, 
        initial_capital=1_000_000.0,
        position_close_bar=77 # 15:55
    )
    
    period_results = {}
    
    for period in periods:
        print(f"\n{'='*50}\nAnalyzing Period: {period.upper()}\n{'='*50}")
        
        df = loader.get_period_data(period)
        
        print("Generating features...")
        alpha = IntradayAlphaFeatures()
        df = alpha.generate_all_features(df)
        season = SeasonalityFeatures()
        df = season.generate_all_features(df)
        
        # --- Pre-calculate Signals ---
        print("Generating Ensemble MoE signals...")
        signals = ensemble_moe.generate_signals(df)
        cols_needed = ['close', 'bar_index', 'signal', 'predicted_return']
        avail_cols = [c for c in cols_needed if c in signals.columns]
        signals_lean = signals[avail_cols].copy()
        
        # --- Run Strategy (Full Universe) ---
        print("Running MultiTrade Strategy (Full)...")
        strat = MultiTradeStrategy(config, max_positions=5, exit_bars=36)
        results = strat.run(df, lambda d: signals_lean, etfs)
        
        # --- Run Strategy (QQQ + IWM Only) ---
        print("Running MultiTrade Strategy (QQQ+IWM)...")
        etfs_subset = ['QQQ', 'IWM']
        strat_subset = MultiTradeStrategy(config, max_positions=5, exit_bars=36)
        results_subset = strat_subset.run(df, lambda d: signals_lean, etfs_subset)
        

        print("Running MultiTrade Strategy (QQQ+IWM Concentrated)...")
        strat_conc = MultiTradeStrategy(config, max_positions=2, exit_bars=36)
        results_conc = strat_conc.run(df, lambda d: signals_lean, etfs_subset)
        
        # --- Run Rebalance Strategy (QQQ + IWM, Dynamic 100%) ---
        print("Running Rebalance Strategy (QQQ+IWM Dynamic)...")
        strat_reb = RebalanceStrategy(config, max_positions=99, exit_bars=36) # Max positions dynamic
        results_reb = strat_reb.run(df, lambda d: signals_lean, etfs_subset)
        
        # --- Run Strategy (QQQ Only Concentrated) ---
        print("Running MultiTrade Strategy (QQQ Only Max 2)...")
        etfs_qqq = ['QQQ']
        strat_conc_qqq = MultiTradeStrategy(config, max_positions=2, exit_bars=36)
        results_conc_qqq = strat_conc_qqq.run(df, lambda d: signals_lean, etfs_qqq)
        
        # --- Run Rebalance Strategy (QQQ Only Dynamic) ---
        print("Running Rebalance Strategy (QQQ Only Dynamic)...")
        strat_reb_qqq = RebalanceStrategy(config, max_positions=99, exit_bars=36)
        results_reb_qqq = strat_reb_qqq.run(df, lambda d: signals_lean, etfs_qqq)
        
        # --- Benchmark Data (QQQ) ---
        spy_df = df.xs('QQQ', level=1)['close']
        spy_daily = spy_df.groupby(spy_df.index.date).last()
        bench_ret = spy_daily.pct_change().dropna()
        
        period_results[period] = {
            'strategy': results,
            'strategy_subset': results_subset,
            'strategy_conc': results_conc,
            'strategy_reb': results_reb,
            'strategy_conc_qqq': results_conc_qqq, # New
            'strategy_reb_qqq': results_reb_qqq, # New
            'benchmark_daily_ret': bench_ret,
            'rec_exposures': strat.daily_exposures,
            'rec_exposures_subset': strat_subset.daily_exposures,
            'rec_exposures_conc': strat_conc.daily_exposures,
            'rec_exposures_reb': strat_reb.daily_exposures,
            'rec_exposures_conc_qqq': strat_conc_qqq.daily_exposures, # New
            'rec_exposures_reb_qqq': strat_reb_qqq.daily_exposures  # New
        }
        
    # --- Aggregation Logic ---
    print("Aggregating Results...")
    
    final_data = {}
    
    # Helper to merge results
    def merge_periods(p_list, name):
        full_dates = []
        full_strat_ret = []
        full_subset_ret = []
        full_conc_ret = []
        full_reb_ret = []
        full_conc_qqq_ret = [] # New
        full_reb_qqq_ret = [] # New
        full_bench_ret = []
        
        merged_exposures = []
        merged_subset_exposures = []
        merged_conc_exposures = []
        merged_reb_exposures = []
        merged_conc_qqq_exposures = [] # New
        merged_reb_qqq_exposures = [] # New
        
        merged_conc_trades = []
        
        for p in p_list:
            res = period_results[p]['strategy']
            res_sub = period_results[p]['strategy_subset']
            res_conc = period_results[p]['strategy_conc']
            res_reb = period_results[p]['strategy_reb']
            res_conc_qqq = period_results[p]['strategy_conc_qqq'] # New
            res_reb_qqq = period_results[p]['strategy_reb_qqq'] # New
            bench = period_results[p]['benchmark_daily_ret']
            
            # Exposures
            exps = period_results[p]['rec_exposures']
            exps_sub = period_results[p]['rec_exposures_subset']
            exps_conc = period_results[p]['rec_exposures_conc']
            exps_reb = period_results[p]['rec_exposures_reb']
            exps_conc_qqq = period_results[p]['rec_exposures_conc_qqq'] # New
            exps_reb_qqq = period_results[p]['rec_exposures_reb_qqq'] # New
            
            # Align dates logic
            u_dates = sorted(list(set(pd.to_datetime(res.equity_curve.index).date)))
            min_len = min(len(res.daily_returns), len(u_dates))
            
            p_dates = u_dates[-min_len:]
            p_strat = res.daily_returns[-min_len:]
            
            # Align exposures (assuming 1-to-1 with daily returns)
            # Safe slice
            p_exps = exps[-min_len:] if len(exps) >= min_len else exps
            
            # Helper: Align Returns
            def align_ret(r_obj, dates_len):
                ret = r_obj.daily_returns[-min(len(r_obj.daily_returns), dates_len):]
                if len(ret) < dates_len:
                    ret = [0.0]*(dates_len-len(ret)) + list(ret)
                return ret
            
            # Helper: Align Exposures
            def align_exp(e_list, dates_len):
                exp = e_list[-min(len(e_list), dates_len):]
                if len(exp) < dates_len:
                    exp = [0.0]*(dates_len-len(exp)) + list(exp)
                return exp
                
            p_subset = align_ret(res_sub, len(p_dates))
            p_conc = align_ret(res_conc, len(p_dates))
            p_reb = align_ret(res_reb, len(p_dates))
            p_conc_qqq = align_ret(res_conc_qqq, len(p_dates)) # New
            p_reb_qqq = align_ret(res_reb_qqq, len(p_dates)) # New
            
            p_exps_sub = align_exp(exps_sub, len(p_dates))
            p_exps_conc = align_exp(exps_conc, len(p_dates))
            p_exps_reb = align_exp(exps_reb, len(p_dates))
            p_exps_conc_qqq = align_exp(exps_conc_qqq, len(p_dates)) # New
            p_exps_reb_qqq = align_exp(exps_reb_qqq, len(p_dates)) # New
            
            # Align benchmark
            p_bench = []
            for d in p_dates:
                if d in bench.index:
                    p_bench.append(bench.loc[d])
                else:
                    p_bench.append(0.0)
            
            full_dates.extend([d.strftime('%Y-%m-%d') for d in p_dates])
            full_strat_ret.extend(p_strat)
            full_subset_ret.extend(p_subset)
            full_conc_ret.extend(p_conc)
            full_reb_ret.extend(p_reb)
            full_conc_qqq_ret.extend(p_conc_qqq) # New
            full_reb_qqq_ret.extend(p_reb_qqq) # New
            full_bench_ret.extend(p_bench)
            
            merged_exposures.extend(p_exps)
            merged_subset_exposures.extend(p_exps_sub)
            merged_conc_exposures.extend(p_exps_conc)
            merged_reb_exposures.extend(p_exps_reb)
            merged_conc_qqq_exposures.extend(p_exps_conc_qqq) # New
            merged_reb_qqq_exposures.extend(p_exps_reb_qqq) # New
            
            if not res_conc.trades.empty:
                merged_conc_trades.append(res_conc.trades)
        
        # Reconstruct Equity Curves (Normalized to 1.0 start)
        def make_eq(rets):
            eq = [1.0]
            for r in rets:
                eq.append(eq[-1] * (1 + r))
            return eq
            
        strat_eq = make_eq(full_strat_ret)
        subset_eq = make_eq(full_subset_ret)
        conc_eq = make_eq(full_conc_ret)
        reb_eq = make_eq(full_reb_ret)
        conc_qqq_eq = make_eq(full_conc_qqq_ret) # New
        reb_qqq_eq = make_eq(full_reb_qqq_ret) # New
        bench_eq = make_eq(full_bench_ret)
            
        # Stats Calculation
        def calc_stats(returns, exposures=None):
            if not returns: return {}
            arr = np.array(returns)
            tot_ret = np.prod(1 + arr) - 1
            n = len(arr)
            ann_ret = (1 + tot_ret) ** (252/n) - 1 if n > 0 else 0
            vol = np.std(arr) * np.sqrt(252)
            sr = (np.mean(arr) / np.std(arr)) * np.sqrt(252) if np.std(arr) > 0 else 0
            
            # Drawdown
            eq = np.cumprod(1 + arr)
            cummax = np.maximum.accumulate(eq)
            dd = np.min((eq - cummax) / cummax)
            
            stats = {
                'Total Return': f"{tot_ret*100:.2f}%",
                'Annualized': f"{ann_ret*100:.2f}%",
                'Sharpe': f"{sr:.2f}",
                'Max DD': f"{dd*100:.2f}%",
                'Vol': f"{vol*100:.2f}%",
                'Avg Daily Ret': f"{np.mean(arr)*100:.2f}%",
                'Total Days': n
            }
            
            if exposures is not None and len(exposures) > 0:
                avg_inv = np.mean(exposures)
                stats['Avg Invested'] = f"{avg_inv*100:.1f}%"
                
            return stats

        strat_stats = calc_stats(full_strat_ret, merged_exposures)
        subset_stats = calc_stats(full_subset_ret, merged_subset_exposures)
        conc_stats = calc_stats(full_conc_ret, merged_conc_exposures)
        reb_stats = calc_stats(full_reb_ret, merged_reb_exposures)
        conc_qqq_stats = calc_stats(full_conc_qqq_ret, merged_conc_qqq_exposures) # New
        reb_qqq_stats = calc_stats(full_reb_qqq_ret, merged_reb_qqq_exposures) # New
        bench_stats = calc_stats(full_bench_ret, [1.0] * len(full_bench_ret))
        
        # Trade Stats (Targeting Concentrated Strategy)
        df_trades = pd.concat(merged_conc_trades) if merged_conc_trades else pd.DataFrame()
        trade_stats = {}
        tod_html = "" # Time of Day stats HTML
        
        if not df_trades.empty and 'return_pct' in df_trades.columns:
            # Overall Stats
            total_pnl = df_trades['pnl'].sum()
            avg_pnl_day = total_pnl / len(full_dates) if full_dates else 0
            trade_stats = {
                'Win Rate': f"{(df_trades['return_pct']>0).mean()*100:.1f}%",
                'Avg Trade': f"{df_trades['return_pct'].mean()*10000:.1f} bp",
                'Num Trades': len(df_trades),
                'Trades/Day': f"{len(df_trades)/len(full_dates):.1f}",
                'Avg PnL/Day': f"${avg_pnl_day:,.0f}"
            }
            conc_stats.update(trade_stats)
            
            # --- Time of Day Analysis ---
            # Extract time from entry_time
            df_trades['entry_time_str'] = df_trades['entry_time'].dt.strftime('%H:%M')
            # Fix order: 09:40, 10:00, 10:30, 12:00, 14:00
            order = ['09:40', '10:00', '10:30', '12:00', '14:00']
            tod_html = "<table><thead><tr><th class='left'>Time</th><th>Total PnL</th><th>Win Rate</th><th>Avg Return</th><th>Trades</th></tr></thead><tbody>"
            
            # Ensure categorical order if possible, otherwise just manual check
            # Simple iteration over known slots to preserve order
            grp = df_trades.groupby('entry_time_str')
            
            for t_slot in order:
                if t_slot in grp.groups:
                    g = grp.get_group(t_slot)
                    pnl = g['pnl'].sum()
                    wr = (g['return_pct'] > 0).mean() * 100
                    avg = g['return_pct'].mean() * 10000
                    cnt = len(g)
                    cls = 'pos' if pnl > 0 else 'neg'
                    tod_html += f"<tr><td class='left'>{t_slot}</td><td class='{cls}'>${pnl:,.0f}</td><td>{wr:.1f}%</td><td class='{cls}'>{avg:.1f} bp</td><td>{cnt}</td></tr>"
            
            tod_html += "</tbody></table>"

        # Instrument Stats
        inst_html = ""
        if not df_trades.empty and 'return_pct' in df_trades.columns:
            grp = df_trades.groupby('instrument')
            inst_html += "<table><thead><tr><th class='left'>Instrument</th><th>Total PnL</th><th>Win Rate</th><th>Avg Return</th><th>Trades</th></tr></thead><tbody>"
            for inst, g in grp:
                pnl = g['pnl'].sum() if 'pnl' in g.columns else 0
                wr = (g['return_pct'] > 0).mean() * 100
                avg = g['return_pct'].mean() * 10000
                cnt = len(g)
                cls = 'pos' if pnl > 0 else 'neg'
                inst_html += f"<tr><td class='left'>{inst}</td><td class='{cls}'>${pnl:,.0f}</td><td>{wr:.1f}%</td><td class='{cls}'>{avg:.1f} bp</td><td>{cnt}</td></tr>"
            inst_html += "</tbody></table>"
            
        return {
            'dates': full_dates,
            'strategy_equity': strat_eq[1:],
            'subset_equity': subset_eq[1:],
            'conc_equity': conc_eq[1:],
            'reb_equity': reb_eq[1:],
            'conc_qqq_equity': conc_qqq_eq[1:], # New
            'reb_qqq_equity': reb_qqq_eq[1:], # New
            'benchmark_equity': bench_eq[1:],
            'strategy_stats': strat_stats,
            'subset_stats': subset_stats,
            'conc_stats': conc_stats,
            'reb_stats': reb_stats,
            'conc_qqq_stats': conc_qqq_stats, # New
            'reb_qqq_stats': reb_qqq_stats, # New
            'benchmark_stats': bench_stats,
            'instrument_html': inst_html,
            'tod_html': tod_html # New
        }

    # Generate Data Bundles
    print("Building View Data...")
    final_data['train'] = merge_periods(['train'], 'Train Period')
    final_data['valid'] = merge_periods(['valid'], 'Valid Period')
    final_data['test'] = merge_periods(['test'], 'Test Period')
    final_data['valid_test'] = merge_periods(['valid', 'test'], 'Valid + Test')
    final_data['all'] = merge_periods(['train', 'valid', 'test'], 'All Periods')
    
    generate_dashboard(final_data)

def generate_dashboard(final_data):
    # Embed data as JSON
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    data_json = json.dumps(final_data, cls=NpEncoder)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Full Day Strategy Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #e8e8e8; padding: 20px; }}
        h1 {{ text-align: center; color: #00d9ff; font-size: 28px; margin-bottom: 20px; }}
        
        .controls {{ display: flex; justify-content: center; gap: 10px; margin-bottom: 30px; }}
        .btn {{ background: #333; color: #ccc; border: 1px solid #555; padding: 10px 20px; cursor: pointer; border-radius: 4px; font-size: 14px; transition: 0.2s; }}
        .btn:hover {{ background: #444; }}
        .btn.active {{ background: #00d9ff; color: #000; font-weight: bold; border-color: #00d9ff; }}
        
        .stats-container {{ display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 30px; }}
        .stats-box {{ flex: 1; min-width: 200px; background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); }}
        .stats-title {{ border-bottom: 1px solid #444; padding-bottom: 10px; margin-bottom: 15px; font-size: 16px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        
        .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }}
        .stat-val {{ font-weight: bold; }}
        
        .chart-container {{ width: 100%; height: 600px; background: rgba(0,0,0,0.2); border-radius: 8px; margin-bottom: 30px; }}
        
        table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.05); font-size: 14px; }}
        th, td {{ padding: 12px; text-align: right; border: 1px solid rgba(255,255,255,0.1); }}
        th {{ background: rgba(0,0,0,0.3); text-align: center; }}
        th.left {{ text-align: left; }}
        
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>🚀 Full Day Strategy Dashboard</h1>
    <p style="text-align:center; color:#888; margin-bottom: 30px;">5 Positions Max | 1/5 Equity per Trade | MoE Ensemble | 1bp Cost</p>
    
    <div class="controls">
        <button class="btn" onclick="render('all')" id="btn-all">All History</button>
        <button class="btn active" onclick="render('valid_test')" id="btn-valid_test">Valid + Test</button>
        <button class="btn" onclick="render('valid')" id="btn-valid">Valid Only</button>
        <button class="btn" onclick="render('test')" id="btn-test">Test Only</button>
    </div>
    
    <div class="stats-container">
        <!-- Benchmark Stats -->
        <div class="stats-box">
            <div class="stats-title" style="color: #888;">Benchmark (QQQ)</div>
            <div id="bench-stats"></div>
        </div>
        
        <!-- Strategy Stats (Concentrated) -->
        <div class="stats-box">
            <div class="stats-title" style="color: #00ff88;">QQQ+IWM (Max 2)</div>
            <div id="conc-stats"></div>
        </div>
        
        <!-- Strategy Stats (Rebalance) -->
        <div class="stats-box">
            <div class="stats-title" style="color: #d100ff;">QQQ+IWM (Dyn)</div>
            <div id="reb-stats"></div>
        </div>
        
        <!-- Strategy Stats (QQQ Only Max 2) -->
        <div class="stats-box">
            <div class="stats-title" style="color: #00ccff;">QQQ Only (Max 2)</div>
            <div id="conc-qqq-stats"></div>
        </div>
        
        <!-- Strategy Stats (QQQ Only Dyn) -->
        <div class="stats-box">
            <div class="stats-title" style="color: #ff00ff;">QQQ Only (Dyn)</div>
            <div id="reb-qqq-stats"></div>
        </div>
    </div>

    
    <div id="prob-dist" class="stats-box" style="margin-bottom:30px;">
        <div class="stats-title">Trading Metrics (Concentrated Strategy - Max 2)</div>
        <div id="trade-stats" style="display:grid; grid-template-columns: repeat(5, 1fr); gap:10px; text-align:center;"></div>
    </div>
    
    <div id="chart" class="chart-container"></div>
    
    <div class="grid-2">
        <div>
            <h3>Instrument Analysis (Concentrated)</h3>
            <div id="inst-table"></div>
        </div>
        <div>
            <h3>Time-of-Day Analysis (Concentrated)</h3>
            <div id="tod-table"></div>
        </div>
    </div>
    
    <div style="text-align:center; color:#666; font-size:12px; margin-top:50px;">
        Generated by Antigravity Agent
    </div>

    <script>
        const data = {data_json};
        
        function render(key) {{
            const d = data[key];
            
            // Highlight Button
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + key).classList.add('active');
            
            // Render Stats
            const sMetrics = ['Total Return', 'Annualized', 'Sharpe', 'Max DD', 'Avg Invested'];
            
            function renderStats(targetId, statsObj) {{
                let html = "";
                sMetrics.forEach(m => {{
                    html += `<div class='stat-row'><span>${{m}}</span><span class='stat-val'>${{statsObj[m] || '-'}}</span></div>`;
                }});
                document.getElementById(targetId).innerHTML = html;
            }}
            
            renderStats('bench-stats', d.benchmark_stats);
            renderStats('conc-stats', d.conc_stats);
            renderStats('reb-stats', d.reb_stats);
            renderStats('conc-qqq-stats', d.conc_qqq_stats);
            renderStats('reb-qqq-stats', d.reb_qqq_stats);
            
            // Render Trade Metrics (Concentrated)
            let tradeHtml = "";
            const tMetrics = ['Win Rate', 'Avg Trade', 'Num Trades', 'Trades/Day', 'Avg PnL/Day'];
            tMetrics.forEach(m => {{
                tradeHtml += `<div><div style='color:#888; font-size:12px;'>${{m}}</div><div style='font-size:20px; font-weight:bold;'>${{d.conc_stats[m] || '-'}}</div></div>`;
            }});
            document.getElementById('trade-stats').innerHTML = tradeHtml;

            // Tables
            document.getElementById('inst-table').innerHTML = d.instrument_html || "<div style='text-align:center; padding:20px'>No Trades</div>";
            document.getElementById('tod-table').innerHTML = d.tod_html || "<div style='text-align:center; padding:20px'>No Trades</div>";
            
            // Render Chart
            const trace_conc = {{
                x: d.dates,
                y: d.conc_equity,
                type: 'scatter',
                mode: 'lines',
                name: 'QQQ+IWM (Max 2)',
                line: {{color: '#00ff88', width: 2}}
            }};
            
            const trace_reb = {{
                x: d.dates,
                y: d.reb_equity,
                type: 'scatter',
                mode: 'lines',
                name: 'QQQ+IWM (Dyn)',
                line: {{color: '#d100ff', width: 2}}
            }};
            
             const trace_conc_qqq = {{
                x: d.dates,
                y: d.conc_qqq_equity,
                type: 'scatter',
                mode: 'lines',
                name: 'QQQ Only (Max 2)',
                line: {{color: '#00ccff', width: 2, dash: 'dash'}}
            }};
            
             const trace_reb_qqq = {{
                x: d.dates,
                y: d.reb_qqq_equity,
                type: 'scatter',
                mode: 'lines',
                name: 'QQQ Only (Dyn)',
                line: {{color: '#ff00ff', width: 2, dash: 'dash'}}
            }};
            
            const trace_bench = {{
                x: d.dates,
                y: d.benchmark_equity,
                type: 'scatter',
                mode: 'lines',
                name: 'Benchmark (QQQ)',
                line: {{color: '#888', width: 1, dash: 'dot'}}
            }};
            
            const layout = {{
                title: 'Cumulative Returns (Normalized)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#ccc' }},
                xaxis: {{ showgrid: false }},
                yaxis: {{ showgrid: true, gridcolor: '#333' }},
                margin: {{ l: 50, r: 20, t: 40, b: 40 }},
                legend: {{ orientation: 'h', y: -0.2 }},
                responsive: true
            }};
            
            Plotly.newPlot('chart', [trace_reb, trace_reb_qqq, trace_conc, trace_conc_qqq, trace_bench], layout, {{responsive: true}});
        }}
        
        // Initial Render
        render('valid_test');
    </script>
</body>
</html>
    """
    
    with open('results/active/full_day_strategy_dashboard_v2.html', 'w') as f:
        f.write(html)
    print("✅ Dashboard updated: results/active/full_day_strategy_dashboard_v2.html")

if __name__ == "__main__":
    run_full_day_strategy()
