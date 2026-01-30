import pandas as pd
import sys
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import numpy as np

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

def run_rolling_validation():
    print("🚀 Starting Walk-Forward Validation (Leverage Edition)...")
    
    # --- Configuration ---
    years = range(2019, 2026) 
    train_start_fixed = "2008-01-01"
    
    base_results_dir = Path.cwd() / 'results' / 'rolling_validation'
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Store returns for equity curves
    accumulated_results = {
        'dates': [],
        'bench_returns': [],
        'Max2_returns': [], 'Max2_exposures': [], 'Max2_max_exposures': [],
        'AE80_returns': [], 'AE80_exposures': [], 'AE80_max_exposures': [],
        'AE100_returns': [], 'AE100_exposures': [], 'AE100_max_exposures': [],
        'AE150_returns': [], 'AE150_exposures': [], 'AE150_max_exposures': [],
        'AE200_returns': [], 'AE200_exposures': [], 'AE200_max_exposures': [],
        'AE100_trades': [] # For Time-of-Day analysis
    }
    
    annual_stats = []
    
    # Global Config
    bt_config = BacktestConfig(
        transaction_cost_bps=1.0, 
        initial_capital=1_000_000.0,
        position_close_bar=77,
        borrow_rate_annual=0.0 # Configurable, explicitly 0 as requested
    )
    etfs = ['QQQ', 'SPY', 'IWM', 'DIA']
    
    # Define Variants
    # Key -> Kwargs for MultiTradeStrategy
    variants = {
        'Max2': {'max_positions': 2},
        'AE80': {'fixed_pos_pct': 0.16},
        'AE100': {'fixed_pos_pct': 0.20},
        'AE150': {'fixed_pos_pct': 0.30},
        'AE200': {'fixed_pos_pct': 0.40}
    }

    loader = IntradayDataLoader('config/intraday_config.yaml')
    
    for year in years:
        print(f"\n{'='*60}\n📅 Processing Validation Year: {year}\n{'='*60}")
        
        valid_start = f"{year}-01-01"
        valid_end = f"{year}-12-31"
        train_end = f"{year-1}-12-31"
        train_start = train_start_fixed
        
        # 1. Train/Load Models
        model_dir = base_results_dir / f"models_{year}"
        if not (model_dir.exists() and (model_dir / "moe_lgb" / "manifest.json").exists()):
            train_moe_models(train_start_date=train_start, train_end_date=train_end, output_dir=model_dir, model_types=['xgb', 'rf', 'lgb'])
            
        lgb_moe = MoEIntradayStrategy({})
        lgb_moe.load_model(str(model_dir / 'moe_lgb'))
        xgb_moe = MoEIntradayStrategy({})
        xgb_moe.load_model(str(model_dir / 'moe_xgb'))
        rf_moe = MoEIntradayStrategy({})
        rf_moe.load_model(str(model_dir / 'moe_rf'))
        
        ensemble = EnsembleStrategy(strategies=[lgb_moe, xgb_moe, rf_moe], weights=[1/3, 1/3, 1/3])
        
        # 2. Prepare Data
        target_period = 'test' if year >= 2022 else 'valid'
        print(f"Loading data from period: {target_period}")
        df_year = loader.get_period_data(target_period, symbols=etfs)
        
        mask = (df_year.index.get_level_values(0) >= pd.Timestamp(valid_start)) & \
               (df_year.index.get_level_values(0) <= pd.Timestamp(valid_end))
        df_year = df_year[mask]
        
        if df_year.empty: continue
            
        alpha = IntradayAlphaFeatures()
        df_year = alpha.generate_all_features(df_year)
        season = SeasonalityFeatures()
        df_year = season.generate_all_features(df_year)
        
        # 3. Generate Signals (Once for all variants)
        signals = ensemble.generate_signals(df_year)
        cols_needed = ['close', 'bar_index', 'signal', 'predicted_return']
        avail_cols = [c for c in cols_needed if c in signals.columns]
        signals_lean = signals[avail_cols].copy()
        
        # 4. Run Variants
        current_year_results = {}
        
        for name, kwargs in variants.items():
            print(f"  Running Strategy: {name}...")
            strat = MultiTradeStrategy(bt_config, exit_bars=36, **kwargs)
            res = strat.run(df_year, lambda d: signals_lean, etfs)
            current_year_results[name] = res
            
            # Record Exposure (Average for the year)
            accumulated_results[f'{name}_exposures'].append(res.daily_exposure)
            accumulated_results[f'{name}_max_exposures'].append(res.max_exposure)
            
            # Store AE100 trades for detailed analysis
            if name == 'AE100':
                accumulated_results['AE100_trades'].extend(res.trades.to_dict('records'))

        # Benchmark
        bench_ret = []
        if 'QQQ' in df_year.index.get_level_values(1).unique():
            spy_df = df_year.xs('QQQ', level=1)['close']
            spy_daily = spy_df.groupby(spy_df.index.date).last()
            bench_ret = spy_daily.pct_change().dropna()
        
        # 5. Store Accumulated Results
        dates = sorted(list(set(pd.to_datetime(current_year_results['Max2'].equity_curve.index).date)))
        accumulated_results['dates'].extend([d.strftime('%Y-%m-%d') for d in dates])
        
        # Helper to align returns
        def get_aligned_ret(res_obj):
            r = res_obj.daily_returns
            needed = len(dates)
            if len(r) < needed: return [0.0]*(needed-len(r)) + r
            return r[-needed:]

        accumulated_results['bench_returns'].extend([bench_ret.loc[d] if d in bench_ret.index else 0.0 for d in dates])
        
        for name in variants.keys():
            accumulated_results[f'{name}_returns'].extend(get_aligned_ret(current_year_results[name]))
            
        # 6. Calc Annual Stats for Table
        def calc_s(r):
            if not r: return 0.0, 0.0
            r = np.array(r)
            return np.prod(1+r)-1, (np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r)>0 else 0.0

        yr_stats = {'Year': year}
        
        # Benchmark
        rb, sb = calc_s([bench_ret.loc[d] if d in bench_ret.index else 0.0 for d in dates])
        yr_stats['Bench_Ret'] = f"{rb*100:.1f}%"
        yr_stats['Bench_SR'] = f"{sb:.2f}"
        
        # Variants
        for name in variants.keys():
            r, s = calc_s(current_year_results[name].daily_returns)
            yr_stats[f'{name}_Ret'] = f"{r*100:.1f}%"
            yr_stats[f'{name}_SR'] = f"{s:.2f}"
            
        annual_stats.append(yr_stats)
        
    # --- Generate Dashboard ---
    generate_rolling_dashboard(accumulated_results, annual_stats)

def generate_rolling_dashboard(data, annual_stats):
    # 1. Calc Full Period Stats
    def calc_full_stats(returns, name, exposures=None, max_exposures=None):
        if not returns: return {}
        arr = np.array(returns)
        tot = np.prod(1+arr)-1
        ann = (1+tot)**(252/len(arr))-1
        vol = np.std(arr)*np.sqrt(252)
        sr = (np.mean(arr)/np.std(arr))*np.sqrt(252) if np.std(arr)>0 else 0
        
        eq = np.cumprod(1+arr)
        dd = np.min((eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq))
        
        inv = "-"
        if exposures:
            inv = f"{np.mean(exposures)*100:.1f}%"
            
        max_inv = "-"
        if max_exposures:
            max_inv = f"{np.max(max_exposures)*100:.1f}%"
        
        return {
            'Strategy': name,
            'Total Return': f"{tot*100:.1f}%",
            'Annualized': f"{ann*100:.1f}%",
            'Sharpe': f"{sr:.2f}",
            'Max DD': f"{dd*100:.1f}%",
            '% Invested': inv,
            'Max Invested': max_inv
        }
        
    stats_list = []
    stats_list.append(calc_full_stats(data['bench_returns'], 'Benchmark', [1.0], [1.0]))
    for key in ['Max2', 'AE80', 'AE100', 'AE150', 'AE200']:
        stats_list.append(calc_full_stats(data[f'{key}_returns'], key, data.get(f'{key}_exposures'), data.get(f'{key}_max_exposures')))

    # 2. AE100 Time-of-Day Analysis
    tod_stats = []
    if data['AE100_trades']:
        df_trades = pd.DataFrame(data['AE100_trades'])
        # Extract time from entry_time
        df_trades['time'] = df_trades['entry_time'].dt.strftime('%H:%M')
        
        tod_grp = df_trades.groupby('time').agg(
            Count=('pnl', 'count'),
            WinRate=('return_pct', lambda x: (x>0).mean()),
            AvgRet=('return_pct', 'mean'),
            TotalPnL=('pnl', 'sum'),
            Sharpe=('return_pct', lambda x: (x.mean()/x.std())*np.sqrt(252*5) if x.std()>0 else 0) # Rough Approx
        )
        
        for t, row in tod_grp.iterrows():
            tod_stats.append({
                'Time': t,
                'Count': row['Count'],
                'WinRate': f"{row['WinRate']*100:.1f}%",
                'AvgRet': f"{row['AvgRet']*100:.2f}%",
                'TotalPnL': f"${row['TotalPnL']:.0f}"
            })

    # 3. HTML Generation
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
            
    # Chart Data
    def make_eq(rets):
        eq = [1.0]
        for v in rets: eq.append(eq[-1]*(1+v))
        return eq
    
    chart_data = {'dates': data['dates']}
    for k in ['bench', 'Max2', 'AE80', 'AE100', 'AE150', 'AE200']:
        chart_data[k] = make_eq(data.get(f'{k}_returns', []))[1:]

    json_data = json.dumps(chart_data, cls=NpEncoder)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Leverage Strategy Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', system-ui; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .card {{ background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 10px; border: 1px solid #334155; text-align: center; }}
        .card h3 {{ color: #94a3b8; font-size: 14px; margin: 0 0 10px 0; }}
        .card .val {{ font-size: 18px; font-weight: bold; color: #f8fafc; }}
        .card .sub {{ font-size: 12px; color: #64748b; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; margin-bottom: 30px; }}
        th, td {{ padding: 12px; text-align: right; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 600; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
        .highlight {{ color: #38bdf8; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Leverage Strategy Analysis (AE80 - AE200)</h1>
        <p style="color: #64748b">Rolling Validation (2019-2025) • Ensemble Model</p>
    </div>
    
    <!-- Snapshots -->
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

    <!-- Chart -->
    <div id="chart" style="height: 600px; background: #1e293b; border-radius: 10px; margin-bottom: 30px;"></div>

    <!-- Annual Table -->
    <h3>📅 Annual Performance (Returns)</h3>
    <table>
        <thead>
            <tr>
                <th>Year</th>
                <th>Benchmark</th>
                <th>Max 2</th>
                <th style="color: #38bdf8">AE 100</th>
                <th>AE 80</th>
                <th>AE 150</th>
                <th>AE 200</th>
            </tr>
        </thead>
        <tbody>
            { "".join([f'''
            <tr>
                <td>{row['Year']}</td>
                <td>{row['Bench_Ret']}</td>
                <td>{row['Max2_Ret']}</td>
                <td class="highlight">{row['AE100_Ret']}</td>
                <td>{row['AE80_Ret']}</td>
                <td>{row['AE150_Ret']}</td>
                <td>{row['AE200_Ret']}</td>
            </tr>
            ''' for row in annual_stats]) }
        </tbody>
    </table>

    <!-- Time of Day -->
    <h3>⏰ AE100 Performance by Entry Time</h3>
    <table style="width: auto; margin: 0 auto;">
        <thead>
            <tr>
                <th>Entry Time</th>
                <th>Count</th>
                <th>Win Rate</th>
                <th>Avg Return</th>
                <th>Total PnL</th>
            </tr>
        </thead>
        <tbody>
            { "".join([f'''
            <tr>
                <td>{r['Time']}</td>
                <td>{r['Count']}</td>
                <td>{r['WinRate']}</td>
                <td>{r['AvgRet']}</td>
                <td>{r['TotalPnL']}</td>
            </tr>
            ''' for r in tod_stats]) }
        </tbody>
    </table>

    <script>
        const data = {json_data};
        const traces = [
            {{ x: data.dates, y: data.bench, name: 'Benchmark', line: {{ color: '#64748b', dash: 'dot' }} }},
            {{ x: data.dates, y: data.Max2, name: 'Max 2', line: {{ color: '#22d3ee' }} }},
            {{ x: data.dates, y: data.AE80, name: 'AE 80 (0.8x)', line: {{ color: '#4ade80' }} }},
            {{ x: data.dates, y: data.AE100, name: 'AE 100 (1.0x)', line: {{ color: '#38bdf8', width: 3 }} }},
            {{ x: data.dates, y: data.AE150, name: 'AE 150 (1.5x)', line: {{ color: '#facc15' }} }},
            {{ x: data.dates, y: data.AE200, name: 'AE 200 (2.0x)', line: {{ color: '#f87171' }} }}
        ];
        
        Plotly.newPlot('chart', traces, {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8' }},
            xaxis: {{ gridcolor: '#334155' }},
            yaxis: {{ gridcolor: '#334155', type: 'log' }},
            margin: {{ t: 20, l: 40, r: 20, b: 40 }},
            legend: {{ orientation: 'h', y: 1.05 }}
        }}, {{responsive: true}});
    </script>
</body>
</html>
    """
    
    with open('results/active/leverage_dashboard.html', 'w') as f:
        f.write(html)
    print("✅ Dashboard generated: results/active/leverage_dashboard.html")

def getColor(name):
    colors = {
        'Benchmark': '#64748b', 'Max2': '#22d3ee', 
        'AE80': '#4ade80', 'AE100': '#38bdf8', 
        'AE150': '#facc15', 'AE200': '#f87171'
    }
    return colors.get(name, '#fff')

if __name__ == "__main__":
    run_rolling_validation()
