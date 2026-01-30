import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def regenerate_quarterly_dashboard():
    print("🚀 Regenerating Quarterly Dashboard...")
    
    # 1. Load Quarterly Trades
    trades_path = Path('results/quarterly_validation/all_trades_quarterly_2013_2025.csv')
    if not trades_path.exists():
        print("❌ Trade log not found!")
        return
        
    df_trades = pd.read_csv(trades_path)
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_trades['exit_date'] = df_trades['exit_time'].dt.date
    
    # Enrich with seasonality info
    df_trades['month'] = df_trades['entry_time'].dt.month
    df_trades['day_of_week'] = df_trades['entry_time'].dt.dayofweek # 0=Mon, 4=Fri
    df_trades['day_of_month'] = df_trades['entry_time'].dt.day
    
    # Reconstruct Daily Returns
    start_date = pd.Timestamp("2013-01-01")
    end_date = pd.Timestamp("2025-12-31")
    all_dates = pd.date_range(start_date, end_date, freq='B')
    
    daily_pnl = df_trades.groupby('exit_date')['pnl'].sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    series_pnl = daily_pnl.reindex(all_dates, fill_value=0.0)
    
    # Reconstruct Daily Returns with Per-Year Capital Reset
    # (Matches original run logic where capital resets to 1M each Jan 1st)
    
    unique_years = sorted(df_trades['entry_time'].dt.year.unique())
    daily_returns_series = pd.Series(dtype=float)
    
    for y in unique_years:
        # Get trades for this year
        mask_y = (df_trades['entry_time'].dt.year == y)
        trades_y = df_trades[mask_y]
        
        # Daily PnL
        pnl_y = trades_y.groupby('exit_date')['pnl'].sum()
        
        # Reindex
        start_y = pd.Timestamp(f"{y}-01-01")
        end_y = pd.Timestamp(f"{y}-12-31")
        days_y = pd.date_range(start_y, end_y, freq='B')
        
        pnl_series = pnl_y.reindex(days_y, fill_value=0.0)
        
        # Calculate Returns on 1M base
        curr_eq = 1_000_000.0
        rets_y = []
        dates_y = []
        
        for d, p in pnl_series.items():
            r = p / curr_eq
            rets_y.append(r)
            dates_y.append(d)
            curr_eq += p
            
        daily_returns_series = pd.concat([
            daily_returns_series, 
            pd.Series(rets_y, index=dates_y)
        ])
    
    df_ret = daily_returns_series.to_frame(name='return')
    df_ret.index.name = 'date'
    df_ret['month'] = df_ret.index.month
    df_ret['day_of_week'] = df_ret.index.dayofweek
    df_ret['day_of_month'] = df_ret.index.day
    
    # 3. Standard Period Stats & Trade Analysis
    periods = {
        '2013-2015': ('2013-01-01', '2015-12-31'),
        '2016-2018': ('2016-01-01', '2018-12-31'),
        '2019-2022': ('2019-01-01', '2022-12-31'),
        '2023-2025': ('2023-01-01', '2025-12-31')
    }
    
    stats_cards = []
    dist_tables = []
    
    for name, (start, end) in periods.items():
        # Returns
        mask_ret = (df_ret.index >= pd.Timestamp(start)) & (df_ret.index <= pd.Timestamp(end))
        sub_ret = df_ret[mask_ret]
        
        # Trades
        mask_trade = (df_trades['entry_time'] >= pd.Timestamp(start)) & (df_trades['entry_time'] <= pd.Timestamp(end))
        t_sub = df_trades[mask_trade]
        
        if sub_ret.empty:
            stats_cards.append({'Period': name, 'Ret': '-', 'SR': '-', 'DD': '-', 'Count': 0})
            continue
            
        r = sub_ret['return'].values
        tot = np.prod(1+r) - 1
        ann = (1+tot)**(252/len(r)) - 1
        sr = (np.mean(r)/np.std(r))*np.sqrt(252) if np.std(r)>0 else 0
        eq = np.cumprod(1+r)
        dd = np.min((eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq))
        
        # Trade Stats
        if not t_sub.empty:
            avg_ret = t_sub['return_pct'].mean()
            best_ret = t_sub['return_pct'].max()
            worst_ret = t_sub['return_pct'].min()
            avg_bars = t_sub['holding_bars'].mean()
            
            # Distribution
            dist_data = {
                '10% (Worst)': t_sub['return_pct'].quantile(0.10),
                '30%': t_sub['return_pct'].quantile(0.30),
                '50% (Median)': t_sub['return_pct'].quantile(0.50),
                '70%': t_sub['return_pct'].quantile(0.70),
                '90% (Best)': t_sub['return_pct'].quantile(0.90)
            }
            dist_tables.append({'Period': name, 'Data': dist_data})
        else:
            avg_ret = 0; best_ret=0; worst_ret=0; avg_bars=0
        
        stats_cards.append({
            'Period': name,
            'Total Return': f"{tot*100:.1f}%",
            'Annualized': f"{ann*100:.1f}%",
            'Sharpe': f"{sr:.2f}",
            'Max DD': f"{dd*100:.1f}%",
            'Trades': len(t_sub),
            'Avg Trade': f"{avg_ret*100:.2f}%",
            'Best Trade': f"{best_ret*100:.2f}%",
            'Worst Trade': f"{worst_ret*100:.2f}%",
            'Avg Duration': f"{avg_bars:.1f} bars"
        })
        
    # 4. Seasonality Analysis
    
    # A. Monthly Seasonality (1-12)
    month_stats = []
    for m in range(1, 13):
        r = df_ret[df_ret['month'] == m]['return']
        t = df_trades[df_trades['month'] == m]
        
        sr = (r.mean()/r.std())*np.sqrt(252) if len(r)>1 and r.std()>0 else 0
        avg_t = t['return_pct'].mean() if not t.empty else 0
        wr = (t['return_pct'] > 0).mean() if not t.empty else 0
        
        month_name = datetime(2000, m, 1).strftime('%b')
        month_stats.append({'Label': month_name, 'Sharpe': sr, 'Avg Trade': avg_t, 'Win Rate': wr, 'Count': len(t)})
        
    # B. Day of Week (0-4)
    dow_stats = []
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    for d in range(5):
        r = df_ret[df_ret['day_of_week'] == d]['return']
        t = df_trades[df_trades['day_of_week'] == d]
        
        sr = (r.mean()/r.std())*np.sqrt(252) if len(r)>1 and r.std()>0 else 0
        avg_t = t['return_pct'].mean() if not t.empty else 0
        wr = (t['return_pct'] > 0).mean() if not t.empty else 0
        
        dow_stats.append({'Label': days[d], 'Sharpe': sr, 'Avg Trade': avg_t, 'Win Rate': wr, 'Count': len(t)})
        
    # C. Day of Month (1-31)
    dom_stats = []
    for d in range(1, 32):
        r = df_ret[df_ret['day_of_month'] == d]['return']
        t = df_trades[df_trades['day_of_month'] == d]
        
        if len(r) < 10: continue 
        
        sr = (r.mean()/r.std())*np.sqrt(252) if len(r)>1 and r.std()>0 else 0
        avg_t = t['return_pct'].mean() if not t.empty else 0
        
        dom_stats.append({'Label': str(d), 'Sharpe': sr, 'Avg Trade': avg_t})

    # 5. Degradation (Month since training)
    # With quarterly training, we compare Month 1 vs 2 vs 3 of the quarter.
    # Logic:
    # Q1: Jan=1, Feb=2, Mar=3
    # Q2: Apr=1, May=2, Jun=3
    # etc.
    # Map month to quarter_month index
    df_ret['q_month'] = (df_ret.index.month - 1) % 3 + 1
    df_trades['q_month'] = (df_trades['month'] - 1) % 3 + 1
    
    deg_stats = []
    for m in [1, 2, 3]:
        r = df_ret[df_ret['q_month'] == m]['return']
        t = df_trades[df_trades['q_month'] == m]
        
        sr = (r.mean()/r.std())*np.sqrt(252) if len(r)>1 and r.std()>0 else 0
        avg_t = t['return_pct'].mean() if not t.empty else 0
        wr = (t['return_pct'] > 0).mean() if not t.empty else 0
        
        deg_stats.append({'Label': f"Month {m}", 'Sharpe': sr, 'Avg Trade': avg_t, 'Win Rate': wr, 'Count': len(t)})

    # HTML Generation
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Quarterly Validation: Seasons & Stats</title>
    <style>
        body {{ font-family: system-ui; background: #0f172a; color: #fff; padding: 20px; }}
        h1, h2 {{ color: #38bdf8; margin-top: 0; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; }}
        .label {{ color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
        td, th {{ padding: 6px 0; border-bottom: 1px solid #334155; text-align: right; }}
        th {{ color: #94a3b8; }}
        th:first-child, td:first-child {{ text-align: left; }}
        .pos {{ color: #4ade80; }}
        .neg {{ color: #f87171; }}
        .section {{ margin-top: 40px; }}
    </style>
</head>
<body>
    <h1>Quarterly Validation (2013-2025)</h1>
    
    <div class="grid">
        { "".join([f'''
        <div class="card">
            <h2>{c['Period']}</h2>
            
            <div class="label">Performance</div>
            <table>
                <tr><td>Return</td><td class="{ 'pos' if '-' not in c['Total Return'] else 'neg' }">{c['Total Return']}</td></tr>
                <tr><td>Annualized</td><td>{c['Annualized']}</td></tr>
                <tr><td>Sharpe</td><td>{c['Sharpe']}</td></tr>
                <tr><td>Max DD</td><td class="neg">{c['Max DD']}</td></tr>
            </table>
            
            <div class="label" style="margin-top: 20px">Trade Stats</div>
            <table>
                <tr><td>Count</td><td>{c['Trades']}</td></tr>
                <tr><td>Avg Return</td><td class="{ 'pos' if '-' not in c['Avg Trade'] else 'neg' }">{c['Avg Trade']}</td></tr>
                <tr><td>Best</td><td class="pos">{c['Best Trade']}</td></tr>
                <tr><td>Worst</td><td class="neg">{c['Worst Trade']}</td></tr>
                <tr><td>Duration</td><td>{c['Avg Duration']}</td></tr>
            </table>
        </div>
        ''' for c in stats_cards]) }
    </div>
    
    <div class="section">
        <h2>📊 Return Distribution (Quantiles)</h2>
        <div class="card" style="width: 100%; overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th style="text-align: left">Period</th>
                        <th>10% (Worst)</th>
                        <th>30%</th>
                        <th>50% (Median)</th>
                        <th>70%</th>
                        <th>90% (Best)</th>
                    </tr>
                </thead>
                <tbody>
                    { "".join([f'''
                    <tr>
                        <td style="text-align: left; font-weight: bold; color: #38bdf8">{d['Period']}</td>
                        <td class="neg">{d['Data']['10% (Worst)']*100:.2f}%</td>
                        <td class="neg">{d['Data']['30%']*100:.2f}%</td>
                        <td>{d['Data']['50% (Median)']*100:.2f}%</td>
                        <td class="pos">{d['Data']['70%']*100:.2f}%</td>
                        <td class="pos">{d['Data']['90% (Best)']*100:.2f}%</td>
                    </tr>
                    ''' for d in dist_tables]) }
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="section">
        <h2>📉 Risk Tail Analysis</h2>
       <div class="card">
           <table>
               <thead>
                   <tr>
                       <th style="text-align: left">Period</th>
                       <th>Avg Loss</th>
                       <th>Avg Win</th>
                       <th>Win/Loss Ratio</th>
                       <th>Win Rate</th>
                   </tr>
               </thead>
               <tbody>
                   { "".join([get_risk_row(df_trades, p, start, end) for p, (start, end) in periods.items()]) }
               </tbody>
           </table>
       </div>
    </div>
    
    <div class="section">
        <h2>🗓 Seasonality & Degradation</h2>
        <div class="grid" style="grid-template-columns: 1fr 1fr;">
            <div class="card">
                <h2>Monthly Seasonality (Calendar)</h2>
                <table>
                    <thead><tr><th>Month</th><th>Sharpe</th><th>Avg Trade</th><th>Win Rate</th></tr></thead>
                    <tbody>
                    { "".join([f"<tr><td>{s['Label']}</td><td>{s['Sharpe']:.2f}</td><td class='{ 'pos' if s['Avg Trade']>0 else 'neg' }'>{s['Avg Trade']*100:.2f}%</td><td>{s['Win Rate']*100:.0f}%</td></tr>" for s in month_stats]) }
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>Model Degradation (Within Quarter)</h2>
                <p style="color: #94a3b8; font-size: 12px">Month 1=Jan/Apr/Jul/Oct</p>
                <table>
                    <thead><tr><th>Month</th><th>Sharpe</th><th>Avg Trade</th><th>Win Rate</th></tr></thead>
                    <tbody>
                    { "".join([f"<tr><td>{s['Label']}</td><td>{s['Sharpe']:.2f}</td><td class='{ 'pos' if s['Avg Trade']>0 else 'neg' }'>{s['Avg Trade']*100:.2f}%</td><td>{s['Win Rate']*100:.0f}%</td></tr>" for s in deg_stats]) }
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>Day of Week Seasonality</h2>
                <table>
                    <thead><tr><th>Day</th><th>Sharpe</th><th>Avg Trade</th><th>Win Rate</th></tr></thead>
                    <tbody>
                    { "".join([f"<tr><td>{s['Label']}</td><td>{s['Sharpe']:.2f}</td><td class='{ 'pos' if s['Avg Trade']>0 else 'neg' }'>{s['Avg Trade']*100:.2f}%</td><td>{s['Win Rate']*100:.0f}%</td></tr>" for s in dow_stats]) }
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                 <h2>Day of Month Seasonality</h2>
                 <div style="height: 300px; overflow-y: auto;">
                 <table>
                    <thead><tr><th>Day</th><th>Sharpe</th><th>Avg Trade</th></tr></thead>
                    <tbody>
                    { "".join([f"<tr><td>{s['Label']}</td><td>{s['Sharpe']:.2f}</td><td class='{ 'pos' if s['Avg Trade']>0 else 'neg' }'>{s['Avg Trade']*100:.2f}%</td></tr>" for s in dom_stats]) }
                    </tbody>
                </table>
                </div>
            </div>
        </div>
    </div>
    
</body>
</html>"""
    
    out_path = Path('results/quarterly_validation/comparison_dashboard_quarterly.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"✅ Dashboard Saved: {out_path}")
    print(f"   (Distinct from Annual Dashboard at results/extended_history/)")

def get_risk_row(df, name, start, end):
    mask = (df['entry_time'] >= pd.Timestamp(start)) & (df['entry_time'] <= pd.Timestamp(end))
    sub = df[mask]
    if sub.empty: return ""
    
    wins = sub[sub['return_pct'] > 0]['return_pct']
    losses = sub[sub['return_pct'] <= 0]['return_pct']
    
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = losses.mean() if not losses.empty else 0
    ratio = abs(avg_win/avg_loss) if avg_loss != 0 else 0
    wr = len(wins) / len(sub)
    
    return f'''
    <tr>
        <td style="text-align: left; font-weight: bold; color: #38bdf8">{name}</td>
        <td class="neg">{avg_loss*100:.2f}%</td>
        <td class="pos">{avg_win*100:.2f}%</td>
        <td>{ratio:.2f}x</td>
        <td>{wr*100:.1f}%</td>
    </tr>
    '''

if __name__ == "__main__":
    regenerate_quarterly_dashboard()
