import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def regenerate_dashboard():
    print("🚀 Regenerating Dashboard with Detailed Trade Stats...")
    
    # 1. Load Trades
    trades_path = Path('results/extended_history/all_trades_2013_2025.csv')
    if not trades_path.exists():
        print("❌ Trade log not found!")
        return
        
    df_trades = pd.read_csv(trades_path)
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_trades['exit_date'] = df_trades['exit_time'].dt.date
    
    # 2. Reconstruct Daily Returns (to preserve Top-Level Stats)
    # Strategy used compounding (fixed_pos_pct), so we reconstruct equity info.
    # Daily PnL is summed from closed trades on that day.
    # Note: This ignores MTM (Mark-to-Market) of open positions, but for stats it's close enough.
    
    start_date = pd.Timestamp("2013-01-01")
    end_date = pd.Timestamp("2025-12-31")
    all_dates = pd.date_range(start_date, end_date, freq='B') # Business days
    
    daily_pnl = df_trades.groupby('exit_date')['pnl'].sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    
    # Reindex to full calendar
    series_pnl = daily_pnl.reindex(all_dates, fill_value=0.0)
    
    # Reconstruct Daily Returns with Per-Year Capital Reset
    # (Matches original run logic where capital resets to 1M each Jan 1st)
    
    unique_years = sorted(df_trades['entry_time'].dt.year.unique())
    daily_returns_series = pd.Series(dtype=float)
    
    for y in unique_years:
        # Get trades for this year
        mask_y = (df_trades['entry_time'].dt.year == y)
        trades_y = df_trades[mask_y]
        
        # Daily PnL for this year
        # Note: exit_date might cross year boundary? 
        # Strategy run loop filtered by year. 
        # But here we have strict trades. 
        # Let's group by exit_date for this subset.
        pnl_y = trades_y.groupby('exit_date')['pnl'].sum()
        
        # Reindex to full year business days
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
            curr_eq += p # Intra-year compounding
            
        # Append to master series
        daily_returns_series = pd.concat([
            daily_returns_series, 
            pd.Series(rets_y, index=dates_y)
        ])
    
    df_ret = daily_returns_series.to_frame(name='return')
    df_ret.index.name = 'date'
    
    # 3. Define Periods
    periods = {
        '2013-2015': ('2013-01-01', '2015-12-31'),
        '2016-2018': ('2016-01-01', '2018-12-31'),
        '2019-2022': ('2019-01-01', '2022-12-31'),
        '2023-2025': ('2023-01-01', '2025-12-31')
    }
    
    stats_cards = []
    dist_tables = []
    
    for name, (start, end) in periods.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        
        # A. Period Returns Analysis
        mask_ret = (df_ret.index >= start_ts) & (df_ret.index <= end_ts)
        sub_ret = df_ret[mask_ret]
        
        r = sub_ret['return'].values
        tot = np.prod(1+r) - 1
        ann = (1+tot)**(252/len(r)) - 1
        vol = np.std(r)*np.sqrt(252)
        sr = (np.mean(r)/np.std(r))*np.sqrt(252) if np.std(r)>0 else 0
        
        # Max DD
        eq_curve = np.cumprod(1+r)
        dd = np.min((eq_curve - np.maximum.accumulate(eq_curve))/np.maximum.accumulate(eq_curve))
        
        # B. Trade Analysis
        mask_trade = (df_trades['entry_time'] >= start_ts) & (df_trades['entry_time'] <= end_ts)
        t_sub = df_trades[mask_trade].copy()
        
        count = len(t_sub)
        avg_ret = t_sub['return_pct'].mean()
        best_ret = t_sub['return_pct'].max()
        worst_ret = t_sub['return_pct'].min()
        avg_bars = t_sub['holding_bars'].mean()
        # Approx minutes (5m bars)
        avg_mins = avg_bars * 5
        
        stats_cards.append({
            'Period': name,
            'Total Return': f"{tot*100:.1f}%",
            'Annualized': f"{ann*100:.1f}%",
            'Sharpe': f"{sr:.2f}",
            'Max DD': f"{dd*100:.1f}%",
            'Trades': count,
            'Avg Trade': f"{avg_ret*100:.2f}%",
            'Best Trade': f"{best_ret*100:.2f}%",
            'Worst Trade': f"{worst_ret*100:.2f}%",
            'Avg Duration': f"{avg_bars:.1f} bars ({avg_mins:.0f}m)"
        })
        
        # C. Distribution (Deciles)
        # Calculate deciles
        if count > 0:
            quantiles = t_sub['return_pct'].quantile(np.linspace(0.1, 0.9, 9))
            
            # Histogram bins (fixed ranges for comparison? or decile values?)
            # User asked for "10% bins" (Deciles) or "Distribution".
            # Let's show Return Distribution by Range to be more readable
            # e.g. < -1%, -1% to -0.5%, ...
            # OR simple deciles table.
            # Let's do Deciles Table.
            
            dist_data = {
                '10% (Worst)': t_sub['return_pct'].quantile(0.10),
                '30%': t_sub['return_pct'].quantile(0.30),
                '50% (Median)': t_sub['return_pct'].quantile(0.50),
                '70%': t_sub['return_pct'].quantile(0.70),
                '90% (Best)': t_sub['return_pct'].quantile(0.90)
            }
            dist_tables.append({'Period': name, 'Data': dist_data})

    # 4. Model Degradation (Monthly Analysis)
    print("  Comparing Performance by Month (1-12)...")
    monthly_stats = []
    
    # We aggregate ALL years for each month index (1=Jan, 12=Dec)
    # This represents "Month relative to training" since we retrain Jan 1
    for m in range(1, 13):
        # Filter Daily Returns for this month
        mask_m = df_ret.index.month == m
        sub_ret = df_ret[mask_m]
        
        # Filter Trades
        # Note: Use entry_time month
        mask_t = df_trades['entry_time'].dt.month == m
        t_sub = df_trades[mask_t]
        
        if sub_ret.empty:
            monthly_stats.append({'Month': m, 'Ret': 0, 'Sharpe': 0, 'WinRate': 0, 'Count': 0})
            continue
            
        # Stats
        r = sub_ret['return'].values
        # Total Return (Sum of returns for this month slot across 12 years)
        # Not compounding "12 Januaries" sequentially, but just sum or cumprod
        tot = np.prod(1+r) - 1
        
        # Ann Ret? Doesn't make sense for disjoint months.
        # Just use Avg Daily Ret * 252 for "Annualized run rate"
        avg_daily = np.mean(r)
        ann_run_rate = (1+avg_daily)**252 - 1
        
        # Sharpe (Annualized)
        sr = (np.mean(r)/np.std(r))*np.sqrt(252) if np.std(r)>0 else 0
        
        # Approx DD? (If you only traded Januaries)
        eq = np.cumprod(1+r)
        dd = np.min((eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq))
        
        # Trade Stats
        wr = (t_sub['return_pct'] > 0).mean() if not t_sub.empty else 0
        avg_t = t_sub['return_pct'].mean() if not t_sub.empty else 0
        
        monthly_stats.append({
            'Month': m,
            'Label': datetime(2000, m, 1).strftime('%B'), # Month Name
            'Total Return': tot,
            'Ann Run Rate': ann_run_rate,
            'Sharpe': sr,
            'Max DD': dd,
            'Win Rate': wr,
            'Trade Count': len(t_sub),
            'Avg Trade': avg_t
        })

    # 5. Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Extended History: Detailed Trade Analysis</title>
    <style>
        body {{ font-family: system-ui; background: #0f172a; color: #fff; padding: 20px; }}
        h1, h2 {{ color: #38bdf8; margin-top: 0; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; }}
        .metric {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        .label {{ color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
        td {{ padding: 6px 0; border-bottom: 1px solid #334155; }}
        td:last-child {{ text-align: right; color: #e2e8f0; font-family: monospace; }}
        th {{ text-align: right; color: #94a3b8; padding-bottom: 10px; }}
        th:first-child {{ text-align: left; }}
        .pos {{ color: #4ade80; }}
        .neg {{ color: #f87171; }}
        .section {{ margin-top: 40px; }}
        .highlight-row:hover {{ background: #334155; }}
    </style>
</head>
<body>
    <h1>🕰 Extended History: Detailed Trade Metrics (2013-2025)</h1>
    
    <div class="grid">
        { "".join([f'''
        <div class="card">
            <h2>{c['Period']}</h2>
            
            <div class="label">Performance</div>
            <table>
                <tr><td>Total Return</td><td class="{ 'pos' if '-' not in c['Total Return'] else 'neg' }">{c['Total Return']}</td></tr>
                <tr><td>Annualized</td><td>{c['Annualized']}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{c['Sharpe']}</td></tr>
                <tr><td>Max Drawdown</td><td class="neg">{c['Max DD']}</td></tr>
            </table>
            
            <div class="label" style="margin-top: 20px">Trade Stats</div>
            <table>
                <tr><td>Count</td><td>{c['Trades']}</td></tr>
                <tr><td>Avg Return</td><td class="{ 'pos' if '-' not in c['Avg Trade'] else 'neg' }">{c['Avg Trade']}</td></tr>
                <tr><td>Best Trade</td><td class="pos">{c['Best Trade']}</td></tr>
                <tr><td>Worst Trade</td><td class="neg">{c['Worst Trade']}</td></tr>
                <tr><td>Avg Duration</td><td>{c['Avg Duration']}</td></tr>
            </table>
        </div>
        ''' for c in stats_cards]) }
    </div>
    
    <div class="section">
        <h2>🗓 Model Degradation Analysis (Months Since Training)</h2>
        <p style="color: #94a3b8; margin-bottom: 20px;">
            Performance aggregated by month relative to model update (Jan 1st). 
            Checks if alpha decays as the model "ages" through the year.
        </p>
        <div class="card" style="width: 100%; overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Status</th>
                        <th>Total Return (Agg)</th>
                        <th>Sharpe Ratio</th>
                        <th>Avg Trade</th>
                        <th>Win Rate</th>
                        <th>Trades</th>
                        <th>Max DD (Monthly)</th>
                    </tr>
                </thead>
                <tbody>
                    { "".join([f'''
                    <tr class="highlight-row">
                        <td style="text-align: left; font-weight: bold; color: {get_month_color(m['Month'])}">
                            {m['Month']} - {m['Label']}
                        </td>
                        <td style="text-align: left; font-size: 12px; color: #94a3b8">{get_degradation_label(m['Sharpe'])}</td>
                        <td class="{ 'pos' if m['Total Return']>0 else 'neg' }">{m['Total Return']*100:.1f}%</td>
                        <td style="font-weight: bold">{m['Sharpe']:.2f}</td>
                        <td class="{ 'pos' if m['Avg Trade']>0 else 'neg' }">{m['Avg Trade']*100:.2f}%</td>
                        <td>{m['Win Rate']*100:.1f}%</td>
                        <td>{m['Trade Count']}</td>
                        <td class="neg">{m['Max DD']*100:.1f}%</td>
                    </tr>
                    ''' for m in monthly_stats]) }
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Return Distribution (Quantiles)</h2>
        <div class="card" style="width: 100%; overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Period</th>
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
                       <th>Period</th>
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

</body>
</html>
    """
    
    out_path = Path('results/extended_history/comparison_dashboard_detailed.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"✅ Detailed Dashboard Saved: {out_path}")

def get_month_color(m):
    # Gradient blue to red? Or just cycle.
    return "#38bdf8"

def get_degradation_label(sharpe):
    if sharpe > 1.5: return "⭐⭐⭐ Excellent"
    if sharpe > 1.0: return "⭐⭐ Good"
    if sharpe > 0.5: return "⭐ Stable"
    return "⚠️ Weak"

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
    regenerate_dashboard()
