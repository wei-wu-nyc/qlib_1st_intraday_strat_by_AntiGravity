def generate_dashboard_html(strategy_name: str, metrics: dict, equity_curve: list, benchmark_return: float) -> str:
    """
    Generate a standalone HTML dashboard for the strategy results.
    """
    # JS data injection
    dates = [x['timestamp'] for x in equity_curve]
    equity_values = [x['equity'] for x in equity_curve]
    
    # Simple downsampling for chart performance if needed
    if len(dates) > 5000:
        step = len(dates) // 2000
        dates = dates[::step]
        equity_values = equity_values[::step]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategy Dashboard: {strategy_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e8e8e8;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2rem;
            margin-bottom: 30px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .metric-label {{ font-size: 0.9rem; color: #888; margin-bottom: 8px; }}
        .metric-value {{ font-size: 1.5rem; font-weight: bold; color: #fff; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            height: 500px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 {strategy_name} Performance</h1>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value { 'positive' if metrics['total_return'] > 0 else 'negative' }">
                    {metrics['total_return'] * 100:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Benchmark (SPY)</div>
                <div class="metric-value { 'positive' if benchmark_return > 0 else 'negative' }">
                    {benchmark_return:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{metrics['win_rate'] * 100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Trades</div>
                <div class="metric-value">{metrics['num_trades']}</div>
            </div>
             <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{metrics['max_drawdown'] * 100:.2f}%</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        const dates = {dates};
        const portfolioEquity = {equity_values};
        
        // Normalize to percentage return
        const initial = portfolioEquity[0];
        const portfolio Pct = portfolioEquity.map(v => (v / initial - 1) * 100);

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: '{strategy_name}',
                        data: portfolioEquity,
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: true,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'top', labels: {{ color: '#fff' }} }},
                    tooltip: {{ mode: 'index', intersect: false }}
                }},
                scales: {{
                    x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#888', maxTicksLimit: 12 }} }},
                    y: {{ 
                        grid: {{ color: 'rgba(255,255,255,0.05)' }}, 
                        ticks: {{ color: '#888', callback: v => '$' + (v/1000).toFixed(0) + 'k' }} 
                    }}
                }},
                interaction: {{ mode: 'nearest', axis: 'x', intersect: false }}
            }}
        }});
    </script>
</body>
</html>"""
    return html
