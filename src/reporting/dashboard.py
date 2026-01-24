"""
HTML Dashboard for Performance Data.

Generates an interactive web dashboard to visualize strategy performance,
including equity curves, metrics tables, and trade statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import json


def generate_dashboard(
    results: Dict[str, Dict[str, Dict]],
    equity_curves: Optional[Dict[str, pd.DataFrame]] = None,
    output_path: str = "results/dashboard.html",
    title: str = "Intraday Trading Strategy Dashboard",
    bench_returns: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate an HTML dashboard for strategy performance.
    
    Args:
        results: Dict of {strategy: {period: metrics}}
        equity_curves: Optional dict of {strategy: equity_df}
        output_path: Path to save HTML file
        title: Dashboard title
        bench_returns: Dict of {period: return_str}
        
    Returns:
        Path to generated HTML file
    """
    # Prepare data for tables
    table_data = prepare_table_data(results)
    
    # Prepare equity curve data for charts
    chart_data = prepare_chart_data(equity_curves) if equity_curves else {}
    
    # (HTML generation code...)
    # ...
    # ...
    # Pass bench_returns to content generation functions
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        :root {{
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --accent: #0f3460;
            --highlight: #e94560;
            --text: #eaeaea;
            --text-muted: #a0a0a0;
            --positive: #00ff88;
            --negative: #ff4444;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--bg-dark) 0%, #0f0f23 100%);
            color: var(--text);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            margin-bottom: 30px;
        }}
        
        h1 {{
            font-size: 2.5rem;
            font-weight: 300;
            letter-spacing: 2px;
            background: linear-gradient(45deg, var(--highlight), #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .timestamp {{
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 10px;
        }}
        
        .section {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }}
        
        .section h2 {{
            font-size: 1.3rem;
            font-weight: 500;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--accent);
        }}
        
        /* Chart specific styles */
        .chart-controls {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        .chart-btn {{
            padding: 8px 16px;
            background: var(--accent);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text);
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }}
        
        .chart-btn:hover, .chart-btn.active {{
            background: var(--highlight);
            border-color: var(--highlight);
        }}
        
        .horizon-select {{
            padding: 8px 16px;
            background: var(--accent);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text);
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85rem;
            margin-left: 20px;
        }}
        
        .chart-wrapper {{
            position: relative;
            height: 500px;
            width: 100%;
        }}

        /* Strategy selector tabs */
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .tab {{
            padding: 10px 20px;
            background: var(--accent);
            border: none;
            border-radius: 8px;
            color: var(--text);
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9rem;
        }}
        
        .tab:hover {{
            background: var(--highlight);
        }}
        
        .tab.active {{
            background: var(--highlight);
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
        }}
        
        /* Performance table - periods as rows */
        .perf-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .perf-table th {{
            background: var(--accent);
            padding: 12px 15px;
            text-align: left;
            font-weight: 500;
            position: sticky;
            top: 0;
        }}
        
        .perf-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .perf-table tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        .perf-table .period {{
            font-weight: 600;
            color: var(--highlight);
        }}
        
        .positive {{
            color: var(--positive);
        }}
        
        .negative {{
            color: var(--negative);
        }}
        
        /* Cards grid */
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, var(--accent) 0%, rgba(15, 52, 96, 0.5) 100%);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}
        
        .metric-card .label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-card .value {{
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 8px;
        }}
        
        /* Chart placeholder */
        .chart-container {{
            height: 400px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
        }}
        
        /* Strategy comparison table */
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .comparison-table th, .comparison-table td {{
            padding: 10px 15px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .comparison-table th {{
            background: var(--accent);
            font-weight: 500;
        }}
        
        .comparison-table th:first-child,
        .comparison-table td:first-child {{
            text-align: left;
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            font-size: 0.8rem;
        }}
        
        @media (max-width: 768px) {{
            h1 {{ font-size: 1.8rem; }}
            .cards-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>
        
        <!-- Interactive Equity Curve Chart -->
        <section class="section">
            <h2>📈 Cumulative Equity Curves</h2>
            <div class="chart-controls">
                <button class="chart-btn active" onclick="updateChart('all')">All Periods</button>
                <button class="chart-btn" onclick="updateChart('valid_test')">Valid + Test</button>
                <button class="chart-btn" onclick="updateChart('test')">Test Only</button>
                <select class="horizon-select" id="horizonSelect" onchange="updateHorizon()">
                    <option value="all">All Horizons</option>
                    <option value="24" selected>24-bar (2hr)</option>
                    <option value="36">36-bar (3hr)</option>
                    <option value="eod">EOD (close)</option>
                </select>
            </div>
            <div class="chart-wrapper">
                <canvas id="equityChart"></canvas>
            </div>
        </section>
        
        <!-- Strategy Summary Cards -->
        <section class="section">
            <h2>📊 Strategy Overview</h2>
            <p style="color: var(--text-muted); margin-bottom: 15px; font-size: 0.85rem;">
                Filtered by horizon selected above. Use "All Horizons" to see all strategies.
            </p>
            <div class="tabs" id="strategyTabs">
                {generate_strategy_tabs(list(results.keys()))}
            </div>
            <div id="strategyContent">
                {generate_strategy_content(results, bench_returns)}
            </div>
        </section>
        
        <!-- Performance Comparison -->
        <section class="section">
            <h2>🏆 Performance Comparison (Test Period)</h2>
            {generate_comparison_table(results, bench_returns)}
        </section>
        
        <!-- Detailed Metrics by Period -->
        <section class="section">
            <h2>📋 Detailed Metrics by Period</h2>
            <p style="color: var(--text-muted); margin-bottom: 15px;">
                Rows: Periods | Columns: Metrics (easier to compare experiments)
            </p>
            {generate_detailed_tables(results, bench_returns)}
        </section>
        
        <footer>
            Intraday Trading Strategy Framework | 
            ETFs: DIA, SPY, QQQ, IWM | 
            5-minute bars | Long Only
        </footer>
    </div>
    
    <script>
        // Strategy tab switching
        document.querySelectorAll('.tab').forEach(tab => {{
            tab.addEventListener('click', function() {{
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                // Add active to clicked
                this.classList.add('active');
                
                // Hide all content
                document.querySelectorAll('.strategy-panel').forEach(p => p.style.display = 'none');
                // Show selected
                const strategy = this.dataset.strategy;
                document.getElementById('panel-' + strategy).style.display = 'block';
            }});
        }});
        
        // Activate first tab
        document.querySelector('.tab')?.click();
        
        // --- Chart.js Implementation ---
        
        const chartData = {json.dumps(chart_data)};
        // Defined periods for filtering
        // Assuming typical ranges based on config - ideally passed from python but simplified here
        // Train: < 2019, Valid: 2019-2021, Test: >= 2022
        
        const ctx = document.getElementById('equityChart').getContext('2d');
        let myChart;
        let currentPeriodMode = 'all';
        let currentHorizon = '8';
        
        function getPeriodColor(dateStr) {{
            const date = new Date(dateStr);
            const year = date.getFullYear();
            if (year < 2019) return 'rgba(100, 100, 100, 0.2)'; // Train (faded)
            if (year < 2022) return 'rgba(255, 206, 86, 0.1)';  // Valid (yellow tint)
            return 'rgba(0, 0, 0, 0)'; // Test (transparent)
        }}
        
        function shouldShowStrategy(strategy, horizon, context) {{
            // context: 'chart' or 'table'
            
            // Rule-based strategies: always show
            const ruleBased = ['MomentumBreakout', 'MeanReversion', 'ORB'];
            if (ruleBased.includes(strategy)) return true;
            
            // Benchmark: always show
            if (strategy.includes('Benchmark')) return true;
            
            // Filter by horizon
            let matchesHorizon = false;
            const horizonSuffix = '_' + horizon + 'bar';
            
            if (horizon === 'all') {{
                matchesHorizon = true;
            }} else if (horizon === 'eod') {{
                matchesHorizon = strategy.includes('_eod');
            }} else {{
                matchesHorizon = strategy.includes(horizonSuffix);
            }}
            
            if (!matchesHorizon) return false;
            
            // Chart specific filtering: unclutter per user request
            if (context === 'chart') {{
                // If it's an ML strategy (LightGBM/XGBoost), ONLY show the '_Base' version
                // or the original version (without underscore suffix besides horizon)
                if (strategy.includes('LightGBM') || strategy.includes('XGBoost')) {{
                    // Check if it is a variant (contains %, Fix, Target, NegGate, Combo)
                    const isVariant = strategy.includes('%') || 
                                      strategy.includes('Fix') || 
                                      strategy.includes('Target') || 
                                      strategy.includes('NegGate') || 
                                      strategy.includes('Combo') ||
                                      strategy.includes('Trail');
                                      
                    if (isVariant) return false;
                }}
            }}
            
            return true;
        }}
        
        function initChart(filterMode, horizonFilter) {{
            if (myChart) myChart.destroy();
            
            const datasets = [];
            const colors = ['#e94560', '#00ff88', '#4cc9f0', '#f72585', '#fee440', '#9b5de5', '#ff9f1c', '#2ec4b6', '#e07be0', '#80ffdb', '#ff595e', '#ffca3a'];
            
            let minDate = new Date('2000-01-01');
            if (filterMode === 'valid_test') minDate = new Date('2019-01-01');
            if (filterMode === 'test') minDate = new Date('2022-01-01');
            
            let colorIndex = 0;
            Object.keys(chartData).forEach((strategy) => {{
                // Filter by horizon AND chart context
                if (!shouldShowStrategy(strategy, horizonFilter, 'chart')) return;
                
                const dataPoints = chartData[strategy].timestamps.map((t, i) => ({{
                    x: t,
                    y: chartData[strategy].values[i]
                }})).filter(pt => new Date(pt.x) >= minDate);
                
                // Re-normalize to 1.0 at start of filtered range
                if (dataPoints.length > 0) {{
                    const startVal = dataPoints[0].y;
                    const normalizedPoints = dataPoints.map(pt => ({{ x: pt.x, y: pt.y / startVal }}));
                    
                    datasets.push({{
                        label: strategy,
                        data: normalizedPoints,
                        borderColor: colors[colorIndex % colors.length],
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1,
                        fill: false
                    }});
                    colorIndex++;
                }}
            }});
            
            // Add background annotations for periods (only if in All mode)
            const annotations = {{}};
            if (filterMode === 'all') {{
                // Can use plugins for this, but simple background color on canvas is tricky with mixed lines
                // Keeping clean lines for now
            }}
            
            myChart = new Chart(ctx, {{
                type: 'line',
                data: {{ datasets: datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false,
                    }},
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#eaeaea' }}
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                            callbacks: {{
                                label: function(context) {{
                                    return context.dataset.label + ': ' + context.parsed.y.toFixed(4);
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{
                                unit: 'month'
                            }},
                            grid: {{ color: 'rgba(255, 255, 255, 0.1)' }},
                            ticks: {{ color: '#a0a0a0' }}
                        }},
                        y: {{
                            grid: {{ color: 'rgba(255, 255, 255, 0.1)' }},
                            ticks: {{ color: '#a0a0a0' }},
                            title: {{
                                display: true,
                                text: 'Normalized Equity',
                                color: '#a0a0a0'
                            }}
                        }}
                    }}
                }}
            }});
            
            // Also update strategy tabs visibility
            updateStrategyTabsVisibility(horizonFilter);
        }}
        
        function updateStrategyTabsVisibility(horizonFilter) {{
            document.querySelectorAll('.tab').forEach(tab => {{
                const strategy = tab.dataset.strategy;
                if (shouldShowStrategy(strategy, horizonFilter)) {{
                    tab.style.display = 'inline-block';
                }} else {{
                    tab.style.display = 'none';
                }}
            }});
            // Activate first visible tab
            const visibleTab = document.querySelector('.tab[style="display: inline-block;"], .tab:not([style])');
            if (visibleTab) visibleTab.click();
        }}
        
        // Expose update function
        window.updateChart = function(mode) {{
            currentPeriodMode = mode;
            
            // Update button states
            document.querySelectorAll('.chart-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            initChart(mode, currentHorizon);
        }};
        
        window.updateHorizon = function() {{
            const select = document.getElementById('horizonSelect');
            currentHorizon = select.value;
            initChart(currentPeriodMode, currentHorizon);
        }};
        
        // Init default
        if (Object.keys(chartData).length > 0) {{
            initChart('all', currentHorizon);
        }} else {{
            console.log("No chart data available");
        }}
    </script>
</body>
</html>
"""
    
    # Save the file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    return str(output_path)


def generate_strategy_tabs(strategies: List[str]) -> str:
    """Generate HTML for strategy tabs."""
    tabs = []
    for i, strategy in enumerate(strategies):
        active = 'active' if i == 0 else ''
        tabs.append(f'<button class="tab {active}" data-strategy="{strategy}">{strategy}</button>')
    return '\n'.join(tabs)


def generate_strategy_content(results: Dict[str, Dict], bench_returns: Optional[Dict] = None) -> str:
    """Generate HTML for strategy content panels."""
    panels = []
    
    for i, (strategy, periods) in enumerate(results.items()):
        display = 'block' if i == 0 else 'none'
        
        # Get test period metrics if available
        test_metrics = periods.get('test', {})
        if hasattr(test_metrics, 'to_dict'):
            test_dict = test_metrics.to_dict()
        elif isinstance(test_metrics, dict):
            test_dict = test_metrics
        else:
            test_dict = {}
        
        cards_html = f"""
        <div class="cards-grid">
            <div class="metric-card">
                <div class="label">Annual Return</div>
                <div class="value {get_color_class(test_dict.get('Ann. Return', '0%'))}">{test_dict.get('Ann. Return', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value">{test_dict.get('Sharpe Ratio', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="label">Max Drawdown</div>
                <div class="value negative">{test_dict.get('Max Drawdown', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="label">Win Rate</div>
                <div class="value">{test_dict.get('Win Rate', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Trades</div>
                <div class="value">{test_dict.get('Num Trades', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="label">Time in Market</div>
                <div class="value">{test_dict.get('Time in Market', 'N/A')}</div>
            </div>
        </div>
        """
        
        # Period comparison table (periods as rows, metrics as columns)
        table_html = generate_period_table(periods, bench_returns)
        
        panels.append(f'''
        <div class="strategy-panel" id="panel-{strategy}" style="display: {display};">
            {cards_html}
            {table_html}
        </div>
        ''')
    
    return '\n'.join(panels)


def generate_period_table(periods: Dict, bench_returns: Optional[Dict] = None) -> str:
    """Generate table with periods as rows and metrics as columns."""
    metrics = ['Ann. Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Num Trades', 'Avg Holding (bars)']
    display_metrics = ['Ann. Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Num Trades', 'Avg Bars']
    
    rows = []
    for period_name in ['train', 'valid', 'test']:
        if period_name not in periods:
            continue
        
        metrics_obj = periods[period_name]
        if hasattr(metrics_obj, 'to_dict'):
            metric_dict = metrics_obj.to_dict()
        elif isinstance(metrics_obj, dict):
            metric_dict = metrics_obj
        else:
            metric_dict = {}
        
        cells = [f'<td class="period">{period_name.capitalize()}</td>']
        
        # Add bench return if available (handle both string and dict formats)
        if bench_returns and period_name in bench_returns:
            bench_data = bench_returns[period_name]
            if isinstance(bench_data, str):
                cells.append(f'<td style="color: #bbb;">{bench_data}</td>')
            else:
                cells.append(f'<td style="color: #bbb;">{bench_data.get("Ann. Return", "-")}</td>')
        else:
            cells.append('<td>-</td>')
            
        for m in metrics:
            value = metric_dict.get(m, 'N/A')
            color_class = get_color_class(value) if m in ['Ann. Return'] else ''
            cells.append(f'<td class="{color_class}">{value}</td>')
        
        rows.append(f'<tr>{"".join(cells)}</tr>')
    
    header_cells = ['<th>Period</th>', '<th>Bench Ret</th>'] + [f'<th>{m}</th>' for m in display_metrics]
    
    return f'''
    <table class="perf-table">
        <thead><tr>{"".join(header_cells)}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    '''


def generate_comparison_table(results: Dict, bench_returns: Optional[Dict] = None) -> str:
    """Generate comparison table across all strategies for test period."""
    metrics = ['Ann. Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Num Trades', 'Avg Holding (bars)']
    display_metrics = ['Ann. Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Num Trades', 'Avg Bars']
    
    rows = []
    
    # Bench row (with all available metrics)
    if bench_returns and 'test' in bench_returns:
        cells = ['<td><strong>Benchmark (SPY)</strong></td>']
        bench_test = bench_returns['test']
        
        # Handle both old format (string) and new format (dict)
        if isinstance(bench_test, str):
            # Old format - just Ann. Return
            cells.append(f'<td style="color: #bbb;">{bench_test}</td>')
            cells.extend(['<td>-</td>'] * (len(metrics) - 1))
        else:
            # New format - dict with multiple metrics
            for m in metrics:
                value = bench_test.get(m, '-')
                cells.append(f'<td style="color: #bbb;">{value}</td>')
        
        rows.append(f'<tr style="background: rgba(255, 255, 255, 0.05); border-bottom: 2px solid var(--accent);">{"".join(cells)}</tr>')

    for strategy, periods in results.items():
        test_metrics = periods.get('test', {})
        if hasattr(test_metrics, 'to_dict'):
            metric_dict = test_metrics.to_dict()
        elif isinstance(test_metrics, dict):
            metric_dict = test_metrics
        else:
            metric_dict = {}
        
        cells = [f'<td><strong>{strategy}</strong></td>']
        
        for m in metrics:
            value = metric_dict.get(m, 'N/A')
            color_class = get_color_class(value) if m in ['Ann. Return'] else ''
            cells.append(f'<td class="{color_class}">{value}</td>')
        
        rows.append(f'<tr>{"".join(cells)}</tr>')
    
    header_cells = ['<th>Strategy</th>'] + [f'<th>{m}</th>' for m in metrics]
    
    return f'''
    <table class="comparison-table">
        <thead><tr>{"".join(header_cells)}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    '''


def generate_detailed_tables(results: Dict, bench_returns: Optional[Dict] = None) -> str:
    """Generate detailed tables for each strategy."""
    tables = []
    for strategy, periods in results.items():
        table_html = f'<h3 style="margin-top: 20px; color: var(--highlight);">{strategy}</h3>'
        table_html += generate_period_table(periods, bench_returns)
        tables.append(table_html)
    return '\n'.join(tables)


def get_color_class(value: str) -> str:
    """Determine CSS color class based on value."""
    if not isinstance(value, str):
        return ''
    value = value.replace('%', '').strip()
    try:
        num = float(value)
        return 'positive' if num > 0 else 'negative' if num < 0 else ''
    except:
        return ''


def prepare_table_data(results: Dict) -> Dict:
    """Prepare table data for JSON."""
    return results


def prepare_chart_data(equity_curves: Dict) -> Dict:
    """Prepare equity curve data for charts."""
    chart_data = {}
    for strategy, df in equity_curves.items():
        if df is not None and 'equity' in df.columns:
            # Downsample to daily resolution to avoid rendering millions of points
            # 5-min data for 20 years is ~2M points. Daily is ~5000.
            # Use the last value of each day
            df_daily = df.resample('D').last().dropna()
            
            chart_data[strategy] = {
                'timestamps': df_daily.index.strftime('%Y-%m-%d %H:%M').tolist(),
                'values': df_daily['equity'].tolist(),
            }
    return chart_data


if __name__ == "__main__":
    print("Testing Dashboard Generator...")
    
    # Create mock results
    from dataclasses import dataclass
    
    @dataclass
    class MockMetrics:
        def to_dict(self):
            return {
                'Total Return': '45.2%',
                'Ann. Return': '12.5%',
                'Ann. Volatility': '18.3%',
                'Sharpe Ratio': '0.92',
                'Max Drawdown': '-15.2%',
                'Calmar Ratio': '0.82',
                'Win Rate': '54.2%',
                'Profit Factor': '1.35',
                'Avg Trade Return': '0.08%',
                'Num Trades': 1254,
                'Time in Market': '68.5%',
            }
    
    mock_results = {
        'MomentumBreakout': {
            'train': MockMetrics(),
            'valid': MockMetrics(),
            'test': MockMetrics(),
        },
        'MeanReversion': {
            'train': MockMetrics(),
            'valid': MockMetrics(),
            'test': MockMetrics(),
        },
        'XGBoost': {
            'train': MockMetrics(),
            'valid': MockMetrics(),
            'test': MockMetrics(),
        },
    }
    
    # Generate dashboard
    output = generate_dashboard(mock_results, output_path='results/dashboard.html')
    print(f"\nDashboard generated: {output}")
    print("\nDashboard test completed!")
