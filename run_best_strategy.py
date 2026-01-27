import pandas as pd
import sys
import json
from pathlib import Path
import warnings

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures
from src.strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy
from src.backtest.strategies.once_per_day import OncePerDayStrategy
from src.backtest.engine import BacktestConfig
from src.reporting.html_generator import generate_dashboard_html

def run():
    print("🚀 Starting Intraday Strategy Run...")
    
    # 1. Load Data
    print("Loading data...")
    loader = IntradayDataLoader('config/intraday_config.yaml')
    df_all = loader.get_period_data('test')
    
    # 2. Features
    print("Generating features...")
    alpha = IntradayAlphaFeatures()
    df_all = alpha.generate_all_features(df_all)
    season = SeasonalityFeatures()
    df_all = season.generate_all_features(df_all)
    
    # 3. Model
    print("Loading model (24-bar horizon)...")
    strategy = LightGBMIntradayStrategy({})
    model_path = Path.cwd() / 'results' / 'models' / 'lightgbmintraday_24bar'
    strategy.load_model(str(model_path))
    
    # 4. Strategy Configuration
    # Winner: 10:00 AM Entry (Bar 6), Max Hold 24 Bars, 1bp Cost
    config = BacktestConfig(
        transaction_cost_bps=1.0,  # 1bp realistic cost
        initial_capital=1_000_000
    )
    
    print("Running Strategy: Once-Per-Day (10am Entry, 24-bar Exit)...")
    bt = OncePerDayStrategy(config, entry_bar=6, exit_bars=36)
    etfs = ['SPY', 'QQQ', 'DIA', 'IWM']
    
    # 5. Execution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = bt.run(df_all, strategy.generate_signals, etfs)
    
    # 6. Benchmark
    spy_df = df_all.xs('SPY', level='instrument')
    bm_ret = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[0] - 1) * 100
    
    print("-" * 50)
    print(f"Total Return: {results.total_return*100:.2f}%")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Win Rate:     {results.win_rate*100:.1f}%")
    print(f"Num Trades:   {results.num_trades}")
    print(f"Benchmark:    {bm_ret:.2f}%")
    print("-" * 50)
    
    # 7. Reporting
    output_dir = Path.cwd() / 'results' / 'active'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON Data
    metrics = {
        'total_return': results.total_return,
        'annualized_return': results.annualized_return,
        'sharpe_ratio': results.sharpe_ratio,
        'max_drawdown': results.max_drawdown,
        'num_trades': results.num_trades,
        'win_rate': results.win_rate
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # HTML Dashboard
    html_content = generate_dashboard_html(
        "10am Volatility Capture (24-bar Exit)",
        metrics,
        [{'timestamp': str(t), 'equity': float(e)} for t, e in results.equity_curve.itertuples()],
        bm_ret
    )
    
    dashboard_path = output_dir / 'dashboard.html'
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
        
    print(f"✅ Results saved to {output_dir}")
    print(f"📊 Dashboard: {dashboard_path}")

if __name__ == "__main__":
    run()
