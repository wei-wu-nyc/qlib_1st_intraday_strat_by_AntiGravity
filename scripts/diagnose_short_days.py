import pandas as pd
import sys
import numpy as np
from pathlib import Path

# Add project root to path BEFORE imports
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.backtest.strategies.multi_trade import MultiTradeStrategy
from src.backtest.engine import BacktestConfig
from src.data.data_loader import IntradayDataLoader
from src.strategies.moe_strategy import MoEIntradayStrategy
from src.strategies.ensemble_strategy import EnsembleStrategy

def run_diagnostic():
    print("🚀 Running Short Day Diagnostic (Year 2024)...")
    
    # 1. Config
    # Uses same config as V2
    entries = [3, 6, 14, 28, 30, 34] # 9:45, 10:00, 10:40, 11:50, 12:00, 12:20
    year = 2024
    
    # 2. Load Data
    loader = IntradayDataLoader('config/intraday_config.yaml')
    etfs = ['QQQ', 'SPY', 'IWM', 'DIA']
    print(f"Loading data for {year}...")
    df = loader.get_period_data('test', symbols=etfs)
    
    # Filter to 2024
    df = df[(df.index.get_level_values(0) >= f"{year}-01-01") & 
            (df.index.get_level_values(0) <= f"{year}-12-31")]
    
    # 3. Detect Short Days
    print("\n🔎 Scanning for Short Trading Days (< 70 bars)...")
    # Group by date and count bars (using SPY as reference)
    spy_data = df.xs('SPY', level=1)
    daily_counts = spy_data.groupby(spy_data.index.date).size()
    short_days = daily_counts[daily_counts < 70]
    
    print(f"Found {len(short_days)} short days:")
    for d, c in short_days.items():
        print(f"  - {d}: {c} bars (Expect ~78)")
        
    # 4. Load Models (Ensemble)
    print("\n🧠 Loading Models...")
    model_dir = Path(f'results/rolling_validation/models_{year}')
    if not model_dir.exists():
        print(f"❌ No models found for {year} in {model_dir}")
        return

    lgb = MoEIntradayStrategy({}); lgb.load_model(str(model_dir / 'moe_lgb'))
    xgb = MoEIntradayStrategy({}); xgb.load_model(str(model_dir / 'moe_xgb'))
    rf = MoEIntradayStrategy({}); rf.load_model(str(model_dir / 'moe_rf'))
    ensemble = EnsembleStrategy([lgb, xgb, rf], [1/3, 1/3, 1/3])
    
    # 5. Generate Signals
    print("Generating signals...")
    # Add features
    from src.features.intraday_alpha import IntradayAlphaFeatures
    from src.features.seasonality_features import SeasonalityFeatures
    df = IntradayAlphaFeatures().generate_all_features(df)
    df = SeasonalityFeatures().generate_all_features(df)
    
    signals = ensemble.generate_signals(df)
    
    # 6. Run Backtest
    print("Running Backtest...")
    bt_config = BacktestConfig(
        transaction_cost_bps=1.0, 
        initial_capital=1_000_000.0,
        position_close_bar=77 # The culprit
    )
    
    strat = MultiTradeStrategy(bt_config, exit_bars=36, allowed_entry_bars=entries, fixed_pos_pct=0.166)
    res = strat.run(df, lambda d: signals, etfs)
    
    # 7. Analyze Overnight Trades
    print("\n🕵️‍♀️ Analyzing Trades for Overnight Holds...")
    overnight_trades = []
    
    for t in res.trades.itertuples():
        if t.entry_time.date() != t.exit_time.date():
            overnight_trades.append(t)
            
    print(f"\n📊 Diagnostic Results:")
    print(f"Total Trades: {len(res.trades)}")
    print(f"Overnight Trades: {len(overnight_trades)}")
    
    if overnight_trades:
        print("\nExamples of Overnight Trades:")
        for t in overnight_trades[:10]:
            print(f"  [{t.instrument}] Entry: {t.entry_time} (Bar {t.entry_bar}) -> Exit: {t.exit_time} ({t.exit_reason})")
            
        print("\nExit Reason Distribution for Overnight Trades:")
        reasons = {}
        for t in overnight_trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print(reasons)
        
        # Correlate with short days
        print("\nCorrelation with Short Days:")
        affected_dates = set(t.entry_time.date() for t in overnight_trades)
        matches = 0
        for d in affected_dates:
            if d in short_days.index:
                print(f"  ✅ Confirmed: Trades on {d} (Short Day) were held overnight.")
                matches += 1
            else:
                print(f"  ❓ Unexplained: Trades on {d} were held overnight (Not a short day?).")
        
        print(f"\n{matches}/{len(affected_dates)} overnight trade dates correspond to Short Days.")

if __name__ == "__main__":
    run_diagnostic()
