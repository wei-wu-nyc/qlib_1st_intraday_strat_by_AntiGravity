
import pandas as pd
import numpy as np
from pathlib import Path
from src.backtest.strategies.once_per_day import OncePerDayStrategy
from src.backtest.engine import BacktestConfig
from src.strategies.ml_models.lightgbm_intraday import LightGBMIntradayStrategy
from src.data.data_loader import IntradayDataLoader
from src.features.intraday_alpha import IntradayAlphaFeatures
from src.features.seasonality_features import SeasonalityFeatures




class DebugOncePerDayStrategy(OncePerDayStrategy):
    def _open(self, inst, timestamp, price, bar_index):
        import numpy as np
        if np.isnan(self.capital):
             print(f"[DEBUG] NAN CAPITAL DETECTED at {timestamp}!")
        elif self.capital < 1000:
             print(f"[DEBUG] LOW CAPITAL ALERT! Capital={self.capital} at {timestamp}")
        
        super()._open(inst, timestamp, price, bar_index)
        if np.isnan(self.capital): # Should be 0, but if NaN propagated...
             print(f"[DEBUG] CAPITAL BECAME NAN after open at {timestamp}!")
    
    def _close(self, inst, timestamp, price, bar_index, reason):
        import numpy as np
        super()._close(inst, timestamp, price, bar_index, reason)
        if np.isnan(self.capital):
             print(f"[DEBUG] CAPITAL BECAME NAN after close at {timestamp}. Price={price}")


def debug_crash():
    print("🚀 Debugging 12:00 Noon Crash...")
    
    # Load Models
    global_strategy = LightGBMIntradayStrategy({})
    global_strategy.load_model(str(Path.cwd() / 'results' / 'models' / 'lightgbmintraday_24bar'))
    
    loader = IntradayDataLoader('config/intraday_config.yaml')
    df = loader.get_period_data('test')
    
    print("Generating features...")
    alpha = IntradayAlphaFeatures()
    df = alpha.generate_all_features(df)
    season = SeasonalityFeatures()
    df = season.generate_all_features(df)
    
    config = BacktestConfig(transaction_cost_bps=1.0)
    etfs = ['SPY', 'QQQ', 'DIA', 'IWM']
    
    print("Running Backtest for 12:00 Noon...")
    # Use Debug Strategy
    bt = DebugOncePerDayStrategy(config, entry_bar=30, exit_bars=24) 
    res = bt.run(df, global_strategy.generate_signals, etfs)
    
    print(f"Total Return: {res.total_return*100:.2f}%")

    print(f"Num Trades: {res.num_trades}")
    
    if res.trades.empty:
        print("No trades!")
        return

    # Analyze Trades
    trades = res.trades.copy()
    print("\nWorst Trades:")
    print(trades.sort_values('return_pct').head(10)[['entry_time', 'instrument', 'entry_price', 'exit_price', 'return_pct']])
    
    min_ret = trades['return_pct'].min()
    print(f"\nMinimum Return: {min_ret*100:.4f}%")
    
    # Analyze Crash Date
    utils_crash_day = pd.Timestamp('2023-11-27').date()
    
    print("\nTrades around Crash Date (2023-11-27):")
    # Show trades BEFORE the crash
    prev_trades = trades[trades['entry_time'].dt.date < utils_crash_day].tail(20)
    print("Previous 20 trades:")
    print(prev_trades[['entry_time', 'instrument', 'return_pct', 'pnl']])
    
    crash_trades = trades[trades['entry_time'].dt.date == utils_crash_day]
    print("\nCrash Day Trades:")
    print(crash_trades)

        
    # Check Equity Curve
    print("\nEquity Curve Head:")
    print(res.equity_curve.head())
    print("\nEquity Curve Tail:")
    print(res.equity_curve.tail())
    
    # Check for zero equity
    zeros = res.equity_curve[res.equity_curve['equity'] <= 0.01]
    if not zeros.empty:
        print("\nEquity hit zero at:")
        print(zeros.head())

if __name__ == "__main__":
    debug_crash()
