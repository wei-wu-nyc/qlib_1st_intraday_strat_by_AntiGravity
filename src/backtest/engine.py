from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime

@dataclass
class BacktestConfig:
    """Configuration for intraday backtest."""
    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 0.0
    position_close_bar: int = 77  # 15:55
    last_entry_bar: int = 67  # 15:00
    bar_duration_minutes: int = 5
    # Strategy specific
    max_holding_bars: int = 24  # Default for 24-bar model

@dataclass
class TradeRecord:
    """Record of a single trade."""
    instrument: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    shares: float
    pnl: float
    return_pct: float
    holding_bars: int
    exit_reason: str
    entry_bar: int = 0  # Bar number at entry

@dataclass
class BacktestResults:
    """Complete results from a backtest run."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_returns: List[float]
    config: BacktestConfig

class IntradayBacktestEngine:
    """
    Base class for intraday backtest strategies.
    Provides utility methods for creating results.
    """
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.daily_returns = []
        self.trades = []
        self.equity_curve = []
    
    def calculate_results(self, capital: float, daily_returns: List[float], 
                         trades: List[TradeRecord], equity_curve: List[Tuple[pd.Timestamp, float]]) -> BacktestResults:
        """Calculate performance metrics from raw backtest data."""
        if trades:
            df = pd.DataFrame([{
                'instrument': t.instrument, 'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'entry_price': t.entry_price, 'exit_price': t.exit_price, 'shares': t.shares,
                'pnl': t.pnl, 'return_pct': t.return_pct, 'holding_bars': t.holding_bars,
                'exit_reason': t.exit_reason, 'entry_bar': t.entry_bar} for t in trades])
        else:
            df = pd.DataFrame()
        
        eq_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity']).set_index('timestamp') if equity_curve else pd.DataFrame()
        
        # Use final equity from curve
        final_equity = eq_df['equity'].iloc[-1] if len(eq_df) > 0 else capital
        tot_ret = final_equity / self.config.initial_capital - 1
        
        n = len(daily_returns)
        yrs = n / 252 if n > 0 else 1
        ann = (1 + tot_ret) ** (1 / yrs) - 1 if yrs > 0 else 0
        
        daily_std = np.std(daily_returns)
        sr = (np.mean(daily_returns) / daily_std) * np.sqrt(252) if len(daily_returns) > 1 and daily_std > 0 else 0
        
        dd = 0
        if len(eq_df) > 0:
            cummax = eq_df['equity'].cummax()
            dd = abs(((eq_df['equity'] - cummax) / cummax).min())
            
        wr = (df['return_pct'] > 0).mean() if len(df) > 0 else 0
        
        return BacktestResults(
            total_return=tot_ret, annualized_return=ann, sharpe_ratio=sr,
            max_drawdown=dd, num_trades=len(trades), win_rate=wr,
            trades=df, equity_curve=eq_df, daily_returns=daily_returns,
            config=self.config
        )
