"""
Performance Metrics for Intraday Trading.

Calculates standard trading metrics plus custom metrics:
- Win rate (only for actual trades)
- Number of trades
- Percentage time in market
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    
    # Risk metrics
    max_drawdown: float
    calmar_ratio: float
    
    # Trade metrics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    
    # Activity metrics
    time_in_market_pct: float
    avg_holding_bars: float
    
    # Benchmark comparison
    excess_return: float
    information_ratio: float
    
    # Charting data
    equity_curve: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'Total Return': f"{self.total_return*100:.2f}%",
            'Ann. Return': f"{self.annualized_return*100:.2f}%",
            'Ann. Volatility': f"{self.annualized_volatility*100:.2f}%",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Max Drawdown': f"{self.max_drawdown*100:.2f}%",
            'Calmar Ratio': f"{self.calmar_ratio:.2f}",
            'Num Trades': self.num_trades,
            'Win Rate': f"{self.win_rate*100:.1f}%",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Avg Trade Return': f"{self.avg_trade_return*100:.3f}%",
            'Avg Win': f"{self.avg_win*100:.3f}%",
            'Avg Loss': f"{self.avg_loss*100:.3f}%",
            'Time in Market': f"{self.time_in_market_pct*100:.1f}%",
            'Avg Holding (bars)': f"{self.avg_holding_bars:.1f}",
            'Excess Return': f"{self.excess_return*100:.2f}%",
            'Info Ratio': f"{self.information_ratio:.2f}",
        }


def calculate_metrics(
    backtest_results: Dict,
    benchmark_returns: Optional[np.ndarray] = None,
    trading_days_per_year: int = 252,
    bars_per_day: int = 78,
) -> PerformanceMetrics:
    """
    Calculate all performance metrics from backtest results.
    
    Args:
        backtest_results: Dict from IntradayBacktest.run()
        benchmark_returns: Optional array of benchmark daily returns
        trading_days_per_year: Trading days for annualization
        bars_per_day: Intraday bars per day
        
    Returns:
        PerformanceMetrics object
    """
    equity_curve = backtest_results['equity_curve']
    trades_df = backtest_results['trades']
    daily_returns = backtest_results['daily_returns']
    
    # Total return
    total_return = backtest_results['total_return']
    
    # Number of days
    if len(daily_returns) > 0:
        n_days = len(daily_returns)
    else:
        n_days = max(1, len(equity_curve) // bars_per_day)
    
    # Annualized return
    years = n_days / trading_days_per_year
    annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
    
    # Annualized volatility
    if len(daily_returns) > 1:
        annualized_volatility = np.std(daily_returns) * np.sqrt(trading_days_per_year)
    else:
        annualized_volatility = 0
    
    # Sharpe ratio (assuming 0% risk-free rate)
    if annualized_volatility > 0:
        sharpe_ratio = annualized_return / annualized_volatility
    else:
        sharpe_ratio = 0
    
    # Max drawdown
    equity = equity_curve['equity'].values
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = abs(np.min(drawdown))
    
    # Calmar ratio
    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = 0
    
    # Trade metrics
    num_trades = len(trades_df) if trades_df is not None and len(trades_df) > 0 else 0
    
    if num_trades > 0:
        trade_returns = trades_df['return_pct'].values
        
        # Win rate (ONLY for actual trades)
        wins = trade_returns > 0
        win_rate = np.mean(wins)
        
        # Profit factor
        gross_profit = np.sum(trade_returns[wins]) if np.any(wins) else 0
        gross_loss = abs(np.sum(trade_returns[~wins])) if np.any(~wins) else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Average trade
        avg_trade_return = np.mean(trade_returns)
        avg_win = np.mean(trade_returns[wins]) if np.any(wins) else 0
        avg_loss = np.mean(trade_returns[~wins]) if np.any(~wins) else 0
        
        # Average holding period
        avg_holding_bars = np.mean(trades_df['holding_bars'].values)
    else:
        win_rate = 0
        profit_factor = 0
        avg_trade_return = 0
        avg_win = 0
        avg_loss = 0
        avg_holding_bars = 0
    
    # Time in market
    # Calculate based on total bars vs bars with position
    total_bars = len(equity_curve)
    if num_trades > 0 and 'holding_bars' in trades_df.columns:
        bars_in_market = trades_df['holding_bars'].sum()
        time_in_market_pct = bars_in_market / total_bars
    else:
        time_in_market_pct = 0
    
    # Benchmark comparison
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        benchmark_total = np.prod(1 + benchmark_returns) - 1
        excess_return = total_return - benchmark_total
        
        # Information ratio
        if len(daily_returns) > 0 and len(benchmark_returns) == len(daily_returns):
            tracking_diff = np.array(daily_returns) - np.array(benchmark_returns)
            tracking_error = np.std(tracking_diff) * np.sqrt(trading_days_per_year)
            if tracking_error > 0:
                information_ratio = (annualized_return - 
                    (np.mean(benchmark_returns) * trading_days_per_year)) / tracking_error
            else:
                information_ratio = 0
        else:
            information_ratio = 0
    else:
        excess_return = 0
        information_ratio = 0
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        num_trades=num_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_return=avg_trade_return,
        avg_win=avg_win,
        avg_loss=avg_loss,
        time_in_market_pct=time_in_market_pct,
        avg_holding_bars=avg_holding_bars,
        excess_return=excess_return,
        information_ratio=information_ratio,
        equity_curve=equity_curve,
    )


def create_performance_table(
    metrics_dict: Dict[str, Dict[str, PerformanceMetrics]],
) -> pd.DataFrame:
    """
    Create a formatted performance table for multiple strategies and periods.
    
    Args:
        metrics_dict: Dict of {strategy_name: {period: PerformanceMetrics}}
        
    Returns:
        Formatted DataFrame for display
    """
    rows = []
    
    for strategy_name, periods in metrics_dict.items():
        row = {'Strategy': strategy_name}
        
        for period, metrics in periods.items():
            metric_dict = metrics.to_dict()
            for metric_name, value in metric_dict.items():
                row[f'{period} {metric_name}'] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def calculate_benchmark_returns(
    benchmark_data: pd.DataFrame,
    price_col: str = 'close',
) -> Tuple[np.ndarray, float]:
    """
    Calculate benchmark buy-and-hold returns.
    
    Args:
        benchmark_data: DataFrame with price data
        price_col: Column name for price
        
    Returns:
        Tuple of (daily_returns array, total_return)
    """
    prices = benchmark_data[price_col].values
    
    # Daily returns from bar data
    # Group by date and get last price of each day
    if isinstance(benchmark_data.index, pd.DatetimeIndex):
        benchmark_data = benchmark_data.copy()
        benchmark_data['_date'] = benchmark_data.index.date
        daily_prices = benchmark_data.groupby('_date')[price_col].last()
    else:
        daily_prices = pd.Series(prices)
    
    daily_returns = daily_prices.pct_change().dropna().values
    total_return = prices[-1] / prices[0] - 1
    
    return daily_returns, total_return


if __name__ == "__main__":
    print("Testing PerformanceMetrics...")
    
    # Create mock backtest results
    np.random.seed(42)
    n_bars = 78 * 20  # 20 trading days
    
    # Mock equity curve
    returns = np.random.randn(n_bars) * 0.001 + 0.00005
    equity = 1000000 * np.cumprod(1 + returns)
    
    equity_curve = pd.DataFrame({
        'equity': equity,
    }, index=pd.date_range('2023-12-01 09:35', periods=n_bars, freq='5min'))
    
    # Mock trades
    n_trades = 50
    trades = pd.DataFrame({
        'instrument': ['SPY'] * n_trades,
        'entry_time': pd.date_range('2023-12-01', periods=n_trades, freq='1h'),
        'exit_time': pd.date_range('2023-12-01 01:00', periods=n_trades, freq='1h'),
        'entry_price': 100 + np.random.randn(n_trades) * 2,
        'exit_price': 100 + np.random.randn(n_trades) * 2,
        'return_pct': np.random.randn(n_trades) * 0.005,
        'pnl': np.random.randn(n_trades) * 1000,
        'holding_bars': np.random.randint(5, 20, n_trades),
        'exit_reason': ['signal'] * n_trades,
    })
    
    # Mock daily returns
    daily_returns = np.random.randn(20) * 0.01
    
    results = {
        'equity_curve': equity_curve,
        'trades': trades,
        'daily_returns': daily_returns,
        'final_capital': equity[-1],
        'total_return': equity[-1] / 1000000 - 1,
        'num_trades': n_trades,
    }
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    print("\nPerformance Metrics:")
    print("-" * 40)
    for name, value in metrics.to_dict().items():
        print(f"  {name}: {value}")
    
    print("\nMetrics test completed!")
