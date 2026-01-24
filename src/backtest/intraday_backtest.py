"""
Intraday Backtest Engine.

Simulates intraday trading with:
- Bar-by-bar execution
- Forced position close at 15:55
- Trade tracking and PnL calculation
- Support for CASH positions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, date, time
from pathlib import Path
import yaml


@dataclass
class Trade:
    """Represents a completed round-trip trade."""
    instrument: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    entry_bar: int
    exit_bar: int
    shares: float
    pnl: float
    return_pct: float
    holding_bars: int
    exit_reason: str  # 'signal', 'stop', 'target', 'eod_close'

if False: # Type checking only
    from .exit_rules import ExitRule


@dataclass
class Position:
    """Represents an open position."""
    instrument: str
    entry_time: pd.Timestamp
    entry_price: float
    entry_bar: int
    shares: float
    
    def close(self, exit_time: pd.Timestamp, exit_price: float, 
              exit_bar: int, exit_reason: str) -> Trade:
        """Close position and return completed trade."""
        pnl = (exit_price - self.entry_price) * self.shares
        return_pct = exit_price / self.entry_price - 1
        holding_bars = exit_bar - self.entry_bar
        
        return Trade(
            instrument=self.instrument,
            entry_time=self.entry_time,
            exit_time=exit_time,
            entry_price=self.entry_price,
            exit_price=exit_price,
            entry_bar=self.entry_bar,
            exit_bar=exit_bar,
            shares=self.shares,
            pnl=pnl,
            return_pct=return_pct,
            holding_bars=holding_bars,
            exit_reason=exit_reason,
        )


class IntradayBacktest:
    """
    Intraday backtesting engine.
    
    Features:
    - Bar-by-bar simulation
    - Forced EOD close at 15:55
    - Cash is an option (no position)
    - Tracks all trades and returns
    """
    
    def __init__(self, config_path: Optional[str] = None, exit_rules: Optional[List['ExitRule']] = None):
        """Initialize backtest engine."""
        self.exit_rules = exit_rules or []
        
        if config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.initial_capital = config['backtest']['initial_capital']
            self.transaction_cost_bps = config['backtest']['transaction_cost_bps']
            self.position_close_bar = 77  # 15:55
            self.last_entry_bar = 67  # 15:00
        else:
            self.initial_capital = 1000000
            self.transaction_cost_bps = 0
            self.position_close_bar = 77
            self.last_entry_bar = 67
        
        self.reset()
    
    def reset(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[float] = []
    
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> Dict:
        """
        Run backtest.
        
        Args:
            data: DataFrame with OHLCV and features, indexed by datetime
            signal_generator: Function that takes features and returns signals
            
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        # Generate signals for all data
        signals = signal_generator(data)
        
        # Ensure we have required columns
        required = ['close', 'bar_index', 'signal']
        for col in required:
            if col not in signals.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add date column for grouping
        signals['_date'] = signals.index.date if isinstance(signals.index, pd.DatetimeIndex) else signals.index.get_level_values(0).date
        
        # Process bar by bar
        prev_date = None
        daily_start_capital = self.capital
        
        for idx, row in signals.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else idx[0]
            current_date = row['_date']
            bar_index = int(row['bar_index'])
            price = row['close']
            signal = row['signal']
            selected_instrument = row.get('selected_instrument', 'ETF')
            
            # Track new day
            if current_date != prev_date:
                if prev_date is not None:
                    daily_return = self.capital / daily_start_capital - 1
                    self.daily_returns.append(daily_return)
                daily_start_capital = self.capital
                prev_date = current_date
            
            # Check for forced EOD close
            if bar_index >= self.position_close_bar and self.position is not None:
                # print(f"DEBUG: Forced close trigger at {timestamp} bar {bar_index}") 
                self._close_position(timestamp, price, bar_index, 'eod_close')
            
            # Handle existing position
            elif self.position is not None:
                # 1. Check custom exit rules first
                rule_exit_triggered = False
                for rule in self.exit_rules:
                    exit_signal = rule.check(self.position, bar_index, price, timestamp)
                    if exit_signal.should_exit:
                        self._close_position(timestamp, price, bar_index, exit_signal.reason)
                        rule_exit_triggered = True
                        break
                
                if rule_exit_triggered:
                    pass # Already closed
                else:
                    # 2. Check for signal exit (legacy/strategy provided)
                    exit_signal = row.get('exit_signal', 0)
                    if exit_signal == 1:
                        # print(f"DEBUG: Signal exit at {timestamp}")
                        self._close_position(timestamp, price, bar_index, 'signal')
            
            # Handle entry
            elif self.position is None and signal == 1:
                # Only enter if not too late
                if bar_index <= self.last_entry_bar:
                    if selected_instrument != 'CASH':
                        # print(f"DEBUG: Opening position at {timestamp} bar {bar_index} price {price} sym {selected_instrument if 'selected_instrument' in row else 'ETF'}")
                        self._open_position(
                            timestamp, selected_instrument, price, bar_index
                        )
            
            # Record equity
            equity = self._get_equity(price)
            self.equity_curve.append((timestamp, equity))
        
        # Close any remaining position at end
        if self.position is not None:
            last_row = signals.iloc[-1]
            self._close_position(
                signals.index[-1] if isinstance(signals.index[-1], pd.Timestamp) else signals.index[-1][0],
                last_row['close'],
                int(last_row['bar_index']),
                'end_of_data'
            )
        
        # Final daily return
        if daily_start_capital != self.capital:
            daily_return = self.capital / daily_start_capital - 1
            self.daily_returns.append(daily_return)
        
        return self._compile_results()
    
    def _open_position(
        self,
        timestamp: pd.Timestamp,
        instrument: str,
        price: float,
        bar_index: int,
    ):
        """Open a new position."""
        # Calculate position size (all-in)
        cost_multiplier = 1 + self.transaction_cost_bps / 10000
        shares = self.capital / (price * cost_multiplier)
        
        self.position = Position(
            instrument=instrument,
            entry_time=timestamp,
            entry_price=price,
            entry_bar=bar_index,
            shares=shares,
        )
        
        # Deduct transaction cost
        self.capital -= self.capital * self.transaction_cost_bps / 10000
    
    def _close_position(
        self,
        timestamp: pd.Timestamp,
        price: float,
        bar_index: int,
        reason: str,
    ):
        """Close current position."""
        if self.position is None:
            return
        
        trade = self.position.close(timestamp, price, bar_index, reason)
        self.trades.append(trade)
        
        # Update capital
        self.capital = self.position.shares * price
        
        # Deduct transaction cost
        self.capital -= self.capital * self.transaction_cost_bps / 10000
        
        self.position = None
    
    def _get_equity(self, current_price: float) -> float:
        """Get current equity value."""
        if self.position is None:
            return self.capital
        else:
            return self.position.shares * current_price
    
    def _compile_results(self) -> Dict:
        """Compile backtest results."""
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df = equity_df.set_index('timestamp')
        
        trades_df = pd.DataFrame([
            {
                'instrument': t.instrument,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'return_pct': t.return_pct,
                'pnl': t.pnl,
                'holding_bars': t.holding_bars,
                'exit_reason': t.exit_reason,
            }
            for t in self.trades
        ]) if self.trades else pd.DataFrame()
        
        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'daily_returns': np.array(self.daily_returns),
            'final_capital': self.capital,
            'total_return': self.capital / self.initial_capital - 1,
            'num_trades': len(self.trades),
        }


class MultiInstrumentBacktest(IntradayBacktest):
    """
    Backtest for multiple instruments with selection.
    
    At each bar, selects the best instrument or CASH.
    """
    
    def run_multi(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signal_generator: Callable[[Dict[str, pd.DataFrame]], pd.DataFrame],
    ) -> Dict:
        """
        Run backtest with multiple instruments.
        
        Args:
            data_dict: Dict of instrument -> DataFrame
            signal_generator: Function that processes all instruments
            
        Returns:
            Backtest results
        """
        self.reset()
        
        # Generate signals across all instruments
        signals = signal_generator(data_dict)
        
        # Merge price data for execution
        # Assume signal_generator returns a DataFrame with 'selected_instrument' column
        
        for idx, row in signals.iterrows():
            timestamp = idx
            bar_index = int(row['bar_index'])
            selected = row['selected_instrument']
            signal = row['signal']
            
            # Get price for selected instrument
            if selected in data_dict and selected != 'CASH':
                price = data_dict[selected].loc[timestamp, 'close']
            else:
                price = None
            
            # EOD forced close
            if bar_index >= self.position_close_bar and self.position is not None:
                pos_price = data_dict[self.position.instrument].loc[timestamp, 'close']
                self._close_position(timestamp, pos_price, bar_index, 'eod_close')
            
            # Position management
            elif self.position is not None:
                current_instrument = self.position.instrument
                
                # Exit if switching instruments or exit signal
                if selected != current_instrument or row.get('exit_signal', 0) == 1:
                    pos_price = data_dict[current_instrument].loc[timestamp, 'close']
                    self._close_position(timestamp, pos_price, bar_index, 'switch')
            
            # Entry
            if self.position is None and signal == 1 and bar_index <= self.last_entry_bar:
                if selected != 'CASH' and price is not None:
                    self._open_position(timestamp, selected, price, bar_index)
            
            # Track equity
            if self.position is not None:
                pos_price = data_dict[self.position.instrument].loc[timestamp, 'close']
                equity = self._get_equity(pos_price)
            else:
                equity = self.capital
            self.equity_curve.append((timestamp, equity))
        
        return self._compile_results()


if __name__ == "__main__":
    print("Testing IntradayBacktest...")
    
    np.random.seed(42)
    
    # Create sample data for multiple days
    dates = pd.date_range('2023-12-18 09:35', periods=78*5, freq='5min')  # 5 days
    
    # Filter to market hours (skip overnight)
    dates = dates[dates.hour < 16]
    dates = dates[(dates.hour > 9) | ((dates.hour == 9) & (dates.minute >= 35))]
    
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    
    sample_df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(len(dates)) * 0.1),
        'low': prices - np.abs(np.random.randn(len(dates)) * 0.1),
        'close': prices,
        'bar_index': [(i % 78) + 1 for i in range(len(dates))],
    }, index=dates)
    
    # Add simple signals
    sample_df['signal'] = (np.random.random(len(dates)) > 0.95).astype(int)
    sample_df['exit_signal'] = (np.random.random(len(dates)) > 0.98).astype(int)
    sample_df['selected_instrument'] = 'SPY'
    
    print(f"\nSample data: {len(sample_df)} bars")
    
    # Run backtest
    bt = IntradayBacktest()
    
    def simple_signal_gen(df):
        return df
    
    results = bt.run(sample_df, simple_signal_gen)
    
    print(f"\nBacktest results:")
    print(f"  Final capital: ${results['final_capital']:,.2f}")
    print(f"  Total return: {results['total_return']*100:.2f}%")
    print(f"  Number of trades: {results['num_trades']}")
    
    if len(results['trades']) > 0:
        print(f"\n  Sample trades:")
        print(results['trades'].head())
    
    print("\nBacktest test completed!")
