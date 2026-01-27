"""
Qlib-based Intraday Backtest Engine with Equal-Weight Portfolio Support.

This module provides:
1. Single-instrument bar-by-bar trading (QlibIntradayBacktest)
2. Multi-instrument equal-weight portfolio (EqualWeightBacktest)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, time


@dataclass
class BacktestConfig:
    """Configuration for intraday backtest."""
    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 0.0
    position_close_bar: int = 77  # 15:55
    last_entry_bar: int = 67  # 15:00
    bar_duration_minutes: int = 5
    allocation_method: str = 'best_only'  # 'best_only', 'equal_weight'
    topk: int = 4


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


class QlibIntradayBacktest:
    """
    Single-instrument bar-by-bar backtest (matches old behavior exactly).
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.reset()
    
    def reset(self):
        self.capital = self.config.initial_capital
        self.position: Optional[Dict] = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[float] = []
    
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame], pd.DataFrame],
        instruments: Optional[List[str]] = None,
    ) -> BacktestResults:
        self.reset()
        
        is_multi = isinstance(data.index, pd.MultiIndex)
        
        if instruments is None:
            if is_multi:
                instruments = data.index.get_level_values('instrument').unique().tolist()
            else:
                instruments = ['ETF']
        
        signals = signal_generator(data)
        
        if isinstance(signals.index, pd.MultiIndex):
            signals = self._select_best_instrument_per_day(signals, instruments)
        
        required = ['close', 'bar_index', 'signal']
        for col in required:
            if col not in signals.columns:
                raise ValueError(f"Missing required column: {col}")
        
        signals = signals.sort_index()
        
        prev_date = None
        daily_start_capital = self.capital
        
        for idx, row in signals.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else idx[0]
            current_date = timestamp.date()
            bar_index = int(row['bar_index'])
            price = row['close']
            signal = int(row['signal'])
            instrument = row.get('selected_instrument', instruments[0])
            
            if current_date != prev_date:
                if prev_date is not None and daily_start_capital > 0:
                    daily_return = self.capital / daily_start_capital - 1
                    self.daily_returns.append(daily_return)
                daily_start_capital = max(self.capital, 1e-10)
                prev_date = current_date
            
            if bar_index >= self.config.position_close_bar and self.position is not None:
                self._close_position(timestamp, price, bar_index, 'eod_close')
            elif self.position is not None:
                exit_signal = int(row.get('exit_signal', 0))
                if exit_signal == 1:
                    self._close_position(timestamp, price, bar_index, 'signal_exit')
            elif self.position is None and signal == 1:
                if bar_index <= self.config.last_entry_bar:
                    if instrument != 'CASH':
                        self._open_position(timestamp, instrument, price, bar_index)
            
            equity = self._get_equity(price)
            self.equity_curve.append((timestamp, equity))
        
        if self.position is not None:
            last_row = signals.iloc[-1]
            last_ts = signals.index[-1] if isinstance(signals.index[-1], pd.Timestamp) else signals.index[-1][0]
            self._close_position(last_ts, last_row['close'], int(last_row['bar_index']), 'end_of_data')
        
        if daily_start_capital != self.capital:
            daily_return = self.capital / daily_start_capital - 1
            self.daily_returns.append(daily_return)
        
        return self._calculate_results()
    
    def _select_best_instrument_per_day(self, signals: pd.DataFrame, instruments: List[str]) -> pd.DataFrame:
        df = signals.reset_index()
        datetime_col = signals.index.names[0] or 'datetime'
        instrument_col = signals.index.names[1] or 'instrument'
        df['_date'] = pd.to_datetime(df[datetime_col]).dt.date
        
        score_col = 'predicted_return' if 'predicted_return' in df.columns else 'signal'
        daily_scores = df.groupby(['_date', instrument_col])[score_col].sum().reset_index()
        best_per_day = daily_scores.loc[daily_scores.groupby('_date')[score_col].idxmax()]
        best_map = dict(zip(best_per_day['_date'], best_per_day[instrument_col]))
        
        df['_best'] = df['_date'].map(best_map)
        best_df = df[df[instrument_col] == df['_best']].copy()
        best_df['selected_instrument'] = best_df[instrument_col]
        best_df = best_df.set_index(datetime_col)
        best_df.index = pd.to_datetime(best_df.index)
        best_df = best_df.sort_index()
        best_df = best_df.drop(columns=['_date', '_best', instrument_col], errors='ignore')
        return best_df
    
    def _open_position(self, timestamp, instrument, price, bar_index):
        cost_rate = self.config.transaction_cost_bps / 10000
        position_value = self.capital
        cost = position_value * cost_rate
        net_value = position_value - cost
        shares = net_value / price
        self.position = {
            'instrument': instrument, 'entry_time': timestamp, 'entry_price': price,
            'entry_bar': bar_index, 'shares': shares, 'entry_value': net_value,
        }
        self.capital = 0
    
    def _close_position(self, timestamp, price, bar_index, reason):
        if self.position is None:
            return
        cost_rate = self.config.transaction_cost_bps / 10000
        gross_value = self.position['shares'] * price
        cost = gross_value * cost_rate
        net_value = gross_value - cost
        pnl = net_value - self.position['entry_value']
        return_pct = pnl / self.position['entry_value'] if self.position['entry_value'] > 0 else 0
        holding_bars = bar_index - self.position['entry_bar']
        
        trade = TradeRecord(
            instrument=self.position['instrument'], entry_time=self.position['entry_time'],
            entry_price=self.position['entry_price'], exit_time=timestamp, exit_price=price,
            shares=self.position['shares'], pnl=pnl, return_pct=return_pct,
            holding_bars=holding_bars, exit_reason=reason
        )
        self.trades.append(trade)
        self.capital = net_value
        self.position = None
    
    def _get_equity(self, current_price):
        if self.position is None:
            return self.capital
        return self.position['shares'] * current_price
    
    def _calculate_results(self):
        if self.trades:
            trades_df = pd.DataFrame([{
                'instrument': t.instrument, 'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'entry_price': t.entry_price, 'exit_price': t.exit_price, 'shares': t.shares,
                'pnl': t.pnl, 'return_pct': t.return_pct, 'holding_bars': t.holding_bars,
                'exit_reason': t.exit_reason} for t in self.trades])
        else:
            trades_df = pd.DataFrame()
        
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        else:
            equity_df = pd.DataFrame()
        
        total_return = self.capital / self.config.initial_capital - 1
        n_days = len(self.daily_returns)
        years = n_days / 252 if n_days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        if self.daily_returns and len(self.daily_returns) > 1:
            sharpe_ratio = (np.mean(self.daily_returns) / np.std(self.daily_returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        if len(equity_df) > 0:
            max_drawdown = abs(((equity_df['equity'] - equity_df['equity'].cummax()) / equity_df['equity'].cummax()).min())
        else:
            max_drawdown = 0
        
        win_rate = (trades_df['return_pct'] > 0).mean() if len(trades_df) > 0 else 0
        
        return BacktestResults(
            total_return=total_return, annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio, max_drawdown=max_drawdown, num_trades=len(self.trades),
            win_rate=win_rate, trades=trades_df, equity_curve=equity_df,
            daily_returns=self.daily_returns, config=self.config
        )


class EqualWeightBacktest:
    """
    Multi-instrument equal-weight portfolio backtest.
    
    Allocates capital equally across all instruments with positive signals,
    holding simultaneous positions.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig(allocation_method='equal_weight')
        self.reset()
    
    def reset(self):
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}  # instrument -> position info
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[float] = []
    
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame], pd.DataFrame],
        instruments: List[str],
    ) -> BacktestResults:
        """
        Run equal-weight portfolio backtest.
        
        Each instrument is traded independently based on its own signals.
        Capital is divided equally among instruments at the start.
        """
        self.reset()
        
        # Generate signals for multi-instrument data
        signals = signal_generator(data)
        
        if not isinstance(signals.index, pd.MultiIndex):
            raise ValueError("EqualWeightBacktest requires MultiIndex (datetime, instrument) data")
        
        required = ['close', 'bar_index', 'signal']
        for col in required:
            if col not in signals.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Allocate capital equally
        n_instruments = len(instruments)
        capital_per_instrument = self.config.initial_capital / n_instruments
        instrument_capital = {inst: capital_per_instrument for inst in instruments}
        
        # Get all timestamps
        all_timestamps = signals.index.get_level_values(0).unique().sort_values()
        
        prev_date = None
        daily_start_equity = self.config.initial_capital
        
        for timestamp in all_timestamps:
            current_date = timestamp.date()
            
            # Track daily returns
            if current_date != prev_date:
                if prev_date is not None and daily_start_equity > 0:
                    current_equity = self._get_total_equity(signals, timestamp, instrument_capital)
                    daily_return = current_equity / daily_start_equity - 1
                    self.daily_returns.append(daily_return)
                daily_start_equity = self._get_total_equity(signals, timestamp, instrument_capital)
                prev_date = current_date
            
            # Get bar data for all instruments
            bar_data = signals.loc[timestamp]
            
            # Process each instrument
            for inst in instruments:
                if inst not in bar_data.index:
                    continue
                
                row = bar_data.loc[inst]
                bar_index = int(row['bar_index'])
                price = row['close']
                signal = int(row['signal'])
                exit_signal = int(row.get('exit_signal', 0))
                
                # EOD close
                if bar_index >= self.config.position_close_bar and inst in self.positions:
                    self._close_position(inst, timestamp, price, bar_index, 'eod_close', instrument_capital)
                
                # Check exit for existing position
                elif inst in self.positions:
                    if exit_signal == 1:
                        self._close_position(inst, timestamp, price, bar_index, 'signal_exit', instrument_capital)
                
                # Entry on signal = 1
                elif inst not in self.positions and signal == 1:
                    if bar_index <= self.config.last_entry_bar:
                        self._open_position(inst, timestamp, price, bar_index, instrument_capital)
            
            # Record equity
            equity = self._get_total_equity(signals, timestamp, instrument_capital)
            self.equity_curve.append((timestamp, equity))
        
        # Close any remaining positions
        last_ts = all_timestamps[-1]
        last_data = signals.loc[last_ts]
        for inst in list(self.positions.keys()):
            if inst in last_data.index:
                row = last_data.loc[inst]
                self._close_position(inst, last_ts, row['close'], int(row['bar_index']), 'end_of_data', instrument_capital)
        
        # Final daily return
        if prev_date is not None:
            final_equity = sum(instrument_capital.values())
            if daily_start_equity > 0 and daily_start_equity != final_equity:
                daily_return = final_equity / daily_start_equity - 1
                self.daily_returns.append(daily_return)
        
        # Calculate final capital
        self.capital = sum(instrument_capital.values())
        
        return self._calculate_results()
    
    def _open_position(self, instrument, timestamp, price, bar_index, instrument_capital):
        if instrument_capital[instrument] <= 0:
            return
        
        cost_rate = self.config.transaction_cost_bps / 10000
        position_value = instrument_capital[instrument]
        cost = position_value * cost_rate
        net_value = position_value - cost
        shares = net_value / price
        
        self.positions[instrument] = {
            'entry_time': timestamp, 'entry_price': price, 'entry_bar': bar_index,
            'shares': shares, 'entry_value': net_value,
        }
        instrument_capital[instrument] = 0  # Capital now in position
    
    def _close_position(self, instrument, timestamp, price, bar_index, reason, instrument_capital):
        if instrument not in self.positions:
            return
        
        pos = self.positions[instrument]
        cost_rate = self.config.transaction_cost_bps / 10000
        gross_value = pos['shares'] * price
        cost = gross_value * cost_rate
        net_value = gross_value - cost
        
        pnl = net_value - pos['entry_value']
        return_pct = pnl / pos['entry_value'] if pos['entry_value'] > 0 else 0
        holding_bars = bar_index - pos['entry_bar']
        
        trade = TradeRecord(
            instrument=instrument, entry_time=pos['entry_time'], entry_price=pos['entry_price'],
            exit_time=timestamp, exit_price=price, shares=pos['shares'], pnl=pnl,
            return_pct=return_pct, holding_bars=holding_bars, exit_reason=reason
        )
        self.trades.append(trade)
        
        instrument_capital[instrument] = net_value
        del self.positions[instrument]
    
    def _get_total_equity(self, signals, timestamp, instrument_capital):
        equity = sum(instrument_capital.values())
        
        if timestamp in signals.index.get_level_values(0):
            bar_data = signals.loc[timestamp]
            for inst, pos in self.positions.items():
                if inst in bar_data.index:
                    price = bar_data.loc[inst, 'close']
                    equity += pos['shares'] * price
        
        return equity
    
    def _calculate_results(self):
        if self.trades:
            trades_df = pd.DataFrame([{
                'instrument': t.instrument, 'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'entry_price': t.entry_price, 'exit_price': t.exit_price, 'shares': t.shares,
                'pnl': t.pnl, 'return_pct': t.return_pct, 'holding_bars': t.holding_bars,
                'exit_reason': t.exit_reason} for t in self.trades])
        else:
            trades_df = pd.DataFrame()
        
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        else:
            equity_df = pd.DataFrame()
        
        total_return = self.capital / self.config.initial_capital - 1
        n_days = len(self.daily_returns)
        years = n_days / 252 if n_days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        if self.daily_returns and len(self.daily_returns) > 1:
            sharpe_ratio = (np.mean(self.daily_returns) / np.std(self.daily_returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        if len(equity_df) > 0:
            max_drawdown = abs(((equity_df['equity'] - equity_df['equity'].cummax()) / equity_df['equity'].cummax()).min())
        else:
            max_drawdown = 0
        
        win_rate = (trades_df['return_pct'] > 0).mean() if len(trades_df) > 0 else 0
        
        return BacktestResults(
            total_return=total_return, annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio, max_drawdown=max_drawdown, num_trades=len(self.trades),
            win_rate=win_rate, trades=trades_df, equity_curve=equity_df,
            daily_returns=self.daily_returns, config=self.config
        )


class BestPerBarBacktest:
    """
    Best-per-bar allocation backtest.
    
    At each bar, selects the instrument with the highest predicted_return,
    allowing intraday switching between instruments. No lookahead bias.
    
    Features:
    - Bar-by-bar best instrument selection
    - Intraday switching (can change positions mid-day)
    - Multi-position support (for future proportional allocation)
    - Tie-breaker: reverse alphabetical (SPY > QQQ > IWM > DIA)
    """
    
    # Tie-breaker order: prefer SPY > QQQ > IWM > DIA
    INSTRUMENT_PRIORITY = ['SPY', 'QQQ', 'IWM', 'DIA']
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.reset()
    
    def reset(self):
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}  # instrument -> position info
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[float] = []
    
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame], pd.DataFrame],
        instruments: List[str],
    ) -> BacktestResults:
        """
        Run best-per-bar backtest.
        
        At each bar:
        1. If no position: enter best instrument with positive signal
        2. If holding position: check if should exit (signal exit or better option)
        3. Track equity and daily returns
        """
        self.reset()
        
        # Generate signals
        signals = signal_generator(data)
        
        if not isinstance(signals.index, pd.MultiIndex):
            raise ValueError("BestPerBarBacktest requires MultiIndex (datetime, instrument)")
        
        required = ['close', 'bar_index', 'signal', 'predicted_return']
        for col in required:
            if col not in signals.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Get all timestamps
        all_timestamps = signals.index.get_level_values(0).unique().sort_values()
        
        prev_date = None
        daily_start_equity = self.config.initial_capital
        
        for timestamp in all_timestamps:
            current_date = timestamp.date()
            
            # Daily return tracking
            if current_date != prev_date:
                if prev_date is not None and daily_start_equity > 0:
                    current_equity = self._get_total_equity(signals, timestamp)
                    daily_return = current_equity / daily_start_equity - 1
                    self.daily_returns.append(daily_return)
                daily_start_equity = self._get_total_equity(signals, timestamp)
                prev_date = current_date
            
            # Get bar data for all instruments at this timestamp
            bar_data = signals.loc[timestamp]
            bar_index = int(bar_data['bar_index'].iloc[0])
            
            # Find best instrument at this bar
            best_inst, best_signal = self._find_best_instrument(bar_data, instruments)
            
            # EOD close
            if bar_index >= self.config.position_close_bar:
                for inst in list(self.positions.keys()):
                    if inst in bar_data.index:
                        price = bar_data.loc[inst, 'close']
                        self._close_position(inst, timestamp, price, bar_index, 'eod_close')
            
            # Handle existing positions
            elif self.positions:
                for inst in list(self.positions.keys()):
                    if inst not in bar_data.index:
                        continue
                    
                    row = bar_data.loc[inst]
                    price = row['close']
                    exit_signal = int(row.get('exit_signal', 0))
                    curr_signal = row['predicted_return']
                    
                    # Exit if: explicit exit signal, or better instrument available
                    should_switch = (best_inst is not None and 
                                    best_inst != inst and 
                                    best_signal > curr_signal)
                    
                    if exit_signal == 1:
                        self._close_position(inst, timestamp, price, bar_index, 'signal_exit')
                    elif should_switch:
                        self._close_position(inst, timestamp, price, bar_index, 'switch')
            
            # Entry: if no position and there's a best instrument
            if not self.positions and best_inst is not None:
                if bar_index <= self.config.last_entry_bar:
                    price = bar_data.loc[best_inst, 'close']
                    self._open_position(best_inst, timestamp, price, bar_index)
            
            # Record equity
            equity = self._get_total_equity(signals, timestamp)
            self.equity_curve.append((timestamp, equity))
        
        # Close remaining positions at end
        last_ts = all_timestamps[-1]
        last_data = signals.loc[last_ts]
        for inst in list(self.positions.keys()):
            if inst in last_data.index:
                self._close_position(inst, last_ts, last_data.loc[inst, 'close'], 
                                    int(last_data['bar_index'].iloc[0]), 'end_of_data')
        
        # Final daily return
        if prev_date is not None and daily_start_equity > 0:
            final_equity = self.capital + sum(
                pos['shares'] * last_data.loc[inst, 'close'] 
                for inst, pos in self.positions.items() if inst in last_data.index
            )
            if self.capital != daily_start_equity:
                self.daily_returns.append(self.capital / daily_start_equity - 1)
        
        return self._calculate_results()
    
    def _find_best_instrument(
        self, 
        bar_data: pd.DataFrame, 
        instruments: List[str]
    ) -> Tuple[Optional[str], float]:
        """
        Find instrument with highest predicted_return at this bar.
        
        Returns (best_instrument, best_signal) or (None, 0) if all negative.
        Tie-breaker: reverse alphabetical (SPY > QQQ > IWM > DIA).
        """
        candidates = []
        
        for inst in instruments:
            if inst not in bar_data.index:
                continue
            
            row = bar_data.loc[inst]
            pred_return = row['predicted_return']
            signal = int(row['signal'])
            
            # Only consider if signal == 1 AND predicted_return > 0
            if signal == 1 and pred_return > 0:
                candidates.append((inst, pred_return))
        
        if not candidates:
            return None, 0.0
        
        # Sort by predicted_return descending, then by priority
        def sort_key(x):
            inst, pred = x
            priority = self.INSTRUMENT_PRIORITY.index(inst) if inst in self.INSTRUMENT_PRIORITY else 999
            return (-pred, priority)  # Higher pred first, then by priority
        
        candidates.sort(key=sort_key)
        return candidates[0]
    
    def _open_position(self, instrument: str, timestamp: pd.Timestamp, 
                       price: float, bar_index: int):
        """Open position in instrument."""
        cost_rate = self.config.transaction_cost_bps / 10000
        position_value = self.capital
        cost = position_value * cost_rate
        net_value = position_value - cost
        shares = net_value / price
        
        self.positions[instrument] = {
            'entry_time': timestamp,
            'entry_price': price,
            'entry_bar': bar_index,
            'shares': shares,
            'entry_value': net_value,
        }
        self.capital = 0
    
    def _close_position(self, instrument: str, timestamp: pd.Timestamp,
                       price: float, bar_index: int, reason: str):
        """Close position and record trade."""
        if instrument not in self.positions:
            return
        
        pos = self.positions[instrument]
        cost_rate = self.config.transaction_cost_bps / 10000
        gross_value = pos['shares'] * price
        cost = gross_value * cost_rate
        net_value = gross_value - cost
        
        pnl = net_value - pos['entry_value']
        return_pct = pnl / pos['entry_value'] if pos['entry_value'] > 0 else 0
        holding_bars = bar_index - pos['entry_bar']
        
        trade = TradeRecord(
            instrument=instrument,
            entry_time=pos['entry_time'],
            entry_price=pos['entry_price'],
            exit_time=timestamp,
            exit_price=price,
            shares=pos['shares'],
            pnl=pnl,
            return_pct=return_pct,
            holding_bars=holding_bars,
            exit_reason=reason
        )
        self.trades.append(trade)
        
        self.capital += net_value
        del self.positions[instrument]
    
    def _get_total_equity(self, signals: pd.DataFrame, timestamp: pd.Timestamp) -> float:
        """Calculate total equity (cash + positions)."""
        equity = self.capital
        
        if timestamp in signals.index.get_level_values(0):
            bar_data = signals.loc[timestamp]
            for inst, pos in self.positions.items():
                if inst in bar_data.index:
                    price = bar_data.loc[inst, 'close']
                    equity += pos['shares'] * price
        
        return equity
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate final backtest metrics."""
        if self.trades:
            trades_df = pd.DataFrame([{
                'instrument': t.instrument, 'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'entry_price': t.entry_price, 'exit_price': t.exit_price, 'shares': t.shares,
                'pnl': t.pnl, 'return_pct': t.return_pct, 'holding_bars': t.holding_bars,
                'exit_reason': t.exit_reason} for t in self.trades])
        else:
            trades_df = pd.DataFrame()
        
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        else:
            equity_df = pd.DataFrame()
        
        total_return = self.capital / self.config.initial_capital - 1
        n_days = len(self.daily_returns)
        years = n_days / 252 if n_days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        if self.daily_returns and len(self.daily_returns) > 1:
            sharpe_ratio = (np.mean(self.daily_returns) / np.std(self.daily_returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        if len(equity_df) > 0:
            max_drawdown = abs(((equity_df['equity'] - equity_df['equity'].cummax()) / equity_df['equity'].cummax()).min())
        else:
            max_drawdown = 0
        
        win_rate = (trades_df['return_pct'] > 0).mean() if len(trades_df) > 0 else 0
        
        return BacktestResults(
            total_return=total_return, annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio, max_drawdown=max_drawdown, num_trades=len(self.trades),
            win_rate=win_rate, trades=trades_df, equity_curve=equity_df,
            daily_returns=self.daily_returns, config=self.config
        )


class ReducedSwitchingBacktest:
    """
    Reduced switching strategies to minimize transaction costs.
    
    Modes:
    - 'min_hold': Only switch after holding >= min_hold_bars
    - 'threshold': Only switch if new signal > current * threshold_mult
    - 'daily': One trade per day (pick best at open)
    - 'proportional': Allocate proportionally to signal strength
    """
    
    INSTRUMENT_PRIORITY = ['SPY', 'QQQ', 'IWM', 'DIA']
    
    def __init__(self, config: Optional[BacktestConfig] = None, 
                 mode: str = 'min_hold',
                 min_hold_bars: int = 6,
                 threshold_mult: float = 1.2,
                 rebalance_bars: int = 12):
        self.config = config or BacktestConfig()
        self.mode = mode
        self.min_hold_bars = min_hold_bars
        self.threshold_mult = threshold_mult
        self.rebalance_bars = rebalance_bars  # For proportional mode
        self.reset()
    
    def reset(self):
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[float] = []
        self.bars_since_entry = 0
        self.last_rebalance_bar = -999
    
    def run(self, data: pd.DataFrame, signal_generator: Callable, 
            instruments: List[str]) -> BacktestResults:
        self.reset()
        signals = signal_generator(data)
        
        if not isinstance(signals.index, pd.MultiIndex):
            raise ValueError("Requires MultiIndex (datetime, instrument)")
        
        all_timestamps = signals.index.get_level_values(0).unique().sort_values()
        prev_date = None
        daily_start_equity = self.config.initial_capital
        daily_best = None  # For daily mode
        
        for timestamp in all_timestamps:
            current_date = timestamp.date()
            bar_data = signals.loc[timestamp]
            bar_index = int(bar_data['bar_index'].iloc[0])
            
            # Daily return tracking
            if current_date != prev_date:
                if prev_date is not None and daily_start_equity > 0:
                    eq = self._get_equity(signals, timestamp)
                    self.daily_returns.append(eq / daily_start_equity - 1)
                daily_start_equity = self._get_equity(signals, timestamp)
                daily_best = None  # Reset for daily mode
                prev_date = current_date
            
            # EOD close
            if bar_index >= self.config.position_close_bar:
                for inst in list(self.positions.keys()):
                    if inst in bar_data.index:
                        self._close(inst, timestamp, bar_data.loc[inst, 'close'], bar_index, 'eod')
                self.bars_since_entry = 0
                self.equity_curve.append((timestamp, self._get_equity(signals, timestamp)))
                continue
            
            # Mode-specific logic
            if self.mode == 'daily':
                self._run_daily(bar_data, bar_index, timestamp, instruments, daily_best)
                if bar_index == 0 or (bar_index == 1 and daily_best is None):
                    daily_best = self._find_best(bar_data, instruments)
            elif self.mode == 'min_hold':
                self._run_min_hold(bar_data, bar_index, timestamp, instruments)
            elif self.mode == 'threshold':
                self._run_threshold(bar_data, bar_index, timestamp, instruments)
            elif self.mode == 'proportional':
                self._run_proportional(bar_data, bar_index, timestamp, instruments)
            
            self.equity_curve.append((timestamp, self._get_equity(signals, timestamp)))
            if self.positions:
                self.bars_since_entry += 1
        
        # Close remaining
        last_ts = all_timestamps[-1]
        last_data = signals.loc[last_ts]
        for inst in list(self.positions.keys()):
            if inst in last_data.index:
                self._close(inst, last_ts, last_data.loc[inst, 'close'], 
                           int(last_data['bar_index'].iloc[0]), 'end')
        
        if prev_date and daily_start_equity > 0:
            self.daily_returns.append(self.capital / daily_start_equity - 1)
        
        return self._results()
    
    def _run_daily(self, bar_data, bar_index, timestamp, instruments, daily_best):
        """One trade per day - pick best at open."""
        if bar_index > self.config.last_entry_bar:
            return
        
        best_inst, best_sig = self._find_best(bar_data, instruments)
        
        if not self.positions and best_inst and bar_index <= 5:
            # Enter at/near market open
            self._open(best_inst, timestamp, bar_data.loc[best_inst, 'close'], bar_index)
    
    def _run_min_hold(self, bar_data, bar_index, timestamp, instruments):
        """Only switch after minimum holding period."""
        if bar_index > self.config.last_entry_bar:
            return
        
        best_inst, best_sig = self._find_best(bar_data, instruments)
        
        if not self.positions:
            if best_inst:
                self._open(best_inst, timestamp, bar_data.loc[best_inst, 'close'], bar_index)
                self.bars_since_entry = 0
        else:
            curr_inst = list(self.positions.keys())[0]
            if curr_inst in bar_data.index:
                curr_sig = bar_data.loc[curr_inst, 'predicted_return']
                # Only switch if held long enough AND better option
                if (self.bars_since_entry >= self.min_hold_bars and 
                    best_inst and best_inst != curr_inst and best_sig > curr_sig):
                    self._close(curr_inst, timestamp, bar_data.loc[curr_inst, 'close'], bar_index, 'switch')
                    self._open(best_inst, timestamp, bar_data.loc[best_inst, 'close'], bar_index)
                    self.bars_since_entry = 0
    
    def _run_threshold(self, bar_data, bar_index, timestamp, instruments):
        """Only switch if significantly better."""
        if bar_index > self.config.last_entry_bar:
            return
        
        best_inst, best_sig = self._find_best(bar_data, instruments)
        
        if not self.positions:
            if best_inst:
                self._open(best_inst, timestamp, bar_data.loc[best_inst, 'close'], bar_index)
        else:
            curr_inst = list(self.positions.keys())[0]
            if curr_inst in bar_data.index:
                curr_sig = bar_data.loc[curr_inst, 'predicted_return']
                # Require 20%+ improvement
                if best_inst and best_inst != curr_inst and best_sig > curr_sig * self.threshold_mult:
                    self._close(curr_inst, timestamp, bar_data.loc[curr_inst, 'close'], bar_index, 'switch')
                    self._open(best_inst, timestamp, bar_data.loc[best_inst, 'close'], bar_index)
    
    def _run_proportional(self, bar_data, bar_index, timestamp, instruments):
        """Allocate proportionally to signal strength, rebalance periodically."""
        if bar_index > self.config.last_entry_bar:
            return
        
        # Only rebalance every N bars
        if bar_index - self.last_rebalance_bar < self.rebalance_bars and self.positions:
            return
        
        # Get signals and weights
        weights = {}
        total_sig = 0
        for inst in instruments:
            if inst not in bar_data.index:
                continue
            sig = bar_data.loc[inst, 'predicted_return']
            if sig > 0 and int(bar_data.loc[inst, 'signal']) == 1:
                weights[inst] = sig
                total_sig += sig
        
        if not weights:
            return
        
        # Normalize weights
        for inst in weights:
            weights[inst] /= total_sig
        
        # Close positions not in new allocation
        for inst in list(self.positions.keys()):
            if inst not in weights and inst in bar_data.index:
                self._close(inst, timestamp, bar_data.loc[inst, 'close'], bar_index, 'rebal')
        
        # Calculate current equity
        equity = self._get_equity_direct(bar_data)
        
        # Open/adjust positions
        for inst, w in weights.items():
            target_value = equity * w
            if inst in bar_data.index:
                price = bar_data.loc[inst, 'close']
                if inst not in self.positions:
                    # Open new position
                    if self.capital > 0:
                        alloc = min(target_value, self.capital)
                        cost = alloc * self.config.transaction_cost_bps / 10000
                        shares = (alloc - cost) / price
                        self.positions[inst] = {
                            'entry_time': timestamp, 'entry_price': price,
                            'entry_bar': bar_index, 'shares': shares,
                            'entry_value': alloc - cost
                        }
                        self.capital -= alloc
        
        self.last_rebalance_bar = bar_index
    
    def _find_best(self, bar_data, instruments) -> Tuple[Optional[str], float]:
        candidates = []
        for inst in instruments:
            if inst not in bar_data.index:
                continue
            sig = bar_data.loc[inst, 'predicted_return']
            if int(bar_data.loc[inst, 'signal']) == 1 and sig > 0:
                pri = self.INSTRUMENT_PRIORITY.index(inst) if inst in self.INSTRUMENT_PRIORITY else 999
                candidates.append((inst, sig, pri))
        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda x: (-x[1], x[2]))
        return candidates[0][0], candidates[0][1]
    
    def _open(self, inst, timestamp, price, bar_index):
        cost = self.capital * self.config.transaction_cost_bps / 10000
        shares = (self.capital - cost) / price
        self.positions[inst] = {
            'entry_time': timestamp, 'entry_price': price,
            'entry_bar': bar_index, 'shares': shares,
            'entry_value': self.capital - cost
        }
        self.capital = 0
    
    def _close(self, inst, timestamp, price, bar_index, reason):
        if inst not in self.positions:
            return
        pos = self.positions[inst]
        gross = pos['shares'] * price
        cost = gross * self.config.transaction_cost_bps / 10000
        net = gross - cost
        pnl = net - pos['entry_value']
        ret = pnl / pos['entry_value'] if pos['entry_value'] > 0 else 0
        
        self.trades.append(TradeRecord(
            instrument=inst, entry_time=pos['entry_time'], entry_price=pos['entry_price'],
            exit_time=timestamp, exit_price=price, shares=pos['shares'],
            pnl=pnl, return_pct=ret, holding_bars=bar_index - pos['entry_bar'],
            exit_reason=reason
        ))
        self.capital += net
        del self.positions[inst]
    
    def _get_equity(self, signals, timestamp):
        eq = self.capital
        if timestamp in signals.index.get_level_values(0):
            bar_data = signals.loc[timestamp]
            for inst, pos in self.positions.items():
                if inst in bar_data.index:
                    eq += pos['shares'] * bar_data.loc[inst, 'close']
        return eq
    
    def _get_equity_direct(self, bar_data):
        eq = self.capital
        for inst, pos in self.positions.items():
            if inst in bar_data.index:
                eq += pos['shares'] * bar_data.loc[inst, 'close']
        return eq
    
    def _results(self):
        if self.trades:
            df = pd.DataFrame([{
                'instrument': t.instrument, 'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'entry_price': t.entry_price, 'exit_price': t.exit_price, 'shares': t.shares,
                'pnl': t.pnl, 'return_pct': t.return_pct, 'holding_bars': t.holding_bars,
                'exit_reason': t.exit_reason} for t in self.trades])
        else:
            df = pd.DataFrame()
        
        eq_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp') if self.equity_curve else pd.DataFrame()
        
        tot_ret = self.capital / self.config.initial_capital - 1
        n = len(self.daily_returns)
        yrs = n / 252 if n > 0 else 1
        ann = (1 + tot_ret) ** (1 / yrs) - 1 if yrs > 0 else 0
        sr = (np.mean(self.daily_returns) / np.std(self.daily_returns)) * np.sqrt(252) if len(self.daily_returns) > 1 else 0
        dd = abs(((eq_df['equity'] - eq_df['equity'].cummax()) / eq_df['equity'].cummax()).min()) if len(eq_df) > 0 else 0
        wr = (df['return_pct'] > 0).mean() if len(df) > 0 else 0
        
        return BacktestResults(
            total_return=tot_ret, annualized_return=ann, sharpe_ratio=sr,
            max_drawdown=dd, num_trades=len(self.trades), win_rate=wr,
            trades=df, equity_curve=eq_df, daily_returns=self.daily_returns,
            config=self.config
        )


class OncePerDayBacktest:
    """
    Once-per-day trading strategy.
    
    - Enter at specified bar (entry_bar) each day
    - Pick best instrument based on signal at that bar
    - Hold until EOD close
    - No intraday switching
    """
    
    INSTRUMENT_PRIORITY = ['SPY', 'QQQ', 'IWM', 'DIA']
    
    def __init__(self, config: Optional[BacktestConfig] = None, 
                 entry_bar: int = 0, exit_bars: int = 24):
        """
        Args:
            config: Backtest configuration
            entry_bar: Bar index to enter (0=open, 6=10am, 12=10:30, etc.)
            exit_bars: Max bars to hold (exit after N bars or EOD)
        """
        self.config = config or BacktestConfig()
        self.entry_bar = entry_bar
        self.exit_bars = exit_bars
        self.reset()
    
    def reset(self):
        self.capital = self.config.initial_capital
        self.position: Optional[Dict] = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[float] = []
    
    def run(self, data: pd.DataFrame, signal_generator: Callable, 
            instruments: List[str]) -> BacktestResults:
        self.reset()
        signals = signal_generator(data)
        
        if not isinstance(signals.index, pd.MultiIndex):
            raise ValueError("Requires MultiIndex (datetime, instrument)")
        
        all_timestamps = signals.index.get_level_values(0).unique().sort_values()
        prev_date = None
        daily_start_equity = self.config.initial_capital
        entered_today = False
        
        for timestamp in all_timestamps:
            current_date = timestamp.date()
            bar_data = signals.loc[timestamp]
            bar_index = int(bar_data['bar_index'].iloc[0])
            
            # New day reset
            if current_date != prev_date:
                if prev_date is not None and daily_start_equity > 0:
                    eq = self._get_equity(bar_data)
                    self.daily_returns.append(eq / daily_start_equity - 1)
                daily_start_equity = self._get_equity(bar_data)
                entered_today = False
                prev_date = current_date
            
            # EOD close
            if bar_index >= self.config.position_close_bar:
                if self.position:
                    inst = self.position['instrument']
                    if inst in bar_data.index:
                        self._close(inst, timestamp, bar_data.loc[inst, 'close'], bar_index, 'eod')
                self.equity_curve.append((timestamp, self._get_equity(bar_data)))
                continue

            # Max holding period exit
            if self.position:
                bars_held = bar_index - self.position['entry_bar']
                if bars_held >= self.exit_bars:
                    inst = self.position['instrument']
                    if inst in bar_data.index:
                        self._close(inst, timestamp, bar_data.loc[inst, 'close'], bar_index, 'max_hold')
                    # Don't re-enter today
                    self.equity_curve.append((timestamp, self._get_equity(bar_data)))
                    continue
            
            # Entry at specified bar (or first available if entry_bar=-1)
            should_try_entry = not entered_today and (
                self.entry_bar == -1 or bar_index == self.entry_bar
            ) and bar_index <= self.config.last_entry_bar
            
            if should_try_entry:
                best_inst, best_sig = self._find_best(bar_data, instruments)
                if best_inst:
                    self._open(best_inst, timestamp, bar_data.loc[best_inst, 'close'], bar_index)
                    entered_today = True
            
            self.equity_curve.append((timestamp, self._get_equity(bar_data)))
        
        # Close remaining position
        if self.position:
            last_ts = all_timestamps[-1]
            last_data = signals.loc[last_ts]
            inst = self.position['instrument']
            if inst in last_data.index:
                self._close(inst, last_ts, last_data.loc[inst, 'close'], 
                           int(last_data['bar_index'].iloc[0]), 'end')
        
        if prev_date and daily_start_equity > 0:
            self.daily_returns.append(self.capital / daily_start_equity - 1)
        
        return self._results()
    
    def _find_best(self, bar_data, instruments) -> Tuple[Optional[str], float]:
        """Find best instrument based on signal."""
        candidates = []
        for inst in instruments:
            if inst not in bar_data.index:
                continue
            sig = bar_data.loc[inst, 'predicted_return']
            signal_val = int(bar_data.loc[inst, 'signal'])
            if signal_val == 1 and sig > 0:
                pri = self.INSTRUMENT_PRIORITY.index(inst) if inst in self.INSTRUMENT_PRIORITY else 999
                candidates.append((inst, sig, pri))
        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda x: (-x[1], x[2]))
        return candidates[0][0], candidates[0][1]
    
    def _open(self, inst, timestamp, price, bar_index):
        cost = self.capital * self.config.transaction_cost_bps / 10000
        shares = (self.capital - cost) / price
        self.position = {
            'instrument': inst,
            'entry_time': timestamp, 
            'entry_price': price,
            'entry_bar': bar_index, 
            'shares': shares,
            'entry_value': self.capital - cost
        }
        self.capital = 0
    
    def _close(self, inst, timestamp, price, bar_index, reason):
        if not self.position or self.position['instrument'] != inst:
            return
        pos = self.position
        gross = pos['shares'] * price
        cost = gross * self.config.transaction_cost_bps / 10000
        net = gross - cost
        pnl = net - pos['entry_value']
        ret = pnl / pos['entry_value'] if pos['entry_value'] > 0 else 0
        
        self.trades.append(TradeRecord(
            instrument=inst, entry_time=pos['entry_time'], entry_price=pos['entry_price'],
            exit_time=timestamp, exit_price=price, shares=pos['shares'],
            pnl=pnl, return_pct=ret, holding_bars=bar_index - pos['entry_bar'],
            exit_reason=reason, entry_bar=pos['entry_bar']
        ))
        self.capital += net
        self.position = None
    
    def _get_equity(self, bar_data):
        eq = self.capital
        if self.position:
            inst = self.position['instrument']
            price = 0
            if inst in bar_data.index:
                price = bar_data.loc[inst, 'close']
                self.position['last_price'] = price # Update last known price
            elif 'last_price' in self.position:
                price = self.position['last_price']
            else:
                price = self.position['entry_price']
            
            eq += self.position['shares'] * price
        return eq
    
    def _results(self):
        if self.trades:
            df = pd.DataFrame([{
                'instrument': t.instrument, 'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'entry_price': t.entry_price, 'exit_price': t.exit_price, 'shares': t.shares,
                'pnl': t.pnl, 'return_pct': t.return_pct, 'holding_bars': t.holding_bars,
                'exit_reason': t.exit_reason, 'entry_bar': t.entry_bar} for t in self.trades])
        else:
            df = pd.DataFrame()
        
        eq_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp') if self.equity_curve else pd.DataFrame()
        
        # Use final equity from curve (not self.capital which may be 0 if position open)
        final_equity = eq_df['equity'].iloc[-1] if len(eq_df) > 0 else self.capital
        tot_ret = final_equity / self.config.initial_capital - 1
        n = len(self.daily_returns)
        yrs = n / 252 if n > 0 else 1
        ann = (1 + tot_ret) ** (1 / yrs) - 1 if yrs > 0 else 0
        sr = (np.mean(self.daily_returns) / np.std(self.daily_returns)) * np.sqrt(252) if len(self.daily_returns) > 1 and np.std(self.daily_returns) > 0 else 0
        dd = abs(((eq_df['equity'] - eq_df['equity'].cummax()) / eq_df['equity'].cummax()).min()) if len(eq_df) > 0 else 0
        wr = (df['return_pct'] > 0).mean() if len(df) > 0 else 0
        
        return BacktestResults(
            total_return=tot_ret, annualized_return=ann, sharpe_ratio=sr,
            max_drawdown=dd, num_trades=len(self.trades), win_rate=wr,
            trades=df, equity_curve=eq_df, daily_returns=self.daily_returns,
            config=self.config
        )


if __name__ == "__main__":
    print("Backtest modules loaded:")
    print("  - QlibIntradayBacktest: Single-instrument bar-by-bar trading")
    print("  - EqualWeightBacktest: Multi-instrument equal-weight portfolio")
    print("  - BestPerBarBacktest: Best instrument per bar with switching")
    print("  - ReducedSwitchingBacktest: Reduced switching strategies")
    print("  - OncePerDayBacktest: Once-per-day entry with EOD exit")
