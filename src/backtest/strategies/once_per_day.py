from typing import List, Optional, Tuple, Dict, Callable
import pandas as pd
from ..engine import IntradayBacktestEngine, BacktestConfig, TradeRecord, BacktestResults

class OncePerDayStrategy(IntradayBacktestEngine):
    """
    Once-per-day trading strategy.
    Targeting specific entry time and max holding period.
    """
    
    INSTRUMENT_PRIORITY = ['SPY', 'QQQ', 'IWM', 'DIA']
    
    def __init__(self, config: Optional[BacktestConfig] = None, 
                 entry_bar: int = 6, exit_bars: int = 36):
        """
        Args:
            config: Backtest configuration
            entry_bar: Bar index to enter (0=9:30, 6=10:00, etc.)
            exit_bars: Max bars to hold (exit after N bars or EOD)
        """
        super().__init__(config or BacktestConfig())
        self.entry_bar = entry_bar
        self.exit_bars = exit_bars
        self.reset()
    
    def reset(self):
        self.capital = self.config.initial_capital
        self.position: Optional[Dict] = None
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def run(self, data: pd.DataFrame, signal_generator: Callable, 
            instruments: List[str]) -> BacktestResults:
        self.reset()
        signals = signal_generator(data)
        
        if not isinstance(signals.index, pd.MultiIndex):
            raise ValueError("Requires MultiIndex (datetime, instrument)")
        
        prev_date = None
        daily_start_equity = self.config.initial_capital
        entered_today = False
        last_timestamp = None
        
        # Optimization: groupby is faster than repeated .loc access for large indices
        for timestamp, chunk in signals.groupby(level=0):
            current_date = timestamp.date()
            last_timestamp = timestamp
            bar_data = chunk.droplevel(0)
            
            # Legacy check (should be safe with groupby)
            # if timestamp not in signals.index: continue
            
            # Ensure bar_index is available (take first)
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
                    self.equity_curve.append((timestamp, self._get_equity(bar_data)))
                    continue
            
            # Entry at specified bar
            should_try_entry = not entered_today and (
                self.entry_bar == -1 or bar_index == self.entry_bar
            ) and bar_index <= self.config.last_entry_bar
            
            if should_try_entry:
                best_inst, best_sig = self._find_best(bar_data, instruments)
                if best_inst:
                    # Fix: If we are already holding a position (e.g. from previous day due to early close),
                    # we must close it before opening a new one to correctly realize PnL.
                    if self.position:
                        p_inst = self.position['instrument']
                        p_price = self.position.get('last_price', self.position['entry_price'])
                        if p_inst in bar_data.index:
                            p_price = bar_data.loc[p_inst, 'close']
                        self._close(p_inst, timestamp, p_price, bar_index, 'force_swap')
                        
                    self._open(best_inst, timestamp, bar_data.loc[best_inst, 'close'], bar_index)
                    entered_today = True
            
            self.equity_curve.append((timestamp, self._get_equity(bar_data)))
        
        # Close remaining
        if self.position and last_timestamp is not None:
            last_ts = last_timestamp
            try:
                last_data = signals.loc[last_ts]
            except KeyError:
                 # Fallback if loc fails (shouldn't if valid ts)
                 # Reconstruct from groupby? No, simpler to just skip or log
                 return self.calculate_results(self.capital, self.daily_returns, self.trades, self.equity_curve)

            inst = self.position['instrument']
            if inst in last_data.index:
                self._close(inst, last_ts, last_data.loc[inst, 'close'], 
                           int(last_data['bar_index'].iloc[0]), 'end')
        
        if prev_date and daily_start_equity > 0:
            self.daily_returns.append(self.capital / daily_start_equity - 1)
        
        return self.calculate_results(self.capital, self.daily_returns, self.trades, self.equity_curve)
    
    def _find_best(self, bar_data, instruments) -> Tuple[Optional[str], float]:
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
        # Correct PnL Base: Should include entry cost to reflect true round-trip friction
        # entry_value was (Capital - EntryCost).
        # We want to measure against (EntryValue + EntryCost) which is the allocated capital.
        # But we don't strictly have entry_cost stored. 
        # However, entry_value / (1 - rate) approx equals allocated capital if we assume linear.
        # Better: Store 'allocated_capital' in position?
        # Or just recalculate entry cost?
        
        # Let's derive it:
        # entry_value = allocated - allocated * rate
        # allocated = entry_value / (1 - bps/10000)
        
        rate = self.config.transaction_cost_bps / 10000
        allocated_capital = pos['entry_value'] / (1 - rate) if rate < 1 else pos['entry_value']
        
        # True PnL = Net Exit Amount - Allocated Capital
        pnl = net - allocated_capital
        ret = pnl / allocated_capital if allocated_capital > 0 else 0
        
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
                self.position['last_price'] = price
            elif 'last_price' in self.position:
                price = self.position['last_price']
            else:
                price = self.position['entry_price']
            
            eq += self.position['shares'] * price
        return eq
