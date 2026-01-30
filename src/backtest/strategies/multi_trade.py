from typing import List, Optional, Tuple, Dict, Callable
import pandas as pd
import numpy as np
from ..engine import IntradayBacktestEngine, BacktestConfig, TradeRecord, BacktestResults

class MultiTradeStrategy(IntradayBacktestEngine):
    """
    Multi-Trade Intraday Strategy.
    Allows up to N simultaneous positions, each with independent exit logic.
    """
    
    INSTRUMENT_PRIORITY = ['SPY', 'QQQ', 'IWM', 'DIA']
    
    # Allowed entry times (approximate bar indices for 5min bars starting 9:30)
    # 0=9:30, 2=9:40, 6=10:00, 12=10:30, 30=12:00, 54=14:00
    # Excluded: 42=13:00, 66=15:00
    # Excluded: 42=13:00, 66=15:00
    DEFAULT_ENTRY_BARS = [2, 6, 12, 30, 54]
    
    def __init__(self, config: Optional[BacktestConfig] = None, 
                 max_positions: int = 5, exit_bars: int = 36,
                 allowed_entry_bars: Optional[List[int]] = None,
                 fixed_pos_pct: Optional[float] = None):
        """
        Args:
            config: Backtest configuration
            max_positions: Maximum number of simultaneous trades
            exit_bars: Holding period for each trade
            allowed_entry_bars: List of bar indices allowed for entry
        """
        super().__init__(config or BacktestConfig())
        self.max_positions = max_positions
        self.exit_bars = exit_bars
        self.allowed_entry_bars = allowed_entry_bars if allowed_entry_bars is not None else self.DEFAULT_ENTRY_BARS
        self.fixed_pos_pct = fixed_pos_pct
        self.reset()
    
    def reset(self):
        self.cash = self.config.initial_capital
        self.positions: List[Dict] = []
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.daily_exposures = [] # Average daily exposure
        self._current_day_exposures = [] # Temp list for intraday
        self.prev_date = None
        self.daily_start_equity = self.config.initial_capital
    
    def run(self, data: pd.DataFrame, signal_generator: Callable, 
            instruments: List[str]) -> BacktestResults:
        self.reset()
        signals = signal_generator(data)
        
        if not isinstance(signals.index, pd.MultiIndex):
            raise ValueError("Requires MultiIndex (datetime, instrument)")
        
        # Optimization: groupby is faster than repeated .loc access for large indices
        for timestamp, chunk in signals.groupby(level=0):
            current_date = timestamp.date()
            bar_data = chunk.droplevel(0)
            
            # Ensure bar_index is available (take first)
            bar_index = int(bar_data['bar_index'].iloc[0])
            
            # --- New Day Logic ---
            if current_date != self.prev_date:
                # Deduct overnight borrowing cost if applicable (cash < 0)
                if self.prev_date is not None and self.cash < 0 and self.config.borrow_rate_annual > 0:
                     daily_rate = self.config.borrow_rate_annual / 365.0
                     interest = abs(self.cash) * daily_rate
                     self.cash -= interest

                # Calculate daily return from previous close
                current_total_equity = self._get_total_equity(bar_data)
                if self.prev_date is not None and self.daily_start_equity > 0:
                    ret = current_total_equity / self.daily_start_equity - 1
                    self.daily_returns.append(ret)
                    
                    # Calculate avg exposure for the previous day
                    if self._current_day_exposures:
                        avg_exp = sum(self._current_day_exposures) / len(self._current_day_exposures)
                        self.daily_exposures.append(avg_exp)
                    else:
                        self.daily_exposures.append(0.0)
                    self._current_day_exposures = []
                
                self.daily_start_equity = current_total_equity
                self.prev_date = current_date
            
                # --- FIX: Flash Close Overnight Positions (Short Day Protection) ---
                # If positions exist at the start of a new day, it means they were missed
                # by the previous day's EOD check (likely a short 13:00 close).
                # prevent holding them through the new day.
                if self.positions:
                    for i in range(len(self.positions) - 1, -1, -1):
                        pos = self.positions[i]
                        inst = pos['instrument']
                        
                        # We must close it using current day's data
                        if inst in bar_data.index:
                            # Use OPEN price to capture overnight gap risk
                            # (If open is missing, fallback to close)
                            exit_price = bar_data.loc[inst, 'open'] if 'open' in bar_data.columns else bar_data.loc[inst, 'close']
                            
                            self._close_position(
                                pos_idx=i, 
                                timestamp=timestamp, 
                                price=exit_price, 
                                bar_index=bar_index, 
                                reason='overnight_flush'
                            )
                        else:
                            # Instrument not trading today? 
                            # Force close at last known to clear the list, assuming flat exit from last price.
                            # (Rare edge case for ETFs)
                            self._close_position(
                                pos_idx=i, 
                                timestamp=timestamp, 
                                price=pos.get('last_known_price', pos['entry_price']), 
                                bar_index=bar_index, 
                                reason='overnight_flush_missing'
                            )
            
            # --- 1. Manage Exits (Independent for each position) ---
            # Iterate backwards to safely remove
            for i in range(len(self.positions) - 1, -1, -1):
                pos = self.positions[i]
                inst = pos['instrument']
                
                # Check exist
                if inst not in bar_data.index:
                    continue
                
                current_price = bar_data.loc[inst, 'close']
                bars_held = bar_index - pos['entry_bar']
                
                reason = None
                if bar_index >= self.config.position_close_bar: # EOD
                    reason = 'eod'
                elif bars_held >= self.exit_bars: # Time Limit
                    reason = 'max_hold'
                
                if reason:
                    self._close_position(i, timestamp, current_price, bar_index, reason)

            # --- 2. Manage Entries ---
            # Check conditions:
            # a) Allowed time slot
            # b) Slots available
            # c) Not EOD
            
            # c) Not EOD
            
            is_entry_time = bar_index in self.allowed_entry_bars
            
            # Capacity Check
            if self.fixed_pos_pct:
                has_capacity = True # Infinite capacity in leverage mode
            else:
                has_capacity = len(self.positions) < self.max_positions
            
            if is_entry_time and has_capacity and bar_index <= self.config.last_entry_bar:
                # Determine allocation size
                total_eq = self._get_total_equity(bar_data)
                
                if self.fixed_pos_pct:
                    target_allocation = total_eq * self.fixed_pos_pct
                    allocation_allowed = True # Allow negative cash
                else:
                    # Classic Max 2 Logic
                    target_allocation = total_eq / self.max_positions
                    cost_buffer = target_allocation * 0.0005 
                    allocation_allowed = self.cash >= (target_allocation + cost_buffer)
                
                if allocation_allowed:
                    # Find best candidate
                    best_inst, best_sig = self._find_best(bar_data, instruments)
                    
                    if best_inst:
                        # Enter new position (Treat as independent even if same instrument)
                        self._open_position(best_inst, timestamp, bar_data.loc[best_inst, 'close'], 
                                          bar_index, target_allocation)
            
            # --- 3. Record Equity & Exposure ---
            current_eq = self._get_total_equity(bar_data)
            self.equity_curve.append((timestamp, current_eq))
            
            invested_value = current_eq - self.cash
            exp_pct = invested_value / current_eq if current_eq > 0 else 0
            self._current_day_exposures.append(exp_pct)
            
        # End of Backtest: Close all remaining
        if self.positions:
            pass 
        
        # Calculate final day return
        final_eq = self.equity_curve[-1][1]
        if self.daily_start_equity > 0:
            self.daily_returns.append(final_eq / self.daily_start_equity - 1)
            # Append final partial day exposure
            if self._current_day_exposures:
                avg_exp = sum(self._current_day_exposures) / len(self._current_day_exposures)
                self.daily_exposures.append(avg_exp)
            else:
                self.daily_exposures.append(0.0)
            
        return self.calculate_results(self.config.initial_capital, self.daily_returns, self.trades, self.equity_curve, self.daily_exposures)

    def _get_total_equity(self, bar_data):
        eq = self.cash
        for pos in self.positions:
            inst = pos['instrument']
            price = pos['entry_price']
            if inst in bar_data.index:
                price = bar_data.loc[inst, 'close']
                pos['last_known_price'] = price
            elif 'last_known_price' in pos:
                price = pos['last_known_price']
            
            val = pos['shares'] * price
            eq += val
        return eq

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

    def _open_position(self, inst, timestamp, price, bar_index, allocation):
        # Transaction Cost Logic:
        # Cost is deduced from allocation. 
        # allocation = entry_value + entry_cost
        # entry_cost = allocation * bps / 10000 (approx, referencing gross)
        # Actually: cost = gross * bps. gross ~= allocation.
        
        rate = self.config.transaction_cost_bps / 10000
        cost = allocation * rate # Estimate based on target gross
        
        if cost > allocation: return # Safety
        
        entry_value = allocation - cost
        shares = entry_value / price
        
        self.cash -= allocation
        
        self.positions.append({
            'instrument': inst,
            'entry_time': timestamp,
            'entry_price': price,
            'entry_bar': bar_index,
            'shares': shares,
            'entry_value': entry_value, # Net invested
            'allocated_capital': allocation # Gross allocated (for PnL calc)
        })

    def _close_position(self, pos_idx, timestamp, price, bar_index, reason):
        pos = self.positions.pop(pos_idx)
        
        gross = pos['shares'] * price
        cost = gross * self.config.transaction_cost_bps / 10000
        net = gross - cost
        
        # PnL against Gross Allocated Capital (to include entry cost)
        allocated = pos.get('allocated_capital', pos['entry_value']) # Fallback if missing
        
        pnl = net - allocated
        ret = pnl / allocated if allocated > 0 else 0
        
        # Add cash back
        self.cash += net
        
        self.trades.append(TradeRecord(
            instrument=pos['instrument'],
            entry_time=pos['entry_time'],
            entry_price=pos['entry_price'],
            exit_time=timestamp,
            exit_price=price,
            shares=pos['shares'],
            pnl=pnl,
            return_pct=ret,
            holding_bars=bar_index - pos['entry_bar'],
            exit_reason=reason,
            entry_bar=pos['entry_bar']
        ))
