from typing import List, Optional, Tuple, Dict, Callable
import pandas as pd
import numpy as np
from ..engine import IntradayBacktestEngine, BacktestConfig, TradeRecord, BacktestResults
from .multi_trade import MultiTradeStrategy

class RelaxedMultiTradeStrategy(MultiTradeStrategy):
    """
    Relaxed Multi-Trade Strategy.
    Allows entry at ANY bar within a defined time block, but restricts to 
    ONE entry per block per day.
    """
    
    # 9:30=0, 9:40=2, 10:00=6, 10:30=12, 12:00=30, 14:00=54, 15:00=66
    BLOCKS = [
        (2, 6),   # 09:40 <= b < 10:00 (Bars 2,3,4,5)
        (6, 12),  # 10:00 <= b < 10:30 (Bars 6..11)
        (12, 30), # 10:30 <= b < 12:00 (Bars 12..29)
        (30, 54), # 12:00 <= b < 14:00 (Bars 30..53)
        (54, 67)  # 14:00 <= b <= 15:00 (Bars 54..66) - Note: 15:00 is bar 66. <67 includes 66.
    ]
    
    def reset(self):
        super().reset()
        self.last_entry_block_idx = -1
        self.daily_entries_count = 0 
        
    def run(self, data: pd.DataFrame, signal_generator: Callable, 
            instruments: List[str]) -> BacktestResults:
        self.reset()
        signals = signal_generator(data)
        
        if not isinstance(signals.index, pd.MultiIndex):
            raise ValueError("Requires MultiIndex (datetime, instrument)")
            
        for timestamp, chunk in signals.groupby(level=0):
            current_date = timestamp.date()
            bar_data = chunk.droplevel(0)
            bar_index = int(bar_data['bar_index'].iloc[0])
            
            # --- New Day Logic ---
            if current_date != self.prev_date:
                # Reset Block Tracker for new day
                self.last_entry_block_idx = -1
                self.daily_entries_count = 0
                
                # Copy standard new day logic from parent
                if self.prev_date is not None and self.cash < 0 and self.config.borrow_rate_annual > 0:
                     daily_rate = self.config.borrow_rate_annual / 365.0
                     interest = abs(self.cash) * daily_rate
                     self.cash -= interest

                current_total_equity = self._get_total_equity(bar_data)
                if self.prev_date is not None and self.daily_start_equity > 0:
                    ret = current_total_equity / self.daily_start_equity - 1
                    self.daily_returns.append(ret)
                    
                    if self._current_day_exposures:
                        avg_exp = sum(self._current_day_exposures) / len(self._current_day_exposures)
                        max_exp = max(self._current_day_exposures)
                        self.daily_exposures.append(avg_exp)
                        # self.max_exposures.append(max_exp) # Parent doesn't have this list yet, need to handle carefuly or just rely on result calculation
                    else:
                        self.daily_exposures.append(0.0)
                    self._current_day_exposures = []
                
                self.daily_start_equity = current_total_equity
                self.prev_date = current_date
            
            # --- 1. Manage Exits ---
            for i in range(len(self.positions) - 1, -1, -1):
                pos = self.positions[i]
                inst = pos['instrument']
                if inst not in bar_data.index: continue
                
                current_price = bar_data.loc[inst, 'close']
                bars_held = bar_index - pos['entry_bar']
                
                reason = None
                if bar_index >= self.config.position_close_bar: reason = 'eod'
                elif bars_held >= self.exit_bars: reason = 'max_hold'
                
                if reason:
                    self._close_position(i, timestamp, current_price, bar_index, reason)
            
            # --- 2. Manage Entries (Relaxed Logic) ---
            # Identify Current Block
            current_block_idx = -1
            for idx, (start, end) in enumerate(self.BLOCKS):
                if start <= bar_index < end:
                    current_block_idx = idx
                    break
            
            is_entry_time = False
            # Rule: Must be in a block, AND haven't entered in this block yet
            if current_block_idx != -1 and current_block_idx != self.last_entry_block_idx:
                is_entry_time = True
            
            # Standard Capacity Check
            if self.fixed_pos_pct:
                has_capacity = True 
            else:
                has_capacity = len(self.positions) < self.max_positions
            
            if is_entry_time and has_capacity and bar_index <= self.config.last_entry_bar:
                # Determine allocation
                total_eq = self._get_total_equity(bar_data)
                
                if self.fixed_pos_pct:
                    target_allocation = total_eq * self.fixed_pos_pct
                    allocation_allowed = True 
                else:
                    target_allocation = total_eq / self.max_positions
                    cost_buffer = target_allocation * 0.0005 
                    allocation_allowed = self.cash >= (target_allocation + cost_buffer)
                
                if allocation_allowed:
                    # Find candidate
                    best_inst, best_sig = self._find_best(bar_data, instruments)
                    
                    if best_inst:
                        # !!! ENTRY TAKEN !!!
                        self._open_position(best_inst, timestamp, bar_data.loc[best_inst, 'close'], 
                                          bar_index, target_allocation)
                        
                        # Mark this block as used
                        self.last_entry_block_idx = current_block_idx
                        self.daily_entries_count += 1
            
            # --- 3. Record Equity ---
            current_eq = self._get_total_equity(bar_data)
            self.equity_curve.append((timestamp, current_eq))
            
            invested_value = current_eq - self.cash
            exp_pct = invested_value / current_eq if current_eq > 0 else 0
            self._current_day_exposures.append(exp_pct)
            
        # Finalize
        final_eq = self.equity_curve[-1][1]
        if self.daily_start_equity > 0:
            self.daily_returns.append(final_eq / self.daily_start_equity - 1)
            if self._current_day_exposures:
                avg_exp = sum(self._current_day_exposures) / len(self._current_day_exposures)
                self.daily_exposures.append(avg_exp)
            else:
                self.daily_exposures.append(0.0)
            
        return self.calculate_results(self.config.initial_capital, self.daily_returns, self.trades, self.equity_curve, self.daily_exposures)
