from typing import List, Optional, Tuple, Dict, Callable
import pandas as pd
from ..engine import IntradayBacktestEngine, BacktestConfig, TradeRecord, BacktestResults

class EqualWeightStrategy(IntradayBacktestEngine):
    """
    Equal-weight portfolio allocation across all instruments.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        super().__init__(config or BacktestConfig())
        self.reset()
    
    def reset(self):
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def run(self, data: pd.DataFrame, signal_generator: Callable, 
            instruments: List[str]) -> BacktestResults:
        self.reset()
        signals = signal_generator(data)
        
        all_timestamps = signals.index.get_level_values(0).unique().sort_values()
        prev_date = None
        daily_start_equity = self.config.initial_capital
        
        for timestamp in all_timestamps:
            current_date = timestamp.date()
            if timestamp not in signals.index:
                continue
            bar_data = signals.loc[timestamp]
            bar_index = int(bar_data['bar_index'].iloc[0])
            
            # New day reset
            if current_date != prev_date:
                if prev_date is not None and daily_start_equity > 0:
                    eq = self._get_equity(bar_data)
                    self.daily_returns.append(eq / daily_start_equity - 1)
                daily_start_equity = self._get_equity(bar_data)
                prev_date = current_date
            
            # EOD Close
            if bar_index >= self.config.position_close_bar:
                for inst in list(self.positions.keys()):
                    if inst in bar_data.index:
                        self._close(inst, timestamp, bar_data.loc[inst, 'close'], bar_index, 'eod')
                self.equity_curve.append((timestamp, self._get_equity(bar_data)))
                continue
            
            # Allocation Logic
            target_holdings = []
            for inst in instruments:
                if inst in bar_data.index:
                    sig = int(bar_data.loc[inst, 'signal'])
                    if sig == 1:
                        target_holdings.append(inst)
            
            if not target_holdings:
                # Close all
                for inst in list(self.positions.keys()):
                    if inst in bar_data.index:
                        self._close(inst, timestamp, bar_data.loc[inst, 'close'], bar_index, 'signal_end')
            else:
                # Rebalance
                current_equity = self._get_equity(bar_data)
                target_value = current_equity / len(target_holdings)
                
                # Close removed
                for inst in list(self.positions.keys()):
                    if inst not in target_holdings and inst in bar_data.index:
                        self._close(inst, timestamp, bar_data.loc[inst, 'close'], bar_index, 'exit')
                
                # Open/Adjust
                for inst in target_holdings:
                    if inst in bar_data.index:
                        price = bar_data.loc[inst, 'close']
                        if inst not in self.positions:
                            # Open new
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
            
            self.equity_curve.append((timestamp, self._get_equity(bar_data)))
        
        # Close remaining
        last_ts = all_timestamps[-1]
        last_data = signals.loc[last_ts]
        for inst in list(self.positions.keys()):
            if inst in last_data.index:
                self._close(inst, last_ts, last_data.loc[inst, 'close'], 
                           int(last_data['bar_index'].iloc[0]), 'end')
        
        if prev_date and daily_start_equity > 0:
            self.daily_returns.append(self.capital / daily_start_equity - 1)
            
        return self.calculate_results(self.capital, self.daily_returns, self.trades, self.equity_curve)

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

    def _get_equity(self, bar_data):
        eq = self.capital
        for inst, pos in self.positions.items():
            price = 0
            if inst in bar_data.index:
                price = bar_data.loc[inst, 'close']
                pos['last_known_price'] = price
            else:
                price = pos.get('last_known_price', pos['entry_price'])
            eq += pos['shares'] * price
        return eq
