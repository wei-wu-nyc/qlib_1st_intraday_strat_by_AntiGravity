"""
Exit Rules for Intraday Trading Strategy.

Defines a framework for modular exit rules:
- Fixed Horizon
- Stop Loss
- Trailing Stop
- Profit Target
- Negative Return Gate (Time-based stop)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import pandas as pd
from dataclasses import dataclass

@dataclass
class ExitSignal:
    should_exit: bool
    reason: str = ""

class ExitRule(ABC):
    """Abstract base class for exit rules."""
    
    @abstractmethod
    def check(self, position, current_bar: int, current_price: float, current_time: pd.Timestamp) -> ExitSignal:
        """
        Check if position should be exited.
        
        Args:
            position: The Position object
            current_bar: Current bar index
            current_price: Current close price
            current_time: Current timestamp
            
        Returns:
            ExitSignal object
        """
        pass

class CompositeExitRule(ExitRule):
    """Combines multiple exit rules (OR logic)."""
    
    def __init__(self, rules: List[ExitRule]):
        self.rules = rules
        
    def check(self, position, current_bar, current_price, current_time) -> ExitSignal:
        for rule in self.rules:
            signal = rule.check(position, current_bar, current_price, current_time)
            if signal.should_exit:
                return signal
        return ExitSignal(False)

class FixedHorizonExit(ExitRule):
    """Exit after N bars."""
    
    def __init__(self, horizon_bars: int):
        self.horizon_bars = horizon_bars
        
    def check(self, position, current_bar, current_price, current_time) -> ExitSignal:
        bars_held = current_bar - position.entry_bar
        if bars_held >= self.horizon_bars:
            return ExitSignal(True, f"fixed_horizon_{self.horizon_bars}")
        return ExitSignal(False)

class StopLossExit(ExitRule):
    """Fixed percentage stop loss."""
    
    def __init__(self, stop_pct: float):
        """
        Args:
            stop_pct: Stop loss percentage (e.g. -0.005 for -0.5%)
                      Must be negative.
        """
        self.stop_pct = stop_pct
        if self.stop_pct > 0:
            raise ValueError("Stop loss percentage must be negative")
            
    def check(self, position, current_bar, current_price, current_time) -> ExitSignal:
        return_pct = current_price / position.entry_price - 1
        if return_pct <= self.stop_pct:
            return ExitSignal(True, f"stop_loss_{self.stop_pct*100:.1f}%")
        return ExitSignal(False)

class ProfitTargetExit(ExitRule):
    """Fixed percentage profit target."""
    
    def __init__(self, target_pct: float):
        """
        Args:
            target_pct: Profit target percentage (e.g. 0.01 for 1%)
                        Must be positive.
        """
        self.target_pct = target_pct
        if self.target_pct < 0:
            raise ValueError("Profit target must be positive")
            
    def check(self, position, current_bar, current_price, current_time) -> ExitSignal:
        return_pct = current_price / position.entry_price - 1
        if return_pct >= self.target_pct:
            return ExitSignal(True, f"profit_target_{self.target_pct*100:.1f}%")
        return ExitSignal(False)

class NegativeReturnGateExit(ExitRule):
    """Exit if return is negative after N bars (Time-based stop)."""
    
    def __init__(self, horizon_bars: int):
        self.horizon_bars = horizon_bars
        
    def check(self, position, current_bar, current_price, current_time) -> ExitSignal:
        bars_held = current_bar - position.entry_bar
        return_pct = current_price / position.entry_price - 1
        
        if bars_held >= self.horizon_bars and return_pct < 0:
            return ExitSignal(True, f"neg_gate_{self.horizon_bars}")
        return ExitSignal(False)

class TrailingStopExit(ExitRule):
    """Trailing stop loss based on high water mark."""
    
    def __init__(self, trail_pct: float):
        """
        Args:
            trail_pct: Trailing distance (e.g. -0.005 for -0.5% drop from peak)
                       Must be negative.
        """
        self.trail_pct = trail_pct
        # Dictionary to store high water mark per position (keyed by entry_time)
        self.high_water_marks = {}
        
        if self.trail_pct > 0:
            raise ValueError("Trailing stop percentage must be negative")
            
    def check(self, position, current_bar, current_price, current_time) -> ExitSignal:
        # Key using instrument and entry time to identify unique positions
        pos_key = (position.instrument, position.entry_time)
        
        # Initialize or update high water mark
        if pos_key not in self.high_water_marks:
            self.high_water_marks[pos_key] = max(position.entry_price, current_price)
        else:
            self.high_water_marks[pos_key] = max(self.high_water_marks[pos_key], current_price)
            
        high_mark = self.high_water_marks[pos_key]
        drawdown_from_peak = current_price / high_mark - 1
        
        if drawdown_from_peak <= self.trail_pct:
            # Clean up memory for closed position logic (handled by backtest reset normally, but good practice)
            return ExitSignal(True, f"trailing_stop_{self.trail_pct*100:.1f}%")
            
        return ExitSignal(False)
