"""
Base Strategy Class for Intraday Trading.

This module defines the abstract base class that all trading strategies
must implement, ensuring consistent interface across rule-based and ML approaches.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Trading signal types."""
    CASH = 0      # No position, hold cash
    LONG = 1      # Long position
    # SHORT = -1  # Short position (if needed later)


@dataclass
class TradeSignal:
    """
    Represents a trading signal at a specific point in time.
    
    Attributes:
        timestamp: When the signal was generated
        instrument: Which ETF to trade (or 'CASH')
        signal: Long or Cash
        strength: Signal strength/confidence (0-1)
        target_bars: Expected holding period
        metadata: Additional signal info
    """
    timestamp: pd.Timestamp
    instrument: str
    signal: Signal
    strength: float
    target_bars: int = 8
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement:
    - generate_signals(): Produce trading signals from features
    - get_name(): Return strategy name for reporting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize strategy.
        
        Args:
            config: Strategy-specific configuration
        """
        self.config = config or {}
        self.name = self.get_name()
        
        # Default trading parameters
        self.position_close_bar = 77  # 15:55
        self.min_bars_to_close = 6    # Minimum 30 min before forced close
        self.target_holding_bars = 8  # Default target holding period
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name."""
        pass
    
    @abstractmethod
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_position: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals from features.
        
        Args:
            features: DataFrame with all features and labels
            current_position: Current held instrument (or None for cash)
            
        Returns:
            DataFrame with 'signal', 'signal_strength', 'selected_instrument' columns
        """
        pass
    
    def filter_tradeable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to only bars where trading is allowed.
        
        Args:
            df: DataFrame with bar_index column
            
        Returns:
            Filtered DataFrame
        """
        if 'bar_index' in df.columns:
            # Must have minimum time before forced close
            last_entry_bar = self.position_close_bar - self.min_bars_to_close
            return df[df['bar_index'] <= last_entry_bar]
        return df
    
    def select_best_instrument(
        self,
        signals: pd.DataFrame,
        instruments: List[str],
    ) -> pd.Series:
        """
        Select the best instrument based on signal strength.
        
        Args:
            signals: DataFrame with signal columns for each instrument
            instruments: List of instrument names
            
        Returns:
            Series with selected instrument for each timestamp
        """
        # Get signal columns for each instrument
        signal_cols = [f'signal_{inst}' for inst in instruments]
        
        if all(col in signals.columns for col in signal_cols):
            # Select instrument with highest signal
            signal_df = signals[signal_cols]
            best_idx = signal_df.idxmax(axis=1)
            best_instrument = best_idx.str.replace('signal_', '')
            
            # Check if best signal exceeds threshold
            max_signal = signal_df.max(axis=1)
            entry_threshold = self.config.get('entry_threshold', 0.0)
            
            # Return CASH if no signal exceeds threshold
            best_instrument = best_instrument.where(
                max_signal > entry_threshold, 
                'CASH'
            )
            
            return best_instrument
        
        return pd.Series('CASH', index=signals.index)
    
    def prepare_output(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        strengths: pd.Series,
        instruments: pd.Series,
    ) -> pd.DataFrame:
        """
        Prepare standardized output format.
        
        Args:
            df: Original DataFrame
            signals: Signal series (1 for LONG, 0 for CASH)
            strengths: Signal strength series
            instruments: Selected instrument series
            
        Returns:
            DataFrame with standardized signal columns
        """
        result = df.copy()
        result['signal'] = signals
        result['signal_strength'] = strengths
        result['selected_instrument'] = instruments
        return result


class RuleBasedStrategy(BaseStrategy):
    """Base class for rule-based strategies with indicator logic."""
    
    def apply_entry_rules(
        self,
        df: pd.DataFrame,
        entry_conditions: pd.Series,
    ) -> pd.Series:
        """
        Apply entry rules with position close time filter.
        
        Args:
            df: DataFrame with bar_index
            entry_conditions: Boolean series of entry conditions
            
        Returns:
            Filtered entry conditions
        """
        # Only allow entries before last entry time
        if 'bar_index' in df.columns:
            last_entry_bar = self.position_close_bar - self.min_bars_to_close
            time_ok = df['bar_index'] <= last_entry_bar
            return entry_conditions & time_ok
        return entry_conditions
    
    def apply_exit_rules(
        self,
        df: pd.DataFrame,
        exit_conditions: pd.Series,
        holding_bars: pd.Series,
    ) -> pd.Series:
        """
        Apply exit rules including forced EOD close.
        
        Args:
            df: DataFrame with bar_index
            exit_conditions: Boolean series of normal exit conditions
            holding_bars: Series of bars held
            
        Returns:
            Combined exit conditions
        """
        # Force exit at position close bar
        if 'bar_index' in df.columns:
            forced_close = df['bar_index'] >= self.position_close_bar
            return exit_conditions | forced_close
        return exit_conditions


class MLStrategy(BaseStrategy):
    """Base class for ML model-based strategies."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model = None
        self.feature_names = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> 'MLStrategy':
        """
        Fit the ML model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional fit parameters
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Prediction array
        """
        pass
    
    def save_model(self, path: str):
        """Save the trained model."""
        import joblib
        joblib.dump(self.model, path)
        
    def load_model(self, path: str):
        """Load a trained model."""
        import joblib
        self.model = joblib.load(path)
        self.is_fitted = True
