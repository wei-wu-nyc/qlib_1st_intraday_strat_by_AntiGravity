"""
Intraday Data Loader for qlib 5-minute data.

This module handles loading and preprocessing of 5-minute bar data
for intraday trading strategies.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, time

import qlib
from qlib.data import D
from qlib.config import REG_US


class IntradayDataLoader:
    """
    Data loader for intraday 5-minute bar data using qlib.
    
    Handles:
    - Initialization of qlib with 5-min data provider
    - Loading OHLCV + tick data for specified ETFs
    - Filtering by date ranges and market hours
    - End-of-day handling for position close requirements
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        if config_path is None:
            # Default config path relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "intraday_config.yaml"
        
        self.config = self._load_config(config_path)
        self._init_qlib()
        
        # Market timing configuration
        self.open_bar = self.config['market']['open_bar']
        self.close_bar = self.config['market']['close_bar']
        self.position_close_bar = self.config['market']['position_close_bar']
        self.bars_per_day = self.config['market']['bars_per_day']
        
        # Instruments
        self.symbols = self.config['instruments']['symbols']
        self.benchmark = self.config['instruments']['benchmark']
        
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_qlib(self):
        """Initialize qlib with 5-min data provider."""
        provider_uri = self.config['data']['provider_uri']
        
        # Initialize qlib
        qlib.init(
            provider_uri=provider_uri,
            region=REG_US,
        )
        print(f"qlib initialized with provider: {provider_uri}")
    
    def get_raw_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load raw OHLCV data for specified symbols and date range.
        
        Args:
            symbols: List of symbols to load. If None, uses configured ETFs.
            start_date: Start date string (YYYY-MM-DD). If None, uses config.
            end_date: End date string (YYYY-MM-DD). If None, uses config.
            fields: List of fields to load. If None, loads all available.
            
        Returns:
            DataFrame with MultiIndex (datetime, instrument) and OHLCV columns.
        """
        if symbols is None:
            symbols = self.symbols
        
        if start_date is None:
            start_date = self.config['periods']['train']['start']
        if end_date is None:
            end_date = self.config['periods']['test']['end']
            
        if fields is None:
            fields = ['$open', '$high', '$low', '$close', '$volume', 
                     '$upticks', '$downticks']
        
        # Format symbols for qlib (uppercase)
        symbols_upper = [s.upper() for s in symbols]
        
        # Load data using qlib Data API
        df = D.features(
            instruments=symbols_upper,
            fields=fields,
            start_time=start_date,
            end_time=end_date,
            freq='5min',
        )
        
        # Clean column names (remove $ prefix)
        df.columns = [col.replace('$', '') for col in df.columns]
        
        # Apply data cleaning
        df = self._clean_data(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data to handle missing values.
        
        Rules:
        1. Volume/Upticks/Downticks: Fill NaN with 0
        2. Close: Forward fill with previous Close
        3. High/Low/Open: Fill NaN with current Close (after ffill)
        
        Args:
            df: Raw DataFrame with OHLCV columns
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # 1. Fill volume-related columns with 0
        vol_cols = ['volume', 'upticks', 'downticks']
        for col in vol_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 2. Forward fill Close price
        # Group by instrument to ensure we don't fill across assets
        if 'close' in df.columns:
            if isinstance(df.index, pd.MultiIndex):
                # Ensure sorted index for ffill
                df = df.sort_index()
                df['close'] = df.groupby(level='instrument')['close'].ffill()
            else:
                df['close'] = df['close'].ffill()
        
        # 3. Fill High/Low/Open with Close where missing
        price_cols = ['high', 'low', 'open']
        for col in price_cols:
            if col in df.columns and 'close' in df.columns:
                df[col] = df[col].fillna(df['close'])
        
        return df
    
    def get_data_with_time_features(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data with additional time-based features.
        
        Adds:
        - bar_index: Bar number within day (1-78)
        - is_close_bar: Whether this is the 15:55 position close bar
        - bars_to_close: Number of bars remaining until position close
        - date: Trading date
        - time: Time of bar
        
        Args:
            symbols: List of symbols to load
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with time features added
        """
        df = self.get_raw_data(symbols, start_date, end_date)
        
        # Extract datetime components from index
        df = df.reset_index()
        
        # Handle the datetime index
        if 'datetime' in df.columns:
            df['date'] = df['datetime'].dt.date
            df['time'] = df['datetime'].dt.time
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
        
        # Calculate bar index within day
        # Market opens at 9:35, each bar is 5 min
        # Bar 1 = 9:35, Bar 2 = 9:40, ..., Bar 78 = 16:00
        df['minutes_from_open'] = (df['hour'] - 9) * 60 + df['minute'] - 35
        df['bar_index'] = (df['minutes_from_open'] / 5).astype(int) + 1
        
        # Position close bar (15:55 = bar 77, since 16:00 is bar 78)
        close_time = time(15, 55)
        df['is_close_bar'] = df['time'] == close_time
        
        # Bars remaining until forced close
        df['bars_to_close'] = 77 - df['bar_index']  # 77 is position close bar
        df['bars_to_close'] = df['bars_to_close'].clip(lower=0)
        
        # Set back to multi-index
        if 'instrument' in df.columns:
            df = df.set_index(['datetime', 'instrument'])
        
        return df
    
    def get_period_data(
        self,
        period: str,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get data for a specific period (train/valid/test).
        
        Args:
            period: One of 'train', 'valid', 'test'
            symbols: List of symbols
            
        Returns:
            DataFrame for the specified period
        """
        period_config = self.config['periods'][period]
        return self.get_data_with_time_features(
            symbols=symbols,
            start_date=period_config['start'],
            end_date=period_config['end'],
        )
    
    def get_benchmark_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get benchmark (SPY) data for comparison.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with benchmark OHLCV data
        """
        return self.get_raw_data(
            symbols=[self.benchmark],
            start_date=start_date,
            end_date=end_date,
        )
    
    def filter_tradeable_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to only bars where new positions can be opened.
        
        Excludes:
        - Bars after last_entry_bar (default 14:30)
        - This ensures minimum holding time before forced close
        
        Args:
            df: DataFrame with bar_index column
            
        Returns:
            Filtered DataFrame
        """
        last_entry_bar_time = self.config['trading']['last_entry_bar']
        hour, minute = map(int, last_entry_bar_time.split(':'))
        last_entry_minutes = (hour - 9) * 60 + minute - 35
        last_entry_bar = last_entry_minutes // 5 + 1
        
        if 'bar_index' in df.columns:
            return df[df['bar_index'] <= last_entry_bar]
        else:
            # Recalculate from index if needed
            df_copy = df.copy()
            if isinstance(df_copy.index, pd.MultiIndex):
                times = df_copy.index.get_level_values(0)
            else:
                times = df_copy.index
            
            hours = times.hour
            minutes = times.minute
            bar_indices = ((hours - 9) * 60 + minutes - 35) // 5 + 1
            return df_copy[bar_indices <= last_entry_bar]
    
    def get_calendar(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        """
        Get trading calendar (list of trading dates).
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading date strings
        """
        if start_date is None:
            start_date = self.config['periods']['train']['start']
        if end_date is None:
            end_date = self.config['periods']['test']['end']
            
        calendar = D.calendar(start_time=start_date, end_time=end_date, freq='day')
        return [str(d.date()) for d in calendar]
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate loaded data for common issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_rows': len(df),
            'symbols': df.index.get_level_values('instrument').unique().tolist() 
                       if isinstance(df.index, pd.MultiIndex) else [],
            'date_range': (df.index.get_level_values(0).min(), 
                          df.index.get_level_values(0).max())
                         if isinstance(df.index, pd.MultiIndex) else None,
            'missing_values': df.isnull().sum().to_dict(),
            'zero_volume_pct': (df['volume'] == 0).mean() * 100 
                              if 'volume' in df.columns else None,
        }
        return results


def load_sample_data(n_days: int = 5) -> pd.DataFrame:
    """
    Quick function to load a sample of data for testing.
    
    Args:
        n_days: Number of recent trading days to load
        
    Returns:
        Sample DataFrame
    """
    loader = IntradayDataLoader()
    
    # Get a sample period
    end_date = "2023-12-31"
    start_date = "2023-12-20"  # About 7-8 trading days
    
    return loader.get_data_with_time_features(
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    # Test the data loader
    print("Testing IntradayDataLoader...")
    
    try:
        loader = IntradayDataLoader()
        
        print("\n1. Loading sample data (last week of 2023)...")
        df = load_sample_data()
        print(f"   Loaded {len(df)} rows")
        
        print("\n2. Data validation:")
        validation = loader.validate_data(df)
        for key, value in validation.items():
            print(f"   {key}: {value}")
        
        print("\n3. Sample of data:")
        print(df.head(10))
        
        print("\n4. Columns available:")
        print(df.columns.tolist())
        
        print("\nData loader test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
