"""
Seasonality Features for Intraday Trading.

This module provides time-based and calendar features that capture
seasonal patterns in market behavior, including:
- Time of day effects (opening/closing)
- Day of week effects
- Month and quarter effects
- Holiday effects
- Economic calendar events
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime, time, date
from pathlib import Path
import yaml

try:
    import pandas_market_calendars as mcal
    HAS_MARKET_CAL = True
except ImportError:
    HAS_MARKET_CAL = False
    print("Warning: pandas_market_calendars not installed. Holiday features limited.")


class SeasonalityFeatures:
    """
    Generate seasonality and time-based features for intraday data.
    """
    
    # US Market Holidays (major ones, for fallback if no calendar)
    US_HOLIDAYS_2023_2025 = [
        # 2023
        "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
        "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
        # 2024
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
        # 2025
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
        "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    ]
    
    # FOMC Meeting dates (approximate - 2-day meetings, using announcement day)
    FOMC_DATES_2023_2025 = [
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
        "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
        "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
        "2025-09-17", "2025-11-05", "2025-12-17",
    ]
    
    # CPI Release dates (usually around 13th of month, 8:30 AM ET)
    CPI_DATES_2023_2025 = [
        # 2023
        "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10",
        "2023-06-13", "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12",
        "2023-11-14", "2023-12-12",
        # 2024
        "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15",
        "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10",
        "2024-11-13", "2024-12-11",
        # 2025
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
        "2025-06-11", "2025-07-10", "2025-08-13", "2025-09-10", "2025-10-14",
        "2025-11-12", "2025-12-10",
    ]
    
    # Non-Farm Payrolls (NFP) - usually first Friday of month, 8:30 AM ET
    NFP_DATES_2023_2025 = [
        # 2023
        "2023-01-06", "2023-02-03", "2023-03-10", "2023-04-07", "2023-05-05",
        "2023-06-02", "2023-07-07", "2023-08-04", "2023-09-01", "2023-10-06",
        "2023-11-03", "2023-12-08",
        # 2024
        "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05", "2024-05-03",
        "2024-06-07", "2024-07-05", "2024-08-02", "2024-09-06", "2024-10-04",
        "2024-11-01", "2024-12-06",
        # 2025
        "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02",
        "2025-06-06", "2025-07-03", "2025-08-01", "2025-09-05", "2025-10-03",
        "2025-11-07", "2025-12-05",
    ]
    
    # GDP Release dates (advance estimate typically late month after quarter)
    GDP_DATES_2023_2025 = [
        # 2023 (Q4 2022, Q1-Q3 2023)
        "2023-01-26", "2023-04-27", "2023-07-27", "2023-10-26",
        # 2024
        "2024-01-25", "2024-04-25", "2024-07-25", "2024-10-30",
        # 2025
        "2025-01-30", "2025-04-30", "2025-07-30", "2025-10-29",
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize seasonality feature generator.
        
        Args:
            config_path: Path to config file
        """
        if config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.opening_minutes = config['seasonality']['opening_minutes']
            self.closing_minutes = config['seasonality']['closing_minutes']
            self.month_boundary_days = config['seasonality']['month_boundary_days']
            self.holiday_proximity_days = config['seasonality']['holiday_proximity_days']
            self.include_economic_events = config['seasonality'].get('include_economic_events', False)
        else:
            self.opening_minutes = 30
            self.closing_minutes = 30
            self.month_boundary_days = 3
            self.holiday_proximity_days = 1
            self.include_economic_events = False  # Disabled by default
        
        # Initialize market calendar
        self.nyse_calendar = mcal.get_calendar('NYSE') if HAS_MARKET_CAL else None
        
        # Convert holiday strings to dates
        self.holidays = set(pd.to_datetime(self.US_HOLIDAYS_2023_2025).date)
        
        # Economic events (only used if include_economic_events is True)
        if self.include_economic_events:
            self.fomc_dates = set(pd.to_datetime(self.FOMC_DATES_2023_2025).date)
            self.cpi_dates = set(pd.to_datetime(self.CPI_DATES_2023_2025).date)
            self.nfp_dates = set(pd.to_datetime(self.NFP_DATES_2023_2025).date)
            self.gdp_dates = set(pd.to_datetime(self.GDP_DATES_2023_2025).date)
        else:
            self.fomc_dates = set()
            self.cpi_dates = set()
            self.nfp_dates = set()
            self.gdp_dates = set()
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all seasonality features.
        
        Args:
            df: DataFrame with datetime index (or 'datetime' column)
            
        Returns:
            DataFrame with seasonality features added
        """
        result = df.copy()
        
        # Ensure we have datetime info
        if isinstance(result.index, pd.MultiIndex):
            datetimes = result.index.get_level_values(0)
        elif isinstance(result.index, pd.DatetimeIndex):
            datetimes = result.index
        elif 'datetime' in result.columns:
            datetimes = pd.to_datetime(result['datetime'])
        else:
            raise ValueError("Cannot find datetime in index or columns")
        
        # Extract time components
        result['_hour'] = datetimes.hour
        result['_minute'] = datetimes.minute
        result['_date'] = datetimes.date
        result['_dayofweek'] = datetimes.dayofweek
        result['_day'] = datetimes.day
        result['_month'] = datetimes.month
        result['_quarter'] = datetimes.quarter
        result['_weekofyear'] = datetimes.isocalendar().week.values
        
        # 1. Time of Day Features
        result = self._add_time_of_day_features(result)
        
        # 2. Day of Week Features
        result = self._add_day_of_week_features(result)
        
        # 3. Month/Quarter Features
        result = self._add_calendar_features(result)
        
        # 4. Holiday Features
        result = self._add_holiday_features(result)
        
        # 5. Special Event Features
        result = self._add_special_event_features(result)
        
        # Remove temporary columns
        temp_cols = [c for c in result.columns if c.startswith('_')]
        result = result.drop(columns=temp_cols)
        
        return result
    
    def _add_time_of_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-of-day features."""
        result = df.copy()
        
        # Minutes since market open (9:30 AM)
        minutes_since_open = (result['_hour'] - 9) * 60 + result['_minute'] - 30
        
        # Normalized time of day (0 = open, 1 = close)
        # Trading day is 9:30 to 16:00 = 390 minutes
        result['time_of_day'] = minutes_since_open / 390.0
        result['time_of_day'] = result['time_of_day'].clip(0, 1)
        
        # Bar index (1-78 for 5-min bars)
        result['bar_index'] = ((minutes_since_open - 5) / 5).astype(int) + 1
        result['bar_index'] = result['bar_index'].clip(1, 78)
        
        # Opening period (first 30 minutes = 6 bars)
        opening_bars = self.opening_minutes // 5
        result['is_opening'] = (result['bar_index'] <= opening_bars).astype(int)
        
        # Closing period (last 30 minutes = 6 bars)
        closing_start_bar = 78 - (self.closing_minutes // 5) + 1
        result['is_closing'] = (result['bar_index'] >= closing_start_bar).astype(int)
        
        # Mid-day (lunch period, roughly 12:00-13:00)
        result['is_midday'] = (
            ((result['_hour'] == 12) | 
             ((result['_hour'] == 11) & (result['_minute'] >= 30)) |
             ((result['_hour'] == 13) & (result['_minute'] <= 30)))
        ).astype(int)
        
        # Specific hour dummies (can be useful for some patterns)
        result['hour_10'] = (result['_hour'] == 10).astype(int)
        result['hour_11'] = (result['_hour'] == 11).astype(int)
        result['hour_14'] = (result['_hour'] == 14).astype(int)
        result['hour_15'] = (result['_hour'] == 15).astype(int)
        
        # Sinusoidal encoding of time (captures cyclical nature)
        result['time_sin'] = np.sin(2 * np.pi * result['time_of_day'])
        result['time_cos'] = np.cos(2 * np.pi * result['time_of_day'])
        
        return result
    
    def _add_day_of_week_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add day-of-week features."""
        result = df.copy()
        
        # Day of week (0=Monday, 4=Friday)
        result['day_of_week'] = result['_dayofweek']
        
        # Monday/Friday effects
        result['is_monday'] = (result['_dayofweek'] == 0).astype(int)
        result['is_friday'] = (result['_dayofweek'] == 4).astype(int)
        
        # Mid-week
        result['is_midweek'] = (result['_dayofweek'].isin([1, 2, 3])).astype(int)
        
        # Sinusoidal encoding
        result['dow_sin'] = np.sin(2 * np.pi * result['_dayofweek'] / 5)
        result['dow_cos'] = np.cos(2 * np.pi * result['_dayofweek'] / 5)
        
        return result
    
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar features (month, quarter, etc)."""
        result = df.copy()
        
        # Month of year
        result['month'] = result['_month']
        
        # Quarter
        result['quarter'] = result['_quarter']
        
        # Month start/end
        result['is_month_start'] = (result['_day'] <= self.month_boundary_days).astype(int)
        result['is_month_end'] = (result['_day'] >= 28 - self.month_boundary_days + 1).astype(int)
        
        # Week of month (approximate)
        result['week_of_month'] = ((result['_day'] - 1) // 7) + 1
        
        # Quarter start/end (first/last month of quarter)
        result['is_quarter_start'] = (result['_month'].isin([1, 4, 7, 10])).astype(int)
        result['is_quarter_end'] = (result['_month'].isin([3, 6, 9, 12])).astype(int)
        
        # Sinusoidal encoding for month
        result['month_sin'] = np.sin(2 * np.pi * result['_month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['_month'] / 12)
        
        return result
    
    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday proximity features."""
        result = df.copy()
        
        # Get trading dates from the data
        trading_dates = pd.Series(result['_date'].unique())
        trading_dates_sorted = sorted(trading_dates)
        date_to_idx = {d: i for i, d in enumerate(trading_dates_sorted)}
        
        # Map each row's date to its index
        result['_date_idx'] = result['_date'].map(date_to_idx)
        
        # Find distances to nearest holiday
        def is_pre_holiday(d):
            """Check if date is before a holiday (next trading day is holiday)."""
            for h in self.holidays:
                diff = (h - d).days
                if 0 < diff <= self.holiday_proximity_days + 2:  # +2 for weekend buffer
                    return 1
            return 0
        
        def is_post_holiday(d):
            """Check if date is after a holiday."""
            for h in self.holidays:
                diff = (d - h).days
                if 0 < diff <= self.holiday_proximity_days + 2:
                    return 1
            return 0
        
        # Apply holiday features (cache by date for efficiency)
        unique_dates = result['_date'].unique()
        pre_holiday_map = {d: is_pre_holiday(d) for d in unique_dates}
        post_holiday_map = {d: is_post_holiday(d) for d in unique_dates}
        
        result['is_pre_holiday'] = result['_date'].map(pre_holiday_map)
        result['is_post_holiday'] = result['_date'].map(post_holiday_map)
        
        return result
    
    def _add_special_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add special event features (FOMC, OpEx, etc)."""
        result = df.copy()
        
        # FOMC announcement days
        result['is_fomc_day'] = result['_date'].isin(self.fomc_dates).astype(int)
        
        # Options Expiration (third Friday of month)
        def is_opex(d):
            """Check if date is monthly options expiration (3rd Friday)."""
            if d.weekday() != 4:  # Not Friday
                return 0
            # Check if it's the 3rd Friday (days 15-21)
            if 15 <= d.day <= 21:
                return 1
            return 0
        
        unique_dates = result['_date'].unique()
        opex_map = {d: is_opex(d) for d in unique_dates}
        result['is_opex'] = result['_date'].map(opex_map)
        
        # Quad witching (3rd Friday of Mar, Jun, Sep, Dec)
        def is_quad_witching(d):
            if is_opex(d) and d.month in [3, 6, 9, 12]:
                return 1
            return 0
        
        quad_map = {d: is_quad_witching(d) for d in unique_dates}
        result['is_quad_witching'] = result['_date'].map(quad_map)
        
        # Economic Data Releases
        # CPI (Consumer Price Index) - major inflation data
        result['is_cpi_day'] = result['_date'].isin(self.cpi_dates).astype(int)
        
        # NFP (Non-Farm Payrolls) - major employment data
        result['is_nfp_day'] = result['_date'].isin(self.nfp_dates).astype(int)
        
        # GDP Release
        result['is_gdp_day'] = result['_date'].isin(self.gdp_dates).astype(int)
        
        # Any major economic event (combined flag)
        result['is_major_econ_day'] = (
            result['is_fomc_day'] | result['is_cpi_day'] | 
            result['is_nfp_day'] | result['is_gdp_day']
        ).astype(int)
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of all seasonality feature names."""
        return [
            # Time of day
            'time_of_day', 'bar_index', 'is_opening', 'is_closing', 'is_midday',
            'hour_10', 'hour_11', 'hour_14', 'hour_15',
            'time_sin', 'time_cos',
            # Day of week
            'day_of_week', 'is_monday', 'is_friday', 'is_midweek',
            'dow_sin', 'dow_cos',
            # Calendar
            'month', 'quarter', 'is_month_start', 'is_month_end', 'week_of_month',
            'is_quarter_start', 'is_quarter_end', 'month_sin', 'month_cos',
            # Holiday
            'is_pre_holiday', 'is_post_holiday',
            # Special events
            'is_fomc_day', 'is_opex', 'is_quad_witching',
            'is_cpi_day', 'is_nfp_day', 'is_gdp_day', 'is_major_econ_day',
        ]


if __name__ == "__main__":
    # Test the seasonality feature generator
    print("Testing SeasonalityFeatures...")
    
    # Create sample data spanning multiple days
    dates = pd.date_range('2023-12-18 09:35', periods=500, freq='5min')
    sample_df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(500) * 0.1),
    }, index=dates)
    sample_df.index.name = 'datetime'
    
    print(f"\nSample data range: {dates[0]} to {dates[-1]}")
    print(f"Sample data shape: {sample_df.shape}")
    
    # Generate features
    feature_gen = SeasonalityFeatures()
    df_features = feature_gen.generate_all_features(sample_df)
    
    print(f"\nFeatures generated: {len(df_features.columns)} columns")
    
    # Show feature names
    feature_names = feature_gen.get_feature_names()
    print(f"\nSeasonality features ({len(feature_names)}):")
    for name in feature_names:
        print(f"  - {name}")
    
    # Show sample values
    print(f"\nSample output (first 10 rows):")
    season_cols = [c for c in df_features.columns if c in feature_names]
    print(df_features[season_cols].head(10))
    
    print("\nSeasonality feature test completed!")
