"""
Intraday Labels for Trading Strategy.

This module handles label generation for intraday trading, with special
attention to:
- Forward return calculation without future data leakage
- End-of-day handling (positions must close at 15:55)
- Multiple horizon labels (6, 8, 12, 24 bars)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime, time
from pathlib import Path
import yaml


class IntradayLabels:
    """
    Generate labels for intraday trading strategies.
    
    Key considerations:
    - All positions must close at 15:55 (bar 77)
    - Labels must not leak future data
    - Different horizons for different strategy types
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize label generator.
        
        Args:
            config_path: Path to config file
        """
        if config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.primary_horizon = config['trading']['primary_label_horizon']
            self.alternative_horizons = config['trading']['alternative_horizons']
            self.position_close_bar = 77  # 15:55 bar
        else:
            self.primary_horizon = 8
            self.alternative_horizons = [6, 12, 24]
            self.position_close_bar = 77
        
        # All horizons to generate
        self.all_horizons = [self.primary_horizon] + self.alternative_horizons
    
    def generate_all_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
    ) -> pd.DataFrame:
        """
        Generate all label types for the input data.
        
        Args:
            df: DataFrame with price data and bar_index column
            price_col: Column name for price (default: 'close')
            
        Returns:
            DataFrame with label columns added
        """
        result = df.copy()
        
        # Ensure bar_index exists
        if 'bar_index' not in result.columns:
            result = self._add_bar_index(result)
        
        # Group by instrument if multi-index
        if isinstance(result.index, pd.MultiIndex):
            grouped = result.groupby(level='instrument')
            label_dfs = []
            
            for name, group in grouped:
                group_labels = self._generate_labels_for_group(
                    group.droplevel('instrument'),
                    price_col
                )
                group_labels['instrument'] = name
                label_dfs.append(group_labels)
            
            result = pd.concat(label_dfs)
            result = result.reset_index().set_index(['datetime', 'instrument'])
        else:
            result = self._generate_labels_for_group(result, price_col)
        
        return result
    
    def _add_bar_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bar_index column based on datetime."""
        result = df.copy()
        
        if isinstance(result.index, pd.DatetimeIndex):
            times = result.index
        elif isinstance(result.index, pd.MultiIndex):
            times = result.index.get_level_values(0)
        else:
            times = pd.to_datetime(result.index)
        
        hours = times.hour
        minutes = times.minute
        
        # Bar 1 = 9:35, Bar 2 = 9:40, etc.
        minutes_from_open = (hours - 9) * 60 + minutes - 35
        result['bar_index'] = (minutes_from_open / 5).astype(int) + 1
        
        return result
    
    def _generate_labels_for_group(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
    ) -> pd.DataFrame:
        """Generate labels for a single instrument."""
        result = df.copy()
        
        # Extract date from index for grouping
        if isinstance(result.index, pd.DatetimeIndex):
            result['_date'] = result.index.date
        else:
            result['_date'] = pd.to_datetime(result.index).date
        
        # Store close price at position close bar (15:55) for each date
        close_prices = result[result['bar_index'] == self.position_close_bar].groupby('_date')[price_col].first()
        result['_close_price'] = result['_date'].map(close_prices)
        
        # 1. Return-to-close label (return to 15:55 close)
        result['ret_to_close'] = result['_close_price'] / result[price_col] - 1
        
        # For bars at or after position close, ret_to_close should be 0
        result.loc[result['bar_index'] >= self.position_close_bar, 'ret_to_close'] = 0
        
        # 2. Fixed horizon returns
        for horizon in self.all_horizons:
            result = self._add_horizon_label(result, price_col, horizon)
        
        # 3. Binary direction labels
        for horizon in self.all_horizons:
            result[f'direction_{horizon}bar'] = (result[f'ret_{horizon}bar'] > 0).astype(int)
        
        result['direction_to_close'] = (result['ret_to_close'] > 0).astype(int)
        
        # 4. Effective return label (what we actually get after EOD close)
        # This is the "realistic" label that accounts for forced close
        result = self._add_effective_return_label(result, price_col)
        
        # 5. Add bars remaining until close
        result['bars_to_close'] = self.position_close_bar - result['bar_index']
        result['bars_to_close'] = result['bars_to_close'].clip(lower=0)
        
        # 6. Can trade flag (enough time for minimum holding)
        min_holding_bars = 6  # At least 30 min before forced close
        result['can_trade'] = (result['bars_to_close'] >= min_holding_bars).astype(int)
        
        # Clean up temporary columns
        result = result.drop(columns=['_date', '_close_price'])
        
        return result
    
    def _add_horizon_label(
        self,
        df: pd.DataFrame,
        price_col: str,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Add forward return label for specific horizon.
        
        Handles end-of-day boundary by using ret_to_close when horizon
        would extend past position close bar.
        """
        result = df.copy()
        
        # Simple forward return (shift by -horizon)
        future_price = result[price_col].shift(-horizon)
        result[f'ret_{horizon}bar'] = future_price / result[price_col] - 1
        
        # For bars where horizon extends past EOD close, use ret_to_close
        cutoff_bar = self.position_close_bar - horizon
        late_bars = result['bar_index'] > cutoff_bar
        result.loc[late_bars, f'ret_{horizon}bar'] = result.loc[late_bars, 'ret_to_close']
        
        # For bars at or after position close, label is NaN (can't trade)
        result.loc[result['bar_index'] >= self.position_close_bar, f'ret_{horizon}bar'] = np.nan
        
        return result
    
    def _add_effective_return_label(
        self,
        df: pd.DataFrame,
        price_col: str,
    ) -> pd.DataFrame:
        """
        Add effective return label that accounts for forced EOD close.
        
        This represents the actual return you would get if you:
        1. Enter at current bar
        2. Exit at target horizon OR at 15:55, whichever comes first
        """
        result = df.copy()
        
        horizon = self.primary_horizon
        
        # Target exit bar
        result['_target_exit_bar'] = result['bar_index'] + horizon
        
        # Actual exit bar (capped at position close)
        result['_actual_exit_bar'] = result['_target_exit_bar'].clip(upper=self.position_close_bar)
        
        # Effective holding period
        result['effective_holding_bars'] = result['_actual_exit_bar'] - result['bar_index']
        
        # Use appropriate return based on whether we hit EOD or not
        result['effective_ret'] = np.where(
            result['_target_exit_bar'] <= self.position_close_bar,
            result[f'ret_{horizon}bar'],  # Normal horizon return
            result['ret_to_close']         # Truncated to EOD
        )
        
        # Clean up
        result = result.drop(columns=['_target_exit_bar', '_actual_exit_bar'])
        
        return result
    
    def get_label_names(self) -> Dict[str, List[str]]:
        """Get dictionary of all label names by category."""
        labels = {
            'returns': [],
            'directions': [],
            'meta': [],
        }
        
        # Return labels
        labels['returns'].append('ret_to_close')
        labels['returns'].append('effective_ret')
        for horizon in self.all_horizons:
            labels['returns'].append(f'ret_{horizon}bar')
        
        # Direction labels
        labels['directions'].append('direction_to_close')
        for horizon in self.all_horizons:
            labels['directions'].append(f'direction_{horizon}bar')
        
        # Meta labels
        labels['meta'].extend([
            'bars_to_close',
            'effective_holding_bars',
            'can_trade',
        ])
        
        return labels
    
    def validate_labels(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate that labels don't leak future data.
        
        Args:
            df: DataFrame with labels
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_rows': len(df),
            'label_coverage': {},
            'issues': [],
        }
        
        label_cols = [c for c in df.columns if c.startswith('ret_') or c.startswith('direction_')]
        
        for col in label_cols:
            non_null = df[col].notna().sum()
            coverage = non_null / len(df) * 100
            results['label_coverage'][col] = f"{coverage:.1f}%"
        
        # Check for obvious issues
        if 'bar_index' in df.columns:
            late_bars = df[df['bar_index'] >= self.position_close_bar]
            for col in label_cols:
                if col.startswith('ret_') and not col.endswith('_close'):
                    non_null_late = late_bars[col].notna().sum()
                    if non_null_late > 0:
                        results['issues'].append(
                            f"{col} has {non_null_late} non-null values after position close bar"
                        )
        
        return results


if __name__ == "__main__":
    # Test the label generator
    print("Testing IntradayLabels...")
    
    # Create sample data for a single day
    np.random.seed(42)
    
    # Simulate a trading day (78 bars from 9:35 to 16:00)
    dates = pd.date_range('2023-12-20 09:35', periods=78, freq='5min')
    prices = 100 + np.cumsum(np.random.randn(78) * 0.1)
    
    sample_df = pd.DataFrame({
        'close': prices,
        'bar_index': range(1, 79),
    }, index=dates)
    sample_df.index.name = 'datetime'
    
    print(f"\nSample data: 1 trading day, {len(sample_df)} bars")
    print(f"Price range: {prices.min():.2f} to {prices.max():.2f}")
    
    # Generate labels
    label_gen = IntradayLabels()
    df_labels = label_gen.generate_all_labels(sample_df)
    
    print(f"\nLabels generated: {len(df_labels.columns)} columns")
    
    # Show label names
    label_names = label_gen.get_label_names()
    print(f"\nLabel categories:")
    for category, names in label_names.items():
        print(f"  {category}: {names}")
    
    # Show labels at different times of day
    print("\n\nLabels at different times of day:")
    print("-" * 80)
    
    key_bars = [1, 20, 40, 60, 70, 75, 77]
    cols_to_show = ['bar_index', 'close', 'ret_8bar', 'ret_to_close', 
                    'effective_ret', 'bars_to_close', 'can_trade']
    
    for bar in key_bars:
        row = df_labels[df_labels['bar_index'] == bar][cols_to_show].iloc[0]
        time_str = dates[bar-1].strftime('%H:%M')
        print(f"\nBar {bar} ({time_str}):")
        for col in cols_to_show:
            if col in ['close']:
                print(f"  {col}: {row[col]:.2f}")
            elif col in ['ret_8bar', 'ret_to_close', 'effective_ret']:
                if pd.notna(row[col]):
                    print(f"  {col}: {row[col]*100:.3f}%")
                else:
                    print(f"  {col}: NaN")
            else:
                print(f"  {col}: {row[col]}")
    
    # Validate
    print("\n\nValidation:")
    validation = label_gen.validate_labels(df_labels)
    print(f"  Label coverage:")
    for label, coverage in validation['label_coverage'].items():
        print(f"    {label}: {coverage}")
    
    if validation['issues']:
        print(f"  Issues found:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    else:
        print("  No issues found!")
    
    print("\n\nLabel generation test completed!")
