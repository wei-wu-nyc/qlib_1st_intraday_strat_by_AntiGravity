"""
Intraday Alpha Features for 5-minute bar data.

This module provides Alpha158-style technical features adapted for 
intraday 5-minute bar data. Features are designed to capture:
- Price momentum and trend
- Volatility and range
- Volume patterns
- Microstructure (using upticks/downticks)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
from pathlib import Path
import yaml


class IntradayAlphaFeatures:
    """
    Generate alpha features for intraday 5-minute bar data.
    
    This class implements various technical indicators commonly used
    in quantitative trading, adapted for high-frequency intraday data.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature generator.
        
        Args:
            config_path: Path to config YAML. If None, uses default periods.
        """
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.return_periods = self.config['features']['return_periods']
            self.rsi_periods = self.config['features']['rsi_periods']
            self.ma_periods = self.config['features']['ma_periods']
            self.macd_params = self.config['features']['macd_params']
            self.bollinger = self.config['features']['bollinger']
            self.atr_periods = self.config['features']['atr_periods']
            self.tick_periods = self.config['features']['tick_periods']
        else:
            # Default configuration
            self.return_periods = [1, 3, 6, 12]
            self.rsi_periods = [6, 12, 24, 48]
            self.ma_periods = [6, 12, 24, 48]
            self.macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
            self.bollinger = {'period': 20, 'std': 2}
            self.atr_periods = [6, 12, 24]
            self.tick_periods = [1, 6, 12]
    
    def generate_all_features(
        self,
        df: pd.DataFrame,
        include_microstructure: bool = True,
    ) -> pd.DataFrame:
        """
        Generate all alpha features for the input data.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
                Should have MultiIndex (datetime, instrument) or single index
            include_microstructure: Whether to include uptick/downtick features
            
        Returns:
            DataFrame with all features added
        """
        result = df.copy()
        
        # Group by instrument if multi-index
        if isinstance(result.index, pd.MultiIndex):
            # Process each instrument separately
            grouped = result.groupby(level='instrument')
            features_list = []
            
            for name, group in grouped:
                group_features = self._generate_features_for_group(
                    group.droplevel('instrument'),
                    include_microstructure
                )
                group_features['instrument'] = name
                features_list.append(group_features)
            
            result = pd.concat(features_list)
            result = result.reset_index().set_index(['datetime', 'instrument'])
        else:
            result = self._generate_features_for_group(result, include_microstructure)
        
        return result
    
    def _generate_features_for_group(
        self,
        df: pd.DataFrame,
        include_microstructure: bool = True,
    ) -> pd.DataFrame:
        """Generate features for a single instrument."""
        result = df.copy()
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in result.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 1. Price/Return Features
        result = self._add_return_features(result)
        
        # 2. Price Position Features
        result = self._add_price_position_features(result)
        
        # 3. Momentum Features
        result = self._add_momentum_features(result)
        
        # 4. Volatility Features
        result = self._add_volatility_features(result)
        
        # 5. Moving Average Features
        result = self._add_ma_features(result)
        
        # 6. Volume Features
        result = self._add_volume_features(result)
        
        # 7. Microstructure Features (if upticks/downticks available)
        if include_microstructure and 'upticks' in result.columns:
            result = self._add_microstructure_features(result)
        
        # 8. Intraday Position Features (return since day open, etc.)
        result = self._add_intraday_position_features(result)
        
        return result
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        result = df.copy()
        
        for period in self.return_periods:
            # Simple return
            result[f'return_{period}bar'] = result['close'].pct_change(period)
            
            # Log return
            result[f'log_return_{period}bar'] = np.log(
                result['close'] / result['close'].shift(period)
            )
            
        # Gap (overnight/intraday gap)
        result['gap'] = result['open'] / result['close'].shift(1) - 1
        
        return result
    
    def _add_price_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price position features."""
        result = df.copy()
        
        # High-Low ratio
        result['high_low_ratio'] = result['high'] / result['low']
        
        # Close position within bar range
        bar_range = result['high'] - result['low']
        result['close_position'] = np.where(
            bar_range > 0,
            (result['close'] - result['low']) / bar_range,
            0.5
        )
        
        # Open position within bar range
        result['open_position'] = np.where(
            bar_range > 0,
            (result['open'] - result['low']) / bar_range,
            0.5
        )
        
        # Bar range as percentage
        result['range_pct'] = bar_range / result['close']
        
        return result
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        result = df.copy()
        
        # RSI for different periods
        for period in self.rsi_periods:
            result[f'rsi_{period}'] = self._calculate_rsi(result['close'], period)
        
        # Rate of Change (ROC)
        for period in self.return_periods:
            result[f'roc_{period}'] = (
                result['close'] / result['close'].shift(period) - 1
            ) * 100
        
        # MACD
        fast = self.macd_params['fast']
        slow = self.macd_params['slow']
        signal = self.macd_params['signal']
        
        ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
        
        result['macd'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # Momentum (price change)
        for period in self.return_periods:
            result[f'momentum_{period}'] = result['close'] - result['close'].shift(period)
        
        return result
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI when undefined
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        result = df.copy()
        
        # Standard deviation of returns
        returns = result['close'].pct_change()
        for period in self.rsi_periods:
            result[f'std_{period}'] = returns.rolling(period).std()
        
        # ATR (Average True Range)
        for period in self.atr_periods:
            result[f'atr_{period}'] = self._calculate_atr(result, period)
        
        # Bollinger Bands
        bb_period = self.bollinger['period']
        bb_std = self.bollinger['std']
        
        sma = result['close'].rolling(bb_period).mean()
        std = result['close'].rolling(bb_period).std()
        
        result['bb_upper'] = sma + bb_std * std
        result['bb_lower'] = sma - bb_std * std
        result['bb_middle'] = sma
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / sma
        
        # Position within Bollinger Bands
        bb_range = result['bb_upper'] - result['bb_lower']
        result['bb_position'] = np.where(
            bb_range > 0,
            (result['close'] - result['bb_lower']) / bb_range,
            0.5
        )
        
        return result
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _add_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        result = df.copy()
        
        for period in self.ma_periods:
            # Simple Moving Average
            result[f'sma_{period}'] = result['close'].rolling(period).mean()
            
            # Exponential Moving Average  
            result[f'ema_{period}'] = result['close'].ewm(span=period, adjust=False).mean()
            
            # Price relative to SMA
            result[f'close_sma_ratio_{period}'] = result['close'] / result[f'sma_{period}']
        
        # MA Crossovers (short/long ratios)
        if 6 in self.ma_periods and 24 in self.ma_periods:
            result['sma_cross_6_24'] = result['sma_6'] / result['sma_24']
            result['ema_cross_6_24'] = result['ema_6'] / result['ema_24']
            
        if 12 in self.ma_periods and 48 in self.ma_periods:
            result['sma_cross_12_48'] = result['sma_12'] / result['sma_48']
            result['ema_cross_12_48'] = result['ema_12'] / result['ema_48']
        
        return result
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        result = df.copy()
        
        # VWAP approximation (cumulative for the calculation period)
        for period in [6, 12, 24]:
            typical_price = (result['high'] + result['low'] + result['close']) / 3
            result[f'vwap_{period}'] = (
                (typical_price * result['volume']).rolling(period).sum() /
                result['volume'].rolling(period).sum()
            )
        
        # Volume relative to MA
        for period in [6, 12, 24]:
            vol_ma = result['volume'].rolling(period).mean()
            result[f'vol_ratio_{period}'] = result['volume'] / vol_ma
        
        # Volume change
        result['vol_change'] = result['volume'].pct_change()
        
        # Price-volume correlation (rolling)
        returns = result['close'].pct_change()
        for period in [12, 24]:
            result[f'price_vol_corr_{period}'] = (
                returns.rolling(period).corr(result['volume'])
            )
        
        return result
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure features using upticks/downticks."""
        result = df.copy()
        
        # Tick ratio (buying pressure)
        total_ticks = result['upticks'] + result['downticks']
        result['tick_ratio'] = np.where(
            total_ticks > 0,
            result['upticks'] / total_ticks,
            0.5
        )
        
        # Tick difference
        result['tick_diff'] = result['upticks'] - result['downticks']
        
        # Rolling tick metrics
        for period in self.tick_periods:
            if period == 1:
                continue
            result[f'tick_ratio_ma_{period}'] = result['tick_ratio'].rolling(period).mean()
            result[f'tick_diff_ma_{period}'] = result['tick_diff'].rolling(period).mean()
        
        # Tick momentum (change in tick ratio)
        for period in [6, 12]:
            result[f'tick_momentum_{period}'] = (
                result['tick_ratio'] - result['tick_ratio'].shift(period)
            )
        
        return result
    
    def _add_intraday_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add intraday position features including return since day open.
        
        These features capture how price has moved relative to the day's open,
        which can be important for intraday momentum/reversion strategies.
        """
        result = df.copy()
        
        # Extract date from index for grouping
        if isinstance(result.index, pd.DatetimeIndex):
            result['_date'] = result.index.date
        else:
            result['_date'] = pd.to_datetime(result.index).date
        
        # Get first bar of each day (opening values)
        day_open_price = result.groupby('_date')['open'].transform('first')
        day_open_high = result.groupby('_date')['high'].transform('first')
        day_open_low = result.groupby('_date')['low'].transform('first')
        
        # Return since day open
        result['ret_since_open'] = result['close'] / day_open_price - 1
        
        # Log return since day open
        result['log_ret_since_open'] = np.log(result['close'] / day_open_price)
        
        # Price relative to day open
        result['close_vs_day_open'] = result['close'] / day_open_price
        
        # Day's cumulative high and low (up to current bar)
        result['day_high'] = result.groupby('_date')['high'].cummax()
        result['day_low'] = result.groupby('_date')['low'].cummin()
        
        # Intraday range (high - low for the day so far)
        result['intraday_range'] = result['day_high'] - result['day_low']
        result['intraday_range_pct'] = result['intraday_range'] / day_open_price
        
        # Position within intraday range
        intraday_span = result['day_high'] - result['day_low']
        result['intraday_position'] = np.where(
            intraday_span > 0,
            (result['close'] - result['day_low']) / intraday_span,
            0.5
        )
        
        # Distance from day high/low
        result['dist_from_day_high'] = (result['day_high'] - result['close']) / result['close']
        result['dist_from_day_low'] = (result['close'] - result['day_low']) / result['close']
        
        # Cumulative volume for the day (relative)
        day_cum_vol = result.groupby('_date')['volume'].cumsum()
        day_total_vol = result.groupby('_date')['volume'].transform('sum')
        result['cum_vol_pct'] = day_cum_vol / day_total_vol
        
        # Clean up temp column
        result = result.drop(columns=['_date'])
        
        return result
    
    def get_feature_names(self, include_microstructure: bool = True) -> List[str]:
        """
        Get list of all feature names that will be generated.
        
        Args:
            include_microstructure: Whether to include microstructure features
            
        Returns:
            List of feature column names
        """
        features = []
        
        # Return features
        for period in self.return_periods:
            features.extend([f'return_{period}bar', f'log_return_{period}bar'])
        features.append('gap')
        
        # Price position
        features.extend(['high_low_ratio', 'close_position', 'open_position', 'range_pct'])
        
        # Momentum
        for period in self.rsi_periods:
            features.append(f'rsi_{period}')
        for period in self.return_periods:
            features.extend([f'roc_{period}', f'momentum_{period}'])
        features.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # Volatility
        for period in self.rsi_periods:
            features.append(f'std_{period}')
        for period in self.atr_periods:
            features.append(f'atr_{period}')
        features.extend(['bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position'])
        
        # MA
        for period in self.ma_periods:
            features.extend([f'sma_{period}', f'ema_{period}', f'close_sma_ratio_{period}'])
        features.extend(['sma_cross_6_24', 'ema_cross_6_24', 'sma_cross_12_48', 'ema_cross_12_48'])
        
        # Volume
        for period in [6, 12, 24]:
            features.extend([f'vwap_{period}', f'vol_ratio_{period}'])
        features.append('vol_change')
        for period in [12, 24]:
            features.append(f'price_vol_corr_{period}')
        
        # Microstructure
        if include_microstructure:
            features.extend(['tick_ratio', 'tick_diff'])
            for period in self.tick_periods:
                if period > 1:
                    features.extend([f'tick_ratio_ma_{period}', f'tick_diff_ma_{period}'])
            for period in [6, 12]:
                features.append(f'tick_momentum_{period}')
        
        # Intraday position features
        features.extend([
            'ret_since_open', 'log_ret_since_open', 'close_vs_day_open',
            'day_high', 'day_low', 'intraday_range', 'intraday_range_pct',
            'intraday_position', 'dist_from_day_high', 'dist_from_day_low',
            'cum_vol_pct'
        ])
        
        return features


if __name__ == "__main__":
    # Test the feature generator
    print("Testing IntradayAlphaFeatures...")
    
    # Create sample data
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2023-01-01 09:35', periods=n, freq='5min')
    sample_df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.1) + np.abs(np.random.randn(n) * 0.05),
        'low': 100 + np.cumsum(np.random.randn(n) * 0.1) - np.abs(np.random.randn(n) * 0.05),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': np.random.randint(1000, 100000, n),
        'upticks': np.random.randint(100, 5000, n),
        'downticks': np.random.randint(100, 5000, n),
    }, index=dates)
    
    # Ensure high >= open, close, low and low <= open, close, high
    sample_df['high'] = sample_df[['open', 'high', 'low', 'close']].max(axis=1)
    sample_df['low'] = sample_df[['open', 'high', 'low', 'close']].min(axis=1)
    
    print(f"\nSample data shape: {sample_df.shape}")
    
    # Generate features
    feature_gen = IntradayAlphaFeatures()
    df_features = feature_gen.generate_all_features(sample_df)
    
    print(f"\nFeatures generated: {len(df_features.columns)} columns")
    print(f"\nFeature names:")
    feature_names = feature_gen.get_feature_names()
    print(f"  Total: {len(feature_names)} features")
    
    # Show sample
    print(f"\nSample output (last 5 rows, first 10 columns):")
    print(df_features.iloc[-5:, :10])
    
    print("\nFeature generation test completed!")
