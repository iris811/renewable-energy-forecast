"""
Data cleaning and preprocessing utilities
Handles missing values, outliers, and data quality issues
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ..utils.logger import log


class DataCleaner:
    """
    Cleans and preprocesses raw data
    """

    def __init__(self):
        """Initialize data cleaner"""
        self.fill_strategies = {
            'temperature': 'interpolate',
            'humidity': 'interpolate',
            'pressure': 'interpolate',
            'wind_speed': 'forward_fill',
            'wind_direction': 'forward_fill',
            'cloud_cover': 'forward_fill',
            'precipitation': 'zero',
            'solar_irradiance': 'zero',
            'ghi': 'zero',
            'dni': 'zero',
            'dhi': 'zero'
        }

    def handle_missing_values(self, df: pd.DataFrame, strategy: Optional[Dict] = None) -> pd.DataFrame:
        """
        Handle missing values in DataFrame

        Args:
            df: Input DataFrame
            strategy: Dictionary mapping column names to fill strategies
                     Options: 'interpolate', 'forward_fill', 'backward_fill', 'mean', 'median', 'zero'

        Returns:
            DataFrame with missing values filled
        """
        df = df.copy()
        strategies = strategy or self.fill_strategies

        for column in df.columns:
            if column not in strategies:
                continue

            missing_count = df[column].isna().sum()
            if missing_count == 0:
                continue

            fill_method = strategies[column]

            if fill_method == 'interpolate':
                df[column] = df[column].interpolate(method='linear', limit_direction='both')
            elif fill_method == 'forward_fill':
                df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
            elif fill_method == 'backward_fill':
                df[column] = df[column].fillna(method='bfill').fillna(method='ffill')
            elif fill_method == 'mean':
                df[column] = df[column].fillna(df[column].mean())
            elif fill_method == 'median':
                df[column] = df[column].fillna(df[column].median())
            elif fill_method == 'zero':
                df[column] = df[column].fillna(0)

            filled_count = missing_count - df[column].isna().sum()
            log.debug(f"Filled {filled_count} missing values in {column} using {fill_method}")

        # Check remaining missing values
        remaining_missing = df.isna().sum()
        if remaining_missing.sum() > 0:
            log.warning(f"Remaining missing values:\n{remaining_missing[remaining_missing > 0]}")

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from specified columns

        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers (None = all numeric columns)
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
                      IQR: typically 1.5 (moderate) or 3.0 (extreme)
                      Z-score: typically 3.0

        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        original_len = len(df)

        for column in columns:
            if column not in df.columns:
                continue

            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)

            elif method == 'zscore':
                mean = df[column].mean()
                std = df[column].std()
                z_scores = np.abs((df[column] - mean) / std)
                mask = z_scores < threshold

            else:
                log.warning(f"Unknown outlier detection method: {method}")
                continue

            outliers_count = (~mask).sum()
            if outliers_count > 0:
                df = df[mask]
                log.debug(f"Removed {outliers_count} outliers from {column}")

        removed_count = original_len - len(df)
        if removed_count > 0:
            log.info(f"Removed {removed_count} rows containing outliers ({removed_count/original_len*100:.2f}%)")

        return df

    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that values are within physically reasonable ranges

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with invalid values replaced
        """
        df = df.copy()

        # Define valid ranges for each parameter
        ranges = {
            'temperature': (-50, 60),      # Celsius
            'humidity': (0, 100),          # Percentage
            'pressure': (800, 1100),       # hPa
            'wind_speed': (0, 50),         # m/s
            'wind_direction': (0, 360),    # degrees
            'cloud_cover': (0, 100),       # Percentage
            'precipitation': (0, 500),     # mm
            'solar_irradiance': (0, 1500), # W/m²
            'ghi': (0, 1500),              # W/m²
            'dni': (0, 1200),              # W/m²
            'dhi': (0, 800),               # W/m²
            'power_output': (0, None)      # kW (no upper limit)
        }

        for column, (min_val, max_val) in ranges.items():
            if column not in df.columns:
                continue

            # Check minimum
            if min_val is not None:
                invalid_min = df[column] < min_val
                if invalid_min.sum() > 0:
                    log.warning(f"Found {invalid_min.sum()} values below minimum for {column}")
                    df.loc[invalid_min, column] = min_val

            # Check maximum
            if max_val is not None:
                invalid_max = df[column] > max_val
                if invalid_max.sum() > 0:
                    log.warning(f"Found {invalid_max.sum()} values above maximum for {column}")
                    df.loc[invalid_max, column] = max_val

        return df

    def resample_timeseries(
        self,
        df: pd.DataFrame,
        freq: str = '1H',
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Resample time series data to regular intervals

        Args:
            df: Input DataFrame with datetime index
            freq: Resampling frequency (e.g., '1H', '15T', '1D')
            aggregation: Aggregation method ('mean', 'sum', 'max', 'min')

        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            log.error("DataFrame index must be DatetimeIndex for resampling")
            return df

        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        # Resample numeric columns
        if aggregation == 'mean':
            resampled_numeric = df[numeric_cols].resample(freq).mean()
        elif aggregation == 'sum':
            resampled_numeric = df[numeric_cols].resample(freq).sum()
        elif aggregation == 'max':
            resampled_numeric = df[numeric_cols].resample(freq).max()
        elif aggregation == 'min':
            resampled_numeric = df[numeric_cols].resample(freq).min()
        else:
            log.warning(f"Unknown aggregation method: {aggregation}, using mean")
            resampled_numeric = df[numeric_cols].resample(freq).mean()

        # For non-numeric columns, use first value
        if len(non_numeric_cols) > 0:
            resampled_non_numeric = df[non_numeric_cols].resample(freq).first()
            resampled = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
        else:
            resampled = resampled_numeric

        log.info(f"Resampled data from {len(df)} to {len(resampled)} records at {freq} frequency")
        return resampled

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with duplicates removed
        """
        original_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        removed = original_len - len(df)

        if removed > 0:
            log.info(f"Removed {removed} duplicate records")

        return df

    def clean(
        self,
        df: pd.DataFrame,
        fill_missing: bool = True,
        remove_outliers: bool = True,
        validate_ranges: bool = True,
        resample: Optional[str] = None,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        Apply full cleaning pipeline

        Args:
            df: Input DataFrame
            fill_missing: Fill missing values
            remove_outliers: Remove outliers
            validate_ranges: Validate value ranges
            resample: Resample frequency (None to skip)
            remove_duplicates: Remove duplicate records

        Returns:
            Cleaned DataFrame
        """
        log.info(f"Starting data cleaning pipeline. Initial records: {len(df)}")

        # Remove duplicates first
        if remove_duplicates:
            df = self.remove_duplicates(df)

        # Resample if requested
        if resample:
            df = self.resample_timeseries(df, freq=resample)

        # Validate ranges
        if validate_ranges:
            df = self.validate_ranges(df)

        # Fill missing values
        if fill_missing:
            df = self.handle_missing_values(df)

        # Remove outliers
        if remove_outliers:
            df = self.remove_outliers(df)

        log.info(f"Data cleaning completed. Final records: {len(df)}")
        return df

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate data quality report

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_records': len(df),
            'missing_values': df.isna().sum().to_dict(),
            'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict(),
            'duplicates': df.index.duplicated().sum(),
            'date_range': {
                'start': df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
                'end': df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None
            }
        }

        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        report['statistics'] = df[numeric_cols].describe().to_dict()

        return report
