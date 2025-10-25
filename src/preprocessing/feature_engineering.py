"""
Feature engineering for renewable energy forecasting
Creates temporal, solar position, and interaction features
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from ..utils.logger import log


class FeatureEngineer:
    """
    Creates features for renewable energy forecasting
    """

    def __init__(self):
        """Initialize feature engineer"""
        pass

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features

        Args:
            df: Input DataFrame with datetime index

        Returns:
            DataFrame with added time features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            log.error("DataFrame index must be DatetimeIndex")
            return df

        df = df.copy()

        # Basic time features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['quarter'] = df.index.quarter

        # Cyclical time features (sine/cosine encoding)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Season
        df['season'] = df['month'].apply(self._get_season)

        log.info(f"Added {18} time features")
        return df

    def add_solar_position(self, df: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
        """
        Calculate solar position (elevation and azimuth)

        Args:
            df: Input DataFrame with datetime index
            latitude: Location latitude in degrees
            longitude: Location longitude in degrees

        Returns:
            DataFrame with solar position features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            log.error("DataFrame index must be DatetimeIndex")
            return df

        df = df.copy()

        # Initialize arrays
        solar_elevation = np.zeros(len(df))
        solar_azimuth = np.zeros(len(df))

        for i, timestamp in enumerate(df.index):
            elevation, azimuth = self._calculate_solar_position(
                timestamp, latitude, longitude
            )
            solar_elevation[i] = elevation
            solar_azimuth[i] = azimuth

        df['solar_elevation'] = solar_elevation
        df['solar_azimuth'] = solar_azimuth

        # Is daytime (solar elevation > 0)
        df['is_daytime'] = (df['solar_elevation'] > 0).astype(int)

        # Solar elevation cyclical features
        df['solar_elevation_sin'] = np.sin(np.radians(df['solar_elevation']))
        df['solar_elevation_cos'] = np.cos(np.radians(df['solar_elevation']))

        log.info(f"Added solar position features for lat={latitude}, lon={longitude}")
        return df

    def add_lag_features(self, df: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
        """
        Add lagged features

        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag values (e.g., [1, 2, 3, 24, 168])

        Returns:
            DataFrame with lagged features
        """
        df = df.copy()
        features_added = 0

        for column in columns:
            if column not in df.columns:
                log.warning(f"Column {column} not found in DataFrame")
                continue

            for lag in lags:
                lag_col_name = f"{column}_lag_{lag}"
                df[lag_col_name] = df[column].shift(lag)
                features_added += 1

        log.info(f"Added {features_added} lag features")
        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: list,
        windows: list,
        functions: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Add rolling window features

        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes (e.g., [3, 6, 12, 24])
            functions: List of functions to apply ('mean', 'std', 'min', 'max')

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        if functions is None:
            functions = ['mean', 'std']

        features_added = 0

        for column in columns:
            if column not in df.columns:
                log.warning(f"Column {column} not found in DataFrame")
                continue

            for window in windows:
                if 'mean' in functions:
                    df[f"{column}_rolling_mean_{window}"] = df[column].rolling(window=window, min_periods=1).mean()
                    features_added += 1
                if 'std' in functions:
                    df[f"{column}_rolling_std_{window}"] = df[column].rolling(window=window, min_periods=1).std()
                    features_added += 1
                if 'min' in functions:
                    df[f"{column}_rolling_min_{window}"] = df[column].rolling(window=window, min_periods=1).min()
                    features_added += 1
                if 'max' in functions:
                    df[f"{column}_rolling_max_{window}"] = df[column].rolling(window=window, min_periods=1).max()
                    features_added += 1

        log.info(f"Added {features_added} rolling window features")
        return df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features specific to renewable energy

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        features_added = 0

        # Temperature-related interactions
        if 'temperature' in df.columns and 'humidity' in df.columns:
            # Heat index (feels-like temperature)
            df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])
            features_added += 1

        # Wind chill
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['wind_chill'] = self._calculate_wind_chill(df['temperature'], df['wind_speed'])
            features_added += 1

        # Solar efficiency factor (considering cloud cover)
        if 'solar_elevation' in df.columns and 'cloud_cover' in df.columns:
            df['solar_efficiency'] = df['solar_elevation'] * (1 - df['cloud_cover'] / 100)
            features_added += 1

        # Clear sky indicator
        if 'cloud_cover' in df.columns:
            df['is_clear_sky'] = (df['cloud_cover'] < 20).astype(int)
            features_added += 1

        # Wind power potential (cubic relationship)
        if 'wind_speed' in df.columns:
            df['wind_power_potential'] = df['wind_speed'] ** 3
            features_added += 1

        # Temperature efficiency for solar panels (performance decreases with high temp)
        if 'temperature' in df.columns:
            # Optimal temperature around 25°C
            df['solar_temp_efficiency'] = 1 - 0.004 * (df['temperature'] - 25)
            df['solar_temp_efficiency'] = df['solar_temp_efficiency'].clip(0, 1.2)
            features_added += 1

        log.info(f"Added {features_added} interaction features")
        return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        latitude: float,
        longitude: float,
        include_lags: bool = True,
        include_rolling: bool = True
    ) -> pd.DataFrame:
        """
        Create all features at once

        Args:
            df: Input DataFrame with datetime index
            latitude: Location latitude
            longitude: Location longitude
            include_lags: Include lag features
            include_rolling: Include rolling window features

        Returns:
            DataFrame with all features
        """
        log.info("Starting feature engineering pipeline")

        # Add time features
        df = self.add_time_features(df)

        # Add solar position
        df = self.add_solar_position(df, latitude, longitude)

        # Add interaction features
        df = self.add_interaction_features(df)

        # Add lag features
        if include_lags:
            lag_columns = ['temperature', 'wind_speed', 'solar_irradiance', 'ghi']
            lag_columns = [col for col in lag_columns if col in df.columns]
            if lag_columns:
                df = self.add_lag_features(df, lag_columns, lags=[1, 2, 3, 24])

        # Add rolling features
        if include_rolling:
            rolling_columns = ['temperature', 'wind_speed', 'solar_irradiance', 'ghi']
            rolling_columns = [col for col in rolling_columns if col in df.columns]
            if rolling_columns:
                df = self.add_rolling_features(df, rolling_columns, windows=[3, 6, 12, 24])

        log.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        return df

    # Helper methods

    @staticmethod
    def _get_season(month: int) -> int:
        """Get season from month (0=Winter, 1=Spring, 2=Summer, 3=Fall)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    @staticmethod
    def _calculate_solar_position(timestamp: datetime, latitude: float, longitude: float):
        """
        Calculate solar elevation and azimuth
        Simplified calculation - for production, consider using pvlib or pysolar
        """
        # Day of year
        day_of_year = timestamp.timetuple().tm_yday

        # Solar declination (degrees)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

        # Hour angle
        hour = timestamp.hour + timestamp.minute / 60.0
        hour_angle = 15 * (hour - 12)

        # Convert to radians
        lat_rad = np.radians(latitude)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)

        # Solar elevation
        sin_elevation = (np.sin(lat_rad) * np.sin(dec_rad) +
                        np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
        elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))

        # Solar azimuth
        cos_azimuth = ((np.sin(dec_rad) - np.sin(lat_rad) * sin_elevation) /
                      (np.cos(lat_rad) * np.cos(np.radians(elevation))))
        azimuth = np.degrees(np.arccos(np.clip(cos_azimuth, -1, 1)))

        if hour_angle > 0:
            azimuth = 360 - azimuth

        return elevation, azimuth

    @staticmethod
    def _calculate_heat_index(temperature: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index (feels-like temperature)"""
        # Simplified heat index calculation
        T = temperature
        RH = humidity
        HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
        return HI

    @staticmethod
    def _calculate_wind_chill(temperature: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind chill temperature"""
        # Wind chill formula (valid for T ≤ 10°C and wind speed > 4.8 km/h)
        T = temperature
        V = wind_speed * 3.6  # Convert m/s to km/h
        WC = 13.12 + 0.6215 * T - 11.37 * (V ** 0.16) + 0.3965 * T * (V ** 0.16)
        # Only apply when conditions are appropriate
        WC = np.where((T <= 10) & (V > 4.8), WC, T)
        return WC
