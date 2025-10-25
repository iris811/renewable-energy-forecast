"""
Data scaling and normalization utilities
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from ..utils.logger import log


class DataScaler:
    """
    Handles data scaling and normalization
    Supports fit-transform pattern for train/test consistency
    """

    def __init__(self, method: str = 'standard'):
        """
        Initialize scaler

        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scalers: Dict[str, any] = {}
        self.feature_columns: List[str] = []

    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> 'DataScaler':
        """
        Fit scalers on data

        Args:
            df: Input DataFrame
            columns: Columns to scale (None = all numeric columns)

        Returns:
            Self for chaining
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_columns = columns

        for column in columns:
            if column not in df.columns:
                log.warning(f"Column {column} not found in DataFrame")
                continue

            # Create appropriate scaler
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()
            elif self.method == 'robust':
                scaler = RobustScaler()
            else:
                log.error(f"Unknown scaling method: {self.method}")
                continue

            # Fit scaler
            data = df[column].values.reshape(-1, 1)
            scaler.fit(data)
            self.scalers[column] = scaler

        log.info(f"Fitted {len(self.scalers)} scalers using {self.method} method")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scalers

        Args:
            df: Input DataFrame

        Returns:
            Scaled DataFrame
        """
        if not self.scalers:
            log.error("Scalers not fitted. Call fit() first.")
            return df

        df = df.copy()

        for column, scaler in self.scalers.items():
            if column not in df.columns:
                log.warning(f"Column {column} not found in DataFrame")
                continue

            data = df[column].values.reshape(-1, 1)
            df[column] = scaler.transform(data).flatten()

        return df

    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform in one step

        Args:
            df: Input DataFrame
            columns: Columns to scale

        Returns:
            Scaled DataFrame
        """
        self.fit(df, columns)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale

        Args:
            df: Scaled DataFrame
            columns: Columns to inverse transform (None = all fitted columns)

        Returns:
            DataFrame in original scale
        """
        if not self.scalers:
            log.error("Scalers not fitted. Cannot inverse transform.")
            return df

        df = df.copy()
        columns = columns or list(self.scalers.keys())

        for column in columns:
            if column not in self.scalers:
                log.warning(f"No scaler found for column {column}")
                continue

            if column not in df.columns:
                log.warning(f"Column {column} not found in DataFrame")
                continue

            scaler = self.scalers[column]
            data = df[column].values.reshape(-1, 1)
            df[column] = scaler.inverse_transform(data).flatten()

        return df

    def save(self, path: str):
        """
        Save fitted scalers to file

        Args:
            path: File path to save scalers
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            'method': self.method,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        log.info(f"Saved scalers to {path}")

    def load(self, path: str) -> 'DataScaler':
        """
        Load fitted scalers from file

        Args:
            path: File path to load scalers from

        Returns:
            Self for chaining
        """
        if not os.path.exists(path):
            log.error(f"Scaler file not found: {path}")
            return self

        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.method = state['method']
        self.scalers = state['scalers']
        self.feature_columns = state['feature_columns']

        log.info(f"Loaded scalers from {path}")
        return self

    def get_scaler_params(self, column: str) -> Dict:
        """
        Get parameters of a fitted scaler

        Args:
            column: Column name

        Returns:
            Dictionary with scaler parameters
        """
        if column not in self.scalers:
            log.warning(f"No scaler found for column {column}")
            return {}

        scaler = self.scalers[column]
        params = {}

        if isinstance(scaler, StandardScaler):
            params['mean'] = scaler.mean_[0]
            params['std'] = scaler.scale_[0]
        elif isinstance(scaler, MinMaxScaler):
            params['min'] = scaler.min_[0]
            params['scale'] = scaler.scale_[0]
        elif isinstance(scaler, RobustScaler):
            params['center'] = scaler.center_[0]
            params['scale'] = scaler.scale_[0]

        return params


class TargetScaler:
    """
    Special scaler for target variable (power output)
    """

    def __init__(self, method: str = 'standard'):
        """
        Initialize target scaler

        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        self.method = method
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def fit(self, y: np.ndarray) -> 'TargetScaler':
        """
        Fit scaler on target values

        Args:
            y: Target values

        Returns:
            Self for chaining
        """
        y = np.array(y).reshape(-1, 1)
        self.scaler.fit(y)
        log.info(f"Fitted target scaler using {self.method} method")
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform target values

        Args:
            y: Target values

        Returns:
            Scaled target values
        """
        y = np.array(y).reshape(-1, 1)
        return self.scaler.transform(y).flatten()

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            y: Target values

        Returns:
            Scaled target values
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled target back to original scale

        Args:
            y: Scaled target values

        Returns:
            Original scale target values
        """
        y = np.array(y).reshape(-1, 1)
        return self.scaler.inverse_transform(y).flatten()

    def save(self, path: str):
        """
        Save fitted scaler to file

        Args:
            path: File path to save scaler
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            'method': self.method,
            'scaler': self.scaler
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        log.info(f"Saved target scaler to {path}")

    def load(self, path: str) -> 'TargetScaler':
        """
        Load fitted scaler from file

        Args:
            path: File path to load scaler from

        Returns:
            Self for chaining
        """
        if not os.path.exists(path):
            log.error(f"Target scaler file not found: {path}")
            return self

        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.method = state['method']
        self.scaler = state['scaler']

        log.info(f"Loaded target scaler from {path}")
        return self
