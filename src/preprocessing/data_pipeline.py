"""
Complete data preprocessing pipeline
Integrates all preprocessing steps into a single workflow
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .feature_engineering import FeatureEngineer
from .scaler import DataScaler, TargetScaler
from .sequence_generator import SequenceGenerator, create_dataloaders
from ..utils.logger import log


class DataPipeline:
    """
    Complete preprocessing pipeline for renewable energy forecasting
    """

    def __init__(
        self,
        location_name: str,
        energy_type: str,
        latitude: float,
        longitude: float,
        sequence_length: int = 168,
        prediction_horizon: int = 24,
        config_path: str = 'configs/config.yaml'
    ):
        """
        Initialize data pipeline

        Args:
            location_name: Location name
            energy_type: Energy type ('solar' or 'wind')
            latitude: Location latitude
            longitude: Location longitude
            sequence_length: Length of input sequences (hours)
            prediction_horizon: Prediction horizon (hours)
            config_path: Path to configuration file
        """
        self.location_name = location_name
        self.energy_type = energy_type
        self.latitude = latitude
        self.longitude = longitude
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Initialize components
        self.data_loader = DataLoader(config_path)
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.sequence_generator = SequenceGenerator(sequence_length, prediction_horizon)

        # Scalers (fitted during pipeline)
        self.feature_scaler: Optional[DataScaler] = None
        self.target_scaler: Optional[TargetScaler] = None

        # Data storage
        self.processed_data: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[List[str]] = None

    def load_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load and merge weather and power data

        Args:
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Merged DataFrame
        """
        log.info(f"Loading data for {self.location_name} ({self.energy_type})")

        df = self.data_loader.merge_weather_and_power(
            location_name=self.location_name,
            energy_type=self.energy_type,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            log.error("No data loaded. Check database and filters.")
            return df

        log.info(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df

    def preprocess(
        self,
        df: pd.DataFrame,
        resample_freq: Optional[str] = '1H',
        include_lags: bool = True,
        include_rolling: bool = True
    ) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline

        Args:
            df: Input DataFrame
            resample_freq: Resampling frequency
            include_lags: Include lag features
            include_rolling: Include rolling features

        Returns:
            Preprocessed DataFrame
        """
        log.info("Starting preprocessing pipeline...")

        # Step 1: Data cleaning
        df = self.data_cleaner.clean(
            df,
            fill_missing=True,
            remove_outliers=True,
            validate_ranges=True,
            resample=resample_freq,
            remove_duplicates=True
        )

        if df.empty:
            log.error("Data is empty after cleaning")
            return df

        # Step 2: Feature engineering
        df = self.feature_engineer.create_all_features(
            df,
            latitude=self.latitude,
            longitude=self.longitude,
            include_lags=include_lags,
            include_rolling=include_rolling
        )

        # Remove rows with NaN values created by lag/rolling features
        df = df.dropna()

        log.info(f"Preprocessing completed. Shape: {df.shape}")
        self.processed_data = df

        return df

    def prepare_features(self, df: pd.DataFrame, target_column: str = 'power_output') -> Tuple[List[str], str]:
        """
        Identify feature and target columns

        Args:
            df: Preprocessed DataFrame
            target_column: Name of target column

        Returns:
            (feature_columns, target_column)
        """
        # Exclude non-feature columns
        exclude_columns = [
            target_column, 'location_name', 'energy_type',
            'capacity', 'capacity_factor', 'is_predicted',
            'latitude', 'longitude', 'source'
        ]

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        log.info(f"Selected {len(feature_columns)} features")
        self.feature_columns = feature_columns

        return feature_columns, target_column

    def scale_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features and target

        Args:
            df: Input DataFrame
            feature_columns: Feature column names
            target_column: Target column name
            fit: Whether to fit scalers (True for training, False for inference)

        Returns:
            Scaled DataFrame
        """
        df = df.copy()

        # Scale features
        if fit:
            self.feature_scaler = DataScaler(method='standard')
            df[feature_columns] = self.feature_scaler.fit_transform(df[feature_columns], feature_columns)
        else:
            if self.feature_scaler is None:
                log.error("Feature scaler not fitted")
                return df
            df[feature_columns] = self.feature_scaler.transform(df[feature_columns])

        # Scale target
        if target_column in df.columns:
            if fit:
                self.target_scaler = TargetScaler(method='standard')
                df[target_column] = self.target_scaler.fit_transform(df[target_column].values)
            else:
                if self.target_scaler is None:
                    log.error("Target scaler not fitted")
                    return df
                df[target_column] = self.target_scaler.transform(df[target_column].values)

        log.info("Data scaling completed")
        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict:
        """
        Create sequences and split into train/val/test

        Args:
            df: Preprocessed and scaled DataFrame
            feature_columns: Feature column names
            target_column: Target column name
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Dictionary with train/val/test data
        """
        log.info("Creating sequences...")

        # Extract features and target
        X = df[feature_columns].values
        y = df[target_column].values

        # Create sequences
        X_seq, y_seq = self.sequence_generator.create_sequences(X, y)

        # Split data
        train_data, val_data, test_data = self.sequence_generator.split_train_val_test(
            X_seq, y_seq, train_ratio, val_ratio, test_ratio
        )

        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'n_features': X.shape[1],
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }

    def run_pipeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 32,
        save_scalers: bool = True,
        scaler_dir: str = 'models/scalers'
    ) -> Dict:
        """
        Run complete pipeline

        Args:
            start_date: Start date for data
            end_date: End date for data
            batch_size: Batch size for DataLoaders
            save_scalers: Save fitted scalers
            scaler_dir: Directory to save scalers

        Returns:
            Dictionary with DataLoaders and metadata
        """
        log.info("=" * 60)
        log.info("Starting complete data pipeline")
        log.info("=" * 60)

        # Step 1: Load data
        df = self.load_data(start_date, end_date)
        if df.empty:
            return {}

        # Step 2: Preprocess
        df = self.preprocess(df)
        if df.empty:
            return {}

        # Step 3: Prepare features
        feature_columns, target_column = self.prepare_features(df)

        # Step 4: Scale data
        df = self.scale_data(df, feature_columns, target_column, fit=True)

        # Step 5: Create sequences
        sequences = self.create_sequences(df, feature_columns, target_column)

        # Step 6: Create DataLoaders
        train_loader, val_loader, test_loader = create_dataloaders(
            sequences['train'],
            sequences['val'],
            sequences['test'],
            batch_size=batch_size
        )

        # Save scalers
        if save_scalers and self.feature_scaler and self.target_scaler:
            os.makedirs(scaler_dir, exist_ok=True)
            feature_scaler_path = os.path.join(scaler_dir, f'{self.location_name}_{self.energy_type}_feature_scaler.pkl')
            target_scaler_path = os.path.join(scaler_dir, f'{self.location_name}_{self.energy_type}_target_scaler.pkl')

            self.feature_scaler.save(feature_scaler_path)
            self.target_scaler.save(target_scaler_path)

        log.info("=" * 60)
        log.info("Pipeline completed successfully!")
        log.info("=" * 60)

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'n_features': sequences['n_features'],
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }

    def load_scalers(self, scaler_dir: str = 'models/scalers') -> bool:
        """
        Load previously fitted scalers

        Args:
            scaler_dir: Directory containing scalers

        Returns:
            True if successful, False otherwise
        """
        feature_scaler_path = os.path.join(scaler_dir, f'{self.location_name}_{self.energy_type}_feature_scaler.pkl')
        target_scaler_path = os.path.join(scaler_dir, f'{self.location_name}_{self.energy_type}_target_scaler.pkl')

        if not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
            log.error(f"Scalers not found in {scaler_dir}")
            return False

        self.feature_scaler = DataScaler().load(feature_scaler_path)
        self.target_scaler = TargetScaler().load(target_scaler_path)

        log.info("Loaded scalers successfully")
        return True

    def get_data_summary(self) -> Dict:
        """
        Get summary of processed data

        Returns:
            Dictionary with data summary
        """
        if self.processed_data is None:
            return {}

        return self.data_cleaner.get_data_quality_report(self.processed_data)
