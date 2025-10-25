"""
Predictor for renewable energy forecasting
Handles real-time inference with trained models
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import os
from datetime import datetime, timedelta

from ..models.base_model import BaseModel
from ..models.model_utils import load_model, get_device
from ..preprocessing import DataLoader, FeatureEngineer, DataScaler, TargetScaler
from ..preprocessing.sequence_generator import OnlineSequenceGenerator
from ..utils.logger import log


class Predictor:
    """
    Real-time predictor for renewable energy forecasting
    """

    def __init__(
        self,
        model: BaseModel,
        feature_scaler: DataScaler,
        target_scaler: TargetScaler,
        sequence_length: int,
        prediction_horizon: int,
        feature_columns: List[str],
        latitude: float,
        longitude: float,
        device: Optional[torch.device] = None
    ):
        """
        Initialize predictor

        Args:
            model: Trained model
            feature_scaler: Fitted feature scaler
            target_scaler: Fitted target scaler
            sequence_length: Input sequence length
            prediction_horizon: Prediction horizon
            feature_columns: List of feature column names
            latitude: Location latitude
            longitude: Location longitude
            device: Computing device
        """
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns
        self.latitude = latitude
        self.longitude = longitude

        # Device
        self.device = device if device is not None else get_device()
        self.model = self.model.to(self.device)
        self.model.eval()

        # Feature engineer for processing new data
        self.feature_engineer = FeatureEngineer()

        # Online sequence generator for real-time prediction
        self.online_generator = OnlineSequenceGenerator(sequence_length)

        log.info(f"Predictor initialized - Device: {self.device}")

    @torch.no_grad()
    def predict(self, input_data: pd.DataFrame) -> Dict:
        """
        Make prediction from input data

        Args:
            input_data: DataFrame with recent weather data

        Returns:
            Dictionary with predictions and metadata
        """
        # Ensure we have enough data
        if len(input_data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} records, got {len(input_data)}"
            )

        # Take last sequence_length records
        input_data = input_data.iloc[-self.sequence_length:].copy()

        # Add features
        input_data = self._add_features(input_data)

        # Ensure all required features exist
        missing_features = set(self.feature_columns) - set(input_data.columns)
        if missing_features:
            log.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                input_data[feature] = 0

        # Select and order features
        features = input_data[self.feature_columns].values

        # Scale features
        features_scaled = self.feature_scaler.transform(
            pd.DataFrame(features, columns=self.feature_columns)
        ).values

        # Convert to tensor
        x = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        # Shape: (1, sequence_length, n_features)

        # Predict
        output = self.model(x)
        # Shape: (1, prediction_horizon)

        # Convert to numpy
        predictions_scaled = output.cpu().numpy().flatten()

        # Inverse transform to original scale
        predictions = self.target_scaler.inverse_transform(predictions_scaled)

        # Generate timestamps for predictions
        last_timestamp = input_data.index[-1]
        prediction_timestamps = [
            last_timestamp + timedelta(hours=i+1)
            for i in range(self.prediction_horizon)
        ]

        result = {
            'predictions': predictions,
            'timestamps': prediction_timestamps,
            'input_start': input_data.index[0],
            'input_end': input_data.index[-1],
            'prediction_start': prediction_timestamps[0],
            'prediction_end': prediction_timestamps[-1],
            'prediction_horizon': self.prediction_horizon
        }

        return result

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        input_data: pd.DataFrame,
        n_samples: int = 100
    ) -> Dict:
        """
        Make prediction with uncertainty estimation using Monte Carlo Dropout

        Args:
            input_data: DataFrame with recent weather data
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary with predictions, mean, std, and confidence intervals
        """
        # Enable dropout for uncertainty estimation
        def enable_dropout(model):
            for module in model.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()

        enable_dropout(self.model)

        # Prepare input
        if len(input_data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} records, got {len(input_data)}"
            )

        input_data = input_data.iloc[-self.sequence_length:].copy()
        input_data = self._add_features(input_data)

        missing_features = set(self.feature_columns) - set(input_data.columns)
        if missing_features:
            for feature in missing_features:
                input_data[feature] = 0

        features = input_data[self.feature_columns].values
        features_scaled = self.feature_scaler.transform(
            pd.DataFrame(features, columns=self.feature_columns)
        ).values
        x = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)

        # Multiple forward passes
        predictions_list = []
        for _ in range(n_samples):
            output = self.model(x)
            predictions_scaled = output.cpu().numpy().flatten()
            predictions = self.target_scaler.inverse_transform(predictions_scaled)
            predictions_list.append(predictions)

        # Stack predictions
        all_predictions = np.stack(predictions_list)  # (n_samples, prediction_horizon)

        # Calculate statistics
        mean_prediction = np.mean(all_predictions, axis=0)
        std_prediction = np.std(all_predictions, axis=0)
        lower_bound = np.percentile(all_predictions, 5, axis=0)   # 5th percentile
        upper_bound = np.percentile(all_predictions, 95, axis=0)  # 95th percentile

        # Generate timestamps
        last_timestamp = input_data.index[-1]
        prediction_timestamps = [
            last_timestamp + timedelta(hours=i+1)
            for i in range(self.prediction_horizon)
        ]

        # Disable dropout again
        self.model.eval()

        result = {
            'predictions': mean_prediction,
            'std': std_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'all_samples': all_predictions,
            'timestamps': prediction_timestamps,
            'input_start': input_data.index[0],
            'input_end': input_data.index[-1],
            'prediction_start': prediction_timestamps[0],
            'prediction_end': prediction_timestamps[-1]
        }

        return result

    def predict_next_hour(self, current_weather: Dict) -> float:
        """
        Predict next hour power output from current weather

        Args:
            current_weather: Dictionary with current weather data

        Returns:
            Predicted power output for next hour
        """
        # Add to online buffer
        # This is a simplified version - in production, maintain a rolling buffer
        raise NotImplementedError(
            "Online prediction requires maintaining a sequence buffer. "
            "Use predict() method with recent DataFrame instead."
        )

    def batch_predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for evaluation

        Args:
            data_loader: PyTorch DataLoader

        Returns:
            Tuple of (predictions, targets)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                # Inverse transform
                predictions_scaled = outputs.cpu().numpy()
                predictions = np.array([
                    self.target_scaler.inverse_transform(pred)
                    for pred in predictions_scaled
                ])

                targets_scaled = targets.cpu().numpy()
                targets_original = np.array([
                    self.target_scaler.inverse_transform(target)
                    for target in targets_scaled
                ])

                all_predictions.append(predictions)
                all_targets.append(targets_original)

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        return predictions, targets

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to input data

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with added features
        """
        # Add time features
        df = self.feature_engineer.add_time_features(df)

        # Add solar position
        df = self.feature_engineer.add_solar_position(df, self.latitude, self.longitude)

        # Add interaction features
        df = self.feature_engineer.add_interaction_features(df)

        # Note: Lag and rolling features should be pre-computed in the input data
        # or maintained in a rolling buffer for online prediction

        return df

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        scaler_dir: str,
        location_name: str,
        energy_type: str,
        latitude: float,
        longitude: float,
        device: Optional[torch.device] = None
    ) -> 'Predictor':
        """
        Create predictor from saved checkpoint

        Args:
            checkpoint_path: Path to model checkpoint
            scaler_dir: Directory containing scalers
            location_name: Location name
            energy_type: Energy type
            latitude: Location latitude
            longitude: Location longitude
            device: Computing device

        Returns:
            Initialized Predictor
        """
        log.info(f"Loading predictor from {checkpoint_path}...")

        # Load checkpoint
        device = device if device is not None else get_device()
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get model info from checkpoint
        input_dim = checkpoint['input_dim']
        output_dim = checkpoint['output_dim']
        sequence_length = checkpoint['sequence_length']

        # Recreate model (need to know model type)
        # This is a limitation - we should save model type in checkpoint
        from ..models import create_model
        model_name = checkpoint.get('model_name', 'LSTMModel')

        # Map model class names to types
        model_type_map = {
            'LSTMModel': 'lstm',
            'LSTMAttentionModel': 'lstm_attention',
            'TransformerModel': 'transformer',
            'TimeSeriesTransformer': 'timeseries'
        }
        model_type = model_type_map.get(model_name, 'lstm')

        model = create_model(
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_length=sequence_length
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Load scalers
        feature_scaler_path = os.path.join(
            scaler_dir,
            f'{location_name.replace(" ", "_")}_{energy_type}_feature_scaler.pkl'
        )
        target_scaler_path = os.path.join(
            scaler_dir,
            f'{location_name.replace(" ", "_")}_{energy_type}_target_scaler.pkl'
        )

        feature_scaler = DataScaler().load(feature_scaler_path)
        target_scaler = TargetScaler().load(target_scaler_path)

        # Get feature columns
        feature_columns = feature_scaler.feature_columns

        log.info(f"✓ Predictor loaded successfully")
        log.info(f"  Model: {model_name}")
        log.info(f"  Features: {len(feature_columns)}")
        log.info(f"  Sequence length: {sequence_length}")
        log.info(f"  Prediction horizon: {output_dim}")

        return cls(
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            sequence_length=sequence_length,
            prediction_horizon=output_dim,
            feature_columns=feature_columns,
            latitude=latitude,
            longitude=longitude,
            device=device
        )

    def save_predictions(self, predictions: Dict, output_path: str):
        """
        Save predictions to file

        Args:
            predictions: Prediction dictionary
            output_path: Output file path
        """
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': predictions['timestamps'],
            'predicted_power': predictions['predictions']
        })

        if 'std' in predictions:
            df['std'] = predictions['std']
            df['lower_bound'] = predictions['lower_bound']
            df['upper_bound'] = predictions['upper_bound']

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        log.info(f"Predictions saved to {output_path}")
