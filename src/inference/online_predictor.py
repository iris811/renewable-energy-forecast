"""
Online real-time predictor with rolling buffer
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Deque
from datetime import datetime
from collections import deque

from .predictor import Predictor
from ..utils.logger import log


class OnlinePredictor:
    """
    Real-time predictor that maintains a rolling buffer of recent data
    Suitable for production deployment with streaming data
    """

    def __init__(
        self,
        predictor: Predictor,
        buffer_size: Optional[int] = None
    ):
        """
        Initialize online predictor

        Args:
            predictor: Base predictor instance
            buffer_size: Size of rolling buffer (default: sequence_length)
        """
        self.predictor = predictor
        self.buffer_size = buffer_size or predictor.sequence_length

        # Rolling buffer for recent data
        self.buffer: Deque[Dict] = deque(maxlen=self.buffer_size)

        # Feature columns needed
        self.required_columns = [
            'temperature', 'humidity', 'pressure',
            'wind_speed', 'wind_direction', 'cloud_cover',
            'precipitation', 'solar_irradiance', 'ghi', 'dni', 'dhi'
        ]

        log.info(f"Online predictor initialized with buffer size: {self.buffer_size}")

    def add_observation(self, timestamp: datetime, weather_data: Dict) -> bool:
        """
        Add new observation to buffer

        Args:
            timestamp: Observation timestamp
            weather_data: Dictionary with weather measurements

        Returns:
            True if observation added successfully
        """
        # Validate required fields
        for field in self.required_columns:
            if field not in weather_data:
                log.warning(f"Missing required field: {field}")
                weather_data[field] = None

        # Add timestamp
        weather_data['timestamp'] = timestamp

        # Add to buffer
        self.buffer.append(weather_data)

        return True

    def can_predict(self) -> bool:
        """
        Check if buffer has enough data for prediction

        Returns:
            True if can make prediction
        """
        return len(self.buffer) >= self.buffer_size

    def predict(self) -> Optional[Dict]:
        """
        Make prediction using current buffer

        Returns:
            Prediction dictionary or None if not ready
        """
        if not self.can_predict():
            log.warning(
                f"Not enough data in buffer: {len(self.buffer)}/{self.buffer_size}"
            )
            return None

        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.buffer))
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Make prediction
        try:
            result = self.predictor.predict(df)
            log.info(f"Prediction made at {datetime.now()}")
            return result
        except Exception as e:
            log.error(f"Prediction failed: {e}")
            return None

    def predict_with_uncertainty(self, n_samples: int = 100) -> Optional[Dict]:
        """
        Make prediction with uncertainty estimation

        Args:
            n_samples: Number of Monte Carlo samples

        Returns:
            Prediction dictionary with uncertainty or None if not ready
        """
        if not self.can_predict():
            log.warning(
                f"Not enough data in buffer: {len(self.buffer)}/{self.buffer_size}"
            )
            return None

        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.buffer))
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Make prediction with uncertainty
        try:
            result = self.predictor.predict_with_uncertainty(df, n_samples=n_samples)
            log.info(f"Prediction with uncertainty made at {datetime.now()}")
            return result
        except Exception as e:
            log.error(f"Prediction failed: {e}")
            return None

    def get_buffer_status(self) -> Dict:
        """
        Get current buffer status

        Returns:
            Dictionary with buffer information
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.buffer_size,
                'filled_percentage': 0.0,
                'can_predict': False,
                'oldest': None,
                'newest': None
            }

        return {
            'size': len(self.buffer),
            'capacity': self.buffer_size,
            'filled_percentage': len(self.buffer) / self.buffer_size * 100,
            'can_predict': self.can_predict(),
            'oldest': self.buffer[0]['timestamp'],
            'newest': self.buffer[-1]['timestamp']
        }

    def clear_buffer(self):
        """Clear the buffer"""
        self.buffer.clear()
        log.info("Buffer cleared")

    def get_buffer_data(self) -> pd.DataFrame:
        """
        Get current buffer as DataFrame

        Returns:
            DataFrame with buffer data
        """
        if len(self.buffer) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(list(self.buffer))
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df


class PredictionService:
    """
    High-level prediction service for production use
    """

    def __init__(
        self,
        predictor: Predictor,
        update_interval: int = 3600  # seconds
    ):
        """
        Initialize prediction service

        Args:
            predictor: Base predictor
            update_interval: How often to make new predictions (seconds)
        """
        self.online_predictor = OnlinePredictor(predictor)
        self.update_interval = update_interval

        # Cache for latest prediction
        self.latest_prediction: Optional[Dict] = None
        self.last_prediction_time: Optional[datetime] = None

        log.info(f"Prediction service initialized")

    def update(self, timestamp: datetime, weather_data: Dict) -> bool:
        """
        Update with new weather data

        Args:
            timestamp: Data timestamp
            weather_data: Weather measurements

        Returns:
            True if update successful
        """
        return self.online_predictor.add_observation(timestamp, weather_data)

    def get_prediction(
        self,
        force_update: bool = False,
        with_uncertainty: bool = False
    ) -> Optional[Dict]:
        """
        Get current prediction

        Args:
            force_update: Force new prediction even if cache is recent
            with_uncertainty: Include uncertainty estimation

        Returns:
            Prediction dictionary
        """
        # Check if we need a new prediction
        need_update = (
            force_update or
            self.latest_prediction is None or
            self.last_prediction_time is None or
            (datetime.now() - self.last_prediction_time).total_seconds() > self.update_interval
        )

        if need_update and self.online_predictor.can_predict():
            # Make new prediction
            if with_uncertainty:
                self.latest_prediction = self.online_predictor.predict_with_uncertainty()
            else:
                self.latest_prediction = self.online_predictor.predict()

            self.last_prediction_time = datetime.now()

        return self.latest_prediction

    def get_status(self) -> Dict:
        """
        Get service status

        Returns:
            Status dictionary
        """
        buffer_status = self.online_predictor.get_buffer_status()

        return {
            'service_status': 'ready' if buffer_status['can_predict'] else 'warming_up',
            'buffer': buffer_status,
            'last_prediction': self.last_prediction_time,
            'prediction_available': self.latest_prediction is not None,
            'update_interval': self.update_interval
        }


def simulate_real_time_prediction(
    predictor: Predictor,
    test_data: pd.DataFrame,
    sequence_length: int
) -> Dict:
    """
    Simulate real-time prediction scenario using historical data

    Args:
        predictor: Base predictor
        test_data: Historical test data
        sequence_length: Length of sequence needed

    Returns:
        Dictionary with simulation results
    """
    log.info("Starting real-time prediction simulation...")

    online_pred = OnlinePredictor(predictor)

    # Warm up buffer with initial data
    warmup_data = test_data.iloc[:sequence_length]

    for idx, row in warmup_data.iterrows():
        weather_data = row.to_dict()
        online_pred.add_observation(idx, weather_data)

    log.info(f"Buffer warmed up with {sequence_length} observations")

    # Simulate real-time predictions
    predictions_list = []
    remaining_data = test_data.iloc[sequence_length:]

    for i, (idx, row) in enumerate(remaining_data.iterrows()):
        # Make prediction
        result = online_pred.predict()

        if result:
            predictions_list.append({
                'prediction_time': idx,
                'predictions': result['predictions'],
                'timestamps': result['timestamps']
            })

        # Add new observation
        weather_data = row.to_dict()
        online_pred.add_observation(idx, weather_data)

        if (i + 1) % 100 == 0:
            log.info(f"Processed {i + 1} predictions...")

    log.info(f"✓ Simulation completed: {len(predictions_list)} predictions made")

    return {
        'predictions': predictions_list,
        'total_predictions': len(predictions_list)
    }
