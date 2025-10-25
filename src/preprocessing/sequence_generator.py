"""
Time series sequence generator for LSTM/Transformer models
Creates sliding window sequences from time series data
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from ..utils.logger import log


class SequenceGenerator:
    """
    Generates sequences for time series forecasting
    """

    def __init__(
        self,
        sequence_length: int = 168,  # 1 week in hours
        prediction_horizon: int = 24,  # 24 hours ahead
        stride: int = 1
    ):
        """
        Initialize sequence generator

        Args:
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of time steps to predict ahead
            stride: Step size for sliding window
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride

    def create_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences using sliding window

        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            Tuple of (X_sequences, y_sequences)
            X_sequences shape: (n_sequences, sequence_length, n_features)
            y_sequences shape: (n_sequences, prediction_horizon) or (n_sequences,)
        """
        X_sequences = []
        y_sequences = [] if y is not None else None

        n_samples = len(X)
        max_start = n_samples - self.sequence_length - self.prediction_horizon + 1

        for i in range(0, max_start, self.stride):
            # Input sequence
            X_seq = X[i:i + self.sequence_length]
            X_sequences.append(X_seq)

            # Target sequence
            if y is not None:
                if self.prediction_horizon == 1:
                    # Single-step prediction
                    y_seq = y[i + self.sequence_length]
                else:
                    # Multi-step prediction
                    y_seq = y[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]

                y_sequences.append(y_seq)

        X_sequences = np.array(X_sequences)

        if y_sequences:
            y_sequences = np.array(y_sequences)
            log.info(f"Created {len(X_sequences)} sequences - X: {X_sequences.shape}, y: {y_sequences.shape}")
        else:
            log.info(f"Created {len(X_sequences)} sequences - X: {X_sequences.shape}")

        return X_sequences, y_sequences

    def split_train_val_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split sequences into train, validation, and test sets
        Uses temporal split (no shuffle) to avoid data leakage

        Args:
            X: Input sequences
            y: Target sequences
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set

        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        log.info(f"Split data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset

        Args:
            X: Input sequences (n_sequences, sequence_length, n_features)
            y: Target sequences (n_sequences,) or (n_sequences, prediction_horizon)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloaders(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets

    Args:
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        test_data: (X_test, y_test)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        (train_loader, val_loader, test_loader)
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle time series data
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    log.info(f"Created DataLoaders - Train batches: {len(train_loader)}, "
             f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


class OnlineSequenceGenerator:
    """
    Generator for online/real-time predictions
    """

    def __init__(self, sequence_length: int = 168):
        """
        Initialize online sequence generator

        Args:
            sequence_length: Number of time steps to look back
        """
        self.sequence_length = sequence_length
        self.buffer = []

    def add_sample(self, sample: np.ndarray):
        """
        Add new sample to buffer

        Args:
            sample: New sample (n_features,)
        """
        self.buffer.append(sample)

        # Keep only last sequence_length samples
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)

    def is_ready(self) -> bool:
        """
        Check if buffer has enough samples for prediction

        Returns:
            True if ready, False otherwise
        """
        return len(self.buffer) >= self.sequence_length

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get current sequence from buffer

        Returns:
            Sequence array (sequence_length, n_features) or None if not ready
        """
        if not self.is_ready():
            return None

        return np.array(self.buffer[-self.sequence_length:])

    def reset(self):
        """Reset buffer"""
        self.buffer = []


def prepare_data_for_training(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list,
    sequence_length: int = 168,
    prediction_horizon: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> dict:
    """
    Complete pipeline to prepare data for training

    Args:
        df: Input DataFrame
        target_column: Name of target column
        feature_columns: List of feature column names
        sequence_length: Length of input sequences
        prediction_horizon: Length of prediction horizon
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Dictionary with train, val, test sequences
    """
    log.info("Preparing data for training...")

    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values

    log.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Create sequences
    generator = SequenceGenerator(
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    X_seq, y_seq = generator.create_sequences(X, y)

    # Split data
    train_data, val_data, test_data = generator.split_train_val_test(
        X_seq, y_seq, train_ratio, val_ratio, test_ratio
    )

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'n_features': X.shape[1]
    }
