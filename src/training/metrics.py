"""
Evaluation metrics for renewable energy forecasting
"""
import numpy as np
import torch
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MetricsCalculator:
    """
    Calculate various evaluation metrics
    """

    def __init__(self):
        """Initialize metrics calculator"""
        self.metrics = {}

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            MAE value
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Mean Absolute Percentage Error

        Args:
            y_true: Ground truth
            y_pred: Predictions
            epsilon: Small value to avoid division by zero

        Returns:
            MAPE value (percentage)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        return mape

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Symmetric Mean Absolute Percentage Error

        Args:
            y_true: Ground truth
            y_pred: Predictions
            epsilon: Small value to avoid division by zero

        Returns:
            SMAPE value (percentage)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        smape = np.mean(numerator / denominator) * 100
        return smape

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R-squared (coefficient of determination)

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            R2 score
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Normalized Root Mean Squared Error (normalized by range)

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            NRMSE value (percentage)
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        range_val = np.max(y_true) - np.min(y_true)
        if range_val == 0:
            return 0.0
        nrmse = (rmse / range_val) * 100
        return nrmse

    @staticmethod
    def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Bias Error (indicates systematic over/under-prediction)

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            MBE value (positive: over-prediction, negative: under-prediction)
        """
        return np.mean(y_pred - y_true)

    @staticmethod
    def nmbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Normalized Mean Bias Error

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            NMBE value (percentage)
        """
        mbe = np.mean(y_pred - y_true)
        mean_true = np.mean(y_true)
        if mean_true == 0:
            return 0.0
        nmbe = (mbe / mean_true) * 100
        return nmbe

    @staticmethod
    def skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> float:
        """
        Forecast Skill Score compared to baseline

        Args:
            y_true: Ground truth
            y_pred: Model predictions
            y_baseline: Baseline predictions (e.g., persistence)

        Returns:
            Skill score (1: perfect, 0: same as baseline, <0: worse than baseline)
        """
        mse_pred = mean_squared_error(y_true, y_pred)
        mse_baseline = mean_squared_error(y_true, y_baseline)

        if mse_baseline == 0:
            return 1.0

        skill = 1 - (mse_pred / mse_baseline)
        return skill

    def calculate_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all metrics

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            Dictionary of all metrics
        """
        # Convert to numpy if tensor
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        # Flatten if multi-dimensional
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        metrics = {
            'rmse': self.rmse(y_true, y_pred),
            'mae': self.mae(y_true, y_pred),
            'mape': self.mape(y_true, y_pred),
            'smape': self.smape(y_true, y_pred),
            'r2': self.r2(y_true, y_pred),
            'nrmse': self.nrmse(y_true, y_pred),
            'mbe': self.mbe(y_true, y_pred),
            'nmbe': self.nmbe(y_true, y_pred)
        }

        return metrics

    def calculate_per_horizon(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon_names: Optional[list] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each prediction horizon

        Args:
            y_true: Ground truth (n_samples, n_horizons)
            y_pred: Predictions (n_samples, n_horizons)
            horizon_names: Names for each horizon (e.g., ['1h', '2h', ..., '24h'])

        Returns:
            Dictionary of horizon -> metrics
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        n_horizons = y_true.shape[1] if len(y_true.shape) > 1 else 1

        if horizon_names is None:
            horizon_names = [f'h{i+1}' for i in range(n_horizons)]

        results = {}

        if len(y_true.shape) == 1:
            # Single horizon
            results[horizon_names[0]] = self.calculate_all(y_true, y_pred)
        else:
            # Multiple horizons
            for i, name in enumerate(horizon_names):
                results[name] = self.calculate_all(y_true[:, i], y_pred[:, i])

        return results

    def print_metrics(self, metrics: Dict[str, float], title: str = "Metrics"):
        """
        Pretty print metrics

        Args:
            metrics: Dictionary of metrics
            title: Title for the metrics table
        """
        print(f"\n{'=' * 50}")
        print(f"{title:^50}")
        print('=' * 50)

        for name, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{name}:")
                for sub_name, sub_value in value.items():
                    print(f"  {sub_name:<20} {sub_value:>12.4f}")
            else:
                print(f"{name:<20} {value:>12.4f}")

        print('=' * 50 + '\n')


class OnlineMetrics:
    """
    Track metrics during training in an online fashion
    """

    def __init__(self):
        """Initialize online metrics tracker"""
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.n_samples = 0
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0
        self.sum_true = 0.0
        self.sum_pred = 0.0
        self.sum_squared_true = 0.0

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Update metrics with new batch

        Args:
            y_true: Ground truth
            y_pred: Predictions
        """
        batch_size = y_true.size(0)
        self.n_samples += batch_size

        # Calculate errors
        squared_error = ((y_pred - y_true) ** 2).sum().item()
        absolute_error = torch.abs(y_pred - y_true).sum().item()

        self.sum_squared_error += squared_error
        self.sum_absolute_error += absolute_error
        self.sum_true += y_true.sum().item()
        self.sum_pred += y_pred.sum().item()
        self.sum_squared_true += (y_true ** 2).sum().item()

    def compute(self) -> Dict[str, float]:
        """
        Compute current metrics

        Returns:
            Dictionary of metrics
        """
        if self.n_samples == 0:
            return {}

        mse = self.sum_squared_error / self.n_samples
        rmse = np.sqrt(mse)
        mae = self.sum_absolute_error / self.n_samples

        return {
            'rmse': rmse,
            'mae': mae,
            'mse': mse
        }

    def get_current_rmse(self) -> float:
        """Get current RMSE"""
        if self.n_samples == 0:
            return float('inf')
        mse = self.sum_squared_error / self.n_samples
        return np.sqrt(mse)

    def get_current_mae(self) -> float:
        """Get current MAE"""
        if self.n_samples == 0:
            return float('inf')
        return self.sum_absolute_error / self.n_samples


def compare_models_metrics(
    models_predictions: Dict[str, tuple],
    metric_names: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare metrics across multiple models

    Args:
        models_predictions: Dictionary of model_name -> (y_true, y_pred)
        metric_names: List of metric names to calculate

    Returns:
        Dictionary of model_name -> metrics
    """
    calculator = MetricsCalculator()
    results = {}

    for model_name, (y_true, y_pred) in models_predictions.items():
        results[model_name] = calculator.calculate_all(y_true, y_pred)

    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """
    Print comparison table for multiple models

    Args:
        results: Dictionary of model_name -> metrics
    """
    if not results:
        return

    # Get all metric names
    metric_names = list(next(iter(results.values())).keys())

    # Header
    print(f"\n{'=' * 80}")
    print(f"{'Model Comparison':^80}")
    print('=' * 80)

    # Column headers
    header = f"{'Model':<20}"
    for metric_name in metric_names:
        header += f"{metric_name:>12}"
    print(header)
    print('-' * 80)

    # Model results
    for model_name, metrics in results.items():
        row = f"{model_name:<20}"
        for metric_name in metric_names:
            value = metrics.get(metric_name, 0.0)
            row += f"{value:>12.4f}"
        print(row)

    print('=' * 80 + '\n')
