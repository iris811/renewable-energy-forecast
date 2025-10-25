"""
Model Evaluator for comprehensive performance assessment
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from src.training.metrics import calculate_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and analysis
    """

    def __init__(self, model, device: str = 'cpu'):
        """
        Initialize evaluator

        Args:
            model: Trained model
            device: Device to use (cpu/cuda)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.predictions = []
        self.actuals = []
        self.timestamps = []

    def evaluate(
        self,
        dataloader,
        scaler=None,
        save_predictions: bool = True,
        output_dir: str = './evaluation_results'
    ) -> Dict:
        """
        Evaluate model on dataset

        Args:
            dataloader: DataLoader with test data
            scaler: Scaler for inverse transform
            save_predictions: Whether to save predictions
            output_dir: Directory to save results

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting model evaluation...")

        all_predictions = []
        all_actuals = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Store predictions and actuals
                all_predictions.append(outputs.cpu().numpy())
                all_actuals.append(targets.cpu().numpy())

        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        actuals = np.concatenate(all_actuals, axis=0)

        # Inverse transform if scaler provided
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
            actuals = scaler.inverse_transform(actuals)

        # Store for later use
        self.predictions = predictions
        self.actuals = actuals

        # Calculate metrics
        metrics = self._calculate_all_metrics(predictions, actuals)

        # Save results if requested
        if save_predictions:
            self._save_results(predictions, actuals, metrics, output_dir)

        logger.info(f"Evaluation completed. RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")

        return metrics

    def _calculate_all_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive metrics

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            Dictionary with all metrics
        """
        # Convert to tensors for metric calculation
        pred_tensor = torch.from_numpy(predictions).float()
        actual_tensor = torch.from_numpy(actuals).float()

        # Basic metrics
        metrics = calculate_metrics(pred_tensor, actual_tensor)

        # Additional metrics per horizon
        if predictions.shape[1] > 1:  # Multi-step prediction
            horizon_metrics = self._calculate_horizon_metrics(predictions, actuals)
            metrics['horizon_metrics'] = horizon_metrics

        # Error distribution
        errors = predictions - actuals
        metrics['error_stats'] = {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'q25': float(np.percentile(errors, 25)),
            'q50': float(np.percentile(errors, 50)),
            'q75': float(np.percentile(errors, 75))
        }

        return metrics

    def _calculate_horizon_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict:
        """
        Calculate metrics for each prediction horizon

        Args:
            predictions: Predictions [batch, horizon]
            actuals: Actuals [batch, horizon]

        Returns:
            Dictionary with metrics per horizon
        """
        horizon_metrics = {}
        num_horizons = predictions.shape[1]

        for h in range(num_horizons):
            pred_h = torch.from_numpy(predictions[:, h:h+1]).float()
            actual_h = torch.from_numpy(actuals[:, h:h+1]).float()

            metrics_h = calculate_metrics(pred_h, actual_h)
            horizon_metrics[f'h_{h+1}'] = metrics_h

        return horizon_metrics

    def _save_results(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        metrics: Dict,
        output_dir: str
    ):
        """
        Save evaluation results to files

        Args:
            predictions: Predicted values
            actuals: Actual values
            metrics: Calculated metrics
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save predictions vs actuals
        results_df = pd.DataFrame({
            'actual': actuals.flatten()[:len(predictions.flatten())],
            'predicted': predictions.flatten()
        })
        results_df['error'] = results_df['predicted'] - results_df['actual']
        results_df['abs_error'] = np.abs(results_df['error'])
        results_df['pct_error'] = (results_df['error'] / results_df['actual']) * 100

        results_file = output_path / f'predictions_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        logger.info(f"Predictions saved to {results_file}")

        # Save metrics
        metrics_file = output_path / f'metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

        # Save summary report
        self._save_summary_report(metrics, output_path, timestamp)

    def _save_summary_report(
        self,
        metrics: Dict,
        output_path: Path,
        timestamp: str
    ):
        """
        Save human-readable summary report

        Args:
            metrics: Evaluation metrics
            output_path: Output directory
            timestamp: Timestamp string
        """
        report_file = output_path / f'evaluation_report_{timestamp}.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Overall Performance Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"RMSE:  {metrics.get('rmse', 0):.6f}\n")
            f.write(f"MAE:   {metrics.get('mae', 0):.6f}\n")
            f.write(f"MAPE:  {metrics.get('mape', 0):.2f}%\n")
            f.write(f"R²:    {metrics.get('r2', 0):.6f}\n\n")

            # Error statistics
            if 'error_stats' in metrics:
                f.write("Error Distribution:\n")
                f.write("-" * 80 + "\n")
                stats = metrics['error_stats']
                f.write(f"Mean Error:   {stats['mean']:.6f}\n")
                f.write(f"Std Error:    {stats['std']:.6f}\n")
                f.write(f"Min Error:    {stats['min']:.6f}\n")
                f.write(f"Max Error:    {stats['max']:.6f}\n")
                f.write(f"25th Percentile: {stats['q25']:.6f}\n")
                f.write(f"Median Error:    {stats['q50']:.6f}\n")
                f.write(f"75th Percentile: {stats['q75']:.6f}\n\n")

            # Horizon-specific metrics
            if 'horizon_metrics' in metrics:
                f.write("Performance by Prediction Horizon:\n")
                f.write("-" * 80 + "\n")
                for horizon, h_metrics in metrics['horizon_metrics'].items():
                    f.write(f"\n{horizon}:\n")
                    f.write(f"  RMSE: {h_metrics.get('rmse', 0):.6f}\n")
                    f.write(f"  MAE:  {h_metrics.get('mae', 0):.6f}\n")
                    f.write(f"  MAPE: {h_metrics.get('mape', 0):.2f}%\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Evaluation report saved to {report_file}")

    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        dataloader,
        scaler=None,
        output_dir: str = './evaluation_results'
    ) -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            models: Dictionary of model_name -> model
            dataloader: Test dataloader
            scaler: Scaler for inverse transform
            output_dir: Output directory

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models...")

        results = []

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")

            # Temporarily set model
            original_model = self.model
            self.model = model
            self.model.to(self.device)
            self.model.eval()

            # Evaluate
            metrics = self.evaluate(
                dataloader,
                scaler=scaler,
                save_predictions=False
            )

            # Store results
            results.append({
                'model': model_name,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'r2': metrics['r2']
            })

            # Restore original model
            self.model = original_model

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('rmse')

        # Save comparison
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = output_path / f'model_comparison_{timestamp}.csv'
        comparison_df.to_csv(comparison_file, index=False)

        logger.info(f"Model comparison saved to {comparison_file}")
        logger.info(f"\nBest model: {comparison_df.iloc[0]['model']} "
                   f"(RMSE: {comparison_df.iloc[0]['rmse']:.6f})")

        return comparison_df
