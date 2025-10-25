"""
Visualization tools for model predictions and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class PredictionVisualizer:
    """
    Visualize model predictions and evaluation results
    """

    def __init__(self, output_dir: str = './evaluation_results/plots'):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_predictions(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamps: Optional[List] = None,
        title: str = "Predictions vs Actuals",
        save_name: Optional[str] = None
    ):
        """
        Plot predictions vs actual values

        Args:
            predictions: Predicted values
            actuals: Actual values
            timestamps: Optional timestamps for x-axis
            title: Plot title
            save_name: Filename to save plot
        """
        # Flatten if multi-step
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = predictions[:, 0]  # Use first horizon
            actuals = actuals[:, 0]
        else:
            predictions = predictions.flatten()
            actuals = actuals.flatten()

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Time series comparison
        x = timestamps if timestamps is not None else range(len(predictions))

        axes[0].plot(x, actuals, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
        axes[0].plot(x, predictions, label='Predicted', color='red', alpha=0.7, linewidth=1.5)
        axes[0].set_xlabel('Time' if timestamps else 'Sample')
        axes[0].set_ylabel('Power Generation')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Scatter plot
        axes[1].scatter(actuals, predictions, alpha=0.5, s=20)
        axes[1].plot([actuals.min(), actuals.max()],
                    [actuals.min(), actuals.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Predicted Values')
        axes[1].set_title('Prediction Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.close()

    def plot_error_analysis(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        save_name: Optional[str] = None
    ):
        """
        Plot error distribution and analysis

        Args:
            predictions: Predicted values
            actuals: Actual values
            save_name: Filename to save plot
        """
        # Flatten arrays
        predictions = predictions.flatten()
        actuals = actuals.flatten()

        # Calculate errors
        errors = predictions - actuals
        abs_errors = np.abs(errors)
        pct_errors = (errors / (actuals + 1e-8)) * 100

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Error distribution
        axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Absolute error over time
        axes[0, 1].plot(abs_errors, alpha=0.6, linewidth=1)
        axes[0, 1].axhline(y=np.mean(abs_errors), color='red',
                          linestyle='--', label=f'Mean: {np.mean(abs_errors):.4f}')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Absolute Error Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Percentage error distribution
        axes[1, 0].hist(pct_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Percentage Error (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Percentage Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Error vs actual values
        axes[1, 1].scatter(actuals, errors, alpha=0.5, s=20)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Prediction Error')
        axes[1, 1].set_title('Error vs Actual Values')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error analysis plot saved to {save_path}")

        plt.close()

    def plot_horizon_performance(
        self,
        horizon_metrics: dict,
        save_name: Optional[str] = None
    ):
        """
        Plot performance metrics across prediction horizons

        Args:
            horizon_metrics: Dictionary with metrics per horizon
            save_name: Filename to save plot
        """
        horizons = []
        rmse_values = []
        mae_values = []
        mape_values = []

        for horizon, metrics in sorted(horizon_metrics.items()):
            horizons.append(horizon.replace('h_', ''))
            rmse_values.append(metrics['rmse'])
            mae_values.append(metrics['mae'])
            mape_values.append(metrics['mape'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # RMSE
        axes[0].plot(horizons, rmse_values, marker='o', linewidth=2)
        axes[0].set_xlabel('Prediction Horizon (hours)')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('RMSE by Horizon')
        axes[0].grid(True, alpha=0.3)

        # MAE
        axes[1].plot(horizons, mae_values, marker='o', linewidth=2, color='green')
        axes[1].set_xlabel('Prediction Horizon (hours)')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('MAE by Horizon')
        axes[1].grid(True, alpha=0.3)

        # MAPE
        axes[2].plot(horizons, mape_values, marker='o', linewidth=2, color='red')
        axes[2].set_xlabel('Prediction Horizon (hours)')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].set_title('MAPE by Horizon')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Horizon performance plot saved to {save_path}")

        plt.close()

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_name: Optional[str] = None
    ):
        """
        Plot comparison of multiple models

        Args:
            comparison_df: DataFrame with model comparison results
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['rmse', 'mae', 'mape', 'r2']
        titles = ['RMSE Comparison', 'MAE Comparison', 'MAPE Comparison', 'R² Comparison']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']

        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[idx // 2, idx % 2]

            sorted_df = comparison_df.sort_values(metric, ascending=(metric != 'r2'))

            ax.barh(sorted_df['model'], sorted_df[metric], color=color, edgecolor='black')
            ax.set_xlabel(metric.upper())
            ax.set_ylabel('Model')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, v in enumerate(sorted_df[metric]):
                ax.text(v, i, f' {v:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.close()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 20,
        save_name: Optional[str] = None
    ):
        """
        Plot feature importance

        Args:
            feature_names: List of feature names
            importance_scores: Importance scores
            top_n: Number of top features to show
            save_name: Filename to save plot
        """
        # Create DataFrame
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })

        # Sort and get top N
        df = df.sort_values('importance', ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(df)), df['importance'], color='steelblue', edgecolor='black')
        plt.yticks(range(len(df)), df['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.close()

    def create_evaluation_dashboard(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        metrics: dict,
        timestamps: Optional[List] = None,
        save_name: str = 'evaluation_dashboard.png'
    ):
        """
        Create comprehensive evaluation dashboard

        Args:
            predictions: Predicted values
            actuals: Actual values
            metrics: Dictionary with metrics
            timestamps: Optional timestamps
            save_name: Filename to save dashboard
        """
        # Flatten arrays
        predictions = predictions.flatten()
        actuals = actuals.flatten()
        errors = predictions - actuals

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Time series comparison
        ax1 = fig.add_subplot(gs[0, :])
        x = timestamps if timestamps else range(len(predictions))
        ax1.plot(x, actuals, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
        ax1.plot(x, predictions, label='Predicted', color='red', alpha=0.7, linewidth=1.5)
        ax1.set_title('Predictions vs Actuals', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time' if timestamps else 'Sample')
        ax1.set_ylabel('Power Generation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(actuals, predictions, alpha=0.5, s=20)
        ax2.plot([actuals.min(), actuals.max()],
                [actuals.min(), actuals.max()],
                'r--', linewidth=2)
        ax2.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.grid(True, alpha=0.3)

        # Error distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # Metrics table
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        metrics_text = f"""
        EVALUATION METRICS

        RMSE:  {metrics.get('rmse', 0):.6f}
        MAE:   {metrics.get('mae', 0):.6f}
        MAPE:  {metrics.get('mape', 0):.2f}%
        R²:    {metrics.get('r2', 0):.6f}

        Error Statistics:
        Mean:  {np.mean(errors):.6f}
        Std:   {np.std(errors):.6f}
        Min:   {np.min(errors):.6f}
        Max:   {np.max(errors):.6f}
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center')

        # Absolute error over time
        ax5 = fig.add_subplot(gs[2, :2])
        abs_errors = np.abs(errors)
        ax5.plot(abs_errors, alpha=0.6, linewidth=1, color='orange')
        ax5.axhline(y=np.mean(abs_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(abs_errors):.4f}')
        ax5.set_title('Absolute Error Over Time', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Sample')
        ax5.set_ylabel('Absolute Error')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Error box plot
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.boxplot([errors], labels=['Errors'])
        ax6.set_title('Error Box Plot', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Error')
        ax6.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Model Evaluation Dashboard',
                    fontsize=16, fontweight='bold', y=0.995)

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Evaluation dashboard saved to {save_path}")

        plt.close()
