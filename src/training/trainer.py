"""
Trainer class for model training
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
import os
from tqdm import tqdm
import time

from ..models.base_model import BaseModel
from ..models.model_utils import get_device
from .losses import get_loss_function
from .metrics import OnlineMetrics, MetricsCalculator
from .early_stopping import EarlyStopping, ReduceLROnPlateau
from ..utils.logger import log


class Trainer:
    """
    Trainer for renewable energy forecasting models
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize trainer

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Computing device
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = device if device is not None else get_device()
        self.model = self.model.to(self.device)

        # Configuration
        self.config = config or {}

        # Loss function
        if criterion is None:
            loss_name = self.config.get('loss', 'mse')
            self.criterion = get_loss_function(loss_name)
        else:
            self.criterion = criterion

        # Optimizer
        if optimizer is None:
            lr = self.config.get('learning_rate', 0.001)
            weight_decay = self.config.get('weight_decay', 0.0)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer

        # Early stopping
        early_stop_config = self.config.get('early_stopping', {})
        if early_stop_config.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=early_stop_config.get('patience', 10),
                min_delta=early_stop_config.get('min_delta', 0.0001),
                mode='min',
                verbose=True
            )
        else:
            self.early_stopping = None

        # Learning rate scheduler
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')

        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-7)
            )
        else:
            self.scheduler = None

        # Metrics
        self.metrics_calculator = MetricsCalculator()
        self.train_metrics = OnlineMetrics()
        self.val_metrics = OnlineMetrics()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # Checkpoints
        checkpoint_config = self.config.get('checkpoint', {})
        self.save_best = checkpoint_config.get('save_best', True)
        self.save_last = checkpoint_config.get('save_last', True)
        self.checkpoint_dir = checkpoint_config.get('save_dir', 'models/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.current_epoch = 0

        log.info(f"Trainer initialized - Device: {self.device}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional)
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Update metrics
            self.train_metrics.update(targets, outputs)

            # Update progress bar
            current_metrics = self.train_metrics.compute()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                rmse=f"{current_metrics.get('rmse', 0):.4f}"
            )

        # Get epoch metrics
        epoch_metrics = self.train_metrics.compute()
        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        for inputs, targets in self.val_loader:
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            # Update metrics
            self.val_metrics.update(targets, outputs)

        # Get validation metrics
        val_metrics = self.val_metrics.compute()
        return val_metrics

    def train(self, epochs: int, resume_from: Optional[str] = None) -> Dict:
        """
        Train model for multiple epochs

        Args:
            epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from

        Returns:
            Training history
        """
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            log.info(f"Resumed training from epoch {self.current_epoch}")

        log.info("=" * 60)
        log.info(f"Starting training for {epochs} epochs")
        log.info("=" * 60)

        start_epoch = self.current_epoch
        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['rmse'])
            self.history['val_loss'].append(val_metrics['rmse'])
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            # Log epoch results
            log.info(
                f"Epoch {self.current_epoch}/{end_epoch} | "
                f"Train RMSE: {train_metrics['rmse']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Learning rate scheduler
            if self.scheduler is not None:
                self.scheduler(val_metrics['rmse'])

            # Save best model
            if self.save_best and val_metrics['rmse'] < self.best_val_loss:
                self.best_val_loss = val_metrics['rmse']
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                self.save_checkpoint(best_path, is_best=True)
                log.info(f"Saved best model (Val RMSE: {self.best_val_loss:.4f})")

            # Save last model
            if self.save_last:
                last_path = os.path.join(self.checkpoint_dir, 'last_model.pth')
                self.save_checkpoint(last_path, is_best=False)

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['rmse'], self.current_epoch):
                    log.info("Early stopping triggered!")
                    break

        log.info("=" * 60)
        log.info("Training completed!")
        log.info("=" * 60)

        return self.history

    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        Save training checkpoint

        Args:
            path: Save path
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.early_stopping is not None:
            checkpoint['early_stopping'] = self.early_stopping.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint

        Args:
            path: Checkpoint path
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if 'early_stopping' in checkpoint and self.early_stopping is not None:
            self.early_stopping.load_state_dict(checkpoint['early_stopping'])

        log.info(f"Loaded checkpoint from {path}")

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with test metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        log.info("Evaluating model on test set...")

        for inputs, targets in tqdm(test_loader, desc="Testing"):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            # Collect predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # Concatenate all batches
        import numpy as np
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all(all_targets, all_predictions)

        # Print metrics
        self.metrics_calculator.print_metrics(metrics, title="Test Metrics")

        return metrics

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Generate predictions

        Args:
            data_loader: Data loader

        Returns:
            Predictions tensor
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu())

        return torch.cat(predictions, dim=0)
