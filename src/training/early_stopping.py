"""
Early stopping to prevent overfitting
"""
import numpy as np
from ..utils.logger import log


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = -np.Inf

    def __call__(self, val_score: float, epoch: int) -> bool:
        """
        Check if should stop training

        Args:
            val_score: Validation metric value
            epoch: Current epoch number

        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                log.info(f"Initial best score: {val_score:.6f}")
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                log.info(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    log.info(f"Early stopping triggered! Best epoch: {self.best_epoch}")
        else:
            improvement = score - self.best_score
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                log.info(f"Validation score improved by {improvement:.6f}")

        return self.early_stop

    def state_dict(self) -> dict:
        """Get state dictionary"""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch,
            'val_score_min': self.val_score_min
        }

    def load_state_dict(self, state_dict: dict):
        """Load state dictionary"""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.best_epoch = state_dict['best_epoch']
        self.val_score_min = state_dict['val_score_min']


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving
    """

    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        verbose: bool = True
    ):
        """
        Initialize ReduceLROnPlateau

        Args:
            optimizer: PyTorch optimizer
            mode: 'min' or 'max'
            factor: Factor by which to reduce learning rate
            patience: Number of epochs to wait before reducing
            min_lr: Minimum learning rate
            verbose: Print messages
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

        self.counter = 0
        self.best_score = None

        if mode == 'min':
            self.best_score = np.Inf
        else:
            self.best_score = -np.Inf

    def __call__(self, val_score: float):
        """
        Check if should reduce learning rate

        Args:
            val_score: Validation metric value
        """
        if self.mode == 'min':
            improved = val_score < self.best_score
        else:
            improved = val_score > self.best_score

        if improved:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0

    def _reduce_lr(self):
        """Reduce learning rate"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)

            if new_lr < old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    log.info(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
