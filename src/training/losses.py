"""
Loss functions for renewable energy forecasting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error Loss
    """

    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate RMSE loss

        Args:
            pred: Predictions (batch_size, output_dim)
            target: Ground truth (batch_size, output_dim)

        Returns:
            RMSE loss
        """
        mse = F.mse_loss(pred, target)
        rmse = torch.sqrt(mse)
        return rmse


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss - gives more weight to certain time steps
    Useful for emphasizing near-term predictions
    """

    def __init__(self, weights: Optional[torch.Tensor] = None):
        """
        Initialize weighted MSE loss

        Args:
            weights: Weight for each output timestep (output_dim,)
                    If None, uses exponential decay (more weight on near-term)
        """
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted MSE loss

        Args:
            pred: Predictions (batch_size, output_dim)
            target: Ground truth (batch_size, output_dim)

        Returns:
            Weighted MSE loss
        """
        mse = (pred - target) ** 2

        if self.weights is None:
            # Exponential decay: more weight on near-term predictions
            output_dim = pred.size(1)
            weights = torch.exp(-0.05 * torch.arange(output_dim, dtype=torch.float32))
            weights = weights / weights.sum()
            weights = weights.to(pred.device)
        else:
            weights = self.weights.to(pred.device)

        weighted_mse = (mse * weights).mean()
        return weighted_mse


class QuantileLoss(nn.Module):
    """
    Quantile Loss for probabilistic forecasting
    Useful for uncertainty estimation
    """

    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        """
        Initialize quantile loss

        Args:
            quantiles: List of quantiles to predict
        """
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate quantile loss

        Args:
            pred: Predictions for each quantile (batch_size, output_dim, n_quantiles)
            target: Ground truth (batch_size, output_dim)

        Returns:
            Quantile loss
        """
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target.unsqueeze(-1) - pred[..., i]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())

        return torch.stack(losses).mean()


class HuberLoss(nn.Module):
    """
    Huber Loss - less sensitive to outliers than MSE
    """

    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber loss

        Args:
            delta: Threshold parameter
        """
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Huber loss

        Args:
            pred: Predictions (batch_size, output_dim)
            target: Ground truth (batch_size, output_dim)

        Returns:
            Huber loss
        """
        return F.huber_loss(pred, target, delta=self.delta)


class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error Loss
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize MAPE loss

        Args:
            epsilon: Small value to avoid division by zero
        """
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate MAPE loss

        Args:
            pred: Predictions (batch_size, output_dim)
            target: Ground truth (batch_size, output_dim)

        Returns:
            MAPE loss
        """
        mape = torch.abs((target - pred) / (target + self.epsilon))
        return mape.mean() * 100


class CombinedLoss(nn.Module):
    """
    Combined loss function - weighted sum of multiple losses
    """

    def __init__(
        self,
        loss_functions: dict,
        loss_weights: Optional[dict] = None
    ):
        """
        Initialize combined loss

        Args:
            loss_functions: Dictionary of loss name -> loss function
            loss_weights: Dictionary of loss name -> weight (default: equal weights)
        """
        super(CombinedLoss, self).__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)

        if loss_weights is None:
            loss_weights = {name: 1.0 for name in loss_functions.keys()}

        self.loss_weights = loss_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss

        Args:
            pred: Predictions (batch_size, output_dim)
            target: Ground truth (batch_size, output_dim)

        Returns:
            Combined loss
        """
        total_loss = 0.0

        for name, loss_fn in self.loss_functions.items():
            weight = self.loss_weights.get(name, 1.0)
            loss = loss_fn(pred, target)
            total_loss += weight * loss

        return total_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss - penalizes under-prediction more than over-prediction
    Useful for renewable energy where under-prediction is more problematic
    """

    def __init__(self, beta: float = 2.0):
        """
        Initialize asymmetric loss

        Args:
            beta: Asymmetry factor (>1: penalize under-prediction more)
        """
        super(AsymmetricLoss, self).__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate asymmetric loss

        Args:
            pred: Predictions (batch_size, output_dim)
            target: Ground truth (batch_size, output_dim)

        Returns:
            Asymmetric loss
        """
        errors = pred - target

        # Under-prediction (pred < target): errors < 0
        under_penalty = self.beta * torch.abs(errors)

        # Over-prediction (pred > target): errors > 0
        over_penalty = torch.abs(errors)

        # Apply appropriate penalty
        loss = torch.where(errors < 0, under_penalty, over_penalty)

        return loss.mean()


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name

    Args:
        loss_name: Name of loss function
        **kwargs: Additional parameters for loss function

    Returns:
        Loss function instance
    """
    losses = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'rmse': RMSELoss,
        'huber': HuberLoss,
        'mape': MAPELoss,
        'weighted_mse': WeightedMSELoss,
        'asymmetric': AsymmetricLoss,
        'quantile': QuantileLoss
    }

    if loss_name.lower() not in losses:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(losses.keys())}")

    loss_class = losses[loss_name.lower()]
    return loss_class(**kwargs)
