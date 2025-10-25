"""
Base model class for renewable energy forecasting
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional
import os
from ..utils.logger import log


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all forecasting models
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sequence_length: int,
        **kwargs
    ):
        """
        Initialize base model

        Args:
            input_dim: Number of input features
            output_dim: Number of output values (prediction horizon)
            sequence_length: Length of input sequence
            **kwargs: Additional model-specific parameters
        """
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        pass

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[Dict] = None, **kwargs):
        """
        Save model checkpoint

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer_state: Optimizer state dict
            **kwargs: Additional information to save
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'sequence_length': self.sequence_length,
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        # Add any additional kwargs
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        log.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict:
        """
        Load model checkpoint

        Args:
            path: Path to checkpoint file
            map_location: Device to map tensors to

        Returns:
            Checkpoint dictionary
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        log.info(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")

        return checkpoint

    def summary(self) -> str:
        """
        Get model summary

        Returns:
            Summary string
        """
        summary = []
        summary.append(f"\n{'='*60}")
        summary.append(f"Model: {self.model_name}")
        summary.append(f"{'='*60}")
        summary.append(f"Input dimension: {self.input_dim}")
        summary.append(f"Output dimension: {self.output_dim}")
        summary.append(f"Sequence length: {self.sequence_length}")
        summary.append(f"Total parameters: {self.get_num_parameters():,}")
        summary.append(f"{'='*60}\n")

        return "\n".join(summary)

    def count_layers(self) -> Dict[str, int]:
        """
        Count number of each layer type

        Returns:
            Dictionary with layer type counts
        """
        layer_counts = {}
        for name, module in self.named_modules():
            module_type = type(module).__name__
            if module_type in layer_counts:
                layer_counts[module_type] += 1
            else:
                layer_counts[module_type] = 1
        return layer_counts

    def freeze_layers(self, freeze: bool = True):
        """
        Freeze or unfreeze all model parameters

        Args:
            freeze: True to freeze, False to unfreeze
        """
        for param in self.parameters():
            param.requires_grad = not freeze

        status = "frozen" if freeze else "unfrozen"
        log.info(f"All model parameters {status}")

    def to_device(self, device: torch.device):
        """
        Move model to device

        Args:
            device: Target device
        """
        self.to(device)
        log.info(f"Model moved to {device}")
        return self


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models
    """

    def __init__(self, models: list, weights: Optional[list] = None):
        """
        Initialize ensemble

        Args:
            models: List of model instances
            weights: Optional weights for each model (default: equal weights)
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        self.weights = torch.tensor(weights, dtype=torch.float32)
        assert len(self.weights) == len(self.models), "Number of weights must match number of models"
        assert abs(self.weights.sum().item() - 1.0) < 1e-6, "Weights must sum to 1.0"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble

        Args:
            x: Input tensor

        Returns:
            Weighted average of model outputs
        """
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)

        # Stack outputs and apply weights
        outputs = torch.stack(outputs, dim=0)  # (n_models, batch_size, output_dim)
        weights = self.weights.view(-1, 1, 1).to(x.device)  # (n_models, 1, 1)

        # Weighted sum
        ensemble_output = (outputs * weights).sum(dim=0)

        return ensemble_output

    def get_num_parameters(self) -> int:
        """Get total number of parameters across all models"""
        return sum(model.get_num_parameters() for model in self.models if hasattr(model, 'get_num_parameters'))
