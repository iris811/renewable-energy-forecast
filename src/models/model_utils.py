"""
Model utility functions and helpers
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Union
import os
import yaml

from .lstm_model import create_lstm_model
from .transformer_model import create_transformer_model
from .base_model import BaseModel
from ..utils.logger import log


def create_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    sequence_length: int,
    config: Optional[Dict] = None,
    config_path: Optional[str] = None
) -> BaseModel:
    """
    Create model from configuration

    Args:
        model_type: Type of model ('lstm', 'transformer', etc.)
        input_dim: Number of input features
        output_dim: Number of output values
        sequence_length: Length of input sequence
        config: Model configuration dictionary
        config_path: Path to configuration file

    Returns:
        Initialized model
    """
    # Load config from file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get('model', {}).get(model_type, {})

    # Default configs for each model type
    default_configs = {
        'lstm': {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': False
        },
        'lstm_attention': {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'attention_dim': 128
        },
        'transformer': {
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 512,
            'dropout': 0.1
        },
        'timeseries': {
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 4,
            'dropout': 0.1
        }
    }

    # Merge with provided config
    if config is None:
        config = {}

    model_config = {**default_configs.get(model_type, {}), **config}

    # Create model based on type
    if model_type in ['lstm', 'lstm_attention', 'multi_output', 'stacked']:
        model = create_lstm_model(
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            model_type=model_type,
            **model_config
        )
    elif model_type in ['transformer', 'timeseries', 'informer']:
        model = create_transformer_model(
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            model_type=model_type,
            **model_config
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def initialize_weights(model: nn.Module, method: str = 'xavier'):
    """
    Initialize model weights

    Args:
        model: PyTorch model
        method: Initialization method ('xavier', 'kaiming', 'normal')
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if method == 'xavier':
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif method == 'kaiming':
                if len(param.shape) >= 2:
                    nn.init.kaiming_uniform_(param)
            elif method == 'normal':
                nn.init.normal_(param, mean=0, std=0.01)
        elif 'bias' in name:
            nn.init.constant_(param, 0)

    log.info(f"Initialized weights using {method} method")


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get computing device

    Args:
        gpu_id: GPU ID (None for auto-detect)

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
        log.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        log.info("CUDA not available, using CPU")

    return device


def save_model(
    model: BaseModel,
    path: str,
    epoch: Optional[int] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    metrics: Optional[Dict] = None
):
    """
    Save model checkpoint

    Args:
        model: Model to save
        path: Save path
        epoch: Current epoch
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        metrics: Training metrics
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
        'sequence_length': model.sequence_length,
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)
    log.info(f"Saved model checkpoint to {path}")


def load_model(
    path: str,
    model: Optional[BaseModel] = None,
    device: Optional[torch.device] = None
) -> Union[BaseModel, Dict]:
    """
    Load model checkpoint

    Args:
        path: Path to checkpoint
        model: Model instance (if None, returns checkpoint dict)
        device: Device to load model to

    Returns:
        Loaded model or checkpoint dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        log.info(f"Loaded model from {path}")
        return model
    else:
        log.info(f"Loaded checkpoint from {path}")
        return checkpoint


def print_model_summary(model: BaseModel):
    """
    Print detailed model summary

    Args:
        model: Model to summarize
    """
    print(model.summary())

    # Parameter counts
    param_counts = count_parameters(model)
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")

    # Layer counts
    layer_counts = model.count_layers()
    print("\nLayer breakdown:")
    for layer_type, count in sorted(layer_counts.items()):
        if count > 1:  # Skip single instance base classes
            print(f"  {layer_type}: {count}")


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate model size in memory

    Args:
        model: PyTorch model

    Returns:
        Dictionary with size in different units
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024 ** 2)
    total_size_gb = total_size_bytes / (1024 ** 3)

    return {
        'bytes': total_size_bytes,
        'mb': total_size_mb,
        'gb': total_size_gb
    }


def compare_models(models: Dict[str, BaseModel]):
    """
    Compare multiple models

    Args:
        models: Dictionary of model name to model instance
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Header
    print(f"{'Model':<20} {'Parameters':<15} {'Size (MB)':<12} {'Type':<20}")
    print("-" * 80)

    # Model details
    for name, model in models.items():
        params = count_parameters(model)
        size = get_model_size(model)

        print(f"{name:<20} {params['total']:>14,} {size['mb']:>11.2f} {model.model_name:<20}")

    print("=" * 80 + "\n")


def freeze_layers(model: nn.Module, layer_names: list):
    """
    Freeze specific layers

    Args:
        model: Model
        layer_names: List of layer names to freeze
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            frozen_count += 1

    log.info(f"Frozen {frozen_count} parameters")


def unfreeze_layers(model: nn.Module, layer_names: list):
    """
    Unfreeze specific layers

    Args:
        model: Model
        layer_names: List of layer names to unfreeze
    """
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            unfrozen_count += 1

    log.info(f"Unfrozen {unfrozen_count} parameters")


def get_gradient_info(model: nn.Module) -> Dict:
    """
    Get gradient statistics

    Args:
        model: Model

    Returns:
        Dictionary with gradient info
    """
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.flatten())

    if not gradients:
        return {}

    all_grads = torch.cat(gradients)

    return {
        'mean': all_grads.mean().item(),
        'std': all_grads.std().item(),
        'min': all_grads.min().item(),
        'max': all_grads.max().item(),
        'norm': all_grads.norm().item()
    }
