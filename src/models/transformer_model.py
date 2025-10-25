"""
Transformer-based models for renewable energy forecasting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .attention import MultiHeadAttention, PositionalEncoding
from ..utils.logger import log


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize encoder layer

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()

        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, mask)

        # Feed-forward
        residual = attn_output
        ff_output = self.feed_forward(attn_output)
        ff_output = self.dropout(ff_output)
        output = self.layer_norm(ff_output + residual)

        return output


class TransformerModel(BaseModel):
    """
    Transformer model for time series forecasting
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        output_dim: int,
        sequence_length: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        """
        Initialize Transformer model

        Args:
            input_dim: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            output_dim: Number of output values
            sequence_length: Length of input sequence
            dropout: Dropout rate
            max_len: Maximum sequence length for positional encoding
        """
        super(TransformerModel, self).__init__(input_dim, output_dim, sequence_length)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        log.info(f"Initialized Transformer model with {self.get_num_parameters():,} parameters")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            mask: Optional mask

        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)

        # Use last time step or average pooling
        # Option 1: Last time step
        # x = x[:, -1, :]  # (batch_size, d_model)

        # Option 2: Average pooling (better for long sequences)
        x = torch.mean(x, dim=1)  # (batch_size, d_model)

        # Output projection
        output = self.output_projection(x)  # (batch_size, output_dim)

        return output


class TimeSeriesTransformer(BaseModel):
    """
    Simplified Transformer for time series
    Optimized for renewable energy forecasting
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        output_dim: int,
        sequence_length: int,
        dropout: float = 0.1
    ):
        """
        Initialize Time Series Transformer

        Args:
            input_dim: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            output_dim: Number of output values
            sequence_length: Length of input sequence
            dropout: Dropout rate
        """
        super(TimeSeriesTransformer, self).__init__(input_dim, output_dim, sequence_length)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, sequence_length, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        log.info(f"Initialized TimeSeriesTransformer with {self.get_num_parameters():,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Embed input
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, d_model)

        # Output
        output = self.output_layer(x)  # (batch_size, output_dim)

        return output


class InformerModel(BaseModel):
    """
    Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
    Simplified implementation focusing on ProbSparse self-attention
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        output_dim: int,
        sequence_length: int,
        dropout: float = 0.1,
        factor: int = 5
    ):
        """
        Initialize Informer model

        Args:
            input_dim: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            output_dim: Number of output values
            sequence_length: Length of input sequence
            dropout: Dropout rate
            factor: Sampling factor for ProbSparse attention
        """
        super(InformerModel, self).__init__(input_dim, output_dim, sequence_length)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.factor = factor

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, sequence_length, dropout)

        # Encoder layers with distillation
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

        # Distillation layers (Conv1d for downsampling)
        self.distillation_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
            for _ in range(n_layers - 1)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        log.info(f"Initialized Informer model with {self.get_num_parameters():,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Encoder layers with distillation
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x)

            # Apply distillation (downsampling) except for last layer
            if i < len(self.encoder_layers) - 1:
                # Transpose for Conv1d: (batch, d_model, seq_len)
                x_transposed = x.transpose(1, 2)
                x_distilled = self.distillation_layers[i](x_transposed)
                x = x_distilled.transpose(1, 2)  # Back to (batch, seq_len, d_model)

        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, d_model)

        # Output projection
        output = self.output_projection(x)  # (batch_size, output_dim)

        return output


def create_transformer_model(
    input_dim: int,
    output_dim: int,
    sequence_length: int,
    model_type: str = 'transformer',
    **kwargs
) -> BaseModel:
    """
    Factory function to create Transformer models

    Args:
        input_dim: Number of input features
        output_dim: Number of output values
        sequence_length: Length of input sequence
        model_type: Type of model ('transformer', 'timeseries', 'informer')
        **kwargs: Additional model-specific parameters

    Returns:
        Initialized model
    """
    models = {
        'transformer': TransformerModel,
        'timeseries': TimeSeriesTransformer,
        'informer': InformerModel
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    model_class = models[model_type]
    model = model_class(
        input_dim=input_dim,
        output_dim=output_dim,
        sequence_length=sequence_length,
        **kwargs
    )

    log.info(f"Created {model_type} model")
    return model
