"""
LSTM-based models for renewable energy forecasting
"""
import torch
import torch.nn as nn
from .base_model import BaseModel
from ..utils.logger import log


class LSTMModel(BaseModel):
    """
    Standard LSTM model for time series forecasting
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        sequence_length: int,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM model

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            output_dim: Number of output values (prediction horizon)
            sequence_length: Length of input sequence
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMModel, self).__init__(input_dim, output_dim, sequence_length)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Direction multiplier
        self.direction_mult = 2 if bidirectional else 1

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * self.direction_mult, output_dim)

        log.info(f"Initialized LSTM model with {self.get_num_parameters():,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.size(0)

        # LSTM forward pass
        # lstm_out: (batch_size, sequence_length, hidden_dim * direction_mult)
        # h_n, c_n: (num_layers * direction_mult, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last time step output
        # If bidirectional, concatenate forward and backward hidden states
        if self.bidirectional:
            # Get last hidden states from both directions
            h_forward = h_n[-2, :, :]  # (batch_size, hidden_dim)
            h_backward = h_n[-1, :, :]  # (batch_size, hidden_dim)
            last_hidden = torch.cat([h_forward, h_backward], dim=1)  # (batch_size, hidden_dim * 2)
        else:
            last_hidden = h_n[-1, :, :]  # (batch_size, hidden_dim)

        # Output layer
        output = self.fc(last_hidden)  # (batch_size, output_dim)

        return output


class LSTMAttentionModel(BaseModel):
    """
    LSTM model with attention mechanism
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        sequence_length: int,
        dropout: float = 0.2,
        attention_dim: int = 128
    ):
        """
        Initialize LSTM with attention

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            output_dim: Number of output values
            sequence_length: Length of input sequence
            dropout: Dropout rate
            attention_dim: Attention mechanism dimension
        """
        super(LSTMAttentionModel, self).__init__(input_dim, output_dim, sequence_length)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_dim = attention_dim

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

        log.info(f"Initialized LSTM-Attention model with {self.get_num_parameters():,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.size(0)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_dim)

        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # Normalize across sequence

        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim)

        # Dropout and output
        context = self.dropout_layer(context)
        output = self.fc(context)  # (batch_size, output_dim)

        return output


class MultiOutputLSTM(BaseModel):
    """
    LSTM with separate output heads for different prediction horizons
    Useful for multi-step forecasting with varying uncertainty
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        sequence_length: int,
        dropout: float = 0.2
    ):
        """
        Initialize Multi-Output LSTM

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            output_dim: Number of output values (prediction horizon)
            sequence_length: Length of input sequence
            dropout: Dropout rate
        """
        super(MultiOutputLSTM, self).__init__(input_dim, output_dim, sequence_length)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Shared layers
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Separate output heads for each time step
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(output_dim)
        ])

        log.info(f"Initialized Multi-Output LSTM with {self.get_num_parameters():,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = h_n[-1, :, :]  # (batch_size, hidden_dim)

        # Shared representation
        shared = self.shared_fc(last_hidden)  # (batch_size, hidden_dim)

        # Generate predictions for each time step
        outputs = []
        for head in self.output_heads:
            out = head(shared)  # (batch_size, 1)
            outputs.append(out)

        # Concatenate all outputs
        output = torch.cat(outputs, dim=1)  # (batch_size, output_dim)

        return output


class StackedLSTM(BaseModel):
    """
    Deep stacked LSTM with residual connections
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        sequence_length: int,
        dropout: float = 0.2,
        use_residual: bool = True
    ):
        """
        Initialize Stacked LSTM

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden dimensions for each LSTM layer
            output_dim: Number of output values
            sequence_length: Length of input sequence
            dropout: Dropout rate
            use_residual: Use residual connections
        """
        super(StackedLSTM, self).__init__(input_dim, output_dim, sequence_length)

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.dropout = dropout
        self.use_residual = use_residual

        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        for i, hidden_dim in enumerate(hidden_dims):
            layer_input_dim = input_dim if i == 0 else hidden_dims[i - 1]

            # LSTM layer
            lstm = nn.LSTM(
                input_size=layer_input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            )
            self.lstm_layers.append(lstm)

            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout))

            # Residual projection (if dimensions don't match)
            if use_residual and layer_input_dim != hidden_dim:
                self.residual_projections.append(nn.Linear(layer_input_dim, hidden_dim))
            else:
                self.residual_projections.append(None)

        # Output layer
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

        log.info(f"Initialized Stacked LSTM with {self.num_layers} layers and {self.get_num_parameters():,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Pass through stacked LSTM layers
        lstm_input = x
        for i, (lstm, dropout, projection) in enumerate(
            zip(self.lstm_layers, self.dropout_layers, self.residual_projections)
        ):
            lstm_out, _ = lstm(lstm_input)

            # Apply dropout
            lstm_out = dropout(lstm_out)

            # Residual connection
            if self.use_residual and projection is not None:
                # Project input to match output dimension
                residual = projection(lstm_input)
                lstm_out = lstm_out + residual
            elif self.use_residual and lstm_input.size(-1) == lstm_out.size(-1):
                lstm_out = lstm_out + lstm_input

            lstm_input = lstm_out

        # Use last time step
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Output layer
        output = self.fc(last_output)  # (batch_size, output_dim)

        return output


def create_lstm_model(
    input_dim: int,
    output_dim: int,
    sequence_length: int,
    model_type: str = 'lstm',
    **kwargs
) -> BaseModel:
    """
    Factory function to create LSTM models

    Args:
        input_dim: Number of input features
        output_dim: Number of output values
        sequence_length: Length of input sequence
        model_type: Type of LSTM model ('lstm', 'lstm_attention', 'multi_output', 'stacked')
        **kwargs: Additional model-specific parameters

    Returns:
        Initialized model
    """
    models = {
        'lstm': LSTMModel,
        'lstm_attention': LSTMAttentionModel,
        'multi_output': MultiOutputLSTM,
        'stacked': StackedLSTM
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
