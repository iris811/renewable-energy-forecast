"""
Attention mechanisms for time series forecasting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, temperature: float, dropout: float = 0.1):
        """
        Initialize attention

        Args:
            temperature: Scaling factor (typically sqrt(d_k))
            dropout: Dropout rate
        """
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass

        Args:
            q: Query (batch_size, n_heads, seq_len, d_k)
            k: Key (batch_size, n_heads, seq_len, d_k)
            v: Value (batch_size, n_heads, seq_len, d_v)
            mask: Optional mask

        Returns:
            (output, attention_weights)
        """
        # Compute attention scores
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Linear projections
        self.w_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * self.d_v, bias=False)
        self.w_o = nn.Linear(n_heads * self.d_v, d_model, bias=False)

        # Attention
        self.attention = ScaledDotProductAttention(
            temperature=math.sqrt(self.d_k),
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass

        Args:
            q: Query (batch_size, seq_len, d_model)
            k: Key (batch_size, seq_len, d_model)
            v: Value (batch_size, seq_len, d_model)
            mask: Optional mask

        Returns:
            (output, attention_weights)
        """
        batch_size, seq_len, _ = q.size()

        # Save residual
        residual = q

        # Linear projections and reshape
        q = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)

        # Expand mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head dimension

        # Apply attention
        output, attn = self.attention(q, k, v, mask=mask)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.w_o(output)
        output = self.dropout(output)

        # Add residual and normalize
        output = self.layer_norm(output + residual)

        return output, attn


class TemporalAttention(nn.Module):
    """
    Temporal attention for time series
    Focuses on important time steps
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize temporal attention

        Args:
            hidden_dim: Hidden dimension size
            attention_dim: Attention mechanism dimension
            dropout: Dropout rate
        """
        super(TemporalAttention, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)

        Returns:
            (context_vector, attention_weights)
        """
        # Compute attention scores
        attention_scores = self.attention(x)  # (batch_size, seq_len, 1)

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)

        # Weighted sum
        context = torch.sum(attention_weights * x, dim=1)  # (batch_size, hidden_dim)

        return context, attention_weights.squeeze(-1)


class SelfAttention(nn.Module):
    """
    Self-attention layer
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize self-attention

        Args:
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
        """
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim

        # Query, Key, Value projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            mask: Optional mask

        Returns:
            (output, attention_weights)
        """
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, hidden_dim)

        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    Adds position information to input embeddings
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initialize learnable positional encoding

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(LearnablePositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
