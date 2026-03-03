"""
MLP classifier for USV disease detection.
"""

from typing import Optional

import torch
import torch.nn as nn


class USVClassifier(nn.Module):
    """
    Multi-layer perceptron for classifying mice based on USV call features.

    Takes concatenated call features (n_max_calls x n_features) and outputs
    a disease probability.
    """

    def __init__(
        self,
        n_max_calls: int = 150,
        n_features: int = 5,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            n_max_calls: Maximum number of calls per sample.
            n_features: Number of features per call.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        self.n_max_calls = n_max_calls
        self.n_features = n_features
        self.input_dim = n_max_calls * n_features

        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (single logit for binary classification)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_max_calls, n_features)
               or (batch, n_max_calls * n_features).

        Returns:
            Logits of shape (batch, 1).
        """
        if x.dim() == 3:
            x = x.flatten(1)  # (batch, n_max_calls * n_features)

        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()


class USVClassifierWithAttention(nn.Module):
    """
    MLP with attention mechanism to weight calls differently.

    This can help the model focus on the most informative calls.
    """

    def __init__(
        self,
        n_max_calls: int = 150,
        n_features: int = 5,
        embed_dim: int = 64,
        hidden_dims: list[int] = [256, 128],
        dropout: float = 0.3,
    ):
        """
        Args:
            n_max_calls: Maximum number of calls per sample.
            n_features: Number of features per call.
            embed_dim: Dimension to embed each call to.
            hidden_dims: Hidden layer dimensions after attention pooling.
            dropout: Dropout probability.
        """
        super().__init__()

        self.n_max_calls = n_max_calls
        self.n_features = n_features

        # Embed each call
        self.call_encoder = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Classifier on pooled representation
        layers = []
        prev_dim = embed_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_max_calls, n_features).

        Returns:
            Logits of shape (batch, 1).
        """
        batch_size = x.shape[0]

        # Create mask for padding (all-zero rows are padding)
        mask = ~torch.all(x == 0, dim=-1)  # (batch, n_max_calls)

        # Encode each call
        encoded = self.call_encoder(x)  # (batch, n_max_calls, embed_dim)

        # Compute attention weights
        attn_scores = self.attention(encoded).squeeze(-1)  # (batch, n_max_calls)

        # Mask out padding
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Softmax to get weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, n_max_calls)

        # Handle case where all calls are masked
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0)

        # Weighted sum of encoded calls
        pooled = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, n_max_calls)
            encoded                       # (batch, n_max_calls, embed_dim)
        ).squeeze(1)  # (batch, embed_dim)

        # Classify
        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        Returns:
            Attention weights of shape (batch, n_max_calls).
        """
        mask = ~torch.all(x == 0, dim=-1)
        encoded = self.call_encoder(x)
        attn_scores = self.attention(encoded).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0)
        return attn_weights


def get_model(
    model_type: str = "mlp",
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: One of "mlp" or "attention".
        **kwargs: Arguments passed to model constructor.

    Returns:
        Model instance.
    """
    if model_type == "mlp":
        return USVClassifier(**kwargs)
    elif model_type == "attention":
        return USVClassifierWithAttention(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
