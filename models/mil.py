"""
Attention-based Multiple Instance Learning (MIL) for USV classification.

Treats each recording as a "bag" of call instances. Learns attention weights
over calls to identify which calls are most discriminative for classification,
then aggregates via weighted sum.

Architecture:
    per-call features (N, D)
        → shared call encoder (D → embed_dim)
        → attention weights (embed_dim → 1, softmax over N calls)
        → weighted sum → (1, embed_dim)
        → classifier MLP → (1, n_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatedAttention(nn.Module):
    """Gated attention mechanism for MIL (Ilse et al., 2018)."""

    def __init__(self, embed_dim: int, attn_dim: int = 64):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, embed_dim) call embeddings.
        Returns:
            (N,) attention weights (softmax-normalised).
        """
        v = self.attention_V(x)  # (N, attn_dim)
        u = self.attention_U(x)  # (N, attn_dim)
        scores = self.attention_w(v * u).squeeze(-1)  # (N,)
        return F.softmax(scores, dim=0)


class MILClassifier(nn.Module):
    """
    Attention-based MIL classifier for USV recordings.

    Each recording is a variable-length set of per-call feature vectors.
    The model learns:
      1. A shared call encoder that transforms raw call features
      2. Attention weights identifying the most informative calls
      3. A classifier operating on the attention-weighted bag representation
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        embed_dim: int = 64,
        attn_dim: int = 32,
        hidden_dims: list[int] = [32],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        # Shared call encoder
        self.call_encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Gated attention
        self.attention = GatedAttention(embed_dim, attn_dim)

        # Classifier
        layers = []
        prev = embed_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, call_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single recording (bag of calls).

        Args:
            call_features: (N_calls, input_dim) tensor.

        Returns:
            logits: (1, n_classes) classification logits.
            attn_weights: (N_calls,) attention weights.
        """
        embeddings = self.call_encoder(call_features)  # (N, embed_dim)
        attn_weights = self.attention(embeddings)       # (N,)

        # Weighted aggregation
        bag_repr = (attn_weights.unsqueeze(1) * embeddings).sum(0, keepdim=True)  # (1, embed_dim)

        logits = self.classifier(bag_repr)  # (1, n_classes)
        return logits, attn_weights

    def predict(self, call_features: torch.Tensor) -> int:
        """Predict class for a single recording."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(call_features)
            return int(logits.argmax(1).item())
