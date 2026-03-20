"""
Average (mean) pooling implementation.

Simple but effective: computes element-wise mean across all calls.
"""

import numpy as np

from .base import CallPooler
from .registry import PoolerRegistry


@PoolerRegistry.register("average")
class AveragePooler(CallPooler):
    """
    Simple mean pooling across all calls.

    For each feature dimension, computes the mean value across all calls
    in the recording. This is equivalent to treating each call equally.

    Attributes:
        n_features: Number of features per call (determines output dimension).
    """

    def __init__(self, n_features: int):
        """
        Initialize AveragePooler.

        Args:
            n_features: Number of features per call. This determines the
                       output dimension of the pooled vector.
        """
        self.n_features = n_features

    def pool(self, call_features: np.ndarray) -> np.ndarray:
        """
        Compute mean of all call features.

        Args:
            call_features: (n_calls, n_features) array.

        Returns:
            (n_features,) mean vector. Returns zeros if input is empty.
        """
        if len(call_features) == 0:
            return np.zeros(self.n_features)

        return np.mean(call_features, axis=0)

    @property
    def output_dim(self) -> int:
        return self.n_features


@PoolerRegistry.register("max")
class MaxPooler(CallPooler):
    """
    Max pooling across all calls.

    For each feature dimension, takes the maximum value across all calls.
    Can capture extreme/outlier call characteristics.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features

    def pool(self, call_features: np.ndarray) -> np.ndarray:
        if len(call_features) == 0:
            return np.zeros(self.n_features)

        return np.max(call_features, axis=0)

    @property
    def output_dim(self) -> int:
        return self.n_features


@PoolerRegistry.register("statistics")
class StatisticsPooler(CallPooler):
    """
    Statistical pooling: computes multiple statistics per feature.

    Computes mean, std, min, max, and median for each feature dimension,
    providing a richer summary at the cost of higher dimensionality.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.n_stats = 5  # mean, std, min, max, median

    def pool(self, call_features: np.ndarray) -> np.ndarray:
        if len(call_features) == 0:
            return np.zeros(self.n_features * self.n_stats)

        stats = []
        for i in range(self.n_features):
            feat = call_features[:, i]
            stats.extend([
                np.mean(feat),
                np.std(feat) if len(feat) > 1 else 0.0,
                np.min(feat),
                np.max(feat),
                np.median(feat),
            ])

        return np.array(stats)

    @property
    def output_dim(self) -> int:
        return self.n_features * self.n_stats
