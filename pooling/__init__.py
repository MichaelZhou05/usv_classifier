"""
Pooling module for aggregating variable-length call sequences.

This module provides strategies for converting variable numbers of USV calls
per recording into fixed-size feature vectors for classification.

Available poolers:
    - average: Simple mean pooling (default)
    - max: Max pooling
    - statistics: Multi-statistic pooling (mean, std, min, max, median)

Usage:
    from usv_classifier.pooling import PoolerRegistry, AveragePooler

    # Get pooler by name
    pooler = PoolerRegistry.get("average", n_features=11)

    # Pool call features
    features = pooler.pool(call_features)  # (n_features,)

    # Register external pooler
    PoolerRegistry.register_external("custom", CustomPooler)
"""

from .base import CallPooler
from .registry import PoolerRegistry
from .average import AveragePooler, MaxPooler, StatisticsPooler

__all__ = [
    "CallPooler",
    "PoolerRegistry",
    "AveragePooler",
    "MaxPooler",
    "StatisticsPooler",
]
