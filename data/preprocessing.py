"""
Preprocessing functions for USV call data.
"""

import numpy as np
from typing import Optional


def pad_or_truncate(
    features: np.ndarray,
    n_max: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad or truncate call features to a fixed length.

    Args:
        features: Array of shape (n_calls, n_features).
        n_max: Target number of calls.
        pad_value: Value to use for padding.

    Returns:
        Array of shape (n_max, n_features).
    """
    n_calls, n_features = features.shape

    if n_calls >= n_max:
        # Truncate - keep first n_max calls (sorted by time)
        return features[:n_max, :]
    else:
        # Pad with zeros
        padded = np.full((n_max, n_features), pad_value, dtype=features.dtype)
        padded[:n_calls, :] = features
        return padded


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features.

    Args:
        features: Array of shape (n_samples, n_max, n_features) or (n_max, n_features).
        mean: Pre-computed mean per feature. If None, computed from data.
        std: Pre-computed std per feature. If None, computed from data.
        eps: Small value to prevent division by zero.

    Returns:
        Tuple of (normalized_features, mean, std).
    """
    if features.ndim == 2:
        # Single sample
        features = features[np.newaxis, ...]

    # Flatten to (total_calls, n_features) ignoring padding
    # For computing stats, we mask out zero-padded rows
    mask = ~np.all(features == 0, axis=-1)  # (n_samples, n_max)

    if mean is None or std is None:
        # Compute stats only on non-padded values
        all_calls = features[mask]  # (total_real_calls, n_features)
        if len(all_calls) == 0:
            mean = np.zeros(features.shape[-1])
            std = np.ones(features.shape[-1])
        else:
            mean = all_calls.mean(axis=0)
            std = all_calls.std(axis=0)

    # Normalize
    std = np.maximum(std, eps)  # Prevent division by zero
    normalized = (features - mean) / std

    # Zero out padded regions again
    normalized[~mask] = 0.0

    if normalized.shape[0] == 1:
        normalized = normalized[0]

    return normalized, mean, std


def compute_dataset_stats(
    all_features: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std across entire dataset for normalization.

    Args:
        all_features: List of feature arrays, each (n_calls, n_features).

    Returns:
        Tuple of (mean, std) arrays of shape (n_features,).
    """
    # Concatenate all calls
    all_calls = np.concatenate(all_features, axis=0)

    mean = all_calls.mean(axis=0)
    std = all_calls.std(axis=0)

    return mean, std


def create_call_mask(features: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask indicating real calls vs padding.

    Args:
        features: Array of shape (n_max, n_features) or (batch, n_max, n_features).

    Returns:
        Boolean array of shape (n_max,) or (batch, n_max).
    """
    return ~np.all(features == 0, axis=-1)


def sort_calls_by_time(features: np.ndarray) -> np.ndarray:
    """
    Sort calls by start time (first feature column).

    Args:
        features: Array of shape (n_calls, n_features).

    Returns:
        Sorted array.
    """
    # Assume first column is start time
    sort_idx = np.argsort(features[:, 0])
    return features[sort_idx]
