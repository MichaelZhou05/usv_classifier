"""
Label mapping and stratified splitting utilities for USV classification.

Labels are inferred from recording filenames. Supported patterns:
  twitcher / twi  → class 0
  wildtype / wt   → class 1
  het / heterozygous → class 2
"""

import numpy as np

LABEL_MAP = {
    'twitcher': 0, 'twi': 0,
    'wildtype': 1, 'wt': 1,
    'heterozygous': 2, 'het': 2,
}

LABEL_NAMES = {0: 'twitcher', 1: 'wildtype', 2: 'heterozygous'}


def infer_label_from_filename(filename: str, n_classes: int = 3) -> int:
    """
    Infer class label from a recording filename.

    Args:
        filename:  Filename (or full path) — only the basename is scanned.
        n_classes: 3 for twitcher/wildtype/het; 2 collapses het+wildtype → 0.

    Returns:
        Integer class label.

    Raises:
        ValueError: If no known pattern is found.
    """
    name = filename.lower()
    for pattern, label in LABEL_MAP.items():
        if pattern in name:
            if n_classes == 2:
                return 1 if label == 0 else 0
            return label
    raise ValueError(f"Cannot infer label from filename: '{filename}'")


def stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split indices into stratified train / val / test sets.

    Splits within each class independently so class balance is preserved in
    all three sets. Guarantees at least 1 sample per class per split when
    the class has at least 3 samples.

    Args:
        labels:      (N,) integer label array.
        train_ratio: Fraction for training.
        val_ratio:   Fraction for validation (test = remainder).
        random_seed: RNG seed for reproducibility.

    Returns:
        (train_idx, val_idx, test_idx) lists of integer indices.
    """
    rng = np.random.default_rng(random_seed)
    train_idx, val_idx, test_idx = [], [], []

    for label in np.unique(labels):
        idx = np.where(labels == label)[0].copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        # Ensure test set gets at least 1 sample
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train: n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    return train_idx, val_idx, test_idx
