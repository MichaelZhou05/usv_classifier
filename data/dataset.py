"""
PyTorch Dataset for USV classification.

Supports both legacy MAT file loading and new enriched CSV features with pooling.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mat_parser import load_deepsqueak_mat, load_all_mat_files, CallData
from .enriched_parser import load_enriched_csv, load_all_enriched_csv, EnrichedCallData, get_recording_name
from .preprocessing import pad_or_truncate, normalize_features, sort_calls_by_time


# 3-class label mapping
LABEL_MAP = {
    'twitcher': 0, 'twi': 0,
    'wildtype': 1, 'wt': 1,
    'heterozygous': 2, 'het': 2,
}

LABEL_NAMES = {0: 'twitcher', 1: 'wildtype', 2: 'heterozygous'}


def infer_label_from_filename(filename: str, n_classes: int = 3) -> int:
    """
    Infer class label from filename.

    Args:
        filename: The filename to parse.
        n_classes: Number of classes (2 or 3).

    Returns:
        Class label (0, 1, or 2 for 3-class; 0 or 1 for 2-class).

    Raises:
        ValueError: If no label pattern is found in filename.
    """
    filename_lower = filename.lower()

    for pattern, label in LABEL_MAP.items():
        if pattern in filename_lower:
            if n_classes == 2:
                # Binary: twitcher=1, others=0
                return 1 if label == 0 else 0
            return label

    raise ValueError(f"Cannot infer label from: {filename}")


class USVDataset(Dataset):
    """
    PyTorch Dataset for USV disease classification.

    Each sample is a single recording (mouse), represented as a fixed-size
    tensor of call features.
    """

    def __init__(
        self,
        mat_directory: str,
        labels_csv: Optional[str] = None,
        n_max_calls: int = 150,
        n_features: int = 5,
        include_score: bool = True,
        normalize: bool = True,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        only_accepted: bool = True,
    ):
        """
        Args:
            mat_directory: Path to directory containing DeepSqueak MAT files.
            labels_csv: Path to CSV with columns [filename, label].
                        If None, tries to infer labels from filenames.
            n_max_calls: Maximum number of calls per sample (pad/truncate to this).
            n_features: Number of features per call (4 without score, 5 with).
            include_score: Whether to include detection score as a feature.
            normalize: Whether to z-score normalize features.
            feature_mean: Pre-computed mean for normalization (for val/test sets).
            feature_std: Pre-computed std for normalization (for val/test sets).
            transform: Optional transform to apply to features.
            only_accepted: Only include calls where Accept=True.
        """
        self.mat_directory = Path(mat_directory)
        self.n_max_calls = n_max_calls
        self.n_features = n_features
        self.include_score = include_score
        self.normalize = normalize
        self.transform = transform
        self.only_accepted = only_accepted

        # Load all MAT files
        self.call_data_list = load_all_mat_files(mat_directory)

        if len(self.call_data_list) == 0:
            raise ValueError(f"No MAT files found in {mat_directory}")

        # Load or infer labels
        self.labels = self._load_labels(labels_csv)

        # Extract features from all recordings
        self.features_list = []
        self.valid_indices = []

        for i, cd in enumerate(self.call_data_list):
            if cd.filename not in self.labels:
                print(f"Warning: No label for {cd.filename}, skipping")
                continue

            features = self._extract_features(cd)
            if features is not None and len(features) > 0:
                self.features_list.append(features)
                self.valid_indices.append(i)

        if len(self.features_list) == 0:
            raise ValueError("No valid samples after filtering")

        # Compute or use provided normalization stats
        if normalize:
            if feature_mean is None or feature_std is None:
                self.feature_mean, self.feature_std = self._compute_stats()
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
        else:
            self.feature_mean = None
            self.feature_std = None

        # Preprocess all features
        self.processed_features = []
        self.processed_labels = []

        for i, features in enumerate(self.features_list):
            orig_idx = self.valid_indices[i]
            filename = self.call_data_list[orig_idx].filename
            label = self.labels[filename]

            # Sort by time
            features = sort_calls_by_time(features)

            # Pad or truncate
            features = pad_or_truncate(features, self.n_max_calls)

            # Normalize
            if self.normalize:
                features, _, _ = normalize_features(
                    features,
                    mean=self.feature_mean,
                    std=self.feature_std
                )

            self.processed_features.append(features)
            self.processed_labels.append(label)

        self.processed_features = np.array(self.processed_features)
        self.processed_labels = np.array(self.processed_labels)

    def _load_labels(self, labels_csv: Optional[str]) -> dict:
        """Load labels from CSV or infer from filenames."""
        labels = {}

        if labels_csv is not None and os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                labels[row['filename']] = int(row['label'])
        else:
            # Infer from filename
            for cd in self.call_data_list:
                filename = cd.filename.lower()
                if 'twitcher' in filename or 'twi' in filename:
                    labels[cd.filename] = 1
                elif 'wildtype' in filename or 'wt' in filename or 'het' in filename:
                    # het (heterozygous) grouped with wildtype as healthy
                    labels[cd.filename] = 0
                else:
                    # Unknown - will be filtered out
                    pass

        return labels

    def _extract_features(self, cd: CallData) -> Optional[np.ndarray]:
        """Extract feature matrix from CallData."""
        if cd.n_calls == 0:
            return None

        features = cd.get_feature_matrix(include_score=self.include_score)

        if self.only_accepted:
            features = features[cd.accept_flags]

        if len(features) == 0:
            return None

        return features

    def _compute_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean and std across all calls in dataset."""
        all_calls = np.concatenate(self.features_list, axis=0)
        mean = all_calls.mean(axis=0)
        std = all_calls.std(axis=0)
        std = np.maximum(std, 1e-8)  # Prevent div by zero
        return mean, std

    def __len__(self) -> int:
        return len(self.processed_features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.processed_features[idx].copy()
        label = self.processed_labels[idx]

        if self.transform is not None:
            features = self.transform(features)

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return features, label

    def get_filename(self, idx: int) -> str:
        """Get original filename for a sample."""
        orig_idx = self.valid_indices[idx]
        return self.call_data_list[orig_idx].filename

    def get_sample_info(self, idx: int) -> dict:
        """Get detailed info about a sample."""
        orig_idx = self.valid_indices[idx]
        cd = self.call_data_list[orig_idx]
        return {
            "filename": cd.filename,
            "n_calls_original": cd.n_calls,
            "audio_duration": cd.audio_duration,
            "label": self.processed_labels[idx],
        }

    @property
    def class_counts(self) -> dict:
        """Count samples per class."""
        unique, counts = np.unique(self.processed_labels, return_counts=True)
        return dict(zip(unique.astype(int), counts))

    @property
    def class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        counts = self.class_counts
        total = sum(counts.values())
        weights = {k: total / (len(counts) * v) for k, v in counts.items()}
        # Return weight for positive class (for BCELoss)
        return torch.tensor(weights.get(1, 1.0), dtype=torch.float32)


class EnrichedUSVDataset(Dataset):
    """
    PyTorch Dataset for USV classification using enriched features with pooling.

    This dataset loads pre-computed enriched features from CSV files (exported
    from MATLAB) and applies a pooling strategy to create fixed-size representations.

    This is the recommended approach for small datasets as it:
    1. Uses richer, domain-specific features from CalculateStats
    2. Applies principled pooling instead of concatenate-and-flatten
    3. Results in much smaller input dimensions, reducing overfitting
    """

    def __init__(
        self,
        csv_directory: str,
        labels_csv: Optional[str] = None,
        pooler: Optional['CallPooler'] = None,
        n_classes: int = 3,
        normalize: bool = True,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            csv_directory: Path to directory containing enriched feature CSV files.
            labels_csv: Path to CSV with columns [filename, label].
                        If None, tries to infer labels from filenames.
            pooler: CallPooler instance for aggregating call features.
                    If None, uses AveragePooler with n_features=11.
            n_classes: Number of output classes (2 or 3).
            normalize: Whether to z-score normalize pooled features.
            feature_mean: Pre-computed mean for normalization (for val/test sets).
            feature_std: Pre-computed std for normalization (for val/test sets).
            transform: Optional transform to apply to features.
        """
        from ..pooling import PoolerRegistry

        self.csv_directory = Path(csv_directory)
        self.n_classes = n_classes
        self.normalize = normalize
        self.transform = transform

        # Set up pooler
        if pooler is None:
            self.pooler = PoolerRegistry.get("average", n_features=11)
        else:
            self.pooler = pooler

        # Load all CSV files
        self.call_data_list = load_all_enriched_csv(csv_directory)

        if len(self.call_data_list) == 0:
            raise ValueError(f"No CSV files found in {csv_directory}")

        # Load or infer labels
        self.labels = self._load_labels(labels_csv)

        # Pool features from all recordings
        self.features_list = []
        self.valid_indices = []

        for i, cd in enumerate(self.call_data_list):
            recording_name = get_recording_name(cd.filename)

            # Try to match by filename or recording name
            label = self.labels.get(cd.filename) or self.labels.get(recording_name)
            if label is None:
                print(f"Warning: No label for {cd.filename}, skipping")
                continue

            if cd.n_calls == 0:
                print(f"Warning: No calls in {cd.filename}, skipping")
                continue

            # Get feature matrix and pool
            features = cd.get_feature_matrix()  # (n_calls, 11)
            pooled = self.pooler.pool(features)  # (output_dim,)

            self.features_list.append(pooled)
            self.valid_indices.append(i)

        if len(self.features_list) == 0:
            raise ValueError("No valid samples after filtering")

        self.features_list = np.array(self.features_list)

        # Compute or use provided normalization stats
        if normalize:
            if feature_mean is None or feature_std is None:
                self.feature_mean = self.features_list.mean(axis=0)
                self.feature_std = self.features_list.std(axis=0)
                self.feature_std = np.maximum(self.feature_std, 1e-8)
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std

            # Apply normalization
            self.processed_features = (
                self.features_list - self.feature_mean
            ) / self.feature_std
        else:
            self.feature_mean = None
            self.feature_std = None
            self.processed_features = self.features_list

        # Extract labels in order
        self.processed_labels = np.array([
            self.labels.get(self.call_data_list[i].filename) or
            self.labels.get(get_recording_name(self.call_data_list[i].filename))
            for i in self.valid_indices
        ])

    def _load_labels(self, labels_csv: Optional[str]) -> dict:
        """Load labels from CSV or infer from filenames."""
        labels = {}

        if labels_csv is not None and os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            for _, row in df.iterrows():
                labels[row['filename']] = int(row['label'])
        else:
            # Infer from filename using 3-class mapping
            for cd in self.call_data_list:
                try:
                    label = infer_label_from_filename(cd.filename, self.n_classes)
                    labels[cd.filename] = label
                    # Also add without _features suffix
                    recording_name = get_recording_name(cd.filename)
                    labels[recording_name] = label
                except ValueError:
                    pass  # Unknown - will be filtered out

        return labels

    def __len__(self) -> int:
        return len(self.processed_features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.processed_features[idx].copy()
        label = self.processed_labels[idx]

        if self.transform is not None:
            features = self.transform(features)

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)  # Long for CrossEntropyLoss

        return features, label

    def get_filename(self, idx: int) -> str:
        """Get original filename for a sample."""
        orig_idx = self.valid_indices[idx]
        return self.call_data_list[orig_idx].filename

    def get_sample_info(self, idx: int) -> dict:
        """Get detailed info about a sample."""
        orig_idx = self.valid_indices[idx]
        cd = self.call_data_list[orig_idx]
        return {
            "filename": cd.filename,
            "n_calls": cd.n_calls,
            "label": int(self.processed_labels[idx]),
            "label_name": LABEL_NAMES.get(int(self.processed_labels[idx]), "unknown"),
        }

    @property
    def class_counts(self) -> dict:
        """Count samples per class."""
        unique, counts = np.unique(self.processed_labels, return_counts=True)
        return dict(zip(unique.astype(int), counts))

    @property
    def class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data (for CrossEntropyLoss)."""
        counts = self.class_counts
        total = sum(counts.values())
        n_classes = len(counts)
        weights = [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)]
        return torch.tensor(weights, dtype=torch.float32)

    @property
    def input_dim(self) -> int:
        """Return the input dimension (pooler output dimension)."""
        return self.pooler.output_dim


def create_data_splits(
    dataset: Union[USVDataset, EnrichedUSVDataset],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """
    Create train/val/test splits.

    Args:
        dataset: USVDataset or EnrichedUSVDataset instance.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        random_seed: Random seed for reproducibility.
        stratify: Whether to stratify by label.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    np.random.seed(random_seed)

    n_samples = len(dataset)
    indices = np.arange(n_samples)

    if stratify:
        # Split within each class
        train_idx, val_idx, test_idx = [], [], []

        for label in np.unique(dataset.processed_labels):
            class_indices = indices[dataset.processed_labels == label]
            np.random.shuffle(class_indices)

            n_class = len(class_indices)
            n_train = int(n_class * train_ratio)
            n_val = int(n_class * val_ratio)

            train_idx.extend(class_indices[:n_train])
            val_idx.extend(class_indices[n_train:n_train + n_val])
            test_idx.extend(class_indices[n_train + n_val:])
    else:
        np.random.shuffle(indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train + n_val].tolist()
        test_idx = indices[n_train + n_val:].tolist()

    return train_idx, val_idx, test_idx
