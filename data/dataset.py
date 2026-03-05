"""
PyTorch Dataset for USV classification.
"""

import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mat_parser import load_deepsqueak_mat, load_all_mat_files, CallData
from .preprocessing import pad_or_truncate, normalize_features, sort_calls_by_time


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


def create_data_splits(
    dataset: USVDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """
    Create train/val/test splits.

    Args:
        dataset: USVDataset instance.
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
