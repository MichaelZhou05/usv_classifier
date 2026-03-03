"""
Parser for DeepSqueak MAT v7.3 output files.

DeepSqueak saves detection results as MATLAB v7.3 MAT files (HDF5 format).
This module extracts call data from these files.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np


@dataclass
class CallData:
    """Container for extracted call data from a single recording."""

    # Per-call features (shape: n_calls x feature)
    start_times: np.ndarray      # seconds
    low_frequencies: np.ndarray  # kHz
    durations: np.ndarray        # seconds
    bandwidths: np.ndarray       # kHz
    scores: np.ndarray           # detection confidence 0-1
    accept_flags: np.ndarray     # boolean

    # Metadata
    filename: str
    audio_duration: float        # seconds
    sample_rate: float           # Hz
    n_calls: int

    def get_feature_matrix(self, include_score: bool = True) -> np.ndarray:
        """
        Get call features as a 2D array.

        Args:
            include_score: Whether to include detection score as a feature.

        Returns:
            Array of shape (n_calls, n_features) where n_features is 4 or 5.
            Features: [start_time, low_freq, duration, bandwidth, (score)]
        """
        features = [
            self.start_times,
            self.low_frequencies,
            self.durations,
            self.bandwidths,
        ]
        if include_score:
            features.append(self.scores)

        return np.column_stack(features)

    def get_derived_features(self) -> dict:
        """
        Compute additional derived features.

        Returns:
            Dictionary with derived feature arrays.
        """
        end_times = self.start_times + self.durations
        high_frequencies = self.low_frequencies + self.bandwidths

        # Inter-call intervals (NaN for first call)
        if self.n_calls > 1:
            inter_call_intervals = np.concatenate([
                [np.nan],
                self.start_times[1:] - end_times[:-1]
            ])
        else:
            inter_call_intervals = np.array([np.nan])

        return {
            "end_times": end_times,
            "high_frequencies": high_frequencies,
            "inter_call_intervals": inter_call_intervals,
            "call_rate": self.n_calls / self.audio_duration if self.audio_duration > 0 else 0,
        }


def load_deepsqueak_mat(filepath: str) -> CallData:
    """
    Load a DeepSqueak MAT file and extract call data.

    Args:
        filepath: Path to the .mat file.

    Returns:
        CallData object containing extracted call information.

    Raises:
        ValueError: If the file format is not recognized.
    """
    filepath = Path(filepath)

    with h5py.File(filepath, 'r') as f:
        # Extract audio metadata
        audiodata = f['audiodata']
        audio_duration = float(audiodata['Duration'][0, 0])
        sample_rate = float(audiodata['SampleRate'][0, 0])

        # The call data is stored in #refs# group
        # Key names vary but typically:
        # - 'g' contains box data (4 x n_calls)
        # - 'h' contains scores (1 x n_calls)
        # - 'j' contains accept flags (1 x n_calls)
        refs = f['#refs#']

        # Find the box data - look for 4xN float64 array
        boxes = None
        scores = None
        accept_flags = None

        for key in refs.keys():
            obj = refs[key]
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype

                # Box data: 4 rows (start, freq, duration, bandwidth) x N calls
                if shape[0] == 4 and len(shape) == 2 and dtype == np.float64:
                    if boxes is None or shape[1] > boxes.shape[1]:
                        boxes = obj[:]

                # Scores: 1 x N float64
                elif len(shape) == 2 and shape[0] == 1 and dtype == np.float64:
                    data = obj[:]
                    # Scores should be between 0 and 1
                    if np.all((data >= 0) & (data <= 1)):
                        if scores is None or data.shape[1] > scores.shape[1]:
                            scores = data

                # Accept flags: 1 x N uint8
                elif len(shape) == 2 and shape[0] == 1 and dtype == np.uint8:
                    data = obj[:]
                    # Accept flags should be 0 or 1
                    if np.all((data == 0) | (data == 1)):
                        if accept_flags is None or data.shape[1] > accept_flags.shape[1]:
                            accept_flags = data

        if boxes is None:
            raise ValueError(f"Could not find call box data in {filepath}")

        n_calls = boxes.shape[1]

        # Handle missing scores/flags
        if scores is None:
            scores = np.ones((1, n_calls))
        if accept_flags is None:
            accept_flags = np.ones((1, n_calls), dtype=np.uint8)

        # Ensure arrays match in length
        scores = scores.flatten()[:n_calls]
        accept_flags = accept_flags.flatten()[:n_calls]

        if len(scores) < n_calls:
            scores = np.pad(scores, (0, n_calls - len(scores)), constant_values=1.0)
        if len(accept_flags) < n_calls:
            accept_flags = np.pad(accept_flags, (0, n_calls - len(accept_flags)), constant_values=1)

        return CallData(
            start_times=boxes[0, :],
            low_frequencies=boxes[1, :],
            durations=boxes[2, :],
            bandwidths=boxes[3, :],
            scores=scores,
            accept_flags=accept_flags.astype(bool),
            filename=filepath.name,
            audio_duration=audio_duration,
            sample_rate=sample_rate,
            n_calls=n_calls,
        )


def load_all_mat_files(
    directory: str,
    recursive: bool = False
) -> list[CallData]:
    """
    Load all MAT files from a directory.

    Args:
        directory: Path to directory containing MAT files.
        recursive: Whether to search subdirectories.

    Returns:
        List of CallData objects.
    """
    directory = Path(directory)

    if recursive:
        mat_files = list(directory.rglob("*.mat"))
    else:
        mat_files = list(directory.glob("*.mat"))

    results = []
    for mat_file in sorted(mat_files):
        try:
            call_data = load_deepsqueak_mat(mat_file)
            results.append(call_data)
        except Exception as e:
            print(f"Warning: Failed to load {mat_file}: {e}")

    return results


def get_dataset_statistics(call_data_list: list[CallData]) -> dict:
    """
    Compute statistics across a dataset of recordings.

    Args:
        call_data_list: List of CallData objects.

    Returns:
        Dictionary with dataset statistics.
    """
    all_n_calls = [cd.n_calls for cd in call_data_list]
    all_durations = np.concatenate([cd.durations for cd in call_data_list])
    all_bandwidths = np.concatenate([cd.bandwidths for cd in call_data_list])
    all_low_freqs = np.concatenate([cd.low_frequencies for cd in call_data_list])

    return {
        "n_recordings": len(call_data_list),
        "total_calls": sum(all_n_calls),
        "calls_per_recording": {
            "min": min(all_n_calls),
            "max": max(all_n_calls),
            "mean": np.mean(all_n_calls),
            "median": np.median(all_n_calls),
            "std": np.std(all_n_calls),
        },
        "duration_ms": {
            "min": all_durations.min() * 1000,
            "max": all_durations.max() * 1000,
            "mean": all_durations.mean() * 1000,
        },
        "bandwidth_khz": {
            "min": all_bandwidths.min(),
            "max": all_bandwidths.max(),
            "mean": all_bandwidths.mean(),
        },
        "low_freq_khz": {
            "min": all_low_freqs.min(),
            "max": all_low_freqs.max(),
            "mean": all_low_freqs.mean(),
        },
    }


if __name__ == "__main__":
    # Test loading
    import sys

    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    else:
        test_dir = "DeepSqueak__output"

    print(f"Loading MAT files from: {test_dir}")
    data = load_all_mat_files(test_dir)

    print(f"\nLoaded {len(data)} recordings")
    for cd in data:
        print(f"  {cd.filename}: {cd.n_calls} calls, {cd.audio_duration:.1f}s duration")

    print("\nDataset statistics:")
    stats = get_dataset_statistics(data)
    for key, value in stats.items():
        print(f"  {key}: {value}")
