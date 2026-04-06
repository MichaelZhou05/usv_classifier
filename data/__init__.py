"""Data loading and feature extraction for USV classification."""

from .dataset import LABEL_MAP, LABEL_NAMES, infer_label_from_filename, stratified_split
from .squeakout_features import (
    generate_call_spectrogram,
    SqueakOutEncoder,
    extract_recording_features,
    extract_recording_spectrograms,
    add_spectrogram_noise,
    augment_to_balance,
    augment_recordings_to_balance,
    augment_by_cross_litter_mixing,
    build_squeakout_dataset,
)

__all__ = [
    # Label utilities
    "LABEL_MAP",
    "LABEL_NAMES",
    "infer_label_from_filename",
    "stratified_split",
    # SqueakOut feature extraction
    "generate_call_spectrogram",
    "SqueakOutEncoder",
    "extract_recording_features",
    "extract_recording_spectrograms",
    "add_spectrogram_noise",
    "augment_to_balance",
    "augment_recordings_to_balance",
    "augment_by_cross_litter_mixing",
    "build_squeakout_dataset",
]
