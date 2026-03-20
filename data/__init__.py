"""Data loading and preprocessing modules."""

from .mat_parser import load_deepsqueak_mat, load_all_mat_files, CallData
from .enriched_parser import (
    load_enriched_csv,
    load_all_enriched_csv,
    EnrichedCallData,
    get_recording_name,
    FEATURE_COLUMNS,
    N_FEATURES,
)
from .dataset import (
    USVDataset,
    EnrichedUSVDataset,
    create_data_splits,
    infer_label_from_filename,
    LABEL_MAP,
    LABEL_NAMES,
)
from .preprocessing import normalize_features, pad_or_truncate

__all__ = [
    # MAT parser
    "load_deepsqueak_mat",
    "load_all_mat_files",
    "CallData",
    # Enriched parser
    "load_enriched_csv",
    "load_all_enriched_csv",
    "EnrichedCallData",
    "get_recording_name",
    "FEATURE_COLUMNS",
    "N_FEATURES",
    # Datasets
    "USVDataset",
    "EnrichedUSVDataset",
    "create_data_splits",
    "infer_label_from_filename",
    "LABEL_MAP",
    "LABEL_NAMES",
    # Preprocessing
    "normalize_features",
    "pad_or_truncate",
]
