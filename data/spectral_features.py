"""
Hand-crafted spectral features for USV call classification.

Extracts call-level acoustic features directly from audio, bypassing the
SqueakOut encoder entirely. These features capture call morphology (shape,
frequency modulation, harmonics) that the segmentation encoder discards.

Per-call features (17-dim):
    Duration, frequency (center, bandwidth, min, max),
    frequency modulation (slope, curvature),
    spectral shape (centroid, bandwidth, flatness, rolloff),
    MFCCs (4 coefficients), energy, wiener entropy

Per-recording aggregation produces summary statistics over all calls.
"""

from __future__ import annotations

import numpy as np
import librosa


# ──────────────────────────────────────────────────────────────────────────────
# Per-call feature extraction
# ──────────────────────────────────────────────────────────────────────────────

CALL_FEATURE_NAMES = [
    "duration",
    "freq_center", "freq_bandwidth", "freq_min", "freq_max",
    "fm_slope", "fm_curvature",
    "spectral_centroid", "spectral_bandwidth", "spectral_flatness", "spectral_rolloff",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4",
    "energy", "wiener_entropy",
]

N_CALL_FEATURES = len(CALL_FEATURE_NAMES)


def extract_call_features(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    freq_min: int = 30_000,
    freq_max: int = 130_000,
    n_fft: int = 512,
    hop_length: int = 64,
) -> np.ndarray:
    """
    Extract hand-crafted acoustic features for a single USV call.

    Args:
        audio:    Full recording waveform (float32).
        sr:       Sample rate in Hz.
        start_sec, end_sec: Call boundaries in seconds.
        freq_min, freq_max: Frequency band for analysis (Hz).
        n_fft:    FFT window size.
        hop_length: STFT hop.

    Returns:
        (N_CALL_FEATURES,) float32 feature vector.
    """
    # Extract call segment with small padding
    pad_sec = 0.005  # 5ms padding
    start_idx = max(0, int((start_sec - pad_sec) * sr))
    end_idx = min(len(audio), int((end_sec + pad_sec) * sr))
    segment = audio[start_idx:end_idx].astype(np.float32)

    if len(segment) < n_fft:
        segment = np.pad(segment, (0, n_fft - len(segment)))

    duration = end_sec - start_sec

    # STFT
    S = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Crop to USV frequency band
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    S_crop = S[freq_mask, :]
    freqs_crop = freqs[freq_mask]

    if S_crop.shape[0] == 0 or S_crop.shape[1] == 0:
        return np.zeros(N_CALL_FEATURES, dtype=np.float32)

    # Power spectrum per frame
    power = S_crop ** 2
    power_sum = power.sum(axis=0) + 1e-12

    # ── Frequency trajectory (weighted mean freq per frame) ──────────────────
    freq_trajectory = (power * freqs_crop[:, None]).sum(axis=0) / power_sum

    freq_center = float(np.mean(freq_trajectory))
    freq_min_val = float(np.min(freq_trajectory))
    freq_max_val = float(np.max(freq_trajectory))
    freq_bw = freq_max_val - freq_min_val

    # ── Frequency modulation ────────────────────────────────────────────────
    n_frames = len(freq_trajectory)
    if n_frames >= 3:
        t = np.arange(n_frames, dtype=np.float32)
        # Linear fit → slope
        coeffs = np.polyfit(t, freq_trajectory, 1)
        fm_slope = float(coeffs[0])  # Hz per frame
        # Quadratic fit → curvature
        coeffs2 = np.polyfit(t, freq_trajectory, 2)
        fm_curvature = float(coeffs2[0])
    elif n_frames == 2:
        fm_slope = float(freq_trajectory[1] - freq_trajectory[0])
        fm_curvature = 0.0
    else:
        fm_slope = 0.0
        fm_curvature = 0.0

    # ── Spectral shape features (from librosa, on full spectrum) ─────────────
    S_full = S + 1e-12
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(
        S=S_full, sr=sr, n_fft=n_fft, hop_length=hop_length)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(
        S=S_full, sr=sr, n_fft=n_fft, hop_length=hop_length)))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(S=S_full)))
    spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(
        S=S_full, sr=sr, n_fft=n_fft, hop_length=hop_length)))

    # ── MFCCs (from power spectrum) ─────────────────────────────────────────
    S_db = librosa.power_to_db(S ** 2 + 1e-12)
    try:
        mfccs = librosa.feature.mfcc(S=S_db, sr=sr, n_mfcc=4, n_fft=n_fft)
        mfcc_means = mfccs.mean(axis=1)  # (4,)
    except Exception:
        mfcc_means = np.zeros(4, dtype=np.float32)

    # ── Energy ──────────────────────────────────────────────────────────────
    energy = float(np.mean(segment ** 2))

    # ── Wiener entropy (spectral flatness in USV band) ──────────────────────
    mean_power_per_bin = power.mean(axis=1) + 1e-12
    geo_mean = np.exp(np.mean(np.log(mean_power_per_bin)))
    arith_mean = np.mean(mean_power_per_bin)
    wiener_entropy = float(geo_mean / (arith_mean + 1e-12))

    return np.array([
        duration,
        freq_center, freq_bw, freq_min_val, freq_max_val,
        fm_slope, fm_curvature,
        spec_centroid, spec_bandwidth, spec_flatness, spec_rolloff,
        *mfcc_means,
        energy, wiener_entropy,
    ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Per-recording extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_recording_spectral_features(
    audio_path: str,
    detections_csv: str,
    freq_min: int = 30_000,
    freq_max: int = 130_000,
    n_fft: int = 512,
    hop_length: int = 64,
    min_confidence: float = 0.0,
) -> np.ndarray:
    """
    Extract per-call spectral features for all detected calls in a recording.

    Args:
        audio_path:      WAV file path.
        detections_csv:  CSV with start_sec, end_sec, confidence columns.
        freq_min/max:    Frequency band (Hz).
        min_confidence:  Skip calls below this confidence threshold.

    Returns:
        (n_calls, N_CALL_FEATURES) float32 array.
    """
    import pandas as pd

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    df = pd.read_csv(detections_csv)

    if df.empty or 'start_sec' not in df.columns:
        return np.empty((0, N_CALL_FEATURES), dtype=np.float32)

    # Confidence filtering
    if 'confidence' in df.columns and min_confidence > 0:
        df = df[df['confidence'] >= min_confidence]

    if df.empty:
        return np.empty((0, N_CALL_FEATURES), dtype=np.float32)

    features = []
    for _, row in df.iterrows():
        feat = extract_call_features(
            audio, sr,
            float(row['start_sec']), float(row['end_sec']),
            freq_min=freq_min, freq_max=freq_max,
            n_fft=n_fft, hop_length=hop_length,
        )
        features.append(feat)

    return np.stack(features).astype(np.float32) if features else \
           np.empty((0, N_CALL_FEATURES), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Per-recording summary statistics
# ──────────────────────────────────────────────────────────────────────────────

STAT_NAMES = ["mean", "std", "min", "max", "p25", "p75"]


def summarize_call_features(call_features: np.ndarray) -> np.ndarray:
    """
    Aggregate per-call features into per-recording summary statistics.

    For each of the N_CALL_FEATURES dimensions, computes:
        mean, std, min, max, 25th percentile, 75th percentile

    Plus 2 extra: call_count, inter_call_interval_mean

    Args:
        call_features: (n_calls, N_CALL_FEATURES) array.

    Returns:
        (N_CALL_FEATURES * 6 + 2,) float32 summary vector.
    """
    n_stats = len(STAT_NAMES)
    n_extra = 2  # call_count, ici_mean

    if len(call_features) == 0:
        return np.zeros(N_CALL_FEATURES * n_stats + n_extra, dtype=np.float32)

    stats = []
    for col in range(N_CALL_FEATURES):
        vals = call_features[:, col]
        stats.extend([
            np.mean(vals),
            np.std(vals) if len(vals) > 1 else 0.0,
            np.min(vals),
            np.max(vals),
            np.percentile(vals, 25),
            np.percentile(vals, 75),
        ])

    # Extra features
    call_count = float(len(call_features))

    # Inter-call interval (from duration column = col 0, but we'd need start times)
    # Approximate from call_count and total recording duration
    durations = call_features[:, 0]  # duration column
    total_call_time = durations.sum()
    ici_mean = 0.0  # placeholder — will be computed from CSV if available

    stats.extend([call_count, ici_mean])

    return np.array(stats, dtype=np.float32)


def summary_feature_dim() -> int:
    """Return dimensionality of summarize_call_features output."""
    return N_CALL_FEATURES * len(STAT_NAMES) + 2
