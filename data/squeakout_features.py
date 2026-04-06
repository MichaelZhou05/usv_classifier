"""
SqueakOut-based latent feature extraction for USV classification.

Replaces hand-crafted summary statistics (duration, min/max freq, slope, etc.)
with 1280-dimensional deep encoder representations learned by the SqueakOut
segmentation model from raw spectrogram data.

Pipeline
────────
  WAV audio + call detection CSV (start_sec, end_sec per call)
      ↓
  generate_call_spectrogram()   — fixed 512×512 window per call
      ↓
  add_spectrogram_noise()       — Gaussian noise at input level
      ↓                           (augmented training copies only, per epoch)
  SqueakOutEncoder.forward()    — 1280-dim latent vector per call
      ↓
  pooler.pool()                 — SWE or average → single recording vector
      ↓
  MLP classifier                — twitcher | wildtype | het

Handling the 512×512 constraint
────────────────────────────────
SqueakOut requires exactly 512×512 input. USV calls have variable duration
(~10ms–300ms). Naively resizing each call's spectrogram to 512×512 would give
different temporal scales per call, corrupting the spatial patterns the encoder
learned. Instead we use a fixed-duration window (default 300ms) per call:

  - Calls shorter than 300ms → zero-padded symmetrically in the time axis
  - Calls longer than 300ms  → centre-cropped in the time axis

Result: every spectrogram pixel represents the same physical time and frequency
interval across the entire dataset, so the encoder sees consistent patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import librosa
import pandas as pd
from PIL import Image

# ── locate squeakout relative to this project ──────────────────────────────────
_SQUEAKOUT_DIR = Path(__file__).parent.parent.parent / "squeakout"
if str(_SQUEAKOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SQUEAKOUT_DIR))

# squeakout.py imports pytorch_lightning at the top level only for the
# SqueakOut_autoencoder Lightning wrapper, which we never use here.
# Mock it out so pytorch_lightning does not need to be installed.
if 'pytorch_lightning' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['pytorch_lightning'] = MagicMock()

from squeakout import SqueakOut  # noqa: E402 — bare nn.Module, no Lightning


# ──────────────────────────────────────────────────────────────────────────────
# 1. Spectrogram generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_call_spectrogram(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    window_duration_sec: float = 0.30,
    freq_min: int = 30_000,
    freq_max: int = 130_000,
    img_size: int = 512,
    n_fft: int = 512,
    hop_length: int = 64,
) -> np.ndarray:
    """
    Generate a fixed-size grayscale spectrogram for a single USV call.

    The call is placed in a fixed-length time window so every output image has
    the same temporal resolution. Short calls are zero-padded symmetrically;
    long calls are centre-cropped. The frequency axis is cropped to
    [freq_min, freq_max] Hz and the whole image is resized to img_size×img_size.

    Args:
        audio:               Raw waveform at sample rate sr.
        sr:                  Sample rate in Hz (e.g. 250_000 for USV recordings).
        start_sec:           Detected call start time (seconds).
        end_sec:             Detected call end time (seconds).
        window_duration_sec: Fixed time window. 300ms covers essentially all
                             mouse USV calls with room to spare.
        freq_min:            Lower frequency bound for the spectrogram crop (Hz).
        freq_max:            Upper frequency bound (Hz).
        img_size:            Output image edge length in pixels (must be 512 for
                             SqueakOut compatibility).
        n_fft:               FFT window length for STFT.
        hop_length:          Hop length for STFT.

    Returns:
        float32 ndarray of shape (img_size, img_size) with values in [0, 1].
    """
    window_samples = int(window_duration_sec * sr)

    call_start_idx = max(0, int(start_sec * sr))
    call_end_idx = min(len(audio), int(end_sec * sr))
    call_samples = audio[call_start_idx:call_end_idx]

    if len(call_samples) >= window_samples:
        # Centre-crop: discard excess symmetrically
        excess = len(call_samples) - window_samples
        trim = excess // 2
        segment = call_samples[trim: trim + window_samples]
    else:
        # Zero-pad symmetrically so the call is centred in the window
        pad_total = window_samples - len(call_samples)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        segment = np.pad(call_samples.astype(np.float32),
                         (pad_left, pad_right), mode='constant')

    # STFT magnitude in dB
    stft = librosa.stft(segment.astype(np.float32),
                        n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Crop frequency axis to the USV band
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    S_cropped = S_db[freq_mask, :]
    if S_cropped.shape[0] == 0:
        S_cropped = S_db  # fallback: keep all frequencies

    # Normalise to [0, 1]
    lo, hi = S_cropped.min(), S_cropped.max()
    S_norm = (S_cropped - lo) / (hi - lo + 1e-8)

    # Resize to (img_size, img_size) — frequency axis is rows, time is columns
    img = Image.fromarray((S_norm * 255).astype(np.uint8), mode='L')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. SqueakOut encoder wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SqueakOutEncoder(nn.Module):
    """
    Extracts latent feature vectors from the pretrained SqueakOut backbone.

    The encoder is the MobileNetV2 portion of SqueakOut, which compresses a
    512×512 spectrogram through 19 inverted-residual blocks. We tap the output
    at different depths and apply global average pooling (GAP) to produce a
    fixed-length vector.

    Extraction points
    ─────────────────
    'deep'  After all 19 blocks → GAP → 1280-dim  (default, most informative)
    'x4'    After blocks 7-13   → GAP →   96-dim  (high-level semantics)
    'x3'    After blocks 4-6    → GAP →   32-dim  (mid-level patterns)
    'multi' Concatenate deep+x4+x3    → 1408-dim  (multi-scale)

    For small datasets (< ~50 recordings) 'x4' or 'x3' may reduce overfitting
    in the downstream classifier.

    All parameters are frozen — we use SqueakOut purely as a feature extractor.
    """

    _OUTPUT_DIMS: dict[str, int] = {
        'deep': 1280,
        'x4': 96,
        'x3': 32,
        'multi': 1408,
    }

    def __init__(
        self,
        weights_path: str,
        extraction_point: Literal['deep', 'x4', 'x3', 'multi'] = 'deep',
        device: str = 'cpu',
    ):
        """
        Args:
            weights_path:      Path to squeakout_weights.ckpt.
            extraction_point:  Which encoder depth to extract from.
            device:            'cpu', 'cuda', or 'mps'.
        """
        super().__init__()
        if extraction_point not in self._OUTPUT_DIMS:
            raise ValueError(
                f"extraction_point must be one of {list(self._OUTPUT_DIMS)}, "
                f"got '{extraction_point}'"
            )
        self.extraction_point = extraction_point
        self.device = torch.device(device)

        # Load directly into SqueakOut (bare nn.Module) — no Lightning needed.
        # Checkpoint was saved by SqueakOut_autoencoder (self.model = SqueakOut()),
        # so state dict keys are prefixed with 'model.'; strip that prefix.
        squeakout = SqueakOut()
        ckpt = torch.load(weights_path, map_location=device)
        state = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}
        squeakout.load_state_dict(state)

        self.backbone = squeakout.backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.to(self.device)

    @property
    def output_dim(self) -> int:
        return self._OUTPUT_DIMS[self.extraction_point]

    def _run_backbone(self, x: torch.Tensor):
        """Run encoder blocks and return (deep, x4, x3) feature maps."""
        for n in range(0, 2):
            x = self.backbone.features[n](x)
        x1 = x                                      # (B, 16, H/2, W/2)

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x                                      # (B, 24, H/4, W/4)

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x                                      # (B, 32, H/8, W/8)

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x                                      # (B, 96, H/16, W/16)

        for n in range(14, 19):
            x = self.backbone.features[n](x)
        deep = x                                    # (B, 1280, H/32, W/32)

        return deep, x4, x3

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 512, 512) normalised spectrograms.
        Returns:
            (B, output_dim) feature vectors.
        """
        x = x.to(self.device)
        deep, x4, x3 = self._run_backbone(x)

        if self.extraction_point == 'deep':
            return deep.mean(dim=[2, 3])
        elif self.extraction_point == 'x4':
            return x4.mean(dim=[2, 3])
        elif self.extraction_point == 'x3':
            return x3.mean(dim=[2, 3])
        else:  # 'multi'
            return torch.cat([
                deep.mean(dim=[2, 3]),
                x4.mean(dim=[2, 3]),
                x3.mean(dim=[2, 3]),
            ], dim=1)

    def encode_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """
        Encode a single (512, 512) float32 spectrogram → (output_dim,) vector.

        Convenience wrapper for single-call inference; batched forward() is
        faster for bulk extraction.
        """
        t = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.forward(t).squeeze(0).cpu().numpy()

    def encode_batch(
        self,
        spectrograms: list[np.ndarray],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Encode a list of (512, 512) spectrograms → (N, output_dim) array.

        Args:
            spectrograms: List of float32 ndarrays, each (512, 512).
            batch_size:   GPU batch size.
        Returns:
            (N, output_dim) float32 numpy array.
        """
        all_feats = []
        for i in range(0, len(spectrograms), batch_size):
            chunk = spectrograms[i: i + batch_size]
            t = torch.tensor(
                np.stack(chunk)[:, np.newaxis], dtype=torch.float32
            )
            all_feats.append(self.forward(t).cpu().numpy())
        return np.concatenate(all_feats, axis=0) if all_feats else \
               np.empty((0, self.output_dim), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Per-recording feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_recording_features(
    audio_path: str,
    detections_csv: str,
    encoder: SqueakOutEncoder,
    window_duration_sec: float = 0.30,
    freq_min: int = 30_000,
    freq_max: int = 130_000,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Extract SqueakOut latent vectors for every detected call in one recording.

    Args:
        audio_path:          Path to the WAV file.
        detections_csv:      CSV with at minimum columns [start_sec, end_sec].
                             One row per detected call.
        encoder:             Initialised SqueakOutEncoder.
        window_duration_sec: Time window for spectrogram generation.
        freq_min / freq_max: Frequency crop bounds in Hz.
        batch_size:          Spectrogram batch size for encoder inference.

    Returns:
        (n_calls, encoder.output_dim) float32 array.
        Shape (0, output_dim) if no calls are detected.
    """
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    detections = pd.read_csv(detections_csv)

    required = {'start_sec', 'end_sec'}
    if not required.issubset(detections.columns):
        raise ValueError(
            f"{detections_csv} must contain columns {required}. "
            f"Found: {list(detections.columns)}"
        )

    spectrograms = [
        generate_call_spectrogram(
            audio, sr,
            start_sec=float(row['start_sec']),
            end_sec=float(row['end_sec']),
            window_duration_sec=window_duration_sec,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        for _, row in detections.iterrows()
    ]

    if not spectrograms:
        return np.empty((0, encoder.output_dim), dtype=np.float32)

    return encoder.encode_batch(spectrograms, batch_size=batch_size)


def extract_recording_spectrograms(
    audio_path: str,
    detections_csv: str,
    window_duration_sec: float = 0.30,
    freq_min: int = 30_000,
    freq_max: int = 130_000,
) -> list[np.ndarray]:
    """
    Extract raw spectrograms for every detected call in one recording.

    Same audio loading and windowing logic as extract_recording_features, but
    returns the spectrograms themselves (before encoding) so that input-level
    augmentation can be applied before the encoder.

    Returns:
        List of (512, 512) float32 arrays with values in [0, 1].
        Empty list if no calls are detected.
    """
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    detections = pd.read_csv(detections_csv)

    required = {'start_sec', 'end_sec'}
    if not required.issubset(detections.columns):
        raise ValueError(
            f"{detections_csv} must contain columns {required}. "
            f"Found: {list(detections.columns)}"
        )

    return [
        generate_call_spectrogram(
            audio, sr,
            start_sec=float(row['start_sec']),
            end_sec=float(row['end_sec']),
            window_duration_sec=window_duration_sec,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        for _, row in detections.iterrows()
    ]


def add_spectrogram_noise(
    spectrograms: list[np.ndarray],
    noise_std: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Add Gaussian noise to spectrograms at the input level (before encoding).

    Each spectrogram is perturbed independently with i.i.d. Gaussian noise,
    then clipped to [0, 1] to stay within valid pixel range.

    Args:
        spectrograms: List of (512, 512) float32 arrays in [0, 1].
        noise_std:    Standard deviation of the additive noise.
        rng:          NumPy random generator for reproducibility.

    Returns:
        List of noisy spectrograms, same shape and dtype.
    """
    return [
        np.clip(
            spec + rng.normal(0.0, noise_std, size=spec.shape).astype(np.float32),
            0.0, 1.0,
        )
        for spec in spectrograms
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 4. Class-balance augmentation
# ──────────────────────────────────────────────────────────────────────────────

def augment_recordings_to_balance(
    recording_meta: list[tuple[int, list[np.ndarray]]],
    noise_std: float = 0.05,
    target_count: Optional[int] = None,
    random_seed: int = 42,
) -> list[tuple[int, list[np.ndarray]]]:
    """
    Balance class representation by synthesising new virtual recordings for
    minority classes.

    Augmentation is done at the **recording** level (before pooling).  For
    each minority-class recording we need to add, we randomly sample one
    existing real recording from that class and apply per-feature Gaussian
    noise to every call vector it contains.  The result is a new virtual
    recording whose pooled representation will differ slightly from the
    original, giving the classifier diverse training examples.

    After this function all classes have the same number of recordings, so
    after pooling each class contributes an equal number of vectors.

    Args:
        recording_meta:  List of (label, call_features) tuples — one entry
                         per recording.  call_features is a list of
                         (output_dim,) float32 arrays.
        noise_std:       Noise amplitude as a fraction of each call's
                         per-feature standard deviation across all calls of
                         that class.  0.05 (5%) perturbs gently while
                         preserving call-level structure.
        target_count:    Desired number of recordings per class.  Defaults to
                         the count of the majority class.
        random_seed:     For reproducibility.

    Returns:
        Extended recording_meta list.  Original entries are unchanged;
        synthetic entries are appended at the end.

    Example
    ───────
    twitcher has 3 recordings, wildtype has 9.
    After augment_recordings_to_balance(), twitcher has 9 recordings (3 real
    + 6 synthetic), wildtype still has 9.  Each pooler.pool() call then yields
    one vector, giving 9 vectors per class for training.
    """
    rng = np.random.default_rng(random_seed)

    # Group existing recordings by class label
    class_recordings: dict[int, list[list[np.ndarray]]] = {}
    for label, calls in recording_meta:
        class_recordings.setdefault(label, []).append(calls)

    if target_count is None:
        target_count = max(len(recs) for recs in class_recordings.values())

    # Pre-compute per-feature std from all calls of each class (for noise scale)
    class_feat_std: dict[int, np.ndarray] = {}
    for label, recs in class_recordings.items():
        all_calls = np.concatenate([np.stack(c) for c in recs], axis=0)
        class_feat_std[label] = np.maximum(all_calls.std(axis=0), 1e-8)

    augmented_meta: list[tuple[int, list[np.ndarray]]] = list(recording_meta)

    for label, recs in class_recordings.items():
        n_real = len(recs)
        if n_real >= target_count:
            continue

        n_to_add = target_count - n_real
        feat_std = class_feat_std[label]

        for _ in range(n_to_add):
            # Pick a random real recording to clone
            src_calls = recs[rng.integers(0, n_real)]
            calls_array = np.stack(src_calls)               # (n_calls, d)
            noise = rng.normal(0.0, noise_std, size=calls_array.shape)
            noise *= feat_std[np.newaxis, :]
            synthetic_calls = list(calls_array + noise)
            augmented_meta.append((label, synthetic_calls))

    return augmented_meta


def augment_to_balance(
    class_call_features: dict[int, list[np.ndarray]],
    noise_std: float = 0.05,
    target_count: Optional[int] = None,
    random_seed: int = 42,
) -> dict[int, list[np.ndarray]]:
    """
    Balance class representation by augmenting minority-class call vectors
    with additive Gaussian noise.

    .. deprecated::
        Prefer :func:`augment_recordings_to_balance`, which works at the
        recording level and guarantees equal pooled-vector counts per class.
        This function operates at the individual-call level and lumps all
        synthetic calls into a single virtual recording per class.

    Args:
        class_call_features: Dict mapping class_label (int) → list of
                             (output_dim,) feature arrays, one per call.
        noise_std:           Noise fraction of per-feature std.
        target_count:        Target call count per class (default: majority).
        random_seed:         For reproducibility.

    Returns:
        Augmented dict with synthetic calls appended for minority classes.
    """
    rng = np.random.default_rng(random_seed)
    if target_count is None:
        target_count = max(len(v) for v in class_call_features.values())

    augmented: dict[int, list[np.ndarray]] = {}
    for label, calls in class_call_features.items():
        n_real = len(calls)
        augmented[label] = list(calls)

        if n_real >= target_count:
            continue

        calls_array = np.stack(calls)                        # (n, d)
        per_feat_std = np.maximum(calls_array.std(axis=0), 1e-8)

        n_to_add = target_count - n_real
        base_indices = rng.integers(0, n_real, size=n_to_add)
        noise = rng.normal(0.0, noise_std, size=(n_to_add, calls_array.shape[1]))
        noise *= per_feat_std[np.newaxis, :]

        synthetic = calls_array[base_indices] + noise
        augmented[label].extend(list(synthetic))

    return augmented


def augment_by_cross_litter_mixing(
    recording_meta: list[tuple[int, list[np.ndarray]]],
    target_count: Optional[int] = None,
    target_multiplier: float = 1.0,
    n_sources: int = 2,
    random_seed: int = 42,
) -> list[tuple[int, list[np.ndarray]]]:
    """
    Create synthetic recordings by mixing calls from different real recordings
    of the same class.

    Unlike noise-based augmentation (which clones a single recording with
    jitter), cross-litter mixing draws calls from *n_sources distinct
    recordings*, producing a virtual recording whose pooled representation
    blends acoustic patterns from multiple litters.  This discourages the
    classifier from memorising litter-specific recording conditions.

    For each synthetic recording to create:
      1. Sample n_sources real recordings (with replacement when n_real < n_sources).
      2. Draw a random 50–80 % subset of calls from each source.
      3. Concatenate the subsets → one new virtual recording.

    Args:
        recording_meta:    List of (label, call_features) tuples.
        target_count:      Desired number of recordings per class after
                           augmentation.  Defaults to
                           max_class_count * target_multiplier.
        target_multiplier: Multiplier applied to the majority class count to
                           set target_count (ignored when target_count is given).
                           1.0 = balance classes; 2.0 = double the majority.
        n_sources:         Number of real recordings to mix per synthetic.
        random_seed:       For reproducibility.

    Returns:
        Extended recording_meta list.  Original entries are unchanged;
        synthetic entries are appended at the end.
    """
    rng = np.random.default_rng(random_seed)

    class_recordings: dict[int, list[list[np.ndarray]]] = {}
    for label, calls in recording_meta:
        class_recordings.setdefault(label, []).append(calls)

    if target_count is None:
        max_count = max(len(recs) for recs in class_recordings.values())
        target_count = max(1, int(round(max_count * target_multiplier)))

    augmented = list(recording_meta)

    for label, recs in class_recordings.items():
        n_real = len(recs)
        if n_real >= target_count:
            continue

        n_to_add = target_count - n_real
        # Sample with replacement when we have fewer recordings than sources
        replace = n_real < n_sources

        for _ in range(n_to_add):
            src_indices = rng.choice(n_real, size=n_sources, replace=replace)
            mixed_calls: list[np.ndarray] = []
            for src_idx in src_indices:
                src_calls = np.stack(recs[int(src_idx)])   # (n_calls, d)
                # Take a random 50–80 % of the calls from this source
                frac = rng.uniform(0.5, 0.8)
                n_take = max(1, int(round(len(src_calls) * frac)))
                chosen = rng.choice(len(src_calls), size=n_take, replace=False)
                mixed_calls.extend(list(src_calls[chosen]))
            augmented.append((label, mixed_calls))

    return augmented


# ──────────────────────────────────────────────────────────────────────────────
# 5. Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_squeakout_dataset(
    data_dir: str,
    encoder: SqueakOutEncoder,
    pooler,
    n_classes: int = 3,
    augment: bool = False,
    noise_std: float = 0.05,
    window_duration_sec: float = 0.30,
    freq_min: int = 30_000,
    freq_max: int = 130_000,
    batch_size: int = 16,
    normalize: bool = True,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a pooled feature matrix from a directory of (audio, detections) pairs.

    Expected directory layout
    ─────────────────────────
    data_dir/
      <recording_name>_twitcher.wav       ← audio file
      <recording_name>_twitcher.csv       ← detections (start_sec, end_sec)
      <recording_name>_wildtype.wav
      <recording_name>_wildtype.csv
      <recording_name>_het.wav
      <recording_name>_het.csv
      ...

    Labels are inferred from the filename (must contain 'twitcher'/'twi',
    'wildtype'/'wt', or 'het'/'heterozygous').

    The function:
      1. Extracts per-call SqueakOut features for each recording.
      2. If augment=True, adds Gaussian noise to minority-class calls to
         balance the class distribution before pooling.
      3. Pools each recording's call features → one vector per recording.
      4. Z-score normalises pooled vectors if normalize=True.

    Args:
        data_dir:            Directory containing WAV + CSV pairs.
        encoder:             Loaded SqueakOutEncoder.
        pooler:              Any pooler implementing .pool(ndarray) → ndarray
                             (AveragePooler, SWEPooler, etc.).
        n_classes:           2 or 3.
        augment:             Whether to augment minority classes. Should be
                             True only for the training split.
        noise_std:           Gaussian noise fraction for augmentation.
        window_duration_sec: Spectrogram time window per call.
        freq_min / freq_max: Frequency crop in Hz.
        batch_size:          Encoder batch size.
        normalize:           Z-score normalise the pooled features.
        feature_mean:        Pre-computed mean (pass from training set for
                             val/test normalisation).
        feature_std:         Pre-computed std (same).
        random_seed:         RNG seed for augmentation.

    Returns:
        features:    (N, pooler.output_dim) float32 array of pooled vectors.
        labels:      (N,) int array.
        feat_mean:   (pooler.output_dim,) normalisation mean (from train set).
        feat_std:    (pooler.output_dim,) normalisation std.
    """
    from .dataset import LABEL_MAP, infer_label_from_filename

    data_path = Path(data_dir)
    wav_files = sorted(data_path.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")

    # ── Step 1: extract per-call features for each recording ─────────────────
    # recording_meta: one entry per recording — (label, [call_vec, ...])
    recording_meta: list[tuple[int, list[np.ndarray]]] = []

    for wav_path in wav_files:
        csv_path = wav_path.with_suffix('.csv')
        if not csv_path.exists():
            print(f"Warning: no detections CSV for {wav_path.name}, skipping")
            continue

        try:
            label = infer_label_from_filename(wav_path.name, n_classes)
        except ValueError:
            print(f"Warning: cannot infer label from {wav_path.name}, skipping")
            continue

        call_feats = extract_recording_features(
            str(wav_path), str(csv_path), encoder,
            window_duration_sec=window_duration_sec,
            freq_min=freq_min,
            freq_max=freq_max,
            batch_size=batch_size,
        )

        if call_feats.shape[0] == 0:
            print(f"Warning: no calls detected in {wav_path.name}, skipping")
            continue

        recording_meta.append((label, list(call_feats)))

    if not recording_meta:
        raise ValueError("No valid recordings found after filtering.")

    # ── Step 2: augment minority classes (training split only) ───────────────
    # Augmentation is done at the recording level: for each missing recording
    # in a minority class we clone a real recording and add per-call Gaussian
    # noise.  After pooling, every class contributes the same number of
    # pooled vectors to the training set.
    if augment:
        recording_meta = augment_recordings_to_balance(
            recording_meta, noise_std=noise_std, random_seed=random_seed
        )

    # ── Step 3: pool each recording → one vector per recording ───────────────
    features_list = []
    labels_list = []

    for label, call_list in recording_meta:
        call_array = np.stack(call_list)           # (n_calls, output_dim)
        pooled = pooler.pool(call_array)           # (pooler.output_dim,)
        features_list.append(pooled)
        labels_list.append(label)

    features = np.stack(features_list).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    # ── Step 4: z-score normalisation ────────────────────────────────────────
    if normalize:
        if feature_mean is None:
            feature_mean = features.mean(axis=0)
            feature_std = np.maximum(features.std(axis=0), 1e-8)
        features = (features - feature_mean) / feature_std

    return features, labels, feature_mean, feature_std
