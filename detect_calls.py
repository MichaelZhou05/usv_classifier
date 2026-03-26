"""
Stage 1: Sliding-window USV call detection using SqueakOut.

Processes a 5-minute (or any length) WAV file by:
  1. Slicing it into overlapping short windows (~500ms each)
  2. Generating a 512×512 spectrogram per window
  3. Running SqueakOut segmentation to get a binary mask
  4. Finding connected components in the mask → individual call bounding boxes
  5. Converting pixel coordinates → time/frequency coordinates
  6. Merging calls that were split across window boundaries
  7. Saving a detections.csv with columns [start_sec, end_sec, freq_low_hz,
     freq_high_hz, confidence]

This script produces the detections.csv that squeakout_features.py expects.

Usage (single file)
───────────────────
    python detect_calls.py \\
        --audio    /path/to/recording.wav \\
        --weights  ../squeakout/squeakout_weights.ckpt \\
        --out_csv  /path/to/detections.csv

Usage (batch — all WAVs in a directory)
────────────────────────────────────────
    python detect_calls.py \\
        --audio_dir /path/to/recordings/ \\
        --weights   ../squeakout/squeakout_weights.ckpt \\
        --out_dir   /path/to/detections/

Duke cluster tip
────────────────
On SLURM, wrap this in a job array — one job per WAV file. Each job only needs
~4GB RAM and one CPU (no GPU required for inference at this scale). Example:

    #SBATCH --array=0-N
    python detect_calls.py --audio ${AUDIO_FILES[$SLURM_ARRAY_TASK_ID]} ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import wave
from PIL import Image

# ── locate squeakout ──────────────────────────────────────────────────────────
_SQUEAKOUT_DIR = Path(__file__).parent.parent / "squeakout"
if str(_SQUEAKOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SQUEAKOUT_DIR))

from squeakout import SqueakOut_autoencoder  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Resolution note
# ─────────────────────────────────────────────────────────────────────────────
# At sr=250kHz with 512px wide image:
#   window=0.5s → 0.98ms/pixel → 50ms call ≈ 51 pixels  (good detection)
#   window=1.0s → 1.95ms/pixel → 50ms call ≈ 26 pixels  (acceptable)
#   window=2.0s → 3.91ms/pixel → 50ms call ≈ 13 pixels  (marginal)
#
# Default WINDOW_SEC=0.5 with OVERLAP=0.1s means a 5-minute file produces
# ~1200 windows. Each window takes ~20ms on CPU → ~24s total per file.
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SEC = 0.50       # duration of each spectrogram window (seconds)
OVERLAP_SEC = 0.10      # overlap between consecutive windows (seconds)
SIGMOID_THRESH = 0.51   # SqueakOut mask threshold
MIN_CALL_PX = 4         # minimum call size in pixels (both axes) to keep
MERGE_GAP_SEC = 0.010   # merge detections separated by < 10ms (same call)

FREQ_MIN_HZ = 30_000    # frequency crop — lower bound
FREQ_MAX_HZ = 130_000   # frequency crop — upper bound
N_FFT = 512
HOP_LENGTH = 64


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading (no librosa dependency — pure stdlib + numpy)
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a 16-bit or 32-bit PCM WAV file. Returns (float32 array, sr)."""
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        n_frames = w.getnframes()
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        raw = w.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    # Downmix to mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Normalise to [-1, 1]
    audio /= float(np.iinfo(dtype).max)
    return audio, sr


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram generation (numpy STFT, no librosa)
# ─────────────────────────────────────────────────────────────────────────────

def _stft(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Minimal magnitude STFT using numpy."""
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(signal) - n_fft) // hop
    if n_frames <= 0:
        # Pad signal so we get at least one frame
        signal = np.pad(signal, (0, n_fft - len(signal)))
        n_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_frames, n_fft),
        strides=(signal.strides[0] * hop, signal.strides[0]),
        writeable=False,
    ).copy()
    frames *= window
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft))  # (n_frames, n_fft//2+1)
    return spectrum.T  # (n_fft//2+1, n_frames)


def _amplitude_to_db(S: np.ndarray) -> np.ndarray:
    """Convert amplitude spectrogram to dB, referenced to max."""
    ref = S.max() if S.max() > 0 else 1.0
    return 20.0 * np.log10(np.maximum(S / ref, 1e-10))


def make_spectrogram_image(
    segment: np.ndarray,
    sr: int,
    freq_min: int = FREQ_MIN_HZ,
    freq_max: int = FREQ_MAX_HZ,
    n_fft: int = N_FFT,
    hop: int = HOP_LENGTH,
    size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a raw audio segment to a (size, size) uint8 grayscale spectrogram.

    Returns
    -------
    img_array : (size, size) uint8 — ready for PIL / SqueakOut
    freqs     : (n_freq_bins_cropped,) Hz — for converting mask rows to Hz
    """
    S = _stft(segment, n_fft, hop)                   # (n_bins, n_frames)
    S_db = _amplitude_to_db(S)

    # Frequency crop
    freqs_all = np.fft.rfftfreq(n_fft, d=1.0 / sr)  # (n_fft//2+1,)
    freq_mask = (freqs_all >= freq_min) & (freqs_all <= freq_max)
    S_crop = S_db[freq_mask, :]
    freqs = freqs_all[freq_mask]

    if S_crop.shape[0] == 0:
        S_crop = S_db
        freqs = freqs_all

    # Normalise to [0, 255]
    lo, hi = S_crop.min(), S_crop.max()
    S_norm = (S_crop - lo) / (hi - lo + 1e-8)
    S_uint8 = (S_norm * 255).astype(np.uint8)

    # Resize to (size, size) — spectrogram is (freq_bins, time_frames)
    # PIL expects (width, height) = (time_frames, freq_bins) → transpose
    img = Image.fromarray(S_uint8)
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8), freqs


# ─────────────────────────────────────────────────────────────────────────────
# SqueakOut inference
# ─────────────────────────────────────────────────────────────────────────────

def load_squeakout(weights_path: str, device: str = 'cpu') -> torch.nn.Module:
    """Load pretrained SqueakOut model in eval mode."""
    model = SqueakOut_autoencoder()
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def segment_image(
    model: torch.nn.Module,
    img_array: np.ndarray,
    device: str = 'cpu',
    threshold: float = SIGMOID_THRESH,
) -> np.ndarray:
    """
    Run SqueakOut on a (512, 512) uint8 image, return binary mask.

    Returns (512, 512) bool array — True where a USV is detected.
    """
    img_f = img_array.astype(np.float32) / 255.0
    t = torch.tensor(img_f).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,512,512)
    logits = model(t)                                              # (1,1,512,512)
    mask = torch.sigmoid(logits).squeeze().cpu().numpy() > threshold
    return mask.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Connected-component analysis → call bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_bboxes(
    mask: np.ndarray,
    min_px: int = MIN_CALL_PX,
) -> list[tuple[int, int, int, int]]:
    """
    Extract bounding boxes of connected components from a binary mask.

    Uses a fast union-find flood-fill without scipy dependency.

    Returns list of (row_min, row_max, col_min, col_max) in pixel coordinates.
    Rows = frequency axis (row 0 = freq_max, row N = freq_min — inverted).
    Cols = time axis (col 0 = window start).
    """
    # Two-pass connected components (simple raster scan)
    labels = np.zeros_like(mask, dtype=np.int32)
    parent = [0]  # index 0 unused

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    next_label = 1
    rows, cols = mask.shape

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            above = labels[r - 1, c] if r > 0 else 0
            left = labels[r, c - 1] if c > 0 else 0
            if above == 0 and left == 0:
                labels[r, c] = next_label
                parent.append(next_label)
                next_label += 1
            elif above != 0 and left == 0:
                labels[r, c] = find(above)
            elif above == 0 and left != 0:
                labels[r, c] = find(left)
            else:
                union(above, left)
                labels[r, c] = find(above)

    # Resolve labels
    for r in range(rows):
        for c in range(cols):
            if labels[r, c]:
                labels[r, c] = find(labels[r, c])

    # Collect bounding boxes
    bboxes = []
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        positions = np.argwhere(labels == lbl)
        rmin, cmin = positions.min(axis=0)
        rmax, cmax = positions.max(axis=0)
        # Filter tiny detections (noise)
        if (rmax - rmin + 1) >= min_px and (cmax - cmin + 1) >= min_px:
            bboxes.append((int(rmin), int(rmax), int(cmin), int(cmax)))
    return bboxes


def pixels_to_seconds(
    col_min: int,
    col_max: int,
    window_start_sec: float,
    window_duration_sec: float,
    img_size: int = 512,
) -> tuple[float, float]:
    """Convert pixel column range → absolute time in seconds."""
    sec_per_pixel = window_duration_sec / img_size
    start = window_start_sec + col_min * sec_per_pixel
    end = window_start_sec + (col_max + 1) * sec_per_pixel
    return start, end


def pixels_to_freq(
    row_min: int,
    row_max: int,
    freq_array: np.ndarray,
    img_size: int = 512,
) -> tuple[float, float]:
    """
    Convert pixel row range → frequency in Hz.

    The image rows are ordered from high frequency (row 0) to low frequency
    (row N) because spectrograms are typically plotted with low freq at the
    bottom but image rows start at the top.
    """
    n_freq_bins = len(freq_array)
    # Map pixel rows to frequency bin indices (inverted)
    bin_max = int(row_min * n_freq_bins / img_size)  # row_min → higher freq
    bin_min = int(row_max * n_freq_bins / img_size)  # row_max → lower freq
    bin_min = max(0, min(bin_min, n_freq_bins - 1))
    bin_max = max(0, min(bin_max, n_freq_bins - 1))
    return float(freq_array[bin_min]), float(freq_array[bin_max])


# ─────────────────────────────────────────────────────────────────────────────
# Merge detections across window boundaries
# ─────────────────────────────────────────────────────────────────────────────

def merge_detections(
    detections: list[dict],
    gap_sec: float = MERGE_GAP_SEC,
) -> list[dict]:
    """
    Merge detections that are temporally adjacent (likely the same call split
    across a sliding window boundary).

    Args:
        detections: List of dicts with keys start_sec, end_sec, freq_low_hz,
                    freq_high_hz, confidence. Must be sorted by start_sec.
        gap_sec:    Detections separated by less than this are merged.

    Returns:
        Merged list of detections.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d['start_sec'])
    merged = [dict(detections[0])]

    for d in detections[1:]:
        prev = merged[-1]
        if d['start_sec'] - prev['end_sec'] <= gap_sec:
            # Extend the previous detection
            prev['end_sec'] = max(prev['end_sec'], d['end_sec'])
            prev['freq_low_hz'] = min(prev['freq_low_hz'], d['freq_low_hz'])
            prev['freq_high_hz'] = max(prev['freq_high_hz'], d['freq_high_hz'])
            prev['confidence'] = max(prev['confidence'], d['confidence'])
        else:
            merged.append(dict(d))

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main detection pipeline
# ─────────────────────────────────────────────────────────────────────────────

def detect_calls_in_file(
    audio_path: str,
    model: torch.nn.Module,
    device: str = 'cpu',
    window_sec: float = WINDOW_SEC,
    overlap_sec: float = OVERLAP_SEC,
    freq_min: int = FREQ_MIN_HZ,
    freq_max: int = FREQ_MAX_HZ,
    sigmoid_threshold: float = SIGMOID_THRESH,
    merge_gap_sec: float = MERGE_GAP_SEC,
    min_call_px: int = MIN_CALL_PX,
    verbose: bool = True,
) -> list[dict]:
    """
    Detect USV calls in a long audio file using a sliding window.

    Args:
        audio_path:        Path to WAV file.
        model:             Loaded SqueakOut model.
        device:            Torch device string.
        window_sec:        Duration of each spectrogram window.
        overlap_sec:       Overlap between consecutive windows.
        freq_min/max:      Frequency crop bounds in Hz.
        sigmoid_threshold: Mask binarisation threshold.
        merge_gap_sec:     Gap below which adjacent detections are merged.
        min_call_px:       Minimum bounding box size to keep (noise filter).
        verbose:           Print progress.

    Returns:
        List of detection dicts, each with:
            start_sec, end_sec, freq_low_hz, freq_high_hz, confidence
    """
    import csv

    audio, sr = load_wav(audio_path)
    duration_sec = len(audio) / sr
    if verbose:
        print(f"  Audio: {duration_sec:.1f}s at {sr}Hz")

    step_sec = window_sec - overlap_sec
    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)

    n_windows = max(1, int(np.ceil((len(audio) - window_samples) / step_samples)) + 1)
    if verbose:
        print(f"  Sliding window: {window_sec}s window, {overlap_sec}s overlap → "
              f"{n_windows} windows")

    all_detections = []

    for i in range(n_windows):
        start_sample = i * step_samples
        end_sample = min(start_sample + window_samples, len(audio))
        segment = audio[start_sample:end_sample]

        # Zero-pad last window if shorter than window_samples
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)))

        window_start_sec = start_sample / sr

        # Generate spectrogram image
        img_array, freq_array = make_spectrogram_image(
            segment, sr, freq_min=freq_min, freq_max=freq_max
        )

        # SqueakOut segmentation
        mask = segment_image(model, img_array, device=device,
                             threshold=sigmoid_threshold)

        # Find call bounding boxes
        bboxes = mask_to_bboxes(mask, min_px=min_call_px)

        for (rmin, rmax, cmin, cmax) in bboxes:
            start_s, end_s = pixels_to_seconds(
                cmin, cmax, window_start_sec, window_sec
            )
            freq_lo, freq_hi = pixels_to_freq(rmin, rmax, freq_array)

            # Confidence: fraction of mask pixels that are True in the bbox
            bbox_mask = mask[rmin:rmax + 1, cmin:cmax + 1]
            confidence = float(bbox_mask.mean())

            all_detections.append({
                'start_sec': round(start_s, 6),
                'end_sec': round(end_s, 6),
                'freq_low_hz': round(freq_lo, 1),
                'freq_high_hz': round(freq_hi, 1),
                'confidence': round(confidence, 4),
            })

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_windows} windows, "
                  f"{len(all_detections)} raw detections so far")

    # Merge calls split across window boundaries
    merged = merge_detections(all_detections, gap_sec=merge_gap_sec)

    if verbose:
        print(f"  Raw detections: {len(all_detections)} → "
              f"after merge: {len(merged)}")

    return merged


def save_detections_csv(detections: list[dict], out_path: str) -> None:
    """Save detections list to CSV. Creates parent directory if needed."""
    import csv
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not detections:
        # Write header-only file so downstream code can handle gracefully
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['start_sec', 'end_sec', 'freq_low_hz',
                               'freq_high_hz', 'confidence']
            )
            writer.writeheader()
        return
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(detections[0].keys()))
        writer.writeheader()
        writer.writerows(detections)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect USV calls in WAV files using SqueakOut."
    )
    parser.add_argument(
        '--weights', required=True,
        help="Path to squeakout_weights.ckpt"
    )
    parser.add_argument(
        '--device', default='cpu',
        help="Torch device: 'cpu', 'cuda', 'mps' (default: cpu)"
    )

    # Input: single file or directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--audio', help="Path to a single WAV file")
    input_group.add_argument('--audio_dir', help="Directory of WAV files")

    # Output
    parser.add_argument(
        '--out_csv', default=None,
        help="Output CSV path (single-file mode only)"
    )
    parser.add_argument(
        '--out_dir', default=None,
        help="Output directory for CSVs (batch mode)"
    )

    # Detection parameters
    parser.add_argument('--window_sec', type=float, default=WINDOW_SEC,
                        help=f"Spectrogram window duration (default {WINDOW_SEC}s)")
    parser.add_argument('--overlap_sec', type=float, default=OVERLAP_SEC,
                        help=f"Window overlap (default {OVERLAP_SEC}s)")
    parser.add_argument('--threshold', type=float, default=SIGMOID_THRESH,
                        help=f"Sigmoid threshold (default {SIGMOID_THRESH})")
    parser.add_argument('--freq_min', type=int, default=FREQ_MIN_HZ,
                        help=f"Min frequency Hz (default {FREQ_MIN_HZ})")
    parser.add_argument('--freq_max', type=int, default=FREQ_MAX_HZ,
                        help=f"Max frequency Hz (default {FREQ_MAX_HZ})")

    args = parser.parse_args()

    print(f"Loading SqueakOut from {args.weights} on {args.device}...")
    model = load_squeakout(args.weights, device=args.device)
    print("Model loaded.")

    if args.audio:
        # Single file
        audio_path = Path(args.audio)
        out_csv = args.out_csv or str(audio_path.with_suffix('.csv'))
        print(f"\nProcessing: {audio_path.name}")
        detections = detect_calls_in_file(
            str(audio_path), model, device=args.device,
            window_sec=args.window_sec, overlap_sec=args.overlap_sec,
            freq_min=args.freq_min, freq_max=args.freq_max,
            sigmoid_threshold=args.threshold,
        )
        save_detections_csv(detections, out_csv)
        print(f"  Saved {len(detections)} calls → {out_csv}")

    else:
        # Batch mode
        audio_dir = Path(args.audio_dir)
        out_dir = Path(args.out_dir) if args.out_dir else audio_dir
        wav_files = sorted(audio_dir.glob("*.wav"))
        if not wav_files:
            print(f"No .wav files found in {audio_dir}")
            sys.exit(1)

        print(f"Found {len(wav_files)} WAV files in {audio_dir}")
        for wav_path in wav_files:
            out_csv = out_dir / (wav_path.stem + '.csv')
            print(f"\nProcessing: {wav_path.name}")
            detections = detect_calls_in_file(
                str(wav_path), model, device=args.device,
                window_sec=args.window_sec, overlap_sec=args.overlap_sec,
                freq_min=args.freq_min, freq_max=args.freq_max,
                sigmoid_threshold=args.threshold,
            )
            save_detections_csv(detections, str(out_csv))
            print(f"  Saved {len(detections)} calls → {out_csv.name}")

        print("\nDone.")


if __name__ == "__main__":
    main()
