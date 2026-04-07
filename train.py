"""
Training script for USV disease classifier — SqueakOut pipeline.

Two-phase workflow
──────────────────
Phase 1 (slow, done once):  detect_calls.py  — sliding-window segmentation
                             → detections.csv per recording (start_sec, end_sec)

Phase 2 (this script):      extract SqueakOut features per call  →
                             pool per recording  →  train MLP classifier

Feature caching
───────────────
Encoder inference is the most expensive step. Pass --cache_dir to save
per-recording feature arrays as .npy files. On re-runs (e.g. when tuning
model hyperparameters) features are loaded from disk instead of recomputed,
cutting runtime from ~minutes to ~seconds.

Usage
─────
    # First run (extracts + caches features, trains)
    python train.py --config config_squeakout.yaml \\
        --data_dir /path/to/recordings/ \\
        --detections_dir /path/to/detections/ \\
        --cache_dir /path/to/feature_cache/

    # Subsequent runs (uses cached features)
    python train.py --config config_squeakout.yaml \\
        --data_dir /path/to/recordings/ \\
        --detections_dir /path/to/detections/ \\
        --cache_dir /path/to/feature_cache/
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Progress logger — key milestones written to stdout AND run_dir/progress.log
# so you can `cat progress.log` for a quick status check without reading the
# full SLURM output log.
# ─────────────────────────────────────────────────────────────────────────────

class ProgressLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def log(self, msg: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = f"[{timestamp}] {msg}"
        print(entry, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(entry + '\n')

import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score as sk_f1
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import yaml

from data import (
    SqueakOutEncoder,
    extract_recording_features,
    extract_recording_spectrograms,
    add_spectrogram_noise,
    augment_recordings_to_balance,
    augment_by_cross_litter_mixing,
    infer_label_from_filename,
    stratified_split,
    LABEL_NAMES,
)
from pooling import PoolerRegistry
from models.mlp import get_model


# ─────────────────────────────────────────────────────────────────────────────
# Recording discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_recordings(
    data_dir: str,
    detections_dir: str,
    n_classes: int = 3,
) -> tuple[list[Path], list[Path], list[int]]:
    """
    Find all (audio, detections) pairs and infer labels from filenames.

    Args:
        data_dir:       Directory containing *.wav files.
        detections_dir: Directory containing *.csv detection files
                        (produced by detect_calls.py). CSV stem must match
                        the WAV stem exactly.
        n_classes:      2 or 3.

    Returns:
        (wav_paths, csv_paths, labels, litter_ids) — parallel lists of matching pairs.
    """
    data_path = Path(data_dir)
    det_path = Path(detections_dir)

    wav_paths, csv_paths, labels, litter_ids = [], [], [], []
    skipped = []

    for wav in sorted(data_path.glob("*.wav")):
        csv = det_path / (wav.stem + ".csv")
        if not csv.exists():
            skipped.append(f"  {wav.name}: no detections CSV at {csv}")
            continue
        try:
            label = infer_label_from_filename(wav.name, n_classes)
        except ValueError:
            skipped.append(f"  {wav.name}: cannot infer label")
            continue
        wav_paths.append(wav)
        csv_paths.append(csv)
        labels.append(label)
        litter_ids.append(extract_litter_id(wav))

    if skipped:
        print(f"Skipped {len(skipped)} recordings:")
        for s in skipped:
            print(s)

    return wav_paths, csv_paths, np.array(labels, dtype=np.int64), litter_ids


def extract_litter_id(wav_path: Path) -> str:
    """Extract litter number from filename.
    e.g. '2025_07_04 48-3 P7 het.wav' → '48'. Falls back to full stem.
    """
    m = re.search(r'\s(\w+)-\d+\s', wav_path.stem)
    return m.group(1) if m else wav_path.stem


def compute_acoustic_stats(csv_path: Path) -> np.ndarray:
    """
    Compute 8 per-recording acoustic summary statistics from a detection CSV.

    Features (in order):
        call_count, call_rate_per_sec,
        mean_duration, std_duration,
        mean_freq_center, std_freq_center,
        mean_freq_bandwidth, std_freq_bandwidth

    Returns (8,) float32 array; all zeros if CSV has no calls.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return np.zeros(8, dtype=np.float32)

    durations = (df['end_sec'] - df['start_sec']).values.astype(np.float32)
    span = float(df['end_sec'].max())
    call_count = float(len(df))
    freq_centers = ((df['freq_low_hz'] + df['freq_high_hz']) / 2.0).values.astype(np.float32)
    freq_bw = np.abs((df['freq_high_hz'] - df['freq_low_hz']).values).astype(np.float32)

    return np.array([
        call_count,
        call_count / max(span, 1.0),
        durations.mean(),
        durations.std() if len(durations) > 1 else 0.0,
        freq_centers.mean(),
        freq_centers.std() if len(freq_centers) > 1 else 0.0,
        freq_bw.mean(),
        freq_bw.std() if len(freq_bw) > 1 else 0.0,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction with optional disk cache
# ─────────────────────────────────────────────────────────────────────────────

def extract_or_load(
    wav_path: Path,
    csv_path: Path,
    encoder: SqueakOutEncoder,
    cache_dir: Path | None,
    window_sec: float,
    freq_min: int,
    freq_max: int,
    batch_size: int,
) -> np.ndarray:
    """
    Return per-call feature array for one recording.

    If cache_dir is set and a cached .npy exists for this recording,
    loads from disk. Otherwise runs the encoder and saves to cache.
    """
    if cache_dir is not None:
        cache_path = cache_dir / f"{wav_path.stem}__{encoder.extraction_point}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    feats = extract_recording_features(
        str(wav_path), str(csv_path), encoder,
        window_duration_sec=window_sec,
        freq_min=freq_min,
        freq_max=freq_max,
        batch_size=batch_size,
    )

    if cache_dir is not None and feats.shape[0] > 0:
        np.save(cache_path, feats)

    return feats


def extract_or_load_spectrograms(
    wav_path: Path,
    csv_path: Path,
    cache_dir: Path | None,
    window_sec: float,
    freq_min: int,
    freq_max: int,
) -> list[np.ndarray]:
    """
    Return list of (512, 512) spectrograms for one recording, with disk caching.
    """
    if cache_dir is not None:
        spec_cache = cache_dir / "spectrograms"
        spec_cache.mkdir(parents=True, exist_ok=True)
        cache_path = spec_cache / f"{wav_path.stem}__specs.npy"
        if cache_path.exists():
            arr = np.load(cache_path)
            return list(arr)

    specs = extract_recording_spectrograms(
        str(wav_path), str(csv_path),
        window_duration_sec=window_sec,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    if cache_dir is not None and specs:
        spec_cache = cache_dir / "spectrograms"
        spec_cache.mkdir(parents=True, exist_ok=True)
        cache_path = spec_cache / f"{wav_path.stem}__specs.npy"
        np.save(cache_path, np.stack(specs).astype(np.float32))

    return specs


def extract_split_features(
    wav_paths: list[Path],
    csv_paths: list[Path],
    labels: np.ndarray,
    encoder: SqueakOutEncoder,
    cache_dir: Path | None,
    cfg_enc: dict,
    cfg_spec: dict,
    augment: bool = False,
    noise_std: float = 0.05,
    random_seed: int = 42,
) -> tuple[dict[int, list[np.ndarray]], list[int]]:
    """
    Extract per-call features for a list of recordings.

    Returns:
        class_calls:  {label: [call_feat_array, ...]} — per-class call list
        valid_labels: parallel list of per-recording labels (some recordings
                      may be dropped if they have 0 valid calls)
    """
    class_calls: dict[int, list[np.ndarray]] = {}
    recording_meta: list[tuple[int, list[np.ndarray]]] = []

    for wav, csv, label in zip(wav_paths, csv_paths, labels):
        feats = extract_or_load(
            wav, csv, encoder, cache_dir,
            window_sec=cfg_spec['window_duration_sec'],
            freq_min=cfg_spec['freq_min'],
            freq_max=cfg_spec['freq_max'],
            batch_size=cfg_enc['batch_size'],
        )
        if feats.shape[0] == 0:
            print(f"  Warning: no calls in {wav.name}, skipping")
            continue

        recording_meta.append((int(label), list(feats)))
        class_calls.setdefault(int(label), []).extend(list(feats))

    if augment and recording_meta:
        recording_meta = augment_recordings_to_balance(
            recording_meta, noise_std=noise_std, random_seed=random_seed
        )

    return recording_meta, class_calls


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def pool_recordings(
    recording_meta: list[tuple[int, list[np.ndarray]]],
    pooler,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-call features → one vector per recording."""
    features, labels = [], []
    for label, calls in recording_meta:
        call_array = np.stack(calls)
        pooled = pooler.pool(call_array)
        features.append(pooled)
        labels.append(label)
    return np.stack(features).astype(np.float32), np.array(labels, dtype=np.int64)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    if len(dataset) == 0:
        raise ValueError("Cannot create DataLoader from an empty dataset.")

    effective_batch_size = min(batch_size, len(dataset))
    # Avoid dropping the only batch when the configured batch size is larger
    # than the fold's training set. Still skip a trailing size-1 batch when
    # shuffling and there is at least one full batch before it.
    drop_last = (
        shuffle
        and len(dataset) > effective_batch_size
        and len(dataset) % effective_batch_size == 1
    )
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    if total == 0:
        raise ValueError("Training loader produced zero samples for this epoch.")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=-1)
        total_loss += loss.item() * X.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    n = len(all_labels)
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    _BINARY_NAMES = {0: 'healthy', 1: 'twitcher'}
    _label_map = _BINARY_NAMES if n_classes == 2 else LABEL_NAMES
    target_names = [_label_map.get(i, f"class_{i}") for i in range(n_classes)]
    report = classification_report(labels, preds,
                                   labels=list(range(n_classes)),
                                   target_names=target_names,
                                   output_dict=True, zero_division=0)
    return {
        "loss": total_loss / n,
        "accuracy": (preds == labels).mean(),
        "macro_f1": report["macro avg"]["f1-score"],
        "report": report,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "preds": preds,
        "labels": labels,
        "probs": probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, data_dir: str, detections_dir: str,
          cache_dir: str | None = None, job_id: str | None = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Output directory (named by job_id when running under SLURM) ──────────
    job_id = job_id or os.environ.get("SLURM_JOB_ID", "")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"job_{job_id}_{timestamp}" if job_id else f"run_{timestamp}"
    out_root = Path(config.get("output_directory", "outputs"))
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    plog = ProgressLogger(run_dir / "progress.log")
    plog.log(f"=== USV Classifier Training ===")
    plog.log(f"Job ID   : {job_id or 'local run'}")
    plog.log(f"Run dir  : {run_dir}")
    plog.log(f"Device   : {device}")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        plog.log(f"Cache dir: {cache_dir}")

    cfg_enc  = config["encoder"]
    cfg_spec = config["spectrogram"]
    cfg_aug  = config.get("augmentation", {})
    cfg_tr   = config.get("training", {})
    cfg_spl  = config.get("splits", {})
    n_classes = config.get("n_classes", 3)

    # ── 1. Discover recordings ────────────────────────────────────────────────
    wav_paths, csv_paths, labels, _litter_ids = find_recordings(
        data_dir, detections_dir, n_classes
    )
    _bin_names = {0: 'healthy', 1: 'twitcher'}
    _display_names = _bin_names if n_classes == 2 else LABEL_NAMES
    class_counts = {_display_names.get(lbl, lbl): int((labels == lbl).sum())
                    for lbl in np.unique(labels)}
    plog.log(f"Recordings: {len(wav_paths)} found | {class_counts}")
    print(f"\nFound {len(wav_paths)} recordings")
    for lbl in np.unique(labels):
        print(f"  {LABEL_NAMES.get(lbl, lbl)}: {(labels == lbl).sum()}")

    if len(wav_paths) == 0:
        raise ValueError("No valid recordings found. Check data_dir and detections_dir.")

    # ── 2. Stratified split on recordings ────────────────────────────────────
    train_idx, val_idx, test_idx = stratified_split(
        labels,
        train_ratio=cfg_spl.get("train_ratio", 0.70),
        val_ratio=cfg_spl.get("val_ratio", 0.15),
        random_seed=cfg_spl.get("random_seed", seed),
    )
    plog.log(f"Split    : train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    def subset(idx):
        w = [wav_paths[i] for i in idx]
        c = [csv_paths[i] for i in idx]
        l = labels[idx]
        return w, c, l

    # ── 3. Load encoder ───────────────────────────────────────────────────────
    weights_path = config.get("squeakout_weights",
                               "../squeakout/squeakout_weights.ckpt")
    print(f"\nLoading SqueakOut encoder ({cfg_enc['extraction_point']})...")
    encoder = SqueakOutEncoder(
        weights_path=weights_path,
        extraction_point=cfg_enc["extraction_point"],
        device=cfg_enc.get("device", "cpu"),
    )
    plog.log(f"Encoder  : {cfg_enc['extraction_point']} ({encoder.output_dim}-dim)")
    print(f"  Output dim: {encoder.output_dim}")

    # ── 4. Set up pooler ──────────────────────────────────────────────────────
    pooler_name = config.get("pooler", "swe")
    pooler_kwargs = {"n_features": encoder.output_dim}
    if pooler_name == "swe":
        swe_cfg = config.get("swe", {})
        pooler_kwargs.update({
            "num_slices":     swe_cfg.get("num_slices", 16),
            "num_ref_points": swe_cfg.get("num_ref_points", 10),
            "freeze_swe":     swe_cfg.get("freeze_swe", False),
            "flatten":        swe_cfg.get("flatten", True),
        })
    pooler = PoolerRegistry.get(pooler_name, **pooler_kwargs)
    print(f"  Pooler: {pooler}")

    # ── 5. Extract features + spectrograms for training split ──────────────
    plog.log("Extracting features (cached recordings will load instantly)...")
    print("\nExtracting features...")

    cache_path = Path(cache_dir) if cache_dir else None

    # Training: extract clean features AND raw spectrograms (for per-epoch noise)
    train_recording_data = []  # [(label, clean_features, spectrograms)]
    tr_w, tr_c, tr_l = subset(train_idx)
    print(f"  train ({len(tr_w)} recordings)...")
    for wav, csv, label in zip(tr_w, tr_c, tr_l):
        feats = extract_or_load(
            wav, csv, encoder, cache_path,
            window_sec=cfg_spec['window_duration_sec'],
            freq_min=cfg_spec['freq_min'],
            freq_max=cfg_spec['freq_max'],
            batch_size=cfg_enc['batch_size'],
        )
        if feats.shape[0] == 0:
            print(f"  Warning: no calls in {wav.name}, skipping")
            continue
        specs = extract_or_load_spectrograms(
            wav, csv, cache_path,
            window_sec=cfg_spec['window_duration_sec'],
            freq_min=cfg_spec['freq_min'],
            freq_max=cfg_spec['freq_max'],
        )
        train_recording_data.append((int(label), feats, specs))

    # Val/test: features only (no augmentation needed)
    def _extract_no_aug(idx, label=""):
        w, c, l = subset(idx)
        print(f"  {label} ({len(w)} recordings)...")
        meta, _ = extract_split_features(
            w, c, l, encoder,
            cache_dir=cache_path,
            cfg_enc=cfg_enc, cfg_spec=cfg_spec,
            augment=False,
        )
        return meta

    val_meta   = _extract_no_aug(val_idx,  label="val")
    test_meta  = _extract_no_aug(test_idx, label="test")

    # ── 6. Balance training classes (duplication only, no noise) ───────────────
    aug_enabled = cfg_aug.get("enabled", True)
    noise_std   = cfg_aug.get("noise_std", 0.05)
    rng_balance = np.random.default_rng(cfg_spl.get("random_seed", seed))

    # Build balanced index: [(src_idx_into_train_recording_data, is_augmented)]
    balanced_indices: list[tuple[int, bool]] = []
    if aug_enabled:
        class_indices: dict[int, list[int]] = {}
        for i, (lbl, _, _) in enumerate(train_recording_data):
            class_indices.setdefault(lbl, []).append(i)
        target_count = max(len(idxs) for idxs in class_indices.values())

        for i in range(len(train_recording_data)):
            balanced_indices.append((i, False))
        for lbl, idxs in class_indices.items():
            n = len(idxs)
            if n >= target_count:
                continue
            for _ in range(target_count - n):
                donor = idxs[rng_balance.integers(0, n)]
                balanced_indices.append((donor, True))
    else:
        balanced_indices = [(i, False) for i in range(len(train_recording_data))]

    n_real = sum(1 for _, aug in balanced_indices if not aug)
    n_aug  = sum(1 for _, aug in balanced_indices if aug)
    plog.log(f"Training : {n_real} real + {n_aug} augmented recordings "
             f"(input-level noise, noise_std={noise_std})")
    print(f"  Balanced: {n_real} real + {n_aug} augmented (input-level noise per epoch)")

    # ── 7. Pool & normalise val/test (fixed, no augmentation) ─────────────────
    print("\nPooling val/test features...")
    X_val,  y_val  = pool_recordings(val_meta,  pooler)
    X_test, y_test = pool_recordings(test_meta, pooler)

    # Normalisation stats from clean (real) training recordings only
    clean_meta = [(lbl, list(feats)) for lbl, feats, _ in train_recording_data]
    X_train_clean, _ = pool_recordings(clean_meta, pooler)
    feat_mean = X_train_clean.mean(axis=0)
    feat_std  = np.maximum(X_train_clean.std(axis=0), 1e-8)

    X_val  = (X_val  - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    plog.log(f"Features : {X_val.shape[1]}-dim | "
             f"train={len(balanced_indices)}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    print(f"\nFeature shape: {X_val.shape[1]}-dim")
    print(f"Train: {len(balanced_indices)} samples  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # ── 8. Build model ────────────────────────────────────────────────────────
    cfg_model = config.get("model", {})
    input_dim = X_val.shape[1]
    model = get_model(
        "enriched",
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dims=cfg_model.get("hidden_dims", [256, 128, 64]),
        dropout=cfg_model.get("dropout", 0.5),
        use_batch_norm=cfg_model.get("use_batch_norm", True),
    ).to(device)
    print(f"\nModel:\n{model}")

    # ── 9. Loss with class weights ─────────────────────────────────────────────
    y_train_balanced = np.array([train_recording_data[si][0] for si, _ in balanced_indices])
    if cfg_model.get("use_class_weights", True):
        counts = np.bincount(y_train_balanced, minlength=n_classes).astype(float)
        weights = len(y_train_balanced) / (n_classes * np.maximum(counts, 1))
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Class weights: {weights.round(3)}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── 10. Optimiser + scheduler ─────────────────────────────────────────────
    batch_size = cfg_tr.get("batch_size", 8)
    val_loader   = make_loader(X_val,   y_val,   batch_size, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_tr.get("learning_rate", 1e-4),
        weight_decay=cfg_tr.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ── 11. Training loop (per-epoch input-level augmentation) ─────────────────
    n_epochs  = cfg_tr.get("n_epochs", 300)
    patience  = cfg_tr.get("early_stopping_patience", 40)
    best_loss = float("inf")
    best_ep   = 0
    stall     = 0

    plog.log(f"Training : up to {n_epochs} epochs, patience={patience}, lr={cfg_tr.get('learning_rate', 1e-4)}")
    print(f"\nTraining for up to {n_epochs} epochs (patience={patience})...")
    print("-" * 65)

    for ep in range(1, n_epochs + 1):
        # Re-encode augmented recordings with fresh noise each epoch
        rng_ep = np.random.default_rng(seed + ep)
        epoch_meta: list[tuple[int, list[np.ndarray]]] = []
        for src_idx, is_aug in balanced_indices:
            label, clean_feats, specs = train_recording_data[src_idx]
            if is_aug:
                noisy_specs = add_spectrogram_noise(specs, noise_std, rng_ep)
                noisy_feats = encoder.encode_batch(
                    noisy_specs, batch_size=cfg_enc['batch_size']
                )
                epoch_meta.append((label, list(noisy_feats)))
            else:
                epoch_meta.append((label, list(clean_feats)))

        X_train_ep, y_train_ep = pool_recordings(epoch_meta, pooler)
        X_train_ep = (X_train_ep - feat_mean) / feat_std
        train_loader = make_loader(X_train_ep, y_train_ep, batch_size, shuffle=True)

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device, n_classes)
        scheduler.step(val_m["loss"])

        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:4d} | tr_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                  f"val_loss={val_m['loss']:.4f} acc={val_m['accuracy']:.3f} "
                  f"f1={val_m['macro_f1']:.3f}")

        if val_m["loss"] < best_loss:
            best_loss = val_m["loss"]
            best_ep   = ep
            stall     = 0
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "config": config,
                "input_dim": input_dim,
                "n_classes": n_classes,
                "feat_mean": feat_mean,
                "feat_std":  feat_std,
                "pooler_name": pooler_name,
                "encoder_extraction_point": cfg_enc["extraction_point"],
            }, run_dir / "best_model.pt")
        else:
            stall += 1
            if stall >= patience:
                print(f"\nEarly stop at epoch {ep} (best={best_ep})")
                plog.log(f"Early stop at epoch {ep} | best_epoch={best_ep}, best_val_loss={best_loss:.4f}")
                break

    print("-" * 65)
    print(f"Best epoch: {best_ep}  val_loss: {best_loss:.4f}")

    # ── 12. Final evaluation on test set ──────────────────────────────────────
    ckpt = torch.load(run_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, criterion, device, n_classes)

    plog.log(f"TEST     : accuracy={test_m['accuracy']:.3f}, macro_F1={test_m['macro_f1']:.3f}")
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_m['accuracy']:.3f}")
    print(f"  Macro F1:  {test_m['macro_f1']:.3f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    _bin = {0: 'healthy', 1: 'twitcher'}
    _lm  = _bin if n_classes == 2 else LABEL_NAMES
    print(f"  Classes: {[_lm.get(i) for i in range(n_classes)]}")
    cm = np.array(test_m["confusion_matrix"])
    print(f"  {cm}")
    print("\nPer-class:")
    for cls in [LABEL_NAMES.get(i, f"class_{i}") for i in range(n_classes)]:
        if cls in test_m["report"]:
            r = test_m["report"][cls]
            print(f"  {cls:15s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")

    # Save results
    results = {
        "best_epoch": best_ep,
        "best_val_loss": float(best_loss),
        "test_accuracy": float(test_m["accuracy"]),
        "test_macro_f1": float(test_m["macro_f1"]),
        "confusion_matrix": test_m["confusion_matrix"],
        "classification_report": test_m["report"],
    }
    with open(run_dir / "results.yaml", "w") as f:
        yaml.dump(results, f)

    plog.log(f"Done     : results saved to {run_dir}")
    print(f"\nRun saved to: {run_dir}")
    return run_dir, results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(config: dict, data_dir: str, detections_dir: str,
                   cache_dir: str | None = None, n_folds: int = 5,
                   job_id: str | None = None):
    """
    Stratified k-fold cross-validation over all recordings.

    Each recording appears in the test set exactly once, giving reliable metrics
    on small datasets where a fixed 70/15/15 split produces noisy results.
    Features are loaded from cache (fast); only pooling + classifier training
    runs per fold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed    = config.get("seed", 42)

    # Output directory for CV run
    job_id = job_id or os.environ.get("SLURM_JOB_ID", "")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"job_{job_id}_{timestamp}_cv{n_folds}" if job_id else f"run_{timestamp}_cv{n_folds}"
    out_root = Path(config.get("output_directory", "outputs"))
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    plog = ProgressLogger(run_dir / "progress.log")
    plog.log(f"=== USV Classifier Cross-Validation ({n_folds} folds) ===")
    plog.log(f"Job ID   : {job_id or 'local run'}")
    plog.log(f"Run dir  : {run_dir}")
    plog.log(f"Device   : {device}")
    n_classes = config.get("n_classes", 3)
    cfg_enc  = config["encoder"]
    cfg_spec = config["spectrogram"]
    cfg_aug  = config.get("augmentation", {})
    cfg_tr   = config.get("training", {})
    cfg_model = config.get("model", {})

    wav_paths, csv_paths, labels, litter_ids = find_recordings(data_dir, detections_dir, n_classes)
    _bin_names = {0: 'healthy', 1: 'twitcher'}
    _display_names = _bin_names if n_classes == 2 else LABEL_NAMES
    class_counts = {_display_names.get(lbl, lbl): int((labels == lbl).sum())
                    for lbl in np.unique(labels)}
    plog.log(f"Recordings: {len(wav_paths)} | {class_counts}")
    print(f"\nCross-validation: {n_folds} folds over {len(wav_paths)} recordings")
    for lbl in np.unique(labels):
        print(f"  {LABEL_NAMES.get(lbl, lbl)}: {(labels == lbl).sum()}")

    # Litter-aware grouping — keeps all siblings in the same fold
    unique_litters = sorted(set(litter_ids))
    litter_to_int  = {l: i for i, l in enumerate(unique_litters)}
    groups = np.array([litter_to_int[l] for l in litter_ids])
    print(f"  Litters: {len(unique_litters)} ({unique_litters})")
    plog.log(f"Litters  : {len(unique_litters)} groups")

    # Acoustic summary features (8-dim) for every recording — loaded from CSV
    print("Computing acoustic summary statistics...")
    acous_all   = np.array([compute_acoustic_stats(p) for p in csv_paths], dtype=np.float32)
    valid_mask  = acous_all[:, 0] > 0   # call_count > 0; False for empty recordings

    weights_path = config.get("squeakout_weights", "../squeakout/squeakout_weights.ckpt")
    encoder = SqueakOutEncoder(
        weights_path=weights_path,
        extraction_point=cfg_enc["extraction_point"],
        device=cfg_enc.get("device", "cpu"),
    )
    pooler_name   = config.get("pooler", "average")
    pooler_kwargs = {"n_features": encoder.output_dim}

    # ── Extract features + spectrograms for all recordings ─────────────────
    cache_path = Path(cache_dir) if cache_dir else None
    noise_std  = cfg_aug.get("noise_std", 0.05)
    aug_enabled = cfg_aug.get("enabled", True)

    print("Extracting features + spectrograms for all recordings...")
    all_feats = []   # per-recording: np.ndarray (n_calls, dim)
    all_specs = []   # per-recording: list of (512, 512) spectrograms
    all_valid = []   # indices into original wav_paths that have valid calls

    for i, (wav, csv) in enumerate(zip(wav_paths, csv_paths)):
        feats = extract_or_load(
            wav, csv, encoder, cache_path,
            window_sec=cfg_spec['window_duration_sec'],
            freq_min=cfg_spec['freq_min'],
            freq_max=cfg_spec['freq_max'],
            batch_size=cfg_enc['batch_size'],
        )
        if feats.shape[0] == 0:
            all_feats.append(None)
            all_specs.append(None)
            continue
        specs = extract_or_load_spectrograms(
            wav, csv, cache_path,
            window_sec=cfg_spec['window_duration_sec'],
            freq_min=cfg_spec['freq_min'],
            freq_max=cfg_spec['freq_max'],
        )
        all_feats.append(feats)
        all_specs.append(specs)
        all_valid.append(i)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_true = np.array(labels, dtype=np.int64)
    preds_svm = np.full(len(labels), -1, dtype=np.int64)
    preds_rf  = np.full(len(labels), -1, dtype=np.int64)
    preds_mlp = np.full(len(labels), -1, dtype=np.int64)

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(range(len(labels)), all_true, groups)):
        plog.log(f"Fold {fold + 1}/{n_folds} (train={len(tr_idx)}, test={len(te_idx)})")
        print(f"\n── Fold {fold + 1}/{n_folds} "
              f"(train={len(tr_idx)}, test={len(te_idx)}) ──")

        pooler = PoolerRegistry.get(pooler_name, **pooler_kwargs)

        # Use last ~15% of train fold as val for MLP early stopping
        n_val = max(1, int(0.15 * len(tr_idx)))
        inner_tr = [i for i in tr_idx[:-n_val] if all_feats[i] is not None]
        inner_val = [i for i in tr_idx[-n_val:] if all_feats[i] is not None]
        te_valid = [i for i in te_idx if all_feats[i] is not None]

        # Build balanced indices for training: [(global_idx, is_augmented)]
        balanced_tr: list[tuple[int, bool]] = [(i, False) for i in inner_tr]
        if aug_enabled:
            rng_bal = np.random.default_rng(seed + fold)
            class_indices: dict[int, list[int]] = {}
            for i in inner_tr:
                class_indices.setdefault(int(all_true[i]), []).append(i)
            target_count = max(len(idxs) for idxs in class_indices.values())
            for lbl, idxs in class_indices.items():
                n = len(idxs)
                if n >= target_count:
                    continue
                for _ in range(target_count - n):
                    donor = idxs[rng_bal.integers(0, n)]
                    balanced_tr.append((donor, True))

        meta_val = [(int(all_true[i]), list(all_feats[i])) for i in inner_val]
        meta_te  = [(int(all_true[i]), list(all_feats[i])) for i in te_valid]

        # ── Acoustic summary features ─────────────────────────────────────────
        acous_real_tr = acous_all[inner_tr]
        train_label_template = np.array([int(all_true[i]) for i, _ in balanced_tr], dtype=np.int64)
        n_orig_tr     = len(inner_tr)
        n_synth       = len(train_label_template) - n_orig_tr
        if n_synth > 0:
            cls_mean = {cls: acous_real_tr[np.array([int(all_true[i]) for i in inner_tr]) == cls].mean(0)
                        for cls in np.unique(np.array([int(all_true[i]) for i in inner_tr]))}
            synth_rows = np.array([cls_mean[lbl] for lbl in train_label_template[n_orig_tr:]])
            acous_tr = np.vstack([acous_real_tr, synth_rows])
        else:
            acous_tr = acous_real_tr

        acous_val = acous_all[inner_val]
        acous_te  = acous_all[te_valid]

        X_val_raw, y_val = pool_recordings(meta_val, pooler)
        X_te_raw,  y_te  = pool_recordings(meta_te,  pooler)
        X_val_raw = np.hstack([X_val_raw, acous_val])
        X_te_raw  = np.hstack([X_te_raw,  acous_te])

        # MLP per fold — per-epoch input-level augmentation
        counts  = np.bincount(train_label_template, minlength=n_classes).astype(float)
        weights = len(train_label_template) / (n_classes * np.maximum(counts, 1))
        crit    = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(device)
        )
        model = get_model(
            "enriched", input_dim=X_val_raw.shape[1], n_classes=n_classes,
            hidden_dims=cfg_model.get("hidden_dims", [64]),
            dropout=cfg_model.get("dropout", 0.2),
            use_batch_norm=False,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(),
                               lr=cfg_tr.get("learning_rate", 1e-3),
                               weight_decay=cfg_tr.get("weight_decay", 1e-4))

        best_svm_score, best_rf_score = -np.inf, -np.inf
        best_svm, best_rf = None, None
        best_svm_norm, best_rf_norm = None, None
        stall_svm, stall_rf = 0, 0

        best_loss, stall_mlp, best_state = float("inf"), 0, None
        best_mlp_norm = None
        patience = cfg_tr.get("early_stopping_patience", 40)
        for ep in range(1, cfg_tr.get("n_epochs", 300) + 1):
            # Re-encode augmented copies with fresh noise each epoch
            rng_ep = np.random.default_rng(seed + fold * 10000 + ep)
            ep_meta: list[tuple[int, list[np.ndarray]]] = []
            for gi, is_aug in balanced_tr:
                label = int(all_true[gi])
                if is_aug:
                    noisy_specs = add_spectrogram_noise(all_specs[gi], noise_std, rng_ep)
                    noisy_feats = encoder.encode_batch(
                        noisy_specs, batch_size=cfg_enc['batch_size']
                    )
                    ep_meta.append((label, list(noisy_feats)))
                else:
                    ep_meta.append((label, list(all_feats[gi])))

            X_tr_ep_raw, y_tr_ep = pool_recordings(ep_meta, pooler)
            X_tr_ep_raw = np.hstack([X_tr_ep_raw, acous_tr])
            mean = X_tr_ep_raw.mean(0)
            std  = np.maximum(X_tr_ep_raw.std(0), 1e-8)
            X_tr_ep = (X_tr_ep_raw - mean) / std
            X_val = (X_val_raw - mean) / std

            if aug_enabled or ep == 1:
                svm = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')
                svm.fit(X_tr_ep, y_tr_ep)
                svm_val_pred = svm.predict(X_val)
                svm_score = sk_f1(y_val, svm_val_pred, average='macro', zero_division=0)
                if svm_score > best_svm_score:
                    best_svm_score = svm_score
                    best_svm = deepcopy(svm)
                    best_svm_norm = (mean.copy(), std.copy())
                    stall_svm = 0
                else:
                    stall_svm += 1

                rf = RandomForestClassifier(
                    n_estimators=200,
                    class_weight='balanced',
                    random_state=seed + fold * 10000 + ep,
                )
                rf.fit(X_tr_ep, y_tr_ep)
                rf_val_pred = rf.predict(X_val)
                rf_score = sk_f1(y_val, rf_val_pred, average='macro', zero_division=0)
                if rf_score > best_rf_score:
                    best_rf_score = rf_score
                    best_rf = deepcopy(rf)
                    best_rf_norm = (mean.copy(), std.copy())
                    stall_rf = 0
                else:
                    stall_rf += 1

            tr_loader = make_loader(X_tr_ep, y_tr_ep, cfg_tr.get("batch_size", 8), shuffle=True)
            val_loader = make_loader(X_val, y_val, cfg_tr.get("batch_size", 8), shuffle=False)

            train_epoch(model, tr_loader, crit, opt, device)
            vm = evaluate(model, val_loader, crit, device, n_classes)
            if vm["loss"] < best_loss:
                best_loss = vm["loss"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_mlp_norm = (mean.copy(), std.copy())
                stall_mlp = 0
            else:
                stall_mlp += 1
            if stall_svm >= patience and stall_rf >= patience and stall_mlp >= patience:
                break

        if best_svm is None or best_rf is None or best_mlp_norm is None:
            raise RuntimeError(f"Fold {fold + 1}: training did not produce valid model state")

        svm_mean, svm_std = best_svm_norm
        rf_mean, rf_std = best_rf_norm
        preds_svm[te_idx[:len(y_te)]] = best_svm.predict((X_te_raw - svm_mean) / svm_std)
        preds_rf[te_idx[:len(y_te)]] = best_rf.predict((X_te_raw - rf_mean) / rf_std)

        if best_state:
            model.load_state_dict(best_state)
        mlp_mean, mlp_std = best_mlp_norm
        X_te = (X_te_raw - mlp_mean) / mlp_std
        te_loader = make_loader(X_te, y_te, cfg_tr.get("batch_size", 8), shuffle=False)
        tm = evaluate(model, te_loader, crit, device, n_classes)
        preds_mlp[te_idx[:len(y_te)]] = tm["preds"]

    # ── Aggregate results ─────────────────────────────────────────────────────
    _BINARY_NAMES = {0: 'healthy', 1: 'twitcher'}
    _label_map = _BINARY_NAMES if n_classes == 2 else LABEL_NAMES
    target_names = [_label_map.get(i, f"class_{i}") for i in range(n_classes)]

    print(f"\n{'='*65}")
    print(f"Cross-validation results ({n_folds} folds, {len(labels)} recordings)")
    print(f"{'='*65}")
    plog.log(f"=== CV Results ({n_folds} folds, {len(labels)} recordings) ===")

    cv_results = {"n_folds": n_folds, "n_recordings": len(labels), "models": {}}
    for name, preds in [("SVM (rbf)", preds_svm), ("Random Forest", preds_rf), ("MLP", preds_mlp)]:
        mask = preds != -1
        t, p = all_true[mask], preds[mask]
        macro = sk_f1(t, p, average='macro', zero_division=0)
        acc   = (t == p).mean()
        report = classification_report(t, p, target_names=target_names,
                                       output_dict=True, zero_division=0)
        plog.log(f"{name:15s}: accuracy={acc:.3f}, macro_F1={macro:.3f}")
        print(f"\n{name}  —  accuracy={acc:.3f}  macro_F1={macro:.3f}")
        print(classification_report(t, p, target_names=target_names, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(t, p))
        cv_results["models"][name] = {
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(t, p).tolist(),
        }
    print(f"{'='*65}\n")

    with open(run_dir / "results.yaml", "w") as f:
        yaml.dump(cv_results, f)
    plog.log(f"Done     : results saved to {run_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train USV classifier using SqueakOut encoder features."
    )
    parser.add_argument("--config", required=True,
                        help="Path to config YAML (e.g. config_squeakout.yaml)")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing *.wav recordings")
    parser.add_argument("--detections_dir", required=True,
                        help="Directory containing *.csv detection files "
                             "(from detect_calls.py)")
    parser.add_argument("--cache_dir", default=None,
                        help="Directory to cache per-recording encoder features. "
                             "Highly recommended on the cluster: extracts once, "
                             "reuses for every subsequent training run.")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Run k-fold cross-validation instead of a fixed split. "
                             "Recommended for small datasets (e.g. --cv_folds 5). "
                             "0 = use fixed train/val/test split.")
    parser.add_argument("--job_id", default=None,
                        help="SLURM job ID for naming the output directory. "
                             "Auto-detected from $SLURM_JOB_ID if not provided.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.cv_folds > 1:
        cross_validate(config, args.data_dir, args.detections_dir,
                       args.cache_dir, n_folds=args.cv_folds, job_id=args.job_id)
    else:
        train(config, args.data_dir, args.detections_dir, args.cache_dir,
              job_id=args.job_id)


if __name__ == "__main__":
    main()
