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
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import yaml

from data import (
    SqueakOutEncoder,
    extract_recording_features,
    augment_to_balance,
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
        (wav_paths, csv_paths, labels) — parallel lists of matching pairs.
    """
    data_path = Path(data_dir)
    det_path = Path(detections_dir)

    wav_paths, csv_paths, labels = [], [], []
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

    if skipped:
        print(f"Skipped {len(skipped)} recordings:")
        for s in skipped:
            print(s)

    return wav_paths, csv_paths, np.array(labels, dtype=np.int64)


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

    if augment and class_calls:
        class_calls = augment_to_balance(
            class_calls, noise_std=noise_std, random_seed=random_seed
        )
        # Append synthetic calls as a virtual recording per minority class
        real_counts = {lbl: sum(len(c) for l2, c in recording_meta if l2 == lbl)
                       for lbl in class_calls}
        for label, all_calls in class_calls.items():
            synthetic = all_calls[real_counts.get(label, 0):]
            if synthetic:
                recording_meta.append((label, synthetic))

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle)  # drop last batch when training to avoid size-1 batches with BatchNorm


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
    target_names = [LABEL_NAMES.get(i, f"class_{i}") for i in range(n_classes)]
    report = classification_report(labels, preds, target_names=target_names,
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
          cache_dir: str | None = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Output directory
    out_root = Path(config.get("output_directory", "outputs"))
    run_dir = out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    cfg_enc  = config["encoder"]
    cfg_spec = config["spectrogram"]
    cfg_aug  = config.get("augmentation", {})
    cfg_tr   = config.get("training", {})
    cfg_spl  = config.get("splits", {})
    n_classes = config.get("n_classes", 3)

    # ── 1. Discover recordings ────────────────────────────────────────────────
    wav_paths, csv_paths, labels = find_recordings(
        data_dir, detections_dir, n_classes
    )
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

    # ── 5. Extract features per split ─────────────────────────────────────────
    print("\nExtracting features...")

    def _extract(idx, augment=False, label=""):
        w, c, l = subset(idx)
        print(f"  {label} ({len(w)} recordings)...")
        meta, _ = extract_split_features(
            w, c, l, encoder,
            cache_dir=Path(cache_dir) if cache_dir else None,
            cfg_enc=cfg_enc, cfg_spec=cfg_spec,
            augment=augment,
            noise_std=cfg_aug.get("noise_std", 0.05),
            random_seed=cfg_spl.get("random_seed", seed),
        )
        return meta

    train_meta = _extract(train_idx, augment=cfg_aug.get("enabled", True), label="train")
    val_meta   = _extract(val_idx,   augment=False, label="val")
    test_meta  = _extract(test_idx,  augment=False, label="test")

    # ── 6. Pool recordings ────────────────────────────────────────────────────
    print("\nPooling features...")
    X_train, y_train = pool_recordings(train_meta, pooler)
    X_val,   y_val   = pool_recordings(val_meta,   pooler)
    X_test,  y_test  = pool_recordings(test_meta,  pooler)

    # ── 7. Normalise (fit on train only) ──────────────────────────────────────
    feat_mean = X_train.mean(axis=0)
    feat_std  = np.maximum(X_train.std(axis=0), 1e-8)
    X_train = (X_train - feat_mean) / feat_std
    X_val   = (X_val   - feat_mean) / feat_std
    X_test  = (X_test  - feat_mean) / feat_std

    print(f"\nFeature shape: {X_train.shape[1]}-dim")
    print(f"Train: {X_train.shape[0]} samples  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # ── 8. Build model ────────────────────────────────────────────────────────
    cfg_model = config.get("model", {})
    input_dim = X_train.shape[1]
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
    if cfg_model.get("use_class_weights", True):
        counts = np.bincount(y_train, minlength=n_classes).astype(float)
        weights = len(y_train) / (n_classes * np.maximum(counts, 1))
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Class weights: {weights.round(3)}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── 10. Optimiser + scheduler ─────────────────────────────────────────────
    batch_size = cfg_tr.get("batch_size", 8)
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
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

    # ── 11. Training loop ──────────────────────────────────────────────────────
    n_epochs  = cfg_tr.get("n_epochs", 300)
    patience  = cfg_tr.get("early_stopping_patience", 40)
    best_loss = float("inf")
    best_ep   = 0
    stall     = 0

    print(f"\nTraining for up to {n_epochs} epochs (patience={patience})...")
    print("-" * 65)

    for ep in range(1, n_epochs + 1):
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
                break

    print("-" * 65)
    print(f"Best epoch: {best_ep}  val_loss: {best_loss:.4f}")

    # ── 11b. SVM baseline (fast sanity check on feature quality) ─────────────
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score as sk_f1
    svm = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_f1 = sk_f1(y_test, svm_preds, average='macro', zero_division=0)
    print(f"\nSVM baseline (rbf, C=1):  macro F1 = {svm_f1:.3f}")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_f1 = sk_f1(y_test, rf_preds, average='macro', zero_division=0)
    print(f"Random forest (200 trees): macro F1 = {rf_f1:.3f}")
    print()

    # ── 12. Final evaluation on test set ──────────────────────────────────────
    ckpt = torch.load(run_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, criterion, device, n_classes)

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_m['accuracy']:.3f}")
    print(f"  Macro F1:  {test_m['macro_f1']:.3f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  Classes: {[LABEL_NAMES.get(i) for i in range(n_classes)]}")
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

    print(f"\nRun saved to: {run_dir}")
    return run_dir, results


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
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.data_dir, args.detections_dir, args.cache_dir)


if __name__ == "__main__":
    main()
