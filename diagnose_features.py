"""
Phase 1 Diagnostic: Feature discriminability analysis.

Answers two key questions:
  1. Do SqueakOut encoder features contain class-separating structure?
  2. Can acoustic summary stats alone outperform encoder features?

Outputs:
  - PCA variance explained
  - Linear probe (LogisticRegression) accuracy + F1 on encoder features
  - Linear probe on acoustic stats only
  - Linear probe on combined features
  - Per-feature Fisher discriminant ratios

Usage:
    python diagnose_features.py \
        --config config_squeakout.yaml \
        --data_dir /path/to/USVRecordingsP7 \
        --detections_dir /path/to/detections \
        --cache_dir /path/to/feature_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import yaml

# ── Locate project modules ──────────────────────────────────────────────────
_SQUEAKOUT_DIR = Path(__file__).parent.parent / "squeakout"
if str(_SQUEAKOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SQUEAKOUT_DIR))

if 'pytorch_lightning' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['pytorch_lightning'] = MagicMock()

from data import infer_label_from_filename, LABEL_NAMES
from data.squeakout_features import SqueakOutEncoder, extract_recording_features
from train import find_recordings, extract_litter_id, compute_acoustic_stats, extract_or_load
from pooling import PoolerRegistry
import re


def fisher_discriminant_ratio(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature Fisher discriminant ratio: (mu1 - mu2)^2 / (var1 + var2)."""
    classes = np.unique(y)
    if len(classes) != 2:
        # Generalise: average pairwise FDR
        from itertools import combinations
        ratios = np.zeros(X.shape[1])
        for c1, c2 in combinations(classes, 2):
            x1, x2 = X[y == c1], X[y == c2]
            mu_diff = (x1.mean(0) - x2.mean(0)) ** 2
            var_sum = x1.var(0) + x2.var(0) + 1e-12
            ratios += mu_diff / var_sum
        return ratios / max(len(list(combinations(classes, 2))), 1)
    x0, x1 = X[y == classes[0]], X[y == classes[1]]
    mu_diff = (x0.mean(0) - x1.mean(0)) ** 2
    var_sum = x0.var(0) + x1.var(0) + 1e-12
    return mu_diff / var_sum


def cv_evaluate(X, y, groups, model_fn, n_folds=5, seed=42):
    """Stratified group k-fold CV, returns macro_f1 and per-fold results."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_preds = np.full(len(y), -1, dtype=np.int64)

    for tr_idx, te_idx in sgkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        model = model_fn()
        model.fit(X_tr, y[tr_idx])
        all_preds[te_idx] = model.predict(X_te)

    mask = all_preds != -1
    t, p = y[mask], all_preds[mask]
    acc = (t == p).mean()
    macro_f1 = f1_score(t, p, average='macro', zero_division=0)
    return acc, macro_f1, t, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--detections_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_classes = config.get("n_classes", 2)
    cfg_enc = config["encoder"]
    cfg_spec = config["spectrogram"]
    seed = config.get("seed", 42)

    # ── Discover recordings ──────────────────────────────────────────────────
    wav_paths, csv_paths, labels, litter_ids = find_recordings(
        args.data_dir, args.detections_dir, n_classes
    )
    unique_litters = sorted(set(litter_ids))
    litter_to_int = {l: i for i, l in enumerate(unique_litters)}
    groups = np.array([litter_to_int[l] for l in litter_ids])

    _names = {0: 'healthy', 1: 'twitcher'} if n_classes == 2 else LABEL_NAMES
    print(f"Recordings: {len(wav_paths)}")
    for lbl in np.unique(labels):
        print(f"  {_names.get(lbl, lbl)}: {(labels == lbl).sum()}")
    print(f"Litters: {len(unique_litters)}")

    # ── Load encoder ─────────────────────────────────────────────────────────
    weights_path = config.get("squeakout_weights", "../squeakout/squeakout_weights.ckpt")

    # Test both extraction points
    for ep in ["x4", "deep"]:
        print(f"\n{'='*65}")
        print(f"  Extraction point: {ep}")
        print(f"{'='*65}")

        encoder = SqueakOutEncoder(
            weights_path=weights_path,
            extraction_point=ep,
            device=cfg_enc.get("device", "cpu"),
        )

        # ── Extract per-call features ────────────────────────────────────────
        all_call_feats = []
        valid_idx = []
        for i, (wav, csv) in enumerate(zip(wav_paths, csv_paths)):
            cache_dir = Path(args.cache_dir) if args.cache_dir else None
            feats = extract_or_load(
                wav, csv, encoder, cache_dir,
                window_sec=cfg_spec['window_duration_sec'],
                freq_min=cfg_spec['freq_min'],
                freq_max=cfg_spec['freq_max'],
                batch_size=cfg_enc['batch_size'],
            )
            if feats.shape[0] > 0:
                all_call_feats.append(feats)
                valid_idx.append(i)

        valid_idx = np.array(valid_idx)
        valid_labels = labels[valid_idx]
        valid_groups = groups[valid_idx]

        # Average-pooled features per recording
        avg_feats = np.array([f.mean(axis=0) for f in all_call_feats], dtype=np.float32)
        print(f"\nEncoder features: {avg_feats.shape} (recordings x features)")

        # ── PCA ──────────────────────────────────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(avg_feats)
        pca = PCA(n_components=min(10, avg_feats.shape[1]))
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        print(f"\nPCA variance explained (first 10 components):")
        for i, (v, cv) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
            print(f"  PC{i+1}: {v:.4f} (cumulative: {cv:.4f})")

        # ── Fisher discriminant ratio ────────────────────────────────────────
        fdr = fisher_discriminant_ratio(avg_feats, valid_labels)
        top_k = min(10, len(fdr))
        top_idx = np.argsort(fdr)[::-1][:top_k]
        print(f"\nTop {top_k} features by Fisher discriminant ratio:")
        for rank, idx in enumerate(top_idx):
            print(f"  Feature {idx}: FDR={fdr[idx]:.4f}")
        print(f"  Mean FDR: {fdr.mean():.4f}, Max FDR: {fdr.max():.4f}")

        # ── CV with encoder features only ────────────────────────────────────
        print(f"\n--- Encoder features only ({ep}, {avg_feats.shape[1]}-dim) ---")
        for name, model_fn in [
            ("LogReg", lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)),
            ("SVM-rbf", lambda: SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')),
            ("RF", lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)),
        ]:
            acc, mf1, t, p = cv_evaluate(avg_feats, valid_labels, valid_groups, model_fn, n_folds=args.n_folds, seed=seed)
            print(f"  {name:10s}: accuracy={acc:.3f}  macro_F1={mf1:.3f}")
            target_names = [_names.get(i, f"class_{i}") for i in range(n_classes)]
            print(classification_report(t, p, target_names=target_names, zero_division=0))

        del encoder  # free memory

    # ── Acoustic stats only ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Acoustic summary stats only (8-dim)")
    print(f"{'='*65}")

    acous = np.array([compute_acoustic_stats(p) for p in csv_paths], dtype=np.float32)
    # Filter to valid (non-empty) recordings
    valid_acous = acous[valid_idx]

    print(f"  Features: {valid_acous.shape}")
    fdr_a = fisher_discriminant_ratio(valid_acous, valid_labels)
    stat_names = ["call_count", "call_rate", "mean_dur", "std_dur",
                  "mean_freq_center", "std_freq_center", "mean_freq_bw", "std_freq_bw"]
    print(f"\n  Fisher discriminant ratios per acoustic stat:")
    for i, (name, f) in enumerate(zip(stat_names, fdr_a)):
        print(f"    {name:20s}: {f:.4f}")

    for name, model_fn in [
        ("LogReg", lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)),
        ("SVM-rbf", lambda: SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')),
        ("RF", lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)),
    ]:
        acc, mf1, t, p = cv_evaluate(valid_acous, valid_labels, valid_groups, model_fn, n_folds=args.n_folds, seed=seed)
        target_names = [_names.get(i, f"class_{i}") for i in range(n_classes)]
        print(f"\n  {name:10s}: accuracy={acc:.3f}  macro_F1={mf1:.3f}")
        print(classification_report(t, p, target_names=target_names, zero_division=0))

    # ── Combined: encoder (x4) + acoustic stats ─────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Combined: x4 encoder + acoustic stats ({avg_feats.shape[1] + 8}-dim)")
    print(f"{'='*65}")

    combined = np.hstack([avg_feats, valid_acous])
    for name, model_fn in [
        ("LogReg", lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)),
        ("SVM-rbf", lambda: SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')),
        ("RF", lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)),
    ]:
        acc, mf1, t, p = cv_evaluate(combined, valid_labels, valid_groups, model_fn, n_folds=args.n_folds, seed=seed)
        target_names = [_names.get(i, f"class_{i}") for i in range(n_classes)]
        print(f"\n  {name:10s}: accuracy={acc:.3f}  macro_F1={mf1:.3f}")
        print(classification_report(t, p, target_names=target_names, zero_division=0))

    print(f"\n{'='*65}")
    print("Diagnostic complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
