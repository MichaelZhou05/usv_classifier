"""
Phase 2.1 test: Hand-crafted spectral features evaluation.

Compares spectral features (alone and combined with encoder features)
against the Phase 1 baselines using the same 5-fold stratified group CV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import yaml

_SQUEAKOUT_DIR = Path(__file__).parent.parent / "squeakout"
if str(_SQUEAKOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SQUEAKOUT_DIR))
if 'pytorch_lightning' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['pytorch_lightning'] = MagicMock()

from data import infer_label_from_filename, LABEL_NAMES
from data.spectral_features import (
    extract_recording_spectral_features, summarize_call_features,
    N_CALL_FEATURES, summary_feature_dim,
)
from train import find_recordings, compute_acoustic_stats
from diagnose_features import cv_evaluate, fisher_discriminant_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--detections_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Filter detections below this confidence")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_classes = config.get("n_classes", 2)
    cfg_spec = config["spectrogram"]
    seed = config.get("seed", 42)

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

    # ── Extract spectral features ────────────────────────────────────────────
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    spectral_cache_dir = cache_dir / "spectral" if cache_dir else None
    if spectral_cache_dir:
        spectral_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting per-call spectral features ({N_CALL_FEATURES}-dim per call)...")
    all_call_feats = []
    all_summaries = []
    valid_idx = []

    for i, (wav, csv) in enumerate(zip(wav_paths, csv_paths)):
        cache_file = spectral_cache_dir / f"{wav.stem}__spectral.npy" if spectral_cache_dir else None

        if cache_file and cache_file.exists():
            feats = np.load(cache_file)
        else:
            feats = extract_recording_spectral_features(
                str(wav), str(csv),
                freq_min=cfg_spec['freq_min'],
                freq_max=cfg_spec['freq_max'],
                n_fft=cfg_spec.get('n_fft', 512),
                hop_length=cfg_spec.get('hop_length', 64),
                min_confidence=args.min_confidence,
            )
            if cache_file:
                np.save(cache_file, feats)

        if feats.shape[0] > 0:
            all_call_feats.append(feats)
            all_summaries.append(summarize_call_features(feats))
            valid_idx.append(i)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(wav_paths)}] {feats.shape[0]} calls", flush=True)
        else:
            print(f"  [{i+1}/{len(wav_paths)}] {wav.stem[:50]}: no calls, skipping")

    valid_idx = np.array(valid_idx)
    valid_labels = labels[valid_idx]
    valid_groups = groups[valid_idx]

    X_spectral = np.stack(all_summaries).astype(np.float32)
    print(f"\nSpectral summary features: {X_spectral.shape}")

    # ── Also get per-call average pooled features ────────────────────────────
    X_avg = np.array([f.mean(axis=0) for f in all_call_feats], dtype=np.float32)
    print(f"Average-pooled call features: {X_avg.shape}")

    # ── Fisher discriminant ratios ───────────────────────────────────────────
    fdr = fisher_discriminant_ratio(X_spectral, valid_labels)
    print(f"\nTop features by FDR (spectral summary):")
    top_idx = np.argsort(fdr)[::-1][:15]
    from data.spectral_features import CALL_FEATURE_NAMES, STAT_NAMES
    for rank, idx in enumerate(top_idx):
        feat_i = idx // len(STAT_NAMES)
        stat_i = idx % len(STAT_NAMES)
        if feat_i < N_CALL_FEATURES:
            name = f"{CALL_FEATURE_NAMES[feat_i]}_{STAT_NAMES[stat_i]}"
        else:
            name = f"extra_{idx}"
        print(f"  {name:35s}: FDR={fdr[idx]:.4f}")
    print(f"  Mean FDR: {fdr.mean():.4f}, Max FDR: {fdr.max():.4f}")

    # ── Evaluate different feature sets ──────────────────────────────────────
    models = [
        ("LogReg", lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)),
        ("SVM-rbf", lambda: SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')),
        ("RF", lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)),
    ]

    feature_sets = {
        f"Spectral summary ({X_spectral.shape[1]}-dim)": X_spectral,
        f"Avg-pooled call feats ({X_avg.shape[1]}-dim)": X_avg,
    }

    # Also try combined with acoustic stats
    acous = np.array([compute_acoustic_stats(csv_paths[i]) for i in valid_idx], dtype=np.float32)
    feature_sets[f"Spectral + acoustic ({X_spectral.shape[1]+8}-dim)"] = np.hstack([X_spectral, acous])
    feature_sets[f"Avg-pooled + acoustic ({X_avg.shape[1]+8}-dim)"] = np.hstack([X_avg, acous])

    # Try loading encoder features for combined test
    try:
        from data.squeakout_features import SqueakOutEncoder
        from train import extract_or_load
        cfg_enc = config["encoder"]
        weights_path = config.get("squeakout_weights", "../squeakout/squeakout_weights.ckpt")
        encoder = SqueakOutEncoder(
            weights_path=weights_path,
            extraction_point=cfg_enc["extraction_point"],
            device=cfg_enc.get("device", "cpu"),
        )
        enc_feats = []
        for i in valid_idx:
            feats = extract_or_load(
                wav_paths[i], csv_paths[i], encoder,
                cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                window_sec=cfg_spec['window_duration_sec'],
                freq_min=cfg_spec['freq_min'],
                freq_max=cfg_spec['freq_max'],
                batch_size=cfg_enc['batch_size'],
            )
            enc_feats.append(feats.mean(axis=0) if feats.shape[0] > 0 else np.zeros(encoder.output_dim))
        X_enc = np.stack(enc_feats).astype(np.float32)
        feature_sets[f"Spectral + encoder ({X_spectral.shape[1]+X_enc.shape[1]}-dim)"] = np.hstack([X_spectral, X_enc])
        feature_sets[f"All combined ({X_spectral.shape[1]+X_enc.shape[1]+8}-dim)"] = np.hstack([X_spectral, X_enc, acous])
        del encoder
    except Exception as e:
        print(f"Skipping encoder features: {e}")

    target_names = [_names.get(i, f"class_{i}") for i in range(n_classes)]

    for feat_name, X in feature_sets.items():
        print(f"\n{'='*65}")
        print(f"  {feat_name}")
        print(f"{'='*65}")
        for model_name, model_fn in models:
            acc, mf1, t, p = cv_evaluate(X, valid_labels, valid_groups, model_fn,
                                          n_folds=args.n_folds, seed=seed)
            print(f"\n  {model_name:10s}: accuracy={acc:.3f}  macro_F1={mf1:.3f}")
            print(classification_report(t, p, target_names=target_names, zero_division=0))

    print(f"\n{'='*65}")
    print("Phase 2.1 evaluation complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
