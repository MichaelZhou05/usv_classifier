"""
Enhanced USV classifier training with spectral features and SWE pooling.

Combines the best approaches from Phase 1-3:
  - Hand-crafted spectral features (17-dim per call)
  - SqueakOut encoder features (96-dim x4 per call)
  - SWE (Sliced-Wasserstein Embedding) pooling
  - Acoustic summary statistics
  - Class-balanced augmentation

Usage:
    python train_enhanced.py \
        --config config_squeakout.yaml \
        --data_dir /path/to/USVRecordingsP7 \
        --detections_dir /path/to/detections \
        --cache_dir /path/to/feature_cache \
        --cv_folds 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score as sk_f1
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yaml

_SQUEAKOUT_DIR = Path(__file__).parent.parent / "squeakout"
if str(_SQUEAKOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SQUEAKOUT_DIR))
if 'pytorch_lightning' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['pytorch_lightning'] = MagicMock()

from data import infer_label_from_filename, LABEL_NAMES
from data.squeakout_features import SqueakOutEncoder
from data.spectral_features import (
    extract_recording_spectral_features, summarize_call_features,
    N_CALL_FEATURES, summary_feature_dim,
)
from train import (
    find_recordings, compute_acoustic_stats, extract_or_load,
    pool_recordings, make_loader, train_epoch, evaluate,
    ProgressLogger,
)
from pooling.swe import SWEPooler
from pooling import PoolerRegistry
from models.mlp import get_model


def extract_combined_features(
    wav_paths, csv_paths, labels, encoder, cache_dir, cfg_enc, cfg_spec,
):
    """
    Extract both SqueakOut encoder features and spectral features per recording.

    Returns:
        enc_meta:   [(label, [call_enc_feats])] for encoder features
        spec_meta:  [(label, [call_spec_feats])] for spectral features
        valid_idx:  indices of recordings with valid calls
    """
    spectral_cache = cache_dir / "spectral" if cache_dir else None
    if spectral_cache:
        spectral_cache.mkdir(parents=True, exist_ok=True)

    enc_meta = []
    spec_meta = []
    valid_idx = []

    for i, (wav, csv) in enumerate(zip(wav_paths, csv_paths)):
        # Encoder features
        enc_feats = extract_or_load(
            wav, csv, encoder, cache_dir,
            window_sec=cfg_spec['window_duration_sec'],
            freq_min=cfg_spec['freq_min'],
            freq_max=cfg_spec['freq_max'],
            batch_size=cfg_enc['batch_size'],
        )

        # Spectral features
        spec_cache_file = spectral_cache / f"{wav.stem}__spectral.npy" if spectral_cache else None
        if spec_cache_file and spec_cache_file.exists():
            spec_feats = np.load(spec_cache_file)
        else:
            spec_feats = extract_recording_spectral_features(
                str(wav), str(csv),
                freq_min=cfg_spec['freq_min'],
                freq_max=cfg_spec['freq_max'],
                n_fft=cfg_spec.get('n_fft', 512),
                hop_length=cfg_spec.get('hop_length', 64),
            )
            if spec_cache_file:
                np.save(spec_cache_file, spec_feats)

        if enc_feats.shape[0] == 0 or spec_feats.shape[0] == 0:
            continue

        label = int(labels[i])
        enc_meta.append((label, list(enc_feats)))
        spec_meta.append((label, list(spec_feats)))
        valid_idx.append(i)

    return enc_meta, spec_meta, np.array(valid_idx)


def augment_meta_pair(enc_meta, spec_meta, noise_std=0.05, seed=42):
    """Augment both enc and spec meta lists in parallel to keep them aligned."""
    from data.squeakout_features import augment_recordings_to_balance

    rng = np.random.default_rng(seed)

    class_recs = {}
    for idx, (label, _) in enumerate(enc_meta):
        class_recs.setdefault(label, []).append(idx)

    target = max(len(idxs) for idxs in class_recs.values())

    aug_enc = list(enc_meta)
    aug_spec = list(spec_meta)

    for label, idxs in class_recs.items():
        n = len(idxs)
        if n >= target:
            continue
        for _ in range(target - n):
            donor_idx = idxs[rng.integers(0, n)]
            # Clone encoder features with noise
            enc_calls = np.stack(enc_meta[donor_idx][1])
            enc_std = np.maximum(enc_calls.std(axis=0), 1e-8)
            enc_noise = rng.normal(0, noise_std, size=enc_calls.shape) * enc_std
            aug_enc.append((label, list(enc_calls + enc_noise)))
            # Clone spectral features with noise
            spec_calls = np.stack(spec_meta[donor_idx][1])
            spec_std = np.maximum(spec_calls.std(axis=0), 1e-8)
            spec_noise = rng.normal(0, noise_std, size=spec_calls.shape) * spec_std
            aug_spec.append((label, list(spec_calls + spec_noise)))

    return aug_enc, aug_spec


def pool_and_combine(enc_meta, spec_meta, enc_pooler, spec_pooler,
                     acous_stats=None):
    """
    Pool both feature types and concatenate into combined vectors.

    Returns: (X_combined, y_labels)
    """
    features = []
    labels = []

    for i, ((lbl_e, enc_calls), (lbl_s, spec_calls)) in enumerate(zip(enc_meta, spec_meta)):
        assert lbl_e == lbl_s, f"Label mismatch at index {i}"

        enc_arr = np.stack(enc_calls)
        spec_arr = np.stack(spec_calls)

        enc_pooled = enc_pooler.pool(enc_arr)
        spec_pooled = spec_pooler.pool(spec_arr)

        combined = np.concatenate([enc_pooled, spec_pooled])

        # Append acoustic stats if available
        if acous_stats is not None and i < len(acous_stats):
            combined = np.concatenate([combined, acous_stats[i]])

        features.append(combined)
        labels.append(lbl_e)

    return np.stack(features).astype(np.float32), np.array(labels, dtype=np.int64)


def cross_validate_enhanced(
    config, data_dir, detections_dir, cache_dir=None, n_folds=5, job_id=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 42)
    n_classes = config.get("n_classes", 2)
    cfg_enc = config["encoder"]
    cfg_spec = config["spectrogram"]
    cfg_aug = config.get("augmentation", {})
    cfg_tr = config.get("training", {})
    cfg_model = config.get("model", {})

    # Output directory
    job_id = job_id or os.environ.get("SLURM_JOB_ID", "")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"job_{job_id}_{timestamp}_enhanced_cv{n_folds}" if job_id else f"run_{timestamp}_enhanced_cv{n_folds}"
    out_root = Path(config.get("output_directory", "outputs"))
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    plog = ProgressLogger(run_dir / "progress.log")
    plog.log(f"=== Enhanced USV Classifier ({n_folds}-fold CV) ===")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    cache_path = Path(cache_dir) if cache_dir else None

    # Discover recordings
    wav_paths, csv_paths, labels, litter_ids = find_recordings(
        data_dir, detections_dir, n_classes
    )
    unique_litters = sorted(set(litter_ids))
    litter_to_int = {l: i for i, l in enumerate(unique_litters)}
    groups = np.array([litter_to_int[l] for l in litter_ids])

    _names = {0: 'healthy', 1: 'twitcher'} if n_classes == 2 else LABEL_NAMES
    class_counts = {_names.get(lbl, lbl): int((labels == lbl).sum())
                    for lbl in np.unique(labels)}
    plog.log(f"Recordings: {len(wav_paths)} | {class_counts}")
    print(f"\n{n_folds}-fold CV | {len(wav_paths)} recordings | {class_counts}")
    print(f"Litters: {len(unique_litters)}")

    # Load encoder
    weights_path = config.get("squeakout_weights", "../squeakout/squeakout_weights.ckpt")
    encoder = SqueakOutEncoder(
        weights_path=weights_path,
        extraction_point=cfg_enc["extraction_point"],
        device=cfg_enc.get("device", "cpu"),
    )

    # Extract all features
    plog.log("Extracting features...")
    print("\nExtracting encoder + spectral features...")
    enc_meta_all, spec_meta_all, valid_idx = extract_combined_features(
        wav_paths, csv_paths, labels, encoder, cache_path, cfg_enc, cfg_spec,
    )
    del encoder  # free memory

    valid_labels = labels[valid_idx]
    valid_groups = groups[valid_idx]
    print(f"Valid recordings: {len(valid_idx)}/{len(wav_paths)}")

    # Acoustic stats for all valid recordings
    acous_all = np.array([compute_acoustic_stats(csv_paths[i]) for i in valid_idx],
                         dtype=np.float32)

    # Pooler configs
    enc_dim = enc_meta_all[0][1][0].shape[0] if enc_meta_all else 96
    spec_dim = spec_meta_all[0][1][0].shape[0] if spec_meta_all else N_CALL_FEATURES
    swe_cfg = config.get("swe", {})

    # Determine pooling approach
    pooler_name = config.get("pooler", "swe")
    print(f"Pooler: {pooler_name}")

    # CV loop
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds_svm = np.full(len(valid_labels), -1, dtype=np.int64)
    preds_rf = np.full(len(valid_labels), -1, dtype=np.int64)
    preds_mlp = np.full(len(valid_labels), -1, dtype=np.int64)

    for fold, (tr_idx, te_idx) in enumerate(
        sgkf.split(range(len(valid_labels)), valid_labels, valid_groups)
    ):
        plog.log(f"Fold {fold+1}/{n_folds} (train={len(tr_idx)}, test={len(te_idx)})")
        print(f"\n── Fold {fold+1}/{n_folds} (train={len(tr_idx)}, test={len(te_idx)}) ──")

        # Split meta
        n_val = max(1, int(0.15 * len(tr_idx)))
        inner_tr = list(tr_idx[:-n_val])
        inner_val = list(tr_idx[-n_val:])

        enc_tr = [enc_meta_all[i] for i in inner_tr]
        spec_tr = [spec_meta_all[i] for i in inner_tr]
        enc_val = [enc_meta_all[i] for i in inner_val]
        spec_val = [spec_meta_all[i] for i in inner_val]
        enc_te = [enc_meta_all[i] for i in te_idx]
        spec_te = [spec_meta_all[i] for i in te_idx]

        # Augment training data
        if cfg_aug.get("enabled", True):
            enc_tr, spec_tr = augment_meta_pair(
                enc_tr, spec_tr,
                noise_std=cfg_aug.get("noise_std", 0.05),
                seed=seed + fold,
            )

        # Create poolers per fold
        if pooler_name == "swe":
            enc_pooler = SWEPooler(
                n_features=enc_dim,
                num_slices=swe_cfg.get("num_slices", 16),
                num_ref_points=swe_cfg.get("num_ref_points", 10),
                freeze_swe=True,
                flatten=swe_cfg.get("flatten", True),
            )
            spec_pooler = SWEPooler(
                n_features=spec_dim,
                num_slices=swe_cfg.get("num_slices", 16),
                num_ref_points=swe_cfg.get("num_ref_points", 10),
                freeze_swe=True,
                flatten=swe_cfg.get("flatten", True),
            )
        else:
            enc_pooler = PoolerRegistry.get("average", n_features=enc_dim)
            spec_pooler = PoolerRegistry.get("average", n_features=spec_dim)

        # Acoustic stats for splits
        acous_tr_real = acous_all[inner_tr]
        n_real = len(inner_tr)
        n_synth = len(enc_tr) - n_real
        if n_synth > 0:
            y_real = np.array([enc_tr[i][0] for i in range(n_real)])
            cls_mean = {c: acous_tr_real[y_real == c].mean(0)
                        for c in np.unique(y_real)}
            synth_acous = np.array([cls_mean[enc_tr[n_real + j][0]]
                                    for j in range(n_synth)])
            acous_tr = np.vstack([acous_tr_real, synth_acous])
        else:
            acous_tr = acous_tr_real

        acous_val = acous_all[inner_val]
        acous_te = acous_all[list(te_idx)]

        # Pool + combine
        X_tr, y_tr = pool_and_combine(enc_tr, spec_tr, enc_pooler, spec_pooler, acous_tr)
        X_val, y_val = pool_and_combine(enc_val, spec_val, enc_pooler, spec_pooler, acous_val)
        X_te, y_te = pool_and_combine(enc_te, spec_te, enc_pooler, spec_pooler, acous_te)

        if fold == 0:
            print(f"  Combined feature dim: {X_tr.shape[1]}")
            plog.log(f"Feature dim: {X_tr.shape[1]}")

        # Normalise
        mean = X_tr.mean(0)
        std = np.maximum(X_tr.std(0), 1e-8)
        X_tr = (X_tr - mean) / std
        X_val = (X_val - mean) / std
        X_te = (X_te - mean) / std

        cls_counts_aug = {}
        for lbl in y_tr:
            cls_counts_aug[lbl] = cls_counts_aug.get(lbl, 0) + 1
        print(f"  Train after aug: {cls_counts_aug}")

        # sklearn (train+val)
        X_sk = np.vstack([X_tr, X_val])
        y_sk = np.concatenate([y_tr, y_val])

        svm = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')
        svm.fit(X_sk, y_sk)
        preds_svm[te_idx[:len(y_te)]] = svm.predict(X_te)

        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                    random_state=seed)
        rf.fit(X_sk, y_sk)
        preds_rf[te_idx[:len(y_te)]] = rf.predict(X_te)

        # MLP
        counts = np.bincount(y_tr, minlength=n_classes).astype(float)
        weights = len(y_tr) / (n_classes * np.maximum(counts, 1))
        crit = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(device)
        )
        model = get_model(
            "enriched", input_dim=X_tr.shape[1], n_classes=n_classes,
            hidden_dims=cfg_model.get("hidden_dims", [64]),
            dropout=cfg_model.get("dropout", 0.2),
            use_batch_norm=False,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(),
                               lr=cfg_tr.get("learning_rate", 1e-3),
                               weight_decay=cfg_tr.get("weight_decay", 1e-4))
        tr_loader = make_loader(X_tr, y_tr, cfg_tr.get("batch_size", 8), shuffle=True)
        val_loader = make_loader(X_val, y_val, cfg_tr.get("batch_size", 8), shuffle=False)

        best_loss, stall, best_state = float("inf"), 0, None
        patience = cfg_tr.get("early_stopping_patience", 40)
        for ep in range(1, cfg_tr.get("n_epochs", 300) + 1):
            train_epoch(model, tr_loader, crit, opt, device)
            vm = evaluate(model, val_loader, crit, device, n_classes)
            if vm["loss"] < best_loss:
                best_loss = vm["loss"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    break
        if best_state:
            model.load_state_dict(best_state)
        te_loader = make_loader(X_te, y_te, cfg_tr.get("batch_size", 8), shuffle=False)
        tm = evaluate(model, te_loader, crit, device, n_classes)
        preds_mlp[te_idx[:len(y_te)]] = tm["preds"]

    # Aggregate results
    _label_map = {0: 'healthy', 1: 'twitcher'} if n_classes == 2 else LABEL_NAMES
    target_names = [_label_map.get(i, f"class_{i}") for i in range(n_classes)]

    print(f"\n{'='*65}")
    print(f"Enhanced CV results ({n_folds} folds, {len(valid_labels)} recordings)")
    print(f"Pooler: {pooler_name} | Features: encoder+spectral+acoustic")
    print(f"{'='*65}")

    cv_results = {"n_folds": n_folds, "n_recordings": len(valid_labels),
                  "pooler": pooler_name, "features": "encoder+spectral+acoustic",
                  "models": {}}

    for name, preds in [("SVM (rbf)", preds_svm), ("Random Forest", preds_rf), ("MLP", preds_mlp)]:
        mask = preds != -1
        t, p = valid_labels[mask], preds[mask]
        macro = sk_f1(t, p, average='macro', zero_division=0)
        acc = (t == p).mean()
        report = classification_report(t, p, target_names=target_names,
                                       output_dict=True, zero_division=0)
        plog.log(f"{name:15s}: accuracy={acc:.3f}, macro_F1={macro:.3f}")
        print(f"\n{name}  —  accuracy={acc:.3f}  macro_F1={macro:.3f}")
        print(classification_report(t, p, target_names=target_names, zero_division=0))
        cm = confusion_matrix(t, p)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        cv_results["models"][name] = {
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    print(f"{'='*65}\n")

    with open(run_dir / "results.yaml", "w") as f:
        yaml.dump(cv_results, f)
    plog.log(f"Done — results saved to {run_dir}")
    print(f"Results saved to: {run_dir}")
    return run_dir, cv_results


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced USV classifier with spectral features and SWE pooling."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--detections_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--job_id", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cross_validate_enhanced(
        config, args.data_dir, args.detections_dir,
        cache_dir=args.cache_dir, n_folds=args.cv_folds, job_id=args.job_id,
    )


if __name__ == "__main__":
    main()
