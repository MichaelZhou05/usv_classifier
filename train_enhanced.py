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
from copy import deepcopy
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score as sk_f1
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import yaml

_SQUEAKOUT_DIR = Path(__file__).parent.parent / "squeakout"
if str(_SQUEAKOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SQUEAKOUT_DIR))
if 'pytorch_lightning' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['pytorch_lightning'] = MagicMock()

from data import LABEL_NAMES
from data.squeakout_features import (
    SqueakOutEncoder, add_spectrogram_noise,
)
from data.spectral_features import (
    compute_acoustic_stats_from_call_features,
    extract_recording_spectral_features,
    extract_spectral_features_from_spectrograms,
    N_CALL_FEATURES,
)
from train import (
    find_recordings, extract_or_load,
    extract_or_load_spectrograms,
    make_loader, train_epoch, evaluate,
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


def balance_class_indices(labels, seed=42):
    """Compute balanced indices: [(src_idx, is_augmented)] for class balancing."""
    rng = np.random.default_rng(seed)
    class_indices: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(i)
    target = max(len(idxs) for idxs in class_indices.values())

    result = [(i, False) for i in range(len(labels))]
    for lbl, idxs in class_indices.items():
        n = len(idxs)
        if n >= target:
            continue
        for _ in range(target - n):
            donor = idxs[rng.integers(0, n)]
            result.append((donor, True))
    return result


def load_detection_durations_and_span(csv_path: Path) -> tuple[np.ndarray, float]:
    """Return per-call durations and overall recording span from a detections CSV."""
    df = pd.read_csv(csv_path)
    if df.empty or "start_sec" not in df.columns or "end_sec" not in df.columns:
        return np.empty(0, dtype=np.float32), 0.0

    durations = (df["end_sec"] - df["start_sec"]).to_numpy(dtype=np.float32)
    span = float(df["end_sec"].max()) if len(df) else 0.0
    return durations, span


def build_epoch_training_meta(
    balanced_tr,
    inner_tr,
    enc_meta_all,
    spec_meta_all,
    acous_all,
    all_specs,
    call_durations_all,
    recording_spans_all,
    encoder,
    cfg_enc,
    cfg_spec,
    noise_std,
    rng,
    encoder_only=False,
):
    """
    Build one fresh training set for the current epoch.

    Augmented copies are regenerated from noised spectrograms so encoder,
    spectral, and acoustic branches stay synchronized.
    """
    enc_tr: list[tuple[int, list[np.ndarray]]] = []
    spec_tr: list[tuple[int, list[np.ndarray]]] = []
    acous_tr: list[np.ndarray] = []

    for src_i, is_aug in balanced_tr:
        real_idx = inner_tr[src_i]
        label = enc_meta_all[real_idx][0]

        if is_aug:
            noisy_specs = add_spectrogram_noise(all_specs[real_idx], noise_std, rng)
            noisy_enc = encoder.encode_batch(
                noisy_specs, batch_size=cfg_enc["batch_size"]
            )
            enc_tr.append((label, list(noisy_enc)))

            if encoder_only:
                spec_tr.append(spec_meta_all[real_idx])
                acous_tr.append(acous_all[real_idx])
            else:
                noisy_spec = extract_spectral_features_from_spectrograms(
                    noisy_specs,
                    durations_sec=call_durations_all[real_idx],
                    freq_min=cfg_spec["freq_min"],
                    freq_max=cfg_spec["freq_max"],
                )
                spec_tr.append((label, list(noisy_spec)))
                acous_tr.append(
                    compute_acoustic_stats_from_call_features(
                        noisy_spec,
                        recording_span_sec=recording_spans_all[real_idx],
                    )
                )
        else:
            enc_tr.append(enc_meta_all[real_idx])
            spec_tr.append(spec_meta_all[real_idx])
            acous_tr.append(acous_all[real_idx])

    return enc_tr, spec_tr, np.stack(acous_tr).astype(np.float32)


def pool_and_combine(enc_meta, spec_meta, enc_pooler, spec_pooler,
                     acous_stats=None, encoder_only=False):
    """
    Pool both feature types and concatenate into combined vectors.

    Args:
        encoder_only: If True, only use encoder features (skip spectral + acoustic).

    Returns: (X_combined, y_labels)
    """
    features = []
    labels = []

    for i, ((lbl_e, enc_calls), (lbl_s, spec_calls)) in enumerate(zip(enc_meta, spec_meta)):
        assert lbl_e == lbl_s, f"Label mismatch at index {i}"

        enc_arr = np.stack(enc_calls)
        enc_pooled = enc_pooler.pool(enc_arr)

        if encoder_only:
            combined = enc_pooled
        else:
            spec_arr = np.stack(spec_calls)
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
    encoder_only=False, mlp_only=False,
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
    aug_tag = "aug" if cfg_aug.get("enabled", True) else "noaug"
    feat_tag = "enconly" if encoder_only else "allfeat"
    pool_tag = config.get("pooler", "swe")
    run_name = f"job_{job_id}_{timestamp}_{aug_tag}_{feat_tag}_{pool_tag}_cv{n_folds}" if job_id else f"run_{timestamp}_{aug_tag}_{feat_tag}_{pool_tag}_cv{n_folds}"
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

    # Extract all features + spectrograms (for input-level augmentation)
    plog.log("Extracting features + spectrograms...")
    print("\nExtracting encoder + spectral features + spectrograms...")
    enc_meta_all, spec_meta_all, valid_idx = extract_combined_features(
        wav_paths, csv_paths, labels, encoder, cache_path, cfg_enc, cfg_spec,
    )

    # Also extract raw spectrograms for per-epoch input-level noise augmentation
    all_specs: list[list[np.ndarray]] = []
    for i in valid_idx:
        specs = extract_or_load_spectrograms(
            wav_paths[i], csv_paths[i], cache_path,
            window_sec=cfg_spec['window_duration_sec'],
            freq_min=cfg_spec['freq_min'],
            freq_max=cfg_spec['freq_max'],
        )
        all_specs.append(specs)
    # Keep encoder loaded for per-epoch re-encoding of augmented copies

    valid_labels = labels[valid_idx]
    valid_groups = groups[valid_idx]
    print(f"Valid recordings: {len(valid_idx)}/{len(wav_paths)}")

    # Metadata needed to rebuild non-encoder branches for noised augmentations
    call_durations_all = []
    recording_spans_all = []
    for i in valid_idx:
        durations, span = load_detection_durations_and_span(csv_paths[i])
        call_durations_all.append(durations)
        recording_spans_all.append(span)
    recording_spans_all = np.asarray(recording_spans_all, dtype=np.float32)

    # Acoustic stats are derived from the call-level spectral features so they
    # can be regenerated for each fresh augmentation.
    acous_all = np.array([
        compute_acoustic_stats_from_call_features(
            np.stack(spec_meta_all[i][1]),
            recording_span_sec=recording_spans_all[i],
        )
        for i in range(len(spec_meta_all))
    ], dtype=np.float32)

    # Pooler configs
    enc_dim = enc_meta_all[0][1][0].shape[0] if enc_meta_all else 96
    spec_dim = spec_meta_all[0][1][0].shape[0] if spec_meta_all else N_CALL_FEATURES
    swe_cfg = config.get("swe", {})

    # Determine pooling approach
    pooler_name = config.get("pooler", "swe")
    feat_label = "encoder_only" if encoder_only else "encoder+spectral+acoustic"
    print(f"Pooler: {pooler_name} | Features: {feat_label}")

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

        enc_val = [enc_meta_all[i] for i in inner_val]
        spec_val = [spec_meta_all[i] for i in inner_val]
        enc_te = [enc_meta_all[i] for i in te_idx]
        spec_te = [spec_meta_all[i] for i in te_idx]

        # Balance training classes (duplication only, fresh features built per epoch)
        tr_labels = [enc_meta_all[i][0] for i in inner_tr]
        noise_std = cfg_aug.get("noise_std", 0.05)
        if cfg_aug.get("enabled", True):
            balanced_tr = balance_class_indices(tr_labels, seed=seed + fold)
        else:
            balanced_tr = [(i, False) for i in range(len(inner_tr))]

        # Create poolers per fold
        if pooler_name == "swe":
            enc_pooler = SWEPooler(
                n_features=enc_dim,
                num_slices=swe_cfg.get("num_slices", 128),
                num_ref_points=swe_cfg.get("num_ref_points", 70),
                freeze_swe=True,
                flatten=swe_cfg.get("flatten", True),
            )
            spec_pooler = SWEPooler(
                n_features=spec_dim,
                num_slices=swe_cfg.get("num_slices", 128),
                num_ref_points=swe_cfg.get("num_ref_points", 70),
                freeze_swe=True,
                flatten=swe_cfg.get("flatten", True),
            )
        else:
            enc_pooler = PoolerRegistry.get("average", n_features=enc_dim)
            spec_pooler = PoolerRegistry.get("average", n_features=spec_dim)

        acous_val = acous_all[inner_val]
        acous_te = acous_all[list(te_idx)]

        # Fixed validation/test matrices; training is rebuilt every epoch.
        X_val_raw, y_val = pool_and_combine(
            enc_val, spec_val, enc_pooler, spec_pooler, acous_val, encoder_only
        )
        X_te_raw, y_te = pool_and_combine(
            enc_te, spec_te, enc_pooler, spec_pooler, acous_te, encoder_only
        )

        if fold == 0:
            print(f"  Combined feature dim: {X_val_raw.shape[1]}")
            plog.log(f"Feature dim: {X_val_raw.shape[1]}")

        train_label_template = np.array(
            [enc_meta_all[inner_tr[src_i]][0] for src_i, _ in balanced_tr],
            dtype=np.int64,
        )
        cls_counts_aug = {
            int(lbl): int((train_label_template == lbl).sum())
            for lbl in np.unique(train_label_template)
        }
        print(f"  Train after aug: {cls_counts_aug}")

        counts = np.bincount(train_label_template, minlength=n_classes).astype(float)
        weights = len(train_label_template) / (n_classes * np.maximum(counts, 1))
        crit = nn.CrossEntropyLoss(
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
        cpu_model_eval_every = max(1, int(cfg_tr.get("cpu_model_eval_every", 1)))
        cpu_patience = max(1, math.ceil(patience / cpu_model_eval_every))
        for ep in range(1, cfg_tr.get("n_epochs", 300) + 1):
            rng_ep = np.random.default_rng(seed + fold * 10000 + ep)
            enc_tr_ep, spec_tr_ep, acous_tr_ep = build_epoch_training_meta(
                balanced_tr,
                inner_tr,
                enc_meta_all,
                spec_meta_all,
                acous_all,
                all_specs,
                call_durations_all,
                recording_spans_all,
                encoder,
                cfg_enc,
                cfg_spec,
                noise_std,
                rng_ep,
                encoder_only=encoder_only,
            )
            X_tr_ep_raw, y_tr_ep = pool_and_combine(
                enc_tr_ep, spec_tr_ep, enc_pooler, spec_pooler, acous_tr_ep, encoder_only
            )
            mean = X_tr_ep_raw.mean(0)
            std = np.maximum(X_tr_ep_raw.std(0), 1e-8)
            X_tr_ep = (X_tr_ep_raw - mean) / std
            X_val = (X_val_raw - mean) / std

            should_fit_cpu_models = (
                not mlp_only
                and (
                    ep == 1
                    or (cfg_aug.get("enabled", True) and ep % cpu_model_eval_every == 0)
                )
            )
            if should_fit_cpu_models:
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

            if ep % 10 == 0 or ep == 1:
                print(
                    f"  Ep {ep:4d} | "
                    f"SVM val_f1={best_svm_score:.3f} | "
                    f"RF val_f1={best_rf_score:.3f} | "
                    f"MLP val_loss={best_loss:.4f}"
                )

            if mlp_only:
                if stall_mlp >= patience:
                    break
            elif stall_svm >= cpu_patience and stall_rf >= cpu_patience and stall_mlp >= patience:
                break

        if best_mlp_norm is None:
            raise RuntimeError(f"Fold {fold + 1}: training did not produce valid model state")
        if not mlp_only and (best_svm is None or best_rf is None):
            raise RuntimeError(f"Fold {fold + 1}: SVM/RF training did not produce valid state")

        if not mlp_only:
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

    # Aggregate results
    _label_map = {0: 'healthy', 1: 'twitcher'} if n_classes == 2 else LABEL_NAMES
    target_names = [_label_map.get(i, f"class_{i}") for i in range(n_classes)]

    print(f"\n{'='*65}")
    print(f"Enhanced CV results ({n_folds} folds, {len(valid_labels)} recordings)")
    print(f"Pooler: {pooler_name} | Features: {feat_label}")
    print(f"{'='*65}")

    cv_results = {"n_folds": n_folds, "n_recordings": len(valid_labels),
                  "pooler": pooler_name, "features": feat_label,
                  "augmentation": cfg_aug.get("enabled", True),
                  "encoder_only": encoder_only,
                  "models": {}}

    model_list = [("MLP", preds_mlp)]
    if not mlp_only:
        model_list = [("SVM (rbf)", preds_svm), ("Random Forest", preds_rf)] + model_list
    for name, preds in model_list:
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
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable Gaussian noise augmentation")
    parser.add_argument("--encoder_only", action="store_true",
                        help="Use only encoder features (no spectral/acoustic)")
    parser.add_argument("--mlp_only", action="store_true",
                        help="Train only MLP; skip SVM and Random Forest")
    parser.add_argument("--pooler", choices=["swe", "average"], default=None,
                        help="Override pooler setting from config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.no_augmentation:
        config.setdefault("augmentation", {})["enabled"] = False
    if args.pooler:
        config["pooler"] = args.pooler

    cross_validate_enhanced(
        config, args.data_dir, args.detections_dir,
        cache_dir=args.cache_dir, n_folds=args.cv_folds, job_id=args.job_id,
        encoder_only=args.encoder_only, mlp_only=args.mlp_only,
    )


if __name__ == "__main__":
    main()
