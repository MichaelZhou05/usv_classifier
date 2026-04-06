"""
Phase 2.3: MIL (Multiple Instance Learning) training for USV classification.

Instead of pooling call features into a single vector, feeds ALL per-call
features through an attention-based MIL model that learns which calls matter.

Uses concatenated encoder + spectral features per call.
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score as sk_f1
from sklearn.model_selection import StratifiedGroupKFold
import yaml

_SQUEAKOUT_DIR = Path(__file__).parent.parent / "squeakout"
if str(_SQUEAKOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SQUEAKOUT_DIR))
if 'pytorch_lightning' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['pytorch_lightning'] = MagicMock()

from data import LABEL_NAMES
from data.squeakout_features import SqueakOutEncoder, add_spectrogram_noise
from data.spectral_features import (
    extract_recording_spectral_features, N_CALL_FEATURES,
)
from train import (
    find_recordings, compute_acoustic_stats, extract_or_load,
    extract_or_load_spectrograms, ProgressLogger,
)
from models.mil import MILClassifier


def load_recording_features(
    wav, csv, encoder, cache_dir, cfg_enc, cfg_spec,
):
    """Load encoder + spectral features for one recording, concatenated per call."""
    enc_feats = extract_or_load(
        wav, csv, encoder, cache_dir,
        window_sec=cfg_spec['window_duration_sec'],
        freq_min=cfg_spec['freq_min'],
        freq_max=cfg_spec['freq_max'],
        batch_size=cfg_enc['batch_size'],
    )

    spectral_cache = cache_dir / "spectral" if cache_dir else None
    if spectral_cache:
        spectral_cache.mkdir(parents=True, exist_ok=True)
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
        return None

    # Align call counts (use min of both)
    n_calls = min(enc_feats.shape[0], spec_feats.shape[0])
    combined = np.hstack([enc_feats[:n_calls], spec_feats[:n_calls]])
    return combined.astype(np.float32)


def train_mil_epoch(
    model, train_data, criterion, optimizer, device,
    noise_std=0.05, rng=None, encoder=None, all_specs=None,
    spec_global_indices=None, cfg_enc=None, spec_feats_all=None,
    global_mean=None, global_std=None,
):
    """
    Train one epoch over bags (one recording at a time).

    For augmented copies, applies Gaussian noise at the spectrogram level
    (before SqueakOut encoding) each epoch, producing fresh features.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    perm = np.random.permutation(len(train_data))

    for idx in perm:
        label, feats, is_aug, global_idx = train_data[int(idx)]

        if is_aug and noise_std > 0 and rng is not None and encoder is not None:
            # Input-level noise: add noise to spectrograms → re-encode
            specs = all_specs[global_idx]
            noisy_specs = add_spectrogram_noise(specs, noise_std, rng)
            enc_feats = encoder.encode_batch(
                noisy_specs, batch_size=cfg_enc['batch_size']
            )
            # Re-combine with spectral features
            if spec_feats_all is not None:
                spec_f = spec_feats_all[global_idx]
                n_calls = min(enc_feats.shape[0], spec_f.shape[0])
                feats = np.hstack([enc_feats[:n_calls], spec_f[:n_calls]])
            else:
                feats = enc_feats
            feats = feats.astype(np.float32)
            # Re-normalise
            if global_mean is not None:
                feats = (feats - global_mean) / global_std

        x = torch.tensor(feats, dtype=torch.float32).to(device)
        y = torch.tensor([label], dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += int((logits.detach().argmax(1) == y).sum().item())
        total += 1

    return total_loss / total, correct / total


@torch.no_grad()
def eval_mil(model, eval_data, criterion, device, n_classes):
    """Evaluate MIL model on a list of (label, features, is_aug) tuples."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for entry in eval_data:
        label, feats = entry[0], entry[1]
        x = torch.tensor(feats, dtype=torch.float32).to(device)
        y = torch.tensor([label], dtype=torch.long).to(device)
        logits, _ = model(x)
        total_loss += criterion(logits, y).item()
        all_preds.append(int(logits.argmax(1).item()))
        all_labels.append(label)

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    return {
        "loss": total_loss / max(len(labels), 1),
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(sk_f1(labels, preds, average='macro', zero_division=0)),
        "preds": preds,
        "labels": labels,
    }


def cross_validate_mil(config, data_dir, detections_dir, cache_dir=None,
                       n_folds=5, job_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 42)
    n_classes = config.get("n_classes", 2)
    cfg_enc = config["encoder"]
    cfg_spec = config["spectrogram"]
    cfg_aug = config.get("augmentation", {})
    cfg_tr = config.get("training", {})

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_id = job_id or os.environ.get("SLURM_JOB_ID", "")
    run_name = f"job_{job_id}_{timestamp}_mil_cv{n_folds}" if job_id else f"run_{timestamp}_mil_cv{n_folds}"
    out_root = Path(config.get("output_directory", "outputs"))
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    plog = ProgressLogger(run_dir / "progress.log")
    plog.log(f"=== MIL USV Classifier ({n_folds}-fold CV) ===")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    cache_path = Path(cache_dir) if cache_dir else None

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
    print(f"\nMIL {n_folds}-fold CV | {len(wav_paths)} recordings | {class_counts}")

    # Load encoder
    weights_path = config.get("squeakout_weights", "../squeakout/squeakout_weights.ckpt")
    encoder = SqueakOutEncoder(
        weights_path=weights_path,
        extraction_point=cfg_enc["extraction_point"],
        device=cfg_enc.get("device", "cpu"),
    )

    # Extract all features + spectrograms (for input-level augmentation)
    print("Extracting combined call features + spectrograms...")
    all_feats = []        # combined encoder+spectral features per recording
    all_enc_feats = []    # encoder-only features per recording
    all_spec_feats = []   # spectral-only features per recording
    all_specs_raw = []    # raw spectrograms per recording (for input-level noise)
    valid_idx = []
    for i, (wav, csv) in enumerate(zip(wav_paths, csv_paths)):
        feats = load_recording_features(wav, csv, encoder, cache_path, cfg_enc, cfg_spec)
        if feats is not None:
            all_feats.append(feats)
            # Also load raw spectrograms for per-epoch noise augmentation
            specs = extract_or_load_spectrograms(
                wav, csv, cache_path,
                window_sec=cfg_spec['window_duration_sec'],
                freq_min=cfg_spec['freq_min'],
                freq_max=cfg_spec['freq_max'],
            )
            all_specs_raw.append(specs)
            # Load individual feature types for re-combining after noisy re-encoding
            enc_f = extract_or_load(
                wav, csv, encoder, cache_path,
                window_sec=cfg_spec['window_duration_sec'],
                freq_min=cfg_spec['freq_min'],
                freq_max=cfg_spec['freq_max'],
                batch_size=cfg_enc['batch_size'],
            )
            all_enc_feats.append(enc_f)
            # Spectral features (the non-encoder part)
            spectral_cache = cache_path / "spectral" if cache_path else None
            if spectral_cache:
                spectral_cache.mkdir(parents=True, exist_ok=True)
            spec_cache_file = spectral_cache / f"{wav.stem}__spectral.npy" if spectral_cache else None
            if spec_cache_file and spec_cache_file.exists():
                spec_f = np.load(spec_cache_file)
            else:
                spec_f = extract_recording_spectral_features(
                    str(wav), str(csv),
                    freq_min=cfg_spec['freq_min'],
                    freq_max=cfg_spec['freq_max'],
                    n_fft=cfg_spec.get('n_fft', 512),
                    hop_length=cfg_spec.get('hop_length', 64),
                )
                if spec_cache_file:
                    np.save(spec_cache_file, spec_f)
            all_spec_feats.append(spec_f)
            valid_idx.append(i)
        else:
            all_specs_raw.append(None)
            all_enc_feats.append(None)
            all_spec_feats.append(None)
        if (i + 1) % 20 == 0:
            n_calls = feats.shape[0] if feats is not None else 0
            print(f"  [{i+1}/{len(wav_paths)}] {n_calls} calls", flush=True)

    # Keep encoder loaded for per-epoch re-encoding
    valid_idx = np.array(valid_idx)
    valid_labels = labels[valid_idx]
    valid_groups = groups[valid_idx]
    input_dim = all_feats[0].shape[1]
    print(f"Valid: {len(valid_idx)} recordings, {input_dim}-dim per call")

    # Normalise per-call features globally (fit on all data, ok for normalisation)
    all_calls = np.vstack(all_feats)
    global_mean = all_calls.mean(axis=0)
    global_std = np.maximum(all_calls.std(axis=0), 1e-8)
    all_feats = [(f - global_mean) / global_std for f in all_feats]

    # CV loop
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_preds = np.full(len(valid_labels), -1, dtype=np.int64)

    for fold, (tr_idx, te_idx) in enumerate(
        sgkf.split(range(len(valid_labels)), valid_labels, valid_groups)
    ):
        plog.log(f"Fold {fold+1}/{n_folds} (train={len(tr_idx)}, test={len(te_idx)})")
        print(f"\n── Fold {fold+1}/{n_folds} (train={len(tr_idx)}, test={len(te_idx)}) ──")

        n_val = max(1, int(0.15 * len(tr_idx)))
        inner_tr = list(tr_idx[:-n_val])
        inner_val = list(tr_idx[-n_val:])

        # Build data lists — include global_idx for spectrogram lookup
        # (label, features, is_augmented, global_valid_idx)
        train_real = [(int(valid_labels[i]), all_feats[i], False, int(valid_idx[i]))
                      for i in inner_tr]
        val_data = [(int(valid_labels[i]), all_feats[i], False, int(valid_idx[i]))
                    for i in inner_val]
        te_data = [(int(valid_labels[i]), all_feats[i], False, int(valid_idx[i]))
                   for i in te_idx]

        # Augment: oversample minority class (duplication only, noise applied per-epoch)
        if cfg_aug.get("enabled", True):
            rng = np.random.default_rng(seed + fold)
            class_recs = {}
            for entry in train_real:
                class_recs.setdefault(entry[0], []).append(entry)
            target = max(len(v) for v in class_recs.values())
            train_data = list(train_real)
            for lbl, recs in class_recs.items():
                n = len(recs)
                if n >= target:
                    continue
                for _ in range(target - n):
                    donor = recs[rng.integers(0, n)]
                    # Mark as augmented — per-epoch noise will be applied at input level
                    train_data.append((lbl, donor[1], True, donor[3]))
        else:
            train_data = train_real

        cls_counts = {}
        for entry in train_data:
            cls_counts[entry[0]] = cls_counts.get(entry[0], 0) + 1
        print(f"  Train: {cls_counts}")

        # Model — small capacity to combat overfitting on tiny dataset
        model = MILClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            embed_dim=32,
            attn_dim=16,
            hidden_dims=[16],
            dropout=0.5,
        ).to(device)

        y_tr = np.array([entry[0] for entry in train_data])
        counts = np.bincount(y_tr, minlength=n_classes).astype(float)
        weights = len(y_tr) / (n_classes * np.maximum(counts, 1))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(device)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15
        )

        noise_std = cfg_aug.get("noise_std", 0.05)
        best_loss, stall, best_state = float("inf"), 0, None
        patience = cfg_tr.get("early_stopping_patience", 40)
        n_epochs = cfg_tr.get("n_epochs", 300)

        for ep in range(1, n_epochs + 1):
            rng_aug = np.random.default_rng(seed + fold + ep * 100)
            tr_loss, tr_acc = train_mil_epoch(
                model, train_data, criterion, optimizer, device,
                noise_std=noise_std, rng=rng_aug,
                encoder=encoder, all_specs=all_specs_raw,
                cfg_enc=cfg_enc, spec_feats_all=all_spec_feats,
                global_mean=global_mean, global_std=global_std,
            )
            vm = eval_mil(model, val_data, criterion, device, n_classes)
            scheduler.step(vm["loss"])

            if ep % 50 == 0 or ep == 1:
                print(f"    Ep {ep:4d} | tr_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                      f"val_loss={vm['loss']:.4f} acc={vm['accuracy']:.3f} "
                      f"f1={vm['macro_f1']:.3f}", flush=True)

            if vm["loss"] < best_loss:
                best_loss = vm["loss"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    print(f"    Early stop at epoch {ep}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Predict test fold
        model.eval()
        with torch.no_grad():
            for rec_idx, entry in zip(te_idx, te_data):
                x = torch.tensor(entry[1], dtype=torch.float32).to(device)
                logits, _ = model(x)
                all_preds[rec_idx] = int(logits.argmax(1).item())

        # Per-fold summary
        fold_mask = all_preds[te_idx] != -1
        if fold_mask.any():
            t_f = valid_labels[te_idx[fold_mask]]
            p_f = all_preds[te_idx[fold_mask]]
            print(f"  Fold {fold+1} — acc={float((t_f == p_f).mean()):.3f}  "
                  f"macro_F1={float(sk_f1(t_f, p_f, average='macro', zero_division=0)):.3f}")

    # Aggregate
    _label_map = {0: 'healthy', 1: 'twitcher'} if n_classes == 2 else LABEL_NAMES
    target_names = [_label_map.get(i, f"class_{i}") for i in range(n_classes)]

    mask = all_preds != -1
    t, p = valid_labels[mask], all_preds[mask]
    macro = float(sk_f1(t, p, average='macro', zero_division=0))
    acc = float((t == p).mean())

    print(f"\n{'='*65}")
    print(f"MIL CV ({n_folds} folds, {len(valid_labels)} recordings)")
    print(f"{'='*65}")
    print(f"accuracy={acc:.3f}  macro_F1={macro:.3f}")
    print(classification_report(t, p, target_names=target_names, zero_division=0))
    cm = confusion_matrix(t, p)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(f"{'='*65}")

    plog.log(f"MIL: accuracy={acc:.3f}, macro_F1={macro:.3f}")

    report = classification_report(t, p, target_names=target_names,
                                   output_dict=True, zero_division=0)
    results = {
        "method": "mil_attention",
        "n_folds": n_folds,
        "n_recordings": int(len(valid_labels)),
        "accuracy": acc,
        "macro_f1": macro,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(run_dir / "results.yaml", "w") as f:
        yaml.dump(results, f)
    plog.log(f"Done — results saved to {run_dir}")
    print(f"Results saved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--detections_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--job_id", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cross_validate_mil(
        config, args.data_dir, args.detections_dir,
        cache_dir=args.cache_dir, n_folds=args.cv_folds, job_id=args.job_id,
    )


if __name__ == "__main__":
    main()
