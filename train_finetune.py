"""
End-to-end fine-tuning of SqueakOut encoder for USV classification.

Performance-critical design
────────────────────────────
Naively passing spectrograms through all backbone blocks each epoch is very
slow on CPU (~35 min/epoch on this dataset).  We exploit the fact that the
early backbone blocks are *frozen* and their outputs never change:

  Phase 1 (one-time, ~4 min):  load spectrograms → run blocks 0..N_frozen-1
                                → save (N_calls, C, H, W) tensors to disk.
  Phase 2 (per epoch, ~5 sec): load cached tensors → run only the 4 unfrozen
                                blocks (10-13) on tiny 32×32 feature maps
                                → pool → classify.

This gives ~30× speedup versus re-running the full backbone every epoch.
The frozen features are stored in `frozen_cache_dir/<stem>__fb<N>.npy`.

Augmentation
────────────
Minority-class recordings are duplicated to balance class counts.  The
augmented copies have Gaussian noise applied at the spectrogram level (before
SqueakOut encoding) each epoch, producing fresh input-level perturbations.
This is done before the frozen backbone blocks, not after the encoder.

Usage
─────
    python train_finetune.py \\
        --config config_squeakout.yaml \\
        --data_dir /path/to/USVs/ \\
        --detections_dir /path/to/detections/ \\
        --spec_cache_dir   ./spec_cache      # raw spectrogram cache
        --frozen_cache_dir ./frozen_cache    # intermediate feature cache (fast)
        --cv_folds 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score as sk_f1
from sklearn.model_selection import StratifiedGroupKFold
import yaml

# ── Locate SqueakOut package ──────────────────────────────────────────────────
_SQUEAKOUT_DIR = Path(__file__).parent.parent / "squeakout"
if str(_SQUEAKOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SQUEAKOUT_DIR))

if 'pytorch_lightning' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['pytorch_lightning'] = MagicMock()

from squeakout import SqueakOut  # noqa: E402

from data import infer_label_from_filename, LABEL_NAMES
from data.squeakout_features import generate_call_spectrogram, add_spectrogram_noise
from models.mlp import EnrichedUSVClassifier


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end model (operates on pre-frozen intermediate features)
# ─────────────────────────────────────────────────────────────────────────────

class FineTuneUSVModel(nn.Module):
    """
    USV classifier that fine-tunes only the last few backbone blocks.

    Operates on *pre-frozen intermediate features* (the output of the frozen
    portion of the backbone) rather than raw spectrograms.  The unfrozen blocks
    transform these cached features into task-adapted representations, which
    are then globally average-pooled and classified by an MLP.

    Forward pass for one recording:
        cached (N, C, H, W) feature maps    ← pre-computed from frozen blocks
            ↓  features[finetune_from..13]  ← fine-tuned (grads flow here)
            ↓  global average pool → (N, 96)
            ↓  mean over N calls → (1, 96)
            ↓  MLP classifier    → (1, n_classes)

    Augmentation is applied at the input (spectrogram) level before frozen
    feature computation, not after the encoder.
    """

    _N_BLOCKS = {'x4': 14, 'deep': 19}
    _OUT_DIM  = {'x4': 96, 'deep': 1280}

    def __init__(
        self,
        backbone,
        n_classes: int = 2,
        hidden_dims: list[int] = [64],
        dropout: float = 0.2,
        finetune_from: int = 10,
        extraction_point: str = 'x4',
        device: str = 'cpu',
    ):
        super().__init__()
        self.backbone     = backbone
        self.ep           = extraction_point
        self.finetune_from = finetune_from
        self.n_classes    = n_classes
        self._device      = torch.device(device)

        n_blocks = self._N_BLOCKS[extraction_point]
        # Freeze early blocks, unfreeze later blocks
        for i in range(n_blocks):
            for p in backbone.features[i].parameters():
                p.requires_grad = (i >= finetune_from)

        d = self._OUT_DIM[extraction_point]
        self.classifier = EnrichedUSVClassifier(
            input_dim=d, n_classes=n_classes,
            hidden_dims=hidden_dims, dropout=dropout, use_batch_norm=False,
        )
        self.backbone.to(self._device)
        self.classifier.to(self._device)

    def forward_from_frozen(
        self,
        frozen_feats: np.ndarray,
    ) -> torch.Tensor:
        """
        Forward pass starting from pre-frozen intermediate features.

        Args:
            frozen_feats:  (N_calls, C, H, W) float32 numpy array —
                           output of backbone.features[0..finetune_from-1].

        Returns:
            (1, n_classes) logit tensor.
        """
        x = torch.from_numpy(frozen_feats.astype(np.float32)).to(self._device)

        # Run the fine-tuned portion of the backbone
        n_blocks = self._N_BLOCKS[self.ep]
        for n in range(self.finetune_from, n_blocks):
            x = self.backbone.features[n](x)

        feats = x.mean(dim=[2, 3])  # global average pool → (N, 96)
        recording_vec = feats.mean(0, keepdim=True)  # (1, 96)
        return self.classifier(recording_vec)         # (1, n_classes)

    @torch.no_grad()
    def predict_from_frozen(self, frozen_feats: np.ndarray) -> int:
        """Eval-mode prediction from cached frozen features."""
        self.eval()
        logits = self.forward_from_frozen(frozen_feats)
        return int(logits.argmax(1).item())


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram generation + frozen feature precomputation
# ─────────────────────────────────────────────────────────────────────────────

def load_spectrograms(
    wav_path: Path,
    csv_path: Path,
    cfg_spec: dict,
    spec_cache_dir: Path | None = None,
) -> list[np.ndarray] | None:
    """
    Generate (or load cached) spectrograms for all calls in one recording.

    Returns list of (512, 512) float32 arrays, or None if no calls found.
    """
    if spec_cache_dir is not None:
        cache_file = spec_cache_dir / f"{wav_path.stem}_specs.npy"
        if cache_file.exists():
            arr = np.load(cache_file)
            return list(arr) if len(arr) > 0 else None

    audio, sr = librosa.load(str(wav_path), sr=None, mono=True)
    df = pd.read_csv(str(csv_path))

    if df.empty or 'start_sec' not in df.columns or 'end_sec' not in df.columns:
        return None

    specs = [
        generate_call_spectrogram(
            audio, sr,
            float(row['start_sec']), float(row['end_sec']),
            window_duration_sec=cfg_spec['window_duration_sec'],
            freq_min=cfg_spec['freq_min'],
            freq_max=cfg_spec['freq_max'],
        )
        for _, row in df.iterrows()
    ]

    if not specs:
        return None

    if spec_cache_dir is not None:
        spec_cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, np.stack(specs).astype(np.float32))

    return specs


def precompute_frozen_features(
    backbone,
    all_specs: list[list[np.ndarray]],
    finetune_from: int,
    extraction_point: str,
    device: torch.device,
    frozen_cache_dir: Path | None = None,
    wav_stems: list[str] | None = None,
) -> list[np.ndarray | None]:
    """
    Run all spectrograms through the frozen backbone blocks once and cache.

    For each recording:
        (N_calls, 1, 512, 512) → features[0..finetune_from-1] → (N_calls, C, H, W)

    The result is cached to disk as `<frozen_cache_dir>/<stem>__fb<N>.npy`.
    Subsequent runs load from cache rather than re-running the backbone.

    Returns:
        List of (N_calls, C, H, W) float32 numpy arrays (one per recording).
        None entries for recordings with no spectrograms.
    """
    n_blocks_total = {'x4': 14, 'deep': 19}[extraction_point]
    frozen_end = min(finetune_from, n_blocks_total)

    all_frozen: list[np.ndarray | None] = []

    print(f"Precomputing frozen features (blocks 0..{frozen_end - 1})...")
    for i, spec_list in enumerate(all_specs):
        stem = wav_stems[i] if wav_stems else str(i)

        if spec_list is None:
            all_frozen.append(None)
            continue

        # Try cache
        if frozen_cache_dir is not None:
            cache_file = frozen_cache_dir / f"{stem}__fb{frozen_end}.npy"
            if cache_file.exists():
                all_frozen.append(np.load(cache_file))
                print(f"  [{i+1:3d}/{len(all_specs)}] {stem}: loaded from cache", flush=True)
                continue

        # Compute
        X = torch.from_numpy(
            np.stack(spec_list)[:, np.newaxis].astype(np.float32)
        ).to(device)

        with torch.no_grad():
            x = X
            for n in range(frozen_end):
                x = backbone.features[n](x)
            frozen = x.cpu().numpy()  # (N, C, H, W)

        if frozen_cache_dir is not None:
            frozen_cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, frozen)

        all_frozen.append(frozen)
        print(f"  [{i+1:3d}/{len(all_specs)}] {stem}: "
              f"{len(spec_list)} calls → {frozen.shape}", flush=True)

    return all_frozen


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_train_data(
    train_real: list[tuple[int, np.ndarray, int]],
    random_seed: int = 42,
) -> list[tuple[int, np.ndarray, bool, int]]:
    """
    Balance class counts by duplicating minority-class recordings.

    Augmented copies are marked is_augmented=True so the training loop knows to
    apply input-level Gaussian noise to their spectrograms before recomputing
    frozen features each epoch.

    Args:
        train_real:   List of (label, frozen_feats, spec_index) for real recordings.
        random_seed:  RNG seed.

    Returns:
        List of (label, frozen_feats, is_augmented, spec_index).
    """
    rng = np.random.default_rng(random_seed)

    class_recs: dict[int, list[tuple[int, np.ndarray, int]]] = {}
    for label, feats, si in train_real:
        class_recs.setdefault(label, []).append((label, feats, si))

    target = max(len(recs) for recs in class_recs.values())

    result: list[tuple[int, np.ndarray, bool, int]] = [
        (label, feats, False, si) for label, feats, si in train_real
    ]
    for label, recs in class_recs.items():
        n = len(recs)
        if n >= target:
            continue
        for _ in range(target - n):
            donor = recs[rng.integers(0, n)]
            result.append((donor[0], donor[1], True, donor[2]))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_e2e(
    model: FineTuneUSVModel,
    train_data: list[tuple[int, np.ndarray, bool, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    noise_std: float,
    accum_steps: int = 8,
    all_specs: list | None = None,
    frozen_backbone=None,
    frozen_end: int = 10,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    One training epoch over pre-frozen features.

    For augmented copies, applies Gaussian noise to the raw spectrograms and
    recomputes frozen features through the frozen backbone blocks, so each
    epoch sees different input-level perturbations.

    Gradients are accumulated over `accum_steps` recordings before each step.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    n = len(train_data)
    perm = np.random.permutation(n)

    for step_i, idx in enumerate(perm, 1):
        label, frozen_feats, is_aug, spec_idx = train_data[int(idx)]
        y = torch.tensor([label], dtype=torch.long, device=model._device)

        if is_aug and rng is not None and all_specs is not None and frozen_backbone is not None:
            # Input-level augmentation: noise → re-run frozen blocks
            noisy_specs = add_spectrogram_noise(all_specs[spec_idx], noise_std, rng)
            X = torch.from_numpy(
                np.stack(noisy_specs)[:, np.newaxis].astype(np.float32)
            ).to(device or model._device)
            with torch.no_grad():
                x = X
                for blk in range(frozen_end):
                    x = frozen_backbone.features[blk](x)
                frozen_feats = x.cpu().numpy()

        logits = model.forward_from_frozen(frozen_feats)
        loss = criterion(logits, y) / n
        loss.backward()

        total_loss += loss.item() * n
        correct += int((logits.detach().argmax(1) == y).sum().item())
        total += 1

        if step_i % accum_steps == 0 or step_i == n:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / total, correct / total


@torch.no_grad()
def eval_e2e(
    model: FineTuneUSVModel,
    eval_data: list[tuple[int, np.ndarray, bool]],
    criterion: nn.Module,
    n_classes: int,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for entry in eval_data:
        label, frozen_feats = entry[0], entry[1]
        y = torch.tensor([label], dtype=torch.long, device=model._device)
        logits = model.forward_from_frozen(frozen_feats)
        total_loss += criterion(logits, y).item()
        all_preds.append(int(logits.argmax(1).item()))
        all_labels.append(label)

    preds  = np.array(all_preds,  dtype=np.int64)
    labels = np.array(all_labels, dtype=np.int64)

    _BINARY_NAMES = {0: 'healthy', 1: 'twitcher'}
    _label_map = _BINARY_NAMES if n_classes == 2 else LABEL_NAMES
    target_names = [_label_map.get(i, f"class_{i}") for i in range(n_classes)]

    report = classification_report(
        labels, preds,
        labels=list(range(n_classes)),
        target_names=target_names,
        output_dict=True, zero_division=0,
    )
    return {
        "loss":     total_loss / max(len(labels), 1),
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "preds":    preds,
        "labels":   labels,
        "report":   report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Recording discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_litter_id(wav_path: Path) -> str:
    m = re.search(r'\s(\w+)-\d+\s', wav_path.stem)
    return m.group(1) if m else wav_path.stem


def find_recordings(data_dir: str, detections_dir: str, n_classes: int):
    data_path = Path(data_dir)
    det_path  = Path(detections_dir)
    wav_paths, csv_paths, labels, litter_ids = [], [], [], []

    for wav in sorted(data_path.glob("*.wav")):
        csv = det_path / (wav.stem + ".csv")
        if not csv.exists():
            continue
        try:
            label = infer_label_from_filename(wav.name, n_classes)
        except ValueError:
            continue
        wav_paths.append(wav)
        csv_paths.append(csv)
        labels.append(label)
        litter_ids.append(extract_litter_id(wav))

    return wav_paths, csv_paths, np.array(labels, dtype=np.int64), litter_ids


# ─────────────────────────────────────────────────────────────────────────────
# Main cross-validation loop
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_finetune(
    config: dict,
    data_dir: str,
    detections_dir: str,
    spec_cache_dir: str | None = None,
    frozen_cache_dir: str | None = None,
    n_folds: int = 5,
):
    """
    Litter-aware k-fold CV with end-to-end fine-tuning of the SqueakOut encoder.

    Two-phase per fold:
      Phase 1 (shared across folds): precompute frozen intermediate features once.
      Phase 2 (per fold):            balance classes → train unfrozen blocks +
                                     MLP → evaluate on held-out test fold.
    """
    device_str = config["encoder"].get("device", "cpu")
    device     = torch.device(device_str)
    seed       = config.get("seed", 42)
    n_classes  = config.get("n_classes", 2)
    cfg_enc    = config["encoder"]
    cfg_spec   = config["spectrogram"]
    cfg_aug    = config.get("augmentation", {})
    cfg_tr     = config.get("training", {})
    cfg_model  = config.get("model", {})
    cfg_ft     = config.get("finetune", {})

    ep            = cfg_enc.get("extraction_point", "x4")
    finetune_from = cfg_ft.get("finetune_from_block", 10)
    lr_enc        = cfg_ft.get("lr_encoder", 1e-5)
    lr_cls        = cfg_ft.get("lr_classifier", 1e-3)
    noise_std     = cfg_aug.get("noise_std", 0.05)
    aug_enabled   = cfg_aug.get("enabled", True)

    # ── Output directory ─────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root  = Path(config.get("output_directory", "outputs"))
    run_dir   = out_root / f"run_{timestamp}_cv{n_folds}_finetune"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    spec_cache_path   = Path(spec_cache_dir)   if spec_cache_dir   else None
    frozen_cache_path = Path(frozen_cache_dir) if frozen_cache_dir else None

    # ── Discover recordings ──────────────────────────────────────────────────
    wav_paths, csv_paths, all_labels, litter_ids = find_recordings(
        data_dir, detections_dir, n_classes
    )

    _bin_names   = {0: 'healthy', 1: 'twitcher'}
    _display     = _bin_names if n_classes == 2 else LABEL_NAMES
    class_counts = {_display.get(lbl, lbl): int((all_labels == lbl).sum())
                    for lbl in np.unique(all_labels)}
    print(f"\nFine-tune CV: {n_folds} folds | {len(wav_paths)} recordings | {class_counts}")
    print(f"  Encoder: {ep}  |  finetune_from_block={finetune_from}")
    print(f"  lr_encoder={lr_enc}  lr_classifier={lr_cls}  noise_std={noise_std}")

    # ── Litter groups ────────────────────────────────────────────────────────
    unique_litters = sorted(set(litter_ids))
    litter_to_int  = {l: i for i, l in enumerate(unique_litters)}
    groups         = np.array([litter_to_int[l] for l in litter_ids])
    print(f"  Litters: {len(unique_litters)}: {unique_litters}")

    # ── Load backbone for precomputation ─────────────────────────────────────
    weights_path = config.get("squeakout_weights", "../squeakout/squeakout_weights.ckpt")

    def load_backbone():
        sq   = SqueakOut()
        ckpt = torch.load(weights_path, map_location='cpu')
        state = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}
        sq.load_state_dict(state)
        return sq.backbone

    # ── Phase 1: Generate spectrograms ───────────────────────────────────────
    print("\nStep 1 — Loading spectrograms (cached if available)...")
    all_specs: list[list[np.ndarray] | None] = []
    for i, (wav, csv) in enumerate(zip(wav_paths, csv_paths)):
        specs = load_spectrograms(wav, csv, cfg_spec, spec_cache_path)
        all_specs.append(specs)
        n_calls = len(specs) if specs else 0
        status  = f"{n_calls} calls" if specs else "EMPTY — will skip"
        print(f"  [{i+1:3d}/{len(wav_paths)}] {wav.stem[:55]}: {status}", flush=True)

    # ── Phase 2: Precompute frozen features ───────────────────────────────────
    print(f"\nStep 2 — Precomputing frozen features (blocks 0..{finetune_from - 1})...")
    backbone_for_precomp = load_backbone().to(device)
    backbone_for_precomp.eval()
    wav_stems = [w.stem for w in wav_paths]

    all_frozen = precompute_frozen_features(
        backbone_for_precomp,
        all_specs,
        finetune_from=finetune_from,
        extraction_point=ep,
        device=device,
        frozen_cache_dir=frozen_cache_path,
        wav_stems=wav_stems,
    )
    del backbone_for_precomp  # free memory; each fold gets a fresh backbone

    # ── Filter to valid recordings ───────────────────────────────────────────
    valid_idx    = np.array([i for i, f in enumerate(all_frozen) if f is not None], dtype=np.int64)
    valid_labels = all_labels[valid_idx]
    valid_groups = groups[valid_idx]
    valid_frozen = [all_frozen[i] for i in valid_idx]
    valid_specs  = [all_specs[i] for i in valid_idx]
    print(f"\n{len(valid_idx)}/{len(wav_paths)} recordings with valid detections")

    # Frozen backbone for re-encoding augmented spectrograms with noise
    frozen_end = min(finetune_from, {'x4': 14, 'deep': 19}[ep])

    # ── CV loop ──────────────────────────────────────────────────────────────
    sgkf      = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_preds = np.full(len(valid_labels), -1, dtype=np.int64)

    for fold, (tr_idx, te_idx) in enumerate(
        sgkf.split(range(len(valid_labels)), valid_labels, valid_groups)
    ):
        print(f"\n── Fold {fold + 1}/{n_folds}  "
              f"(train={len(tr_idx)}, test={len(te_idx)}) ──")

        # Inner val split for MLP early stopping
        n_val     = max(1, int(0.15 * len(tr_idx)))
        inner_tr  = list(tr_idx[:-n_val])
        inner_val = list(tr_idx[-n_val:])

        # Include spec index for input-level augmentation lookup
        train_real = [(int(valid_labels[i]), valid_frozen[i], i) for i in inner_tr]
        val_data   = [(int(valid_labels[i]), valid_frozen[i], False, i) for i in inner_val]
        te_data    = [(int(valid_labels[i]), valid_frozen[i], False, i) for i in te_idx]

        # Balance minority class
        if aug_enabled:
            train_data = augment_train_data(train_real, random_seed=seed + fold)
        else:
            train_data = [(lbl, feats, False, si) for lbl, feats, si in train_real]

        cls_counts_aug = {}
        for entry in train_data:
            cls_counts_aug[entry[0]] = cls_counts_aug.get(entry[0], 0) + 1
        print(f"  Train after aug: "
              f"{ {_display.get(k, k): v for k, v in cls_counts_aug.items()} }")

        # Fresh model per fold (with its own backbone copy for unfrozen blocks)
        backbone = load_backbone()
        model = FineTuneUSVModel(
            backbone=backbone,
            n_classes=n_classes,
            hidden_dims=cfg_model.get("hidden_dims", [64]),
            dropout=cfg_model.get("dropout", 0.2),
            finetune_from=finetune_from,
            extraction_point=ep,
            device=device_str,
        )

        # Separate frozen backbone for re-encoding noisy spectrograms
        frozen_backbone = load_backbone().to(device)
        frozen_backbone.eval()
        for p in frozen_backbone.parameters():
            p.requires_grad = False

        enc_params = [p for p in model.backbone.parameters() if p.requires_grad]
        cls_params = list(model.classifier.parameters())
        print(f"  Trainable params — encoder: {sum(p.numel() for p in enc_params):,}  "
              f"classifier: {sum(p.numel() for p in cls_params):,}")

        optimizer = torch.optim.Adam(
            [{"params": enc_params, "lr": lr_enc},
             {"params": cls_params, "lr": lr_cls}],
            weight_decay=cfg_tr.get("weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        # Class-weighted cross-entropy
        y_tr    = np.array([entry[0] for entry in train_data])
        counts  = np.bincount(y_tr, minlength=n_classes).astype(float)
        weights = len(y_tr) / (n_classes * np.maximum(counts, 1))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )

        # Training loop — per-epoch input-level augmentation
        best_loss  = float("inf")
        best_state = None
        stall      = 0
        patience   = cfg_tr.get("early_stopping_patience", 40)
        n_epochs   = cfg_tr.get("n_epochs", 300)

        for ep_num in range(1, n_epochs + 1):
            rng_ep = np.random.default_rng(seed + fold * 10000 + ep_num)
            tr_loss, tr_acc = train_epoch_e2e(
                model, train_data, criterion, optimizer, noise_std,
                all_specs=valid_specs,
                frozen_backbone=frozen_backbone,
                frozen_end=frozen_end,
                device=device,
                rng=rng_ep,
            )
            vm = eval_e2e(model, val_data, criterion, n_classes)
            scheduler.step(vm["loss"])

            if ep_num % 20 == 0 or ep_num == 1:
                print(f"    Ep {ep_num:4d} | "
                      f"tr_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                      f"val_loss={vm['loss']:.4f} acc={vm['accuracy']:.3f} "
                      f"f1={vm['macro_f1']:.3f}", flush=True)

            if vm["loss"] < best_loss:
                best_loss  = vm["loss"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    print(f"    Early stop at epoch {ep_num} "
                          f"(best val_loss={best_loss:.4f})")
                    break

        if best_state:
            model.load_state_dict(best_state)
        del frozen_backbone  # free memory

        # Predict on test fold
        model.eval()
        for rec_idx, entry in zip(te_idx, te_data):
            all_preds[rec_idx] = model.predict_from_frozen(entry[1])

        # Per-fold summary
        fold_mask = all_preds[te_idx] != -1
        if fold_mask.any():
            t_f = valid_labels[te_idx[fold_mask]]
            p_f = all_preds[te_idx[fold_mask]]
            print(f"  Fold {fold + 1} test — "
                  f"acc={float((t_f == p_f).mean()):.3f}  "
                  f"macro_F1={float(sk_f1(t_f, p_f, average='macro', zero_division=0)):.3f}")

    # ── Aggregate results ─────────────────────────────────────────────────────
    _BINARY_NAMES = {0: 'healthy', 1: 'twitcher'}
    _label_map    = _BINARY_NAMES if n_classes == 2 else LABEL_NAMES
    target_names  = [_label_map.get(i, f"class_{i}") for i in range(n_classes)]

    mask  = all_preds != -1
    t, p  = valid_labels[mask], all_preds[mask]
    macro = float(sk_f1(t, p, average='macro', zero_division=0))
    acc   = float((t == p).mean())

    print(f"\n{'='*65}")
    print(f"Fine-tune CV  ({n_folds} folds, {len(valid_labels)} recordings)")
    print(f"{'='*65}")
    print(f"accuracy={acc:.3f}  macro_F1={macro:.3f}")
    print(classification_report(t, p, target_names=target_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(t, p)
    print(cm)
    print(f"{'='*65}")

    report = classification_report(
        t, p, target_names=target_names, output_dict=True, zero_division=0
    )
    results = {
        "method":            "finetune_e2e",
        "n_folds":           n_folds,
        "n_recordings":      int(len(valid_labels)),
        "extraction_point":  ep,
        "finetune_from":     finetune_from,
        "lr_encoder":        lr_enc,
        "lr_classifier":     lr_cls,
        "noise_std":         noise_std,
        "accuracy":          acc,
        "macro_f1":          macro,
        "classification_report": report,
        "confusion_matrix":  cm.tolist(),
    }
    with open(run_dir / "results.yaml", "w") as f:
        yaml.dump(results, f)
    print(f"\nResults saved to: {run_dir}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SqueakOut encoder end-to-end for USV classification."
    )
    parser.add_argument("--config",           required=True)
    parser.add_argument("--data_dir",         required=True)
    parser.add_argument("--detections_dir",   required=True)
    parser.add_argument("--spec_cache_dir",   default=None,
                        help="Cache raw spectrograms (saves ~4 min on first run)")
    parser.add_argument("--frozen_cache_dir", default=None,
                        help="Cache frozen backbone features (saves ~2 min per run). "
                             "Highly recommended — invalidated only if finetune_from changes.")
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cross_validate_finetune(
        config,
        args.data_dir,
        args.detections_dir,
        spec_cache_dir=args.spec_cache_dir,
        frozen_cache_dir=args.frozen_cache_dir,
        n_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
