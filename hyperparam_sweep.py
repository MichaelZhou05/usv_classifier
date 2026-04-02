"""
Hyperparameter sweep on the best pipeline (SWE + spectral + encoder + acoustic).

Tests different SVM C values, SWE configurations, and feature combinations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import yaml

_SQUEAKOUT_DIR = Path(__file__).parent.parent / "squeakout"
if str(_SQUEAKOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SQUEAKOUT_DIR))
if 'pytorch_lightning' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['pytorch_lightning'] = MagicMock()

from data import LABEL_NAMES
from data.squeakout_features import SqueakOutEncoder
from data.spectral_features import (
    extract_recording_spectral_features, N_CALL_FEATURES,
)
from train import find_recordings, compute_acoustic_stats, extract_or_load
from train_enhanced import extract_combined_features, augment_meta_pair, pool_and_combine
from pooling.swe import SWEPooler
from pooling import PoolerRegistry


def cv_score(X, y, groups, C=1.0, gamma='scale', kernel='rbf', n_folds=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_preds = np.full(len(y), -1, dtype=np.int64)
    for tr_idx, te_idx in sgkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        svm = SVC(kernel=kernel, class_weight='balanced', C=C, gamma=gamma)
        svm.fit(X_tr, y[tr_idx])
        all_preds[te_idx] = svm.predict(X_te)
    mask = all_preds != -1
    t, p = y[mask], all_preds[mask]
    acc = (t == p).mean()
    macro_f1 = f1_score(t, p, average='macro', zero_division=0)
    tw_f1 = f1_score(t, p, average='binary', pos_label=1, zero_division=0)
    return acc, macro_f1, tw_f1


def main():
    with open("config_squeakout.yaml") as f:
        config = yaml.safe_load(f)

    n_classes = config.get("n_classes", 2)
    cfg_enc = config["encoder"]
    cfg_spec = config["spectrogram"]
    seed = config.get("seed", 42)
    cache_dir = Path("/hpc/group/naderilab/zz394/Mice/feature_cache")
    data_dir = "/hpc/group/naderilab/zz394/Mice/USVs/USVRecordingsP7"
    det_dir = "/hpc/group/naderilab/zz394/Mice/dectections_squeakout"

    wav_paths, csv_paths, labels, litter_ids = find_recordings(data_dir, det_dir, n_classes)
    unique_litters = sorted(set(litter_ids))
    litter_to_int = {l: i for i, l in enumerate(unique_litters)}
    groups = np.array([litter_to_int[l] for l in litter_ids])

    weights_path = config.get("squeakout_weights", "../squeakout/squeakout_weights.ckpt")
    encoder = SqueakOutEncoder(
        weights_path=weights_path,
        extraction_point=cfg_enc["extraction_point"],
        device=cfg_enc.get("device", "cpu"),
    )

    print("Extracting features...")
    enc_meta, spec_meta, valid_idx = extract_combined_features(
        wav_paths, csv_paths, labels, encoder, cache_dir, cfg_enc, cfg_spec,
    )
    del encoder

    valid_labels = labels[valid_idx]
    valid_groups = groups[valid_idx]
    acous_all = np.array([compute_acoustic_stats(csv_paths[i]) for i in valid_idx], dtype=np.float32)

    enc_dim = enc_meta[0][1][0].shape[0]
    spec_dim = spec_meta[0][1][0].shape[0]

    print(f"\n{'Pooler':<15} {'C':<6} {'gamma':<10} {'kernel':<8} {'Acc':<8} {'Macro_F1':<10} {'Tw_F1':<8}")
    print("-" * 75)

    results = []

    for num_slices, num_refs in [(8, 8), (16, 10), (16, 16), (24, 12), (32, 16)]:
        enc_pooler = SWEPooler(n_features=enc_dim, num_slices=num_slices,
                               num_ref_points=num_refs, freeze_swe=True, flatten=True)
        spec_pooler = SWEPooler(n_features=spec_dim, num_slices=num_slices,
                                num_ref_points=num_refs, freeze_swe=True, flatten=True)

        X, y = pool_and_combine(enc_meta, spec_meta, enc_pooler, spec_pooler, acous_all)
        pooler_desc = f"SWE-{num_slices}x{num_refs}"

        for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
            for gamma in ['scale', 'auto']:
                for kernel in ['rbf', 'linear']:
                    if kernel == 'linear' and gamma == 'auto':
                        continue  # gamma irrelevant for linear
                    acc, mf1, tw_f1 = cv_score(X, y, valid_groups, C=C,
                                                gamma=gamma, kernel=kernel, seed=seed)
                    print(f"{pooler_desc:<15} {C:<6} {gamma:<10} {kernel:<8} {acc:<8.3f} {mf1:<10.3f} {tw_f1:<8.3f}")
                    results.append((pooler_desc, C, gamma, kernel, acc, mf1, tw_f1))

    # Also test average pooling for comparison
    enc_pooler = PoolerRegistry.get("average", n_features=enc_dim)
    spec_pooler = PoolerRegistry.get("average", n_features=spec_dim)
    X, y = pool_and_combine(enc_meta, spec_meta, enc_pooler, spec_pooler, acous_all)
    for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
        for gamma in ['scale']:
            acc, mf1, tw_f1 = cv_score(X, y, valid_groups, C=C, gamma=gamma, seed=seed)
            print(f"{'Average':<15} {C:<6} {gamma:<10} {'rbf':<8} {acc:<8.3f} {mf1:<10.3f} {tw_f1:<8.3f}")
            results.append(('Average', C, gamma, 'rbf', acc, mf1, tw_f1))

    # Sort by macro_f1
    results.sort(key=lambda x: x[5], reverse=True)
    print(f"\n{'='*75}")
    print("Top 10 configurations by macro_F1:")
    print(f"{'='*75}")
    for i, (pooler, C, gamma, kernel, acc, mf1, tw_f1) in enumerate(results[:10]):
        print(f"  {i+1}. {pooler:<15} C={C:<5} gamma={gamma:<6} {kernel:<7} "
              f"acc={acc:.3f} macro_F1={mf1:.3f} tw_F1={tw_f1:.3f}")


if __name__ == "__main__":
    main()
