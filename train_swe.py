"""Training with Sliced-Wasserstein Embedding (SWE) pooling.

SWE is an optimal transport-based pooling method that creates fixed-dimensional
embeddings from sets of arbitrary size. It captures the distribution of calls
in a principled way.

Usage:
    python train_swe.py --csv-dir /path/to/enriched_features --binary
    python train_swe.py --csv-dir /path/to/enriched_features --binary --num-slices 16 --num-ref-points 20
"""

import argparse
import glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from pooling.swe import SWE_Pooling


LABEL_NAMES_BINARY = ['twitcher', 'healthy']
LABEL_NAMES_3CLASS = ['twitcher', 'wildtype', 'heterozygous']

FEATURE_NAMES = [
    'duration', 'low_freq', 'high_freq', 'principal_freq', 'bandwidth',
    'freq_stdev', 'slope', 'sinuosity', 'snr', 'entropy', 'mean_power'
]


def infer_label(filename: str, binary: bool = False) -> int:
    """Infer class label from filename."""
    fname = filename.lower()
    if 'twitcher' in fname or 'twi' in fname:
        return 0
    elif 'wildtype' in fname or ' wt ' in fname or '_wt_' in fname:
        return 1 if binary else 1
    elif 'het' in fname:
        return 1 if binary else 2
    else:
        raise ValueError(f"Cannot infer label from: {filename}")


def load_and_pool_swe(csv_dir: str, swe_pooler: SWE_Pooling, binary: bool = False, device: str = 'cpu'):
    """Load CSV files and pool using SWE.

    Args:
        csv_dir: Directory containing enriched feature CSVs
        swe_pooler: SWE_Pooling module
        binary: If True, use binary labels
        device: torch device

    Returns:
        X: Feature matrix (n_samples, swe_output_dim)
        y: Labels
        names: Filenames
    """
    files = sorted(glob.glob(f"{csv_dir}/*.csv"))

    if len(files) == 0:
        raise ValueError(f"No CSV files found in {csv_dir}")

    print(f"Found {len(files)} files")

    X_list = []
    y_list = []
    names = []

    swe_pooler.eval()

    with torch.no_grad():
        for f in files:
            df = pd.read_csv(f)
            features = df.drop('call_id', axis=1).values

            if len(features) == 0:
                print(f"Warning: No calls in {f}, skipping")
                continue

            # Handle NaN/Inf
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                label = infer_label(Path(f).stem, binary=binary)
            except ValueError as e:
                print(f"Warning: {e}, skipping")
                continue

            # Convert to tensor: (1, n_calls, n_features)
            X_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.ones(1, X_tensor.shape[1], dtype=torch.bool, device=device)

            # SWE pooling
            pooled = swe_pooler(X_tensor, mask)  # (1, output_dim)
            pooled = pooled.squeeze(0).cpu().numpy()

            X_list.append(pooled)
            y_list.append(label)
            names.append(Path(f).stem)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, names


def main():
    parser = argparse.ArgumentParser(description='Train with SWE pooling')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Directory containing enriched feature CSVs')
    parser.add_argument('--binary', action='store_true',
                        help='Use binary classification (disease vs healthy)')
    parser.add_argument('--num-slices', type=int, default=8,
                        help='Number of SWE slices (projection directions)')
    parser.add_argument('--num-ref-points', type=int, default=10,
                        help='Number of reference points per slice')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--freeze-swe', action='store_true',
                        help='Freeze SWE parameters (non-learnable)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_names = LABEL_NAMES_BINARY if args.binary else LABEL_NAMES_3CLASS

    print(f"Device: {device}")
    print(f"Loading data from: {args.csv_dir}")
    print(f"Classification: {'binary' if args.binary else '3-class'}")
    print(f"SWE config: {args.num_slices} slices x {args.num_ref_points} ref points")

    # Create SWE pooler
    swe_pooler = SWE_Pooling(
        d_in=11,  # 11 call features
        num_slices=args.num_slices,
        num_ref_points=args.num_ref_points,
        freeze_swe=args.freeze_swe,
        flatten=True
    ).to(device)

    output_dim = swe_pooler.output_dim
    print(f"SWE output dimension: {output_dim}")

    # Load and pool data
    X, y, names = load_and_pool_swe(args.csv_dir, swe_pooler, binary=args.binary, device=device)

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution:")
    for i, name in enumerate(label_names):
        count = sum(y == i)
        if count > 0:
            print(f"  {name}: {count}")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models to try
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=2,
            class_weight='balanced', random_state=42
        ),
        'SVM_Linear': SVC(
            kernel='linear', C=0.1, class_weight='balanced', random_state=42
        ),
        'SVM_RBF': SVC(
            kernel='rbf', C=1.0, class_weight='balanced', random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=0.1, max_iter=1000, class_weight='balanced', random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=50, max_depth=3, min_samples_leaf=2, random_state=42
        ),
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    print("\n" + "=" * 70)
    print(f"5-Fold Cross-Validation Results (SWE Pooling)")
    print("=" * 70)

    results = {}

    for name, model in models.items():
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average='macro')

        results[name] = {'accuracy': acc, 'macro_f1': macro_f1, 'y_pred': y_pred}

    # Summary table
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 45)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)

    for name, metrics in sorted_results:
        print(f"{name:<25} {metrics['accuracy']:>10.3f} {metrics['macro_f1']:>10.3f}")

    # Detailed results for best model
    best_name, best_metrics = sorted_results[0]
    print(f"\n{'=' * 70}")
    print(f"Best Model: {best_name}")
    print(f"{'=' * 70}")

    y_pred = best_metrics['y_pred']
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print(f"\n{classification_report(y, y_pred, target_names=label_names, zero_division=0)}")

    # Compare with statistics pooling baseline
    print("\n" + "=" * 70)
    print("Comparison with Statistics Pooling")
    print("=" * 70)
    print("\nRun `python train_sklearn.py --binary` to compare with statistics pooling.")
    print("SWE pooling captures distributional information via optimal transport,")
    print("which may better represent the variability in call patterns.")


if __name__ == '__main__':
    main()
