"""Call-level training for USV classification.

Instead of pooling calls per recording, this trains on individual calls.
This gives many more training samples but requires careful cross-validation
to avoid data leakage (calls from same mouse in train and test).

Usage:
    python train_call_level.py --csv-dir /path/to/enriched_features --binary
"""

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


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


def load_call_level_data(csv_dir: str, binary: bool = False):
    """Load individual calls (not pooled).

    Returns:
        X: Feature matrix (n_total_calls, 11)
        y: Labels (n_total_calls,)
        mouse_ids: Which mouse each call belongs to (for grouped CV)
        mouse_names: List of mouse names
    """
    files = sorted(glob.glob(f"{csv_dir}/*.csv"))

    if len(files) == 0:
        raise ValueError(f"No CSV files found in {csv_dir}")

    print(f"Found {len(files)} recordings")

    X_list = []
    y_list = []
    mouse_ids = []
    mouse_names = []

    for mouse_id, f in enumerate(files):
        df = pd.read_csv(f)
        features = df.drop('call_id', axis=1).values

        if len(features) == 0:
            continue

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            label = infer_label(Path(f).stem, binary=binary)
        except ValueError as e:
            print(f"Warning: {e}, skipping")
            continue

        # Add all calls from this mouse
        n_calls = len(features)
        X_list.append(features)
        y_list.extend([label] * n_calls)
        mouse_ids.extend([mouse_id] * n_calls)
        mouse_names.append(Path(f).stem)

    X = np.vstack(X_list)
    y = np.array(y_list)
    mouse_ids = np.array(mouse_ids)

    return X, y, mouse_ids, mouse_names


def grouped_cross_validation(X, y, mouse_ids, model, n_splits=5, random_state=42):
    """Cross-validation that keeps all calls from same mouse together.

    This prevents data leakage - we never have calls from the same mouse
    in both training and test sets.
    """
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)

    y_pred_all = np.zeros_like(y)
    y_true_all = np.zeros_like(y)

    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=mouse_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and predict
        model_clone = clone_model(model)
        model_clone.fit(X_train_scaled, y_train)
        y_pred = model_clone.predict(X_test_scaled)

        y_pred_all[test_idx] = y_pred
        y_true_all[test_idx] = y_test

        fold_acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_acc)

        # Count unique mice in train/test
        n_train_mice = len(np.unique(mouse_ids[train_idx]))
        n_test_mice = len(np.unique(mouse_ids[test_idx]))
        print(f"  Fold {fold+1}: {n_train_mice} mice train, {n_test_mice} mice test, "
              f"acc={fold_acc:.3f}")

    return y_pred_all, y_true_all, fold_accuracies


def clone_model(model):
    """Create a fresh copy of a model."""
    from sklearn.base import clone
    return clone(model)


def evaluate_with_voting(X, y, mouse_ids, mouse_names, model, binary=False):
    """Evaluate using call-level predictions, then majority vote per mouse.

    This combines call-level training with mouse-level evaluation.
    """
    from sklearn.model_selection import GroupKFold

    label_names = LABEL_NAMES_BINARY if binary else LABEL_NAMES_3CLASS
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    mouse_predictions = defaultdict(list)
    mouse_true_labels = {}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=mouse_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_mouse_ids = mouse_ids[test_idx]

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and predict
        model_clone = clone_model(model)
        model_clone.fit(X_train_scaled, y_train)
        y_pred = model_clone.predict(X_test_scaled)

        # Collect predictions by mouse
        for i, (pred, true_label, mid) in enumerate(zip(y_pred, y_test, test_mouse_ids)):
            mouse_predictions[mid].append(pred)
            mouse_true_labels[mid] = true_label

    # Majority vote per mouse
    mouse_final_pred = {}
    for mid, preds in mouse_predictions.items():
        # Majority vote
        counts = np.bincount(preds)
        mouse_final_pred[mid] = np.argmax(counts)

    # Compute mouse-level metrics
    y_true_mice = []
    y_pred_mice = []
    for mid in sorted(mouse_true_labels.keys()):
        y_true_mice.append(mouse_true_labels[mid])
        y_pred_mice.append(mouse_final_pred[mid])

    y_true_mice = np.array(y_true_mice)
    y_pred_mice = np.array(y_pred_mice)

    return y_true_mice, y_pred_mice


def main():
    parser = argparse.ArgumentParser(description='Call-level USV training')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Directory containing enriched feature CSVs')
    parser.add_argument('--binary', action='store_true',
                        help='Use binary classification (disease vs healthy)')
    args = parser.parse_args()

    label_names = LABEL_NAMES_BINARY if args.binary else LABEL_NAMES_3CLASS

    print(f"Loading data from: {args.csv_dir}")
    print(f"Classification: {'binary' if args.binary else '3-class'}")
    print()

    # Load call-level data
    X, y, mouse_ids, mouse_names = load_call_level_data(args.csv_dir, binary=args.binary)

    n_mice = len(np.unique(mouse_ids))
    print(f"\nDataset statistics:")
    print(f"  Total calls: {len(X)}")
    print(f"  Total mice: {n_mice}")
    print(f"  Features per call: {X.shape[1]}")
    print(f"  Avg calls per mouse: {len(X) / n_mice:.1f}")
    print(f"\nClass distribution (calls):")
    for i, name in enumerate(label_names):
        print(f"  {name}: {sum(y == i)} calls")

    # Count mice per class
    mice_per_class = defaultdict(int)
    for mid in np.unique(mouse_ids):
        label = y[mouse_ids == mid][0]
        mice_per_class[label] += 1
    print(f"\nClass distribution (mice):")
    for i, name in enumerate(label_names):
        print(f"  {name}: {mice_per_class[i]} mice")

    # Models to try
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'SVM_Linear': SVC(
            kernel='linear', C=0.1, class_weight='balanced', random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=0.1, max_iter=1000, class_weight='balanced', random_state=42
        ),
    }

    print("\n" + "=" * 70)
    print("Call-Level Training with Grouped Cross-Validation")
    print("(All calls from same mouse kept together)")
    print("=" * 70)

    results = {}

    for name, model in models.items():
        print(f"\n{name}:")

        y_pred, y_true, fold_accs = grouped_cross_validation(
            X, y, mouse_ids, model, n_splits=5
        )

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        results[name] = {'accuracy': acc, 'macro_f1': macro_f1}

        print(f"  Call-level accuracy: {acc:.3f}")
        print(f"  Call-level macro F1: {macro_f1:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary (Call-Level)")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 45)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['macro_f1'], reverse=True):
        print(f"{name:<25} {metrics['accuracy']:>10.3f} {metrics['macro_f1']:>10.3f}")

    # Now do mouse-level evaluation with majority voting
    print("\n" + "=" * 70)
    print("Mouse-Level Evaluation (Majority Vote)")
    print("=" * 70)

    best_model_name = max(results, key=lambda x: results[x]['macro_f1'])
    best_model = models[best_model_name]

    print(f"\nUsing best model: {best_model_name}")

    y_true_mice, y_pred_mice = evaluate_with_voting(
        X, y, mouse_ids, mouse_names, best_model, binary=args.binary
    )

    acc = accuracy_score(y_true_mice, y_pred_mice)
    macro_f1 = f1_score(y_true_mice, y_pred_mice, average='macro')

    print(f"\nMouse-level accuracy: {acc:.3f}")
    print(f"Mouse-level macro F1: {macro_f1:.3f}")
    print(f"\nConfusion Matrix (mice):")
    print(confusion_matrix(y_true_mice, y_pred_mice))
    print(f"\n{classification_report(y_true_mice, y_pred_mice, target_names=label_names, zero_division=0)}")

    # Feature importance
    print("\n" + "=" * 70)
    print("Feature Importance")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf.fit(X_scaled, y)

    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]

    print(f"\nFeature ranking:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {FEATURE_NAMES[idx]}: {importance[idx]:.4f}")


if __name__ == '__main__':
    main()
