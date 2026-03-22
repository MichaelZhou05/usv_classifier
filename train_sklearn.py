"""Traditional ML baseline for small USV dataset.

For small datasets (<100 samples), traditional ML methods like RandomForest
and SVM typically outperform neural networks due to:
- Fewer parameters (less overfitting)
- Built-in regularization
- Better suited for tabular/structured features

Usage:
    python train_sklearn.py --csv-dir /path/to/enriched_features
"""

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


LABEL_NAMES = ['twitcher', 'wildtype', 'heterozygous']


def infer_label(filename: str) -> int:
    """Infer class label from filename."""
    fname = filename.lower()
    if 'twitcher' in fname or 'twi' in fname:
        return 0
    elif 'wildtype' in fname or ' wt ' in fname or '_wt_' in fname:
        return 1
    elif 'het' in fname:
        return 2
    else:
        raise ValueError(f"Cannot infer label from: {filename}")


def load_data(csv_dir: str, pooling: str = 'statistics'):
    """Load and pool features from CSV files.

    Args:
        csv_dir: Directory containing enriched feature CSVs.
        pooling: Pooling strategy ('statistics', 'average', 'full').

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        names: Sample filenames
    """
    files = sorted(glob.glob(f"{csv_dir}/*.csv"))

    if len(files) == 0:
        raise ValueError(f"No CSV files found in {csv_dir}")

    print(f"Found {len(files)} files")

    X_list = []
    y_list = []
    names = []

    for f in files:
        df = pd.read_csv(f)
        features = df.drop('call_id', axis=1).values

        if len(features) == 0:
            print(f"Warning: No calls in {f}, skipping")
            continue

        # Pool call-level features to recording-level
        if pooling == 'statistics':
            # Mean, std, min, max, median (5 * 11 = 55 features)
            pooled = np.concatenate([
                features.mean(axis=0),
                features.std(axis=0),
                features.min(axis=0),
                features.max(axis=0),
                np.median(features, axis=0)
            ])
        elif pooling == 'average':
            pooled = features.mean(axis=0)
        elif pooling == 'full':
            # Mean, std, min, max, median, 10th/90th percentile
            pooled = np.concatenate([
                features.mean(axis=0),
                features.std(axis=0),
                features.min(axis=0),
                features.max(axis=0),
                np.median(features, axis=0),
                np.percentile(features, 10, axis=0),
                np.percentile(features, 90, axis=0),
            ])
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # Handle NaN/Inf values
        pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            label = infer_label(Path(f).stem)
            X_list.append(pooled)
            y_list.append(label)
            names.append(Path(f).stem)
        except ValueError as e:
            print(f"Warning: {e}, skipping")
            continue

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, names


def evaluate_models(X, y, cv_folds: int = 5):
    """Evaluate multiple models using cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        cv_folds: Number of cross-validation folds

    Returns:
        results: Dict of model_name -> metrics
    """
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models to evaluate
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42
        ),
        'SVM_RBF': SVC(
            kernel='rbf',
            C=1.0,
            class_weight='balanced',
            random_state=42
        ),
        'SVM_Linear': SVC(
            kernel='linear',
            C=0.1,
            class_weight='balanced',
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results = {}

    for name, model in models.items():
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average='macro')
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, target_names=LABEL_NAMES, zero_division=0)

        results[name] = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'confusion_matrix': cm,
            'report': report,
            'predictions': y_pred,
        }

    return results, scaler


def print_results(results):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("5-Fold Cross-Validation Results")
    print("=" * 70)

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
    print(f"\nConfusion Matrix:")
    print(best_metrics['confusion_matrix'])
    print(f"\n{best_metrics['report']}")


def main():
    parser = argparse.ArgumentParser(description='Train sklearn models on USV features')
    parser.add_argument('--csv-dir', type=str,
                        default='/hpc/group/naderilab/zz394/Mice/enriched_features',
                        help='Directory containing enriched feature CSVs')
    parser.add_argument('--pooling', type=str, default='statistics',
                        choices=['average', 'statistics', 'full'],
                        help='Pooling strategy for call features')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    args = parser.parse_args()

    print(f"Loading data from: {args.csv_dir}")
    print(f"Pooling strategy: {args.pooling}")

    # Load data
    X, y, names = load_data(args.csv_dir, pooling=args.pooling)

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {sum(y == i)}")

    # Evaluate models
    results, scaler = evaluate_models(X, y, cv_folds=args.cv_folds)

    # Print results
    print_results(results)

    # Feature importance for best tree-based model
    print("\n" + "=" * 70)
    print("Feature Importance (RandomForest)")
    print("=" * 70)

    # Retrain RandomForest on full data for feature importance
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42
    )
    X_scaled = scaler.transform(X)
    rf.fit(X_scaled, y)

    # Feature names
    base_features = ['duration', 'low_freq', 'high_freq', 'principal_freq',
                     'bandwidth', 'freq_stdev', 'slope', 'sinuosity',
                     'snr', 'entropy', 'mean_power']

    if args.pooling == 'statistics':
        stats = ['mean', 'std', 'min', 'max', 'median']
        feature_names = [f"{s}_{f}" for s in stats for f in base_features]
    elif args.pooling == 'full':
        stats = ['mean', 'std', 'min', 'max', 'median', 'p10', 'p90']
        feature_names = [f"{s}_{f}" for s in stats for f in base_features]
    else:
        feature_names = base_features

    # Top 10 features
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1][:10]

    print(f"\nTop 10 most important features:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")


if __name__ == '__main__':
    main()
