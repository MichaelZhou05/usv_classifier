"""
Evaluation and inference script for trained USV classifier.

Usage:
    python evaluate.py --model outputs/run_xxx/best_model.pt --mat_dir /path/to/mat/files
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from data import USVDataset
from models.mlp import get_model


def load_trained_model(model_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model_config = {
        "n_max_calls": config.get("n_max_calls", 150),
        "n_features": config.get("n_features", 5),
        "hidden_dims": config.get("hidden_dims", [512, 256, 128]),
        "dropout": config.get("dropout", 0.3),
    }

    model = get_model(config.get("model_type", "mlp"), **model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def predict_single(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Make prediction for a single sample.

    Args:
        model: Trained model.
        features: Feature array of shape (n_max_calls, n_features).
        device: Torch device.
        threshold: Classification threshold.

    Returns:
        Dictionary with prediction, probability, and label.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        prob = model.predict_proba(x).item()
        pred = int(prob >= threshold)

    return {
        "prediction": pred,
        "probability": prob,
        "label": "twitcher" if pred == 1 else "wildtype",
    }


def evaluate_dataset(
    model: torch.nn.Module,
    dataset: USVDataset,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on entire dataset."""
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for i in range(len(dataset)):
            features, label = dataset[i]
            features = features.unsqueeze(0).to(device)

            prob = model.predict_proba(features).item()
            pred = int(prob >= threshold)

            all_preds.append(pred)
            all_probs.append(prob)
            all_labels.append(int(label.item()))
            all_filenames.append(dataset.get_filename(i))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = (all_preds == all_labels).mean()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
        "per_sample": [
            {
                "filename": fn,
                "true_label": int(tl),
                "prediction": int(p),
                "probability": float(pr),
                "correct": tl == p,
            }
            for fn, tl, p, pr in zip(all_filenames, all_labels, all_preds, all_probs)
        ],
    }


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {results['accuracy']:.3f}")
    print(f"  Precision:   {results['precision']:.3f}")
    print(f"  Recall:      {results['recall']:.3f}")
    print(f"  Specificity: {results['specificity']:.3f}")
    print(f"  F1 Score:    {results['f1']:.3f}")

    cm = results["confusion_matrix"]
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              WT      TWI")
    print(f"  Actual WT   {cm['tn']:3d}     {cm['fp']:3d}")
    print(f"  Actual TWI  {cm['fn']:3d}     {cm['tp']:3d}")

    print(f"\nPer-Sample Results:")
    print(f"{'Filename':<50} {'True':>6} {'Pred':>6} {'Prob':>6} {'OK':>4}")
    print("-" * 72)
    for s in results["per_sample"]:
        label_str = "TWI" if s["true_label"] == 1 else "WT"
        pred_str = "TWI" if s["prediction"] == 1 else "WT"
        ok_str = "yes" if s["correct"] else "NO"
        print(f"{s['filename'][:48]:<50} {label_str:>6} {pred_str:>6} {s['probability']:>6.3f} {ok_str:>4}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate USV classifier")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mat_dir", type=str, required=True, help="Path to MAT files")
    parser.add_argument("--labels", type=str, help="Path to labels CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--output", type=str, help="Path to save results YAML")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, config = load_trained_model(args.model, device)
    print(f"Loaded model from: {args.model}")

    # Load normalization stats
    model_dir = Path(args.model).parent
    stats_path = model_dir / "normalization_stats.npz"
    if stats_path.exists():
        stats = np.load(stats_path)
        feature_mean = stats["mean"]
        feature_std = stats["std"]
        print(f"Loaded normalization stats from: {stats_path}")
    else:
        feature_mean = None
        feature_std = None
        print("Warning: No normalization stats found, computing from data")

    # Load dataset
    dataset = USVDataset(
        mat_directory=args.mat_dir,
        labels_csv=args.labels,
        n_max_calls=config.get("n_max_calls", 150),
        n_features=config.get("n_features", 5),
        include_score=config.get("include_score", True),
        normalize=config.get("normalize", True),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    print(f"Loaded {len(dataset)} samples")

    # Evaluate
    results = evaluate_dataset(model, dataset, device, threshold=args.threshold)

    # Print results
    print_results(results)

    # Save if requested
    if args.output:
        # Convert numpy types for YAML serialization
        results_serializable = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in results.items()
            if k != "per_sample"
        }
        results_serializable["per_sample"] = results["per_sample"]

        with open(args.output, "w") as f:
            yaml.dump(results_serializable, f)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
