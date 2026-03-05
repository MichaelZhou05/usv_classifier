"""
Training script for USV disease classifier.

Usage:
    python train.py --config config.yaml
    python train.py --mat_dir /path/to/mat/files --labels /path/to/labels.csv
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from data import USVDataset, create_data_splits
from models import USVClassifier
from models.mlp import get_model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += features.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(features)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * features.size(0)
            all_preds.extend((probs >= 0.5).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = (all_preds == all_labels).mean()

    # Precision, recall, F1
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def train(config: dict) -> dict:
    """
    Full training pipeline.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with training results and best model path.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load dataset
    print(f"Loading data from: {config['mat_directory']}")
    dataset = USVDataset(
        mat_directory=config["mat_directory"],
        labels_csv=config.get("labels_csv"),
        n_max_calls=config.get("n_max_calls", 150),
        n_features=config.get("n_features", 5),
        include_score=config.get("include_score", True),
        normalize=config.get("normalize", True),
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.class_counts}")

    # Create splits
    train_idx, val_idx, test_idx = create_data_splits(
        dataset,
        train_ratio=config.get("train_ratio", 0.7),
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.15),
        random_seed=seed,
        stratify=True,
    )

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Create dataloaders
    batch_size = config.get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model_config = {
        "n_max_calls": config.get("n_max_calls", 150),
        "n_features": config.get("n_features", 5),
        "hidden_dims": config.get("hidden_dims", [512, 256, 128]),
        "dropout": config.get("dropout", 0.3),
        "use_batch_norm": config.get("use_batch_norm", True),
    }

    model_type = config.get("model_type", "mlp")
    model = get_model(model_type, **model_config)
    model = model.to(device)

    print(f"\nModel architecture:")
    print(model)

    # Loss function with optional class weighting
    if config.get("use_class_weights", True):
        pos_weight = dataset.class_weights.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using class weight: {pos_weight.item():.3f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    lr = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Training loop
    n_epochs = config.get("n_epochs", 100)
    early_stopping_patience = config.get("early_stopping_patience", 20)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {"train": [], "val": []}

    print(f"\nStarting training for {n_epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train"].append(train_metrics)
        history["val"].append({k: v for k, v in val_metrics.items() if k not in ["predictions", "labels", "probabilities"]})

        scheduler.step(val_metrics["loss"])

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}, F1: {val_metrics['f1']:.3f}"
            )

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "config": config,
            }, run_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("-" * 60)
    print(f"Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")

    # Load best model and evaluate on test set
    checkpoint = torch.load(run_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall:    {test_metrics['recall']:.3f}")
    print(f"  F1:        {test_metrics['f1']:.3f}")

    # Save results
    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in test_metrics.items() if k not in ["predictions", "labels", "probabilities"]},
        "history": history,
    }

    with open(run_dir / "results.yaml", "w") as f:
        yaml.dump(results, f)

    # Save normalization stats for inference
    np.savez(
        run_dir / "normalization_stats.npz",
        mean=dataset.feature_mean,
        std=dataset.feature_std,
    )

    print(f"\nResults saved to: {run_dir}")

    return {
        "run_dir": str(run_dir),
        "test_metrics": test_metrics,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train USV disease classifier")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--mat_dir", type=str, help="Path to MAT files directory")
    parser.add_argument("--labels", type=str, help="Path to labels CSV")
    parser.add_argument("--n_max_calls", type=int, default=150, help="Max calls per sample")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    # Load config from file or build from args
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "mat_directory": args.mat_dir,
            "labels_csv": args.labels,
            "n_max_calls": args.n_max_calls,
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "output_dir": args.output_dir,
        }

    if not config.get("mat_directory"):
        parser.error("Either --config or --mat_dir is required")

    train(config)


if __name__ == "__main__":
    main()
