"""
Training script for USV disease classifier.

Supports both legacy binary classification (MAT files) and multi-class
classification with enriched features (CSV files with pooling).

Usage:
    # Legacy mode (MAT files, binary classification)
    python train.py --config config.yaml

    # Enriched mode (CSV files, 3-class classification)
    python train.py --config config_enriched.yaml
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import yaml

from data import USVDataset, EnrichedUSVDataset, create_data_splits, LABEL_NAMES
from models import USVClassifier
from models.mlp import get_model


def train_epoch_binary(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch (binary classification)."""
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


def train_epoch_multiclass(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch (multi-class classification)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)  # Long tensor for CrossEntropyLoss

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += features.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def evaluate_binary(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model (binary classification)."""
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


def evaluate_multiclass(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int = 3,
) -> dict:
    """Evaluate model (multi-class classification)."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)

            total_loss += loss.item() * features.size(0)
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = (all_preds == all_labels).mean()

    # Per-class metrics using sklearn
    target_names = [LABEL_NAMES.get(i, f"class_{i}") for i in range(n_classes)]
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Macro-averaged F1
    macro_f1 = report.get('macro avg', {}).get('f1-score', 0)

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
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

    # Determine mode: enriched (CSV) or legacy (MAT)
    use_enriched = config.get("csv_directory") is not None
    n_classes = config.get("n_classes", 3 if use_enriched else 2)

    if use_enriched:
        # Enriched mode: CSV files with pooling
        print(f"Loading enriched features from: {config['csv_directory']}")

        # Set up pooler
        from pooling import PoolerRegistry
        pooler_name = config.get("pooler", "average")
        n_features = config.get("n_features", 11)
        pooler = PoolerRegistry.get(pooler_name, n_features=n_features)
        print(f"Using pooler: {pooler}")

        dataset = EnrichedUSVDataset(
            csv_directory=config["csv_directory"],
            labels_csv=config.get("labels_csv"),
            pooler=pooler,
            n_classes=n_classes,
            normalize=config.get("normalize", True),
        )

        input_dim = dataset.input_dim
        model_type = "enriched"

    else:
        # Legacy mode: MAT files
        print(f"Loading data from: {config['mat_directory']}")
        dataset = USVDataset(
            mat_directory=config["mat_directory"],
            labels_csv=config.get("labels_csv"),
            n_max_calls=config.get("n_max_calls", 150),
            n_features=config.get("n_features", 5),
            include_score=config.get("include_score", True),
            normalize=config.get("normalize", True),
        )

        input_dim = None  # Will be computed from n_max_calls * n_features
        model_type = config.get("model_type", "mlp")

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
    if use_enriched:
        model_config = {
            "input_dim": input_dim,
            "n_classes": n_classes,
            "hidden_dims": config.get("hidden_dims", [32, 16]),
            "dropout": config.get("dropout", 0.5),
            "use_batch_norm": config.get("use_batch_norm", False),
        }
    else:
        model_config = {
            "n_max_calls": config.get("n_max_calls", 150),
            "n_features": config.get("n_features", 5),
            "hidden_dims": config.get("hidden_dims", [512, 256, 128]),
            "dropout": config.get("dropout", 0.3),
            "use_batch_norm": config.get("use_batch_norm", True),
        }

    model = get_model(model_type, **model_config)
    model = model.to(device)

    print(f"\nModel architecture:")
    print(model)

    # Loss function
    if use_enriched or n_classes > 2:
        # Multi-class: CrossEntropyLoss
        if config.get("use_class_weights", True):
            class_weights = dataset.class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using class weights: {class_weights.cpu().numpy()}")
        else:
            criterion = nn.CrossEntropyLoss()

        train_fn = train_epoch_multiclass
        eval_fn = lambda m, dl, c, d: evaluate_multiclass(m, dl, c, d, n_classes)
        metric_key = "macro_f1"
    else:
        # Binary: BCEWithLogitsLoss
        if config.get("use_class_weights", True):
            pos_weight = dataset.class_weights.to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using class weight: {pos_weight.item():.3f}")
        else:
            criterion = nn.BCEWithLogitsLoss()

        train_fn = train_epoch_binary
        eval_fn = evaluate_binary
        metric_key = "f1"

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
    print("-" * 70)

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_fn(model, train_loader, criterion, optimizer, device)
        val_metrics = eval_fn(model, val_loader, criterion, device)

        history["train"].append(train_metrics)

        # Store val metrics (excluding large arrays)
        val_store = {k: v for k, v in val_metrics.items()
                     if k not in ["predictions", "labels", "probabilities", "classification_report", "confusion_matrix"]}
        history["val"].append(val_store)

        scheduler.step(val_metrics["loss"])

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            val_metric = val_metrics.get(metric_key, val_metrics.get("f1", 0))
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}, "
                f"{metric_key}: {val_metric:.3f}"
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

    print("-" * 70)
    print(f"Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")

    # Load best model and evaluate on test set
    checkpoint = torch.load(run_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = eval_fn(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")

    if use_enriched or n_classes > 2:
        print(f"  Macro F1:  {test_metrics['macro_f1']:.3f}")
        print(f"\nConfusion Matrix:")
        conf_mat = np.array(test_metrics['confusion_matrix'])
        print(f"  {conf_mat}")
        print(f"\nPer-class metrics:")
        for class_name in [LABEL_NAMES.get(i, f"class_{i}") for i in range(n_classes)]:
            if class_name in test_metrics['classification_report']:
                class_metrics = test_metrics['classification_report'][class_name]
                print(f"  {class_name}: precision={class_metrics['precision']:.3f}, "
                      f"recall={class_metrics['recall']:.3f}, f1={class_metrics['f1-score']:.3f}")
    else:
        print(f"  Precision: {test_metrics['precision']:.3f}")
        print(f"  Recall:    {test_metrics['recall']:.3f}")
        print(f"  F1:        {test_metrics['f1']:.3f}")

    # Save results
    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_metrics": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in test_metrics.items()
            if k not in ["predictions", "labels", "probabilities"]
        },
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
    parser.add_argument("--mat_dir", type=str, help="Path to MAT files directory (legacy mode)")
    parser.add_argument("--csv_dir", type=str, help="Path to enriched CSV files directory")
    parser.add_argument("--labels", type=str, help="Path to labels CSV")
    parser.add_argument("--n_max_calls", type=int, default=150, help="Max calls per sample (legacy)")
    parser.add_argument("--n_classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--pooler", type=str, default="average", help="Pooling strategy")
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
            "csv_directory": args.csv_dir,
            "labels_csv": args.labels,
            "n_max_calls": args.n_max_calls,
            "n_classes": args.n_classes,
            "pooler": args.pooler,
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "output_dir": args.output_dir,
        }

    if not config.get("mat_directory") and not config.get("csv_directory"):
        parser.error("Either --config with mat_directory/csv_directory, or --mat_dir/--csv_dir is required")

    train(config)


if __name__ == "__main__":
    main()
