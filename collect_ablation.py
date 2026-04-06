"""Collect and compare ablation study results."""
import sys
from pathlib import Path
import yaml


def collect(output_dir="outputs"):
    results_dir = Path(output_dir)
    runs = sorted(results_dir.glob("job_*_ablation*")) or sorted(results_dir.glob("job_*"))

    rows = []
    for run in runs:
        rfile = run / "results.yaml"
        if not rfile.exists():
            continue
        with open(rfile) as f:
            r = yaml.safe_load(f)
        if r is None:
            continue

        aug = r.get("augmentation", "?")
        enc_only = r.get("encoder_only", "?")
        pooler = r.get("pooler", "?")
        features = r.get("features", "?")

        for model_name, metrics in r.get("models", {}).items():
            rows.append({
                "run": run.name,
                "augmentation": aug,
                "features": features,
                "pooler": pooler,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "twitcher_f1": metrics.get("classification_report", {}).get("twitcher", {}).get("f1-score", 0),
            })

    if not rows:
        print("No results found yet.")
        return

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"{'Condition':<45} {'Model':<18} {'Accuracy':>8} {'Macro_F1':>9} {'Twitch_F1':>10}")
    print(f"{'='*100}")

    # Sort by macro_f1 descending
    rows.sort(key=lambda x: x["macro_f1"], reverse=True)
    for r in rows:
        cond = f"aug={r['augmentation']:<5} feat={r['features']:<30} pool={r['pooler']}"
        print(f"{cond:<45} {r['model']:<18} {r['accuracy']:>8.3f} {r['macro_f1']:>9.3f} {r['twitcher_f1']:>10.3f}")

    print(f"{'='*100}")

    # Best per model type
    print("\nBest configuration per model:")
    for model in ["SVM (rbf)", "Random Forest", "MLP"]:
        model_rows = [r for r in rows if r["model"] == model]
        if model_rows:
            best = max(model_rows, key=lambda x: x["macro_f1"])
            print(f"  {model}: macro_F1={best['macro_f1']:.3f} twitcher_F1={best['twitcher_f1']:.3f} "
                  f"[aug={best['augmentation']}, {best['features']}, {best['pooler']}]")


if __name__ == "__main__":
    collect(sys.argv[1] if len(sys.argv) > 1 else "outputs")
