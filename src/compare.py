# ============================================================
# Stage 6 — compare.py
# Purpose : Once every approach has been trained (train.py) and
#           evaluated (evaluate.py), aggregate their results into
#           ONE results table and comparison figures — required by
#           Section 5.D of the assignment (side-by-side comparison
#           on the same test set, at least two comparison figures).
#
# How to run:
#   venv\Scripts\python.exe src\compare.py
# ============================================================

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
MODEL_TAGS = ["lstm", "gru", "transformer"]


def load_run(tag: str):
    run_dir = os.path.join(RESULTS_DIR, tag)
    meta_path = os.path.join(run_dir, "meta.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    history_path = os.path.join(run_dir, "history.json")
    if not (os.path.exists(meta_path) and os.path.exists(metrics_path)):
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    history = None
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    return meta, metrics, history


def main():
    rows = []
    histories = {}

    for tag in MODEL_TAGS:
        run = load_run(tag)
        if run is None:
            print(f"Skipping '{tag}' — run train.py and evaluate.py for it first.")
            continue
        meta, metrics, history = run
        rows.append({
            "approach": tag.upper(),
            "test_accuracy": metrics["test_accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "trainable_params": meta["trainable_params"],
            "training_time_s": meta["training_time_seconds"],
            "epochs_run": meta["epochs_run"],
            "device": meta["device"],
        })
        if history:
            histories[tag] = history

    if not rows:
        print("No completed runs found. Train and evaluate at least two approaches first.")
        return

    df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    md_path = os.path.join(RESULTS_DIR, "comparison_table.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))

    print("\n=== Comparison table ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")

    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ── Figure 1: bar chart of test accuracy / F1 per approach ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["test_accuracy"], width, label="Test Accuracy", color="steelblue")
    ax.bar([i + width / 2 for i in x], df["f1_macro"], width, label="Macro F1", color="tomato")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["approach"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test Accuracy & F1 by Approach")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    acc_fig_path = os.path.join(figures_dir, "accuracy_comparison.png")
    plt.savefig(acc_fig_path, dpi=150)
    plt.close()
    print(f"Saved: {acc_fig_path}")

    # ── Figure 2: overlaid validation-accuracy learning curves ──
    if histories:
        fig, ax = plt.subplots(figsize=(9, 5))
        for tag, history in histories.items():
            ax.plot(history["val_acc"], label=f"{tag.upper()} val_acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title("Validation Accuracy Learning Curves")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        curves_fig_path = os.path.join(figures_dir, "learning_curves.png")
        plt.savefig(curves_fig_path, dpi=150)
        plt.close()
        print(f"Saved: {curves_fig_path}")

    print("\nComparison complete! Paste comparison_table.md into the README and the slides.")


if __name__ == "__main__":
    main()
