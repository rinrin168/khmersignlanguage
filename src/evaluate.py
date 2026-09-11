# ============================================================
# Stage 5 — evaluate.py
# Purpose : Load a trained approach's best checkpoint, score it on
#           the held-out test set (same test set for every
#           approach), and save metrics/plots/error analysis.
#
# How to run:
#   venv\Scripts\python.exe src\evaluate.py --model lstm
# ============================================================

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device  # noqa: E402
from dataset import LandmarkSequenceDataset  # noqa: E402
from models import build_model  # noqa: E402

DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate one trained approach on the test set.")
    p.add_argument("--model", required=True, choices=["lstm", "gru", "transformer"])
    p.add_argument("--tag", default=None)
    return p.parse_args()


def plot_training_history(history: dict, out_path: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes[0].plot(history["train_acc"], label="Train Accuracy", color="steelblue")
    axes[0].plot(history["val_acc"], label="Val Accuracy", color="tomato")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_loss"], label="Train Loss", color="steelblue")
    axes[1].plot(history["val_loss"], label="Val Loss", color="tomato")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved training curves : {out_path}")


def plot_confusion_matrix(y_true, y_pred, labels, out_path: str, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix : {out_path}")


def main():
    args = parse_args()
    tag = args.tag or args.model
    run_dir = os.path.join(RESULTS_DIR, tag)
    set_seed(42)
    device = get_device()

    model_path = os.path.join(MODELS_DIR, f"{tag}_khmer_sign.pt")
    if not os.path.exists(model_path):
        print(f"{model_path} not found. Run: train.py --model {args.model}")
        return

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True)
    num_classes = len(labels)
    input_size = X_test.shape[-1]

    model = build_model(args.model, input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    loader = DataLoader(LandmarkSequenceDataset(X_test, y_test), batch_size=32, shuffle=False)

    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.argmax(dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)

    accuracy = float(np.mean(y_pred == y_true))
    print(f"\nTest Accuracy ({args.model}): {accuracy * 100:.2f}%")

    report_dict = classification_report(y_true, y_pred, target_names=labels,
                                         output_dict=True, zero_division=0)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    # ── Save metrics.json (used later by compare.py) ──
    metrics = {
        "model": args.model,
        "test_accuracy": accuracy,
        "precision_macro": report_dict["macro avg"]["precision"],
        "recall_macro": report_dict["macro avg"]["recall"],
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "precision_weighted": report_dict["weighted avg"]["precision"],
        "recall_weighted": report_dict["weighted avg"]["recall"],
        "f1_weighted": report_dict["weighted avg"]["f1-score"],
        "per_class": {cls: report_dict[cls] for cls in labels},
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ──
    history_path = os.path.join(run_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(history, os.path.join(run_dir, "training_history.png"),
                               f"{args.model.upper()} Training History")

    plot_confusion_matrix(y_true, y_pred, labels,
                           os.path.join(run_dir, "confusion_matrix.png"),
                           f"Confusion Matrix — {args.model.upper()}")

    # ── Error analysis: which test clips were misclassified, and how confident
    #    was the (wrong) prediction? Helps explain likely causes (Section 5.C). ──
    wrong_idx = np.where(y_pred != y_true)[0]
    error_rows = [{
        "test_index": int(i),
        "true_label": labels[y_true[i]],
        "predicted_label": labels[y_pred[i]],
        "predicted_confidence": float(y_prob[i, y_pred[i]]),
    } for i in wrong_idx]
    error_df = pd.DataFrame(error_rows)
    error_path = os.path.join(run_dir, "errors.csv")
    error_df.to_csv(error_path, index=False)
    print(f"\nMisclassified {len(wrong_idx)}/{len(y_true)} test samples -> {error_path}")

    print("\nEvaluation complete! Check the results/ folder.")


if __name__ == "__main__":
    main()
