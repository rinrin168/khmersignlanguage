# ============================================================
# Stage 4 — train.py
# Purpose : Train ONE of the three distinct deep-learning
#           approaches (LSTM / GRU / Transformer encoder), all
#           trained from scratch on the identical data/ split
#           produced by preprocess.py.
#
# How to run:
#   venv\Scripts\python.exe src\train.py --model lstm
#   venv\Scripts\python.exe src\train.py --model gru
#   venv\Scripts\python.exe src\train.py --model transformer
#
# Each run saves, under results/<model>/:
#   checkpoint.pt      — best model + optimizer state (torch.save)
#   history.json        — per-epoch train/val loss & accuracy
#   meta.json            — trainable params, training time, hardware
# and under models/:
#   <model>_khmer_sign.pt
# ============================================================

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, device_name, count_trainable_params, save_checkpoint  # noqa: E402
from dataset import LandmarkSequenceDataset  # noqa: E402
from models import build_model  # noqa: E402

DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"


def parse_args():
    p = argparse.ArgumentParser(description="Train one Khmer Sign Language approach.")
    p.add_argument("--model", required=True, choices=["lstm", "gru", "transformer"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization")
    p.add_argument("--patience", type=int, default=25, help="early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None, help="optional run name (defaults to --model)")
    return p.parse_args()


def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    tag = args.tag or args.model
    run_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    set_seed(args.seed)
    device = get_device()

    # ── Load the fixed split (identical for every approach) ──
    required = ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy", "labels.npy"]
    for fname in required:
        if not os.path.exists(os.path.join(DATA_DIR, fname)):
            print(f"{fname} not found. Run preprocess.py first.")
            return

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True)

    num_classes = len(labels)
    input_size = X_train.shape[-1]

    train_loader = DataLoader(LandmarkSequenceDataset(X_train, y_train),
                               batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LandmarkSequenceDataset(X_val, y_val),
                             batch_size=args.batch_size, shuffle=False)

    print(f"Training '{args.model}' on {len(X_train)} samples, "
          f"validating on {len(X_val)} samples, {num_classes} classes")
    print(f"Device: {device_name(device)}")

    model = build_model(args.model, input_size=input_size, num_classes=num_classes).to(device)
    n_params = count_trainable_params(model)
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    best_model_path = os.path.join(MODELS_DIR, f"{tag}_khmer_sign.pt")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    start_time = __import__("time").time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_acc)
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        flag = " *" if improved else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f}{flag}")

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping — no val_acc improvement for {args.patience} epochs.")
            break

    training_time = __import__("time").time() - start_time

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "model": args.model,
            "trainable_params": n_params,
            "training_time_seconds": round(training_time, 1),
            "device": device_name(device),
            "epochs_run": len(history["train_loss"]),
            "best_val_acc": best_val_acc,
            "hyperparameters": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "seed": args.seed,
            },
        }, f, indent=2)

    print(f"\nBest val_acc        : {best_val_acc:.3f}")
    print(f"Best model saved to : {best_model_path}")
    print(f"Checkpoint saved to : {checkpoint_path}")
    print(f"History/meta saved  : {run_dir}/")
    print(f"Next step           : run evaluate.py --model {args.model}")


if __name__ == "__main__":
    main()
