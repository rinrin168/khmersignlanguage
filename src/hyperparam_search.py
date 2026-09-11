# ============================================================
# hyperparam_search.py
# Purpose : Small hyperparameter search (learning rate + weight
#           decay / regularization) for the best-performing
#           approach, as required by Section 5.C: "Perform
#           hyperparameter tuning (at minimum learning rate and
#           one regularization choice) for at least your
#           best-performing approach, and report what you tried."
#
# Run compare.py FIRST to see which approach currently wins, then:
#   venv\Scripts\python.exe src\hyperparam_search.py --model transformer
# ============================================================

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, get_device, count_trainable_params  # noqa: E402
from dataset import LandmarkSequenceDataset  # noqa: E402
from models import build_model  # noqa: E402

DATA_DIR = "data"
RESULTS_DIR = "results"

LEARNING_RATES = [1e-3, 5e-4, 1e-4]
WEIGHT_DECAYS = [0.0, 1e-4]     # 0.0 = no L2 regularization, 1e-4 = with it
SEARCH_EPOCHS = 40               # short budget per combo — this is a search, not final training
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Learning-rate / weight-decay search for one approach.")
    p.add_argument("--model", required=True, choices=["lstm", "gru", "transformer"])
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def run_one(model_name, lr, weight_decay, X_train, y_train, X_val, y_val,
            num_classes, input_size, device, batch_size):
    set_seed(SEED)
    model = build_model(model_name, input_size=input_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(LandmarkSequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LandmarkSequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    start = time.time()
    for _ in range(SEARCH_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)

    elapsed = time.time() - start
    return best_val_acc, elapsed


def main():
    args = parse_args()
    device = get_device()

    for fname in ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy", "labels.npy"]:
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

    results = []
    for lr, wd in itertools.product(LEARNING_RATES, WEIGHT_DECAYS):
        print(f"\n--- {args.model}: lr={lr}, weight_decay={wd} ---")
        val_acc, elapsed = run_one(args.model, lr, wd, X_train, y_train, X_val, y_val,
                                    num_classes, input_size, device, args.batch_size)
        print(f"    best val_acc over {SEARCH_EPOCHS} epochs: {val_acc:.3f}  ({elapsed:.1f}s)")
        results.append({"lr": lr, "weight_decay": wd, "best_val_acc": val_acc, "seconds": round(elapsed, 1)})

    results.sort(key=lambda r: r["best_val_acc"], reverse=True)

    run_dir = os.path.join(RESULTS_DIR, args.model)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "hparam_search.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "search_epochs": SEARCH_EPOCHS,
            "results": results,
            "best": results[0],
        }, f, indent=2)

    print(f"\nBest combo: lr={results[0]['lr']}, weight_decay={results[0]['weight_decay']} "
          f"-> val_acc {results[0]['best_val_acc']:.3f}")
    print(f"Saved: {out_path}")
    print(f"\nNow retrain the full run with these values:")
    print(f"  venv\\Scripts\\python.exe src\\train.py --model {args.model} "
          f"--lr {results[0]['lr']} --weight_decay {results[0]['weight_decay']}")


if __name__ == "__main__":
    main()
