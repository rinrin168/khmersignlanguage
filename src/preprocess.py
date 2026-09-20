# ============================================================
# Stage 3 — preprocess.py
# Purpose : Load X.npy / y.npy, create ONE fixed train/val/test
#           split, and save it so every approach (LSTM/GRU/
#           Transformer) trains and evaluates on identical data.
#
# How to run:
#   venv\Scripts\python.exe src\preprocess.py
# ============================================================
#
# WHY A VALIDATION SET TOO?
#   Train  : the model learns from this.
#   Val    : used during training to pick the best checkpoint,
#            tune hyperparameters, and drive early stopping —
#            never used for the final reported numbers.
#   Test   : held out completely until the very end. Every
#            approach is scored on the SAME test set so the
#            comparison in Section 4/D of the assignment is fair.
#
# Split sizes: 70% train / 15% val / 15% test, stratified by class.
# ============================================================

import json
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed  # noqa: E402

DATA_DIR = "data"
RESULTS_DIR = "results"
SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def main():
    set_seed(SEED)

    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")

    if not os.path.exists(X_path):
        print("X.npy not found. Run extract_landmarks.py first.")
        return

    X = np.load(X_path)              # shape (N, 90, 258)
    y = np.load(y_path)              # shape (N,) integer labels
    labels = np.load(labels_path, allow_pickle=True)

    num_classes = len(labels)
    print(f"Loaded X shape : {X.shape}")
    print(f"Loaded y shape : {y.shape}")
    print(f"Classes ({num_classes}): {list(labels)}")

    # ── Split off the test set first (held out until final evaluation) ──
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    # ── Split the remainder into train / val ──
    val_fraction_of_temp = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction_of_temp,
        random_state=SEED,
        stratify=y_temp
    )

    print(f"\nTrain samples : {X_train.shape[0]}")
    print(f"Val samples   : {X_val.shape[0]}")
    print(f"Test samples  : {X_test.shape[0]}")

    # ── Save preprocessed arrays (raw integer labels — PyTorch's
    #    CrossEntropyLoss expects class indices, not one-hot) ──
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)

    # ── Record the split recipe for the README / reproducibility ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "split_info.json"), "w") as f:
        json.dump({
            "seed": SEED,
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
            "num_classes": num_classes,
            "classes": list(labels),
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
        }, f, indent=2)

    print("\nSaved: X_train/X_val/X_test, y_train/y_val/y_test to data/")
    print("Saved: results/split_info.json")
    print("Next step: run train.py --model lstm  (then gru, then transformer)")


if __name__ == "__main__":
    main()
