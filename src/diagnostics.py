# ============================================================
# diagnostics.py
# Purpose : Sanity checks on the trained models and the dataset.
#           Answers: do the models actually use frame ORDER, how
#           much does a no-order baseline already solve, and how
#           uncertain is a 50-clip test accuracy?
#
# How to run (after train.py / evaluate.py for all three models):
#   venv\Scripts\python.exe src\diagnostics.py
# Writes results/diagnostics.json
# ============================================================

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model  # noqa: E402

DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
APPROACHES = ["lstm", "gru", "transformer"]
N_PERMUTATIONS = 20
SEED = 0


def wilson_interval(k, n, z=1.96):
    """95% confidence interval for an accuracy of k correct out of n."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def accuracy(net, X, y):
    with torch.no_grad():
        pred = net(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return float((pred == y).mean())


def hand_presence(X):
    left = (np.abs(X[:, :, 132:195]).sum(2) > 0).mean(1)
    right = (np.abs(X[:, :, 195:258]).sum(2) > 0).mean(1)
    return np.stack([left, right], 1)


def main():
    labels = list(np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True))
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    n_test = len(y_test)
    rng = np.random.default_rng(SEED)

    out = {"n_test": n_test, "models": {}, "baselines": {}}

    for name in APPROACHES:
        net = build_model(name, input_size=X_test.shape[-1], num_classes=len(labels))
        net.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}_khmer_sign.pt"), map_location="cpu"))
        net.eval()

        original = accuracy(net, X_test, y_test)
        shuffled = [accuracy(net, X_test[:, rng.permutation(X_test.shape[1]), :], y_test)
                    for _ in range(N_PERMUTATIONS)]
        reversed_ = accuracy(net, X_test[:, ::-1, :].copy(), y_test)
        low, high = wilson_interval(round(original * n_test), n_test)

        out["models"][name] = {
            "test_accuracy": original,
            "test_accuracy_ci95": [low, high],
            "shuffled_frames_mean": float(np.mean(shuffled)),
            "shuffled_frames_min": float(np.min(shuffled)),
            "shuffled_frames_max": float(np.max(shuffled)),
            "reversed_time": reversed_,
        }

    lr_hands = LogisticRegression(max_iter=2000).fit(hand_presence(X_train), y_train)
    out["baselines"]["logreg_hand_presence_only"] = float(lr_hands.score(hand_presence(X_test), y_test))
    lr_mean = LogisticRegression(max_iter=3000).fit(X_train.mean(1), y_train)
    out["baselines"]["logreg_clip_mean_no_order"] = float(lr_mean.score(X_test.mean(1), y_test))

    errors = pd.read_csv(os.path.join(RESULTS_DIR, "lstm", "errors.csv"))
    full_X = np.load(os.path.join(DATA_DIR, "X.npy"))
    full_y = np.load(os.path.join(DATA_DIR, "y.npy"))
    one_handed = [labels[i] for i in range(len(labels))
                  if (np.abs(full_X[full_y == i][:, :, 195:258]).sum(2) > 0).mean() < 0.05]
    guessed = errors[errors.predicted_label == "understand"]
    out["lstm_error_structure"] = {
        "one_handed_signs": one_handed,
        "predicted_understand_total": int(len(guessed)),
        "predicted_understand_from_one_handed": int(guessed.true_label.isin(one_handed).sum()),
        "mean_confidence_of_errors": float(errors.predicted_confidence.mean()),
        "chance_confidence": 1 / len(labels),
    }

    with open(os.path.join(RESULTS_DIR, "diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'model':12s} {'test':>6s} {'95% CI':>14s} {'shuffled':>9s} {'reversed':>9s}")
    for name, m in out["models"].items():
        ci = m["test_accuracy_ci95"]
        print(f"{name:12s} {m['test_accuracy']:6.2f} [{ci[0]:.2f}, {ci[1]:.2f}] "
              f"{m['shuffled_frames_mean']:9.2f} {m['reversed_time']:9.2f}")
    print("baselines:", out["baselines"])
    print("lstm error structure:", out["lstm_error_structure"])
    print("Saved results/diagnostics.json")


if __name__ == "__main__":
    main()
