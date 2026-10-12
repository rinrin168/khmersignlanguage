# ============================================================
# make_slide_figures.py
# Purpose : Build the presentation figures from the saved results
#           (no numbers are typed by hand). Output: slides/figures/*.png
#
# How to run (after train/evaluate/diagnostics):
#   venv\Scripts\python.exe src\make_slide_figures.py
# ============================================================

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

DATA_DIR = "data"
RESULTS_DIR = "results"
OUT_DIR = os.path.join("slides", "figures")
APPROACHES = ["gru", "transformer", "lstm"]
NAMES = {"lstm": "LSTM", "gru": "GRU", "transformer": "Transformer"}
COLORS = {"lstm": "#E69F00", "gru": "#0072B2", "transformer": "#009E73"}
INK, MUTED, ACCENT = "#222222", "#6B7280", "#D1495B"

plt.rcParams.update({
    "svg.fonttype": "path", "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 16, "axes.titleweight": "bold", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def load(name, fname):
    with open(os.path.join(RESULTS_DIR, name, fname)) as f:
        return json.load(f)


def pretty(label):
    return label.replace("dont_understand", "don't understand").replace("_", " ")


def save(fig, fname):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    canva = os.path.join("slides", "art", "canva", "charts")
    os.makedirs(canva, exist_ok=True)
    stem = os.path.splitext(fname)[0]
    fig.savefig(os.path.join(canva, stem + ".png"), dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(os.path.join(canva, stem + ".svg"), bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("saved", path)


def box(ax, x, y, w, h, text, fc="#EEF2F6", ec="#9CA3AF", fs=13, bold=True, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 lw=2, color=MUTED))


# ---------------------------------------------------------------- pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.set_xlim(0, 15); ax.set_ylim(0, 4.4); ax.axis("off")
    steps = [
        ("Webcam", "OpenCV,\nmirrored view", "#EEF2F6"),
        ("MediaPipe\nHolistic", "pose + both hands\n(pretrained, frozen)", "#EEF2F6"),
        ("258 numbers\nper frame", "33 pose x 4\n+ 2 hands x 21 x 3", "#EEF2F6"),
        ("Clip of\n90 frames", "90 x 258 array\n(about 3 s at 30 fps)", "#EEF2F6"),
        ("LSTM / GRU /\nTransformer", "trained from scratch\nin PyTorch", "#D6E9F8"),
        ("1 of 11 signs", "softmax over\n11 classes", "#DDF2E6"),
    ]
    w, gap, x = 2.0, 0.5, 0.2
    for i, (title, sub, fc) in enumerate(steps):
        box(ax, x, 1.75, w, 1.5, title, fc=fc, fs=13)
        ax.text(x + w / 2, 1.3, sub, ha="center", va="top", fontsize=11, color=MUTED, linespacing=1.3)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.04, 2.5, x + w + gap - 0.04, 2.5)
        x += w + gap
    ax.text(7.5, 4.05, "Same input and the same train / validation / test split for all three models",
            ha="center", fontsize=14, color=INK, fontweight="bold")
    save(fig, "fig_pipeline.png")


# ---------------------------------------------------------------- dataset
def fig_dataset_split(labels):
    counts = {s: np.bincount(np.load(os.path.join(DATA_DIR, f"y_{s}.npy")), minlength=len(labels))
              for s in ["train", "val", "test"]}
    fig, ax = plt.subplots(figsize=(11, 6))
    ypos = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    colors = {"train": "#5D6D7E", "val": "#AAB7B8", "test": "#8E44AD"}
    for s in ["train", "val", "test"]:
        ax.barh(ypos, counts[s], left=left, color=colors[s], label=s, height=0.72)
        left += counts[s]
    for yv, tot in zip(ypos, left):
        ax.text(tot + 0.4, yv, f"{int(tot)}", va="center", fontsize=12, color=MUTED)
    ax.set_yticks(ypos); ax.set_yticklabels([pretty(l) for l in labels])
    ax.set_xlabel("clips"); ax.set_xlim(0, 34)
    n = [int(counts[s].sum()) for s in ["train", "val", "test"]]
    ax.set_title(f"{sum(n)} clips: {n[0]} train / {n[1]} validation / {n[2]} test (stratified)")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, "fig_dataset_split.png")


def fig_hand_detection(labels):
    X, y = np.load(os.path.join(DATA_DIR, "X.npy")), np.load(os.path.join(DATA_DIR, "y.npy"))
    rows = []
    for i, w in enumerate(labels):
        c = X[y == i]
        left = (np.abs(c[:, :, 132:195]).sum(2) > 0)
        right = (np.abs(c[:, :, 195:258]).sum(2) > 0)
        rows.append((w, (left | right).mean() * 100, right.mean() * 100 < 5))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (w, pct, one) in enumerate(rows):
        ax.barh(i, pct, color=ACCENT if pct < 85 else "#7F8FA6", height=0.72)
        ax.text(pct + 0.8, i, f"{pct:.0f}%" + ("  (one hand)" if one else "  (two hands)"),
                va="center", fontsize=12, color=MUTED)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([pretty(r[0]) for r in rows])
    ax.set_xlim(0, 125); ax.set_xlabel("% of frames with at least one hand detected")
    ax.set_title("Hand detection quality per sign (body pose found in 100% of frames)")
    save(fig, "fig_hand_detection.png")


# ---------------------------------------------------------------- architectures
def stack(ax, x0, title, color, layers, footer):
    w, h, gap = 4.2, 0.78, 0.26
    for i, (txt, kind) in enumerate(layers):
        y = 1.1 + i * (h + gap)
        fc = {"io": "#EEF2F6", "core": color, "head": "#F3F4F6"}[kind]
        tc = "white" if kind == "core" else INK
        box(ax, x0, y, w, h, txt, fc=fc, ec=color if kind == "core" else "#9CA3AF", fs=11.5, tc=tc)
        if i < len(layers) - 1:
            arrow(ax, x0 + w / 2, y + h + 0.02, x0 + w / 2, y + h + gap - 0.02)
    top = 1.1 + len(layers) * (h + gap)
    ax.text(x0 + w / 2, top + 0.1, title, ha="center", fontsize=17, fontweight="bold", color=color)
    ax.text(x0 + w / 2, 0.8, footer, ha="center", va="top", fontsize=12, color=MUTED, linespacing=1.4)


def fig_architectures():
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9.6); ax.axis("off")
    meta = {m: load(m, "meta.json") for m in ["lstm", "gru", "transformer"]}
    rec = lambda cell: [("Input  90 x 258", "io"), (f"{cell} 64", "core"), (f"{cell} 128", "core"),
                        (f"{cell} 64  (last hidden state)", "core"), ("Dropout 0.2", "head"),
                        ("Dense 64 + ReLU", "head"), ("Dense 11  (softmax)", "head")]
    stack(ax, 0.2, "LSTM", COLORS["lstm"], rec("LSTM"),
          f"{meta['lstm']['trainable_params']:,} parameters\nrecurrent, 3 gates + cell state\nDropout 0.2 after every block")
    stack(ax, 5.4, "GRU", COLORS["gru"], rec("GRU"),
          f"{meta['gru']['trainable_params']:,} parameters\nrecurrent, 2 gates\nDropout 0.2 after every block")
    stack(ax, 10.6, "Transformer", COLORS["transformer"],
          [("Input  90 x 258", "io"), ("Linear 258 -> 128", "core"), ("+ positional encoding", "core"),
           ("2 encoder layers: 4-head\nself-attention + feed-forward 256", "core"),
           ("Average over the 90 frames", "core"), ("Dense 64 + ReLU", "head"),
           ("Dense 11  (softmax)", "head")],
          f"{meta['transformer']['trainable_params']:,} parameters\nno recurrence, attention only\nDropout 0.2 inside layers")
    save(fig, "fig_architectures.png")


# ---------------------------------------------------------------- curves
def fig_curves():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    for j, m in enumerate(["lstm", "gru", "transformer"]):
        h = load(m, "history.json")
        ep = np.arange(1, len(h["val_acc"]) + 1)
        best = int(np.argmax(h["val_acc"])) + 1
        ax = axes[0, j]
        ax.plot(ep, h["train_acc"], color="#374151", lw=2, label="train")
        ax.plot(ep, h["val_acc"], color=COLORS[m], lw=2.4, label="validation")
        ax.axvline(best, color=MUTED, ls="--", lw=1.2)
        ax.text(best + 0.6, 0.05, f"best epoch {best}", fontsize=11, color=MUTED)
        ax.set_ylim(0, 1.03); ax.set_title(NAMES[m], color=COLORS[m])
        ax.set_ylabel("accuracy" if j == 0 else "")
        ax = axes[1, j]
        ax.plot(ep, h["train_loss"], color="#374151", lw=2, label="train")
        ax.plot(ep, h["val_loss"], color=COLORS[m], lw=2.4, label="validation")
        ax.axvline(best, color=MUTED, ls="--", lw=1.2)
        ax.set_yscale("log"); ax.set_xlabel("epoch")
        ax.yaxis.set_minor_formatter(NullFormatter())
        if m == "lstm":
            ax.set_yticks([1.2, 1.6, 2.0, 2.4])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylabel("loss (log scale)" if j == 0 else "")
    handles = [Line2D([0], [0], color="#374151", lw=2, label="train (dark grey)"),
               Line2D([0], [0], color="#9AA5B1", lw=2.6, label="validation (model colour)")]
    names = [h.get_label() for h in handles]
    fig.tight_layout()
    fig.legend(handles, names, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_curves_train_val.png")


# ---------------------------------------------------------------- results
def fig_results():
    diag = json.load(open(os.path.join(RESULTS_DIR, "diagnostics.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(3); w = 0.36
    acc = [load(m, "metrics.json")["test_accuracy"] for m in APPROACHES]
    f1 = [load(m, "metrics.json")["f1_macro"] for m in APPROACHES]
    ci = np.array([diag["models"][m]["test_accuracy_ci95"] for m in APPROACHES])
    err = np.vstack([np.array(acc) - ci[:, 0], ci[:, 1] - np.array(acc)])
    a.bar(x - w / 2, acc, w, color=[COLORS[m] for m in APPROACHES], yerr=err, capsize=5,
          error_kw={"lw": 1.6, "color": INK}, label="test accuracy (95% CI)")
    a.bar(x + w / 2, f1, w, color=[COLORS[m] for m in APPROACHES], alpha=0.45, label="macro F1")
    for xi, v in zip(x - w / 2, acc):
        a.text(xi, 0.04, f"{v*100:.0f}%", ha="center", color="white", fontweight="bold", fontsize=14)
    for xi, v in zip(x + w / 2, f1):
        a.text(xi, 0.04, f"{v:.2f}", ha="center", color=INK, fontweight="bold", fontsize=13)
    a.set_xticks(x); a.set_xticklabels([NAMES[m] for m in APPROACHES]); a.set_ylim(0, 1.08)
    a.set_title("Test set (50 clips): solid = accuracy, light = macro F1"); a.set_ylabel("score")
    params = [load(m, "meta.json")["trainable_params"] for m in APPROACHES]
    secs = [load(m, "meta.json")["training_time_seconds"] for m in APPROACHES]
    eps = [load(m, "meta.json")["epochs_run"] for m in APPROACHES]
    b.bar(x, params, color=[COLORS[m] for m in APPROACHES], width=0.55)
    for xi, p, s, e in zip(x, params, secs, eps):
        b.text(xi, p + 6000, f"{p:,}\n{s:.1f} s, {e} epochs", ha="center", fontsize=12)
    b.set_xticks(x); b.set_xticklabels([NAMES[m] for m in APPROACHES])
    b.set_ylim(0, max(params) * 1.28); b.set_ylabel("trainable parameters")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    b.set_title("Model size and training cost (Tesla T4)")
    fig.tight_layout()
    save(fig, "fig_results.png")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_accuracy_bars():
    tags = list(APPROACHES)
    if os.path.exists(os.path.join(RESULTS_DIR, "lstm_tuned", "metrics.json")):
        tags.append("lstm_tuned")
    names = {**NAMES, "lstm_tuned": "LSTM (tuned)"}
    colors = {**COLORS, "lstm_tuned": "#F2C14E"}
    n = len(np.load(os.path.join(DATA_DIR, "y_test.npy")))
    acc = np.array([load(t, "metrics.json")["test_accuracy"] for t in tags])
    ci = np.array([wilson(round(a * n), n) for a in acc])
    err = np.vstack([acc - ci[:, 0], ci[:, 1] - acc])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(range(len(tags)), acc, color=[colors[t] for t in tags], width=0.62,
           yerr=err, capsize=6, error_kw={"lw": 1.8, "color": INK})
    for i, a in enumerate(acc):
        ax.text(i, 0.04, f"{a*100:.0f}%", ha="center", color="white", fontweight="bold", fontsize=17)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels([names[t] for t in tags], fontsize=13)
    ax.set_ylim(0, 1.08); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_title("Test accuracy (50 clips, 95% interval)", fontsize=14)
    save(fig, "fig_accuracy_bars.png")


# ---------------------------------------------------------------- tuning
def fig_tuning():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, m in zip(axes, ["gru", "lstm"]):
        res = load(m, "hparam_search.json")["results"]
        lrs, wds = [1e-3, 5e-4, 1e-4], [0.0, 1e-4]
        grid = np.zeros((2, 3))
        for r in res:
            grid[wds.index(r["weight_decay"]), lrs.index(r["lr"])] = r["best_val_acc"]
        im = ax.imshow(grid, vmin=0.4, vmax=1.0, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(3):
                ax.text(j, i, f"{grid[i, j]*100:.0f}%", ha="center", va="center", fontsize=17,
                        fontweight="bold", color="white" if grid[i, j] > 0.75 else INK)
        bi, bj = np.unravel_index(np.argmax(grid), grid.shape)
        best = load(m, "hparam_search.json")["best"]
        bi, bj = wds.index(best["weight_decay"]), lrs.index(best["lr"])
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=ACCENT, lw=4))
        ax.set_xticks(range(3)); ax.set_xticklabels(["1e-3", "5e-4", "1e-4"])
        ax.set_yticks(range(2)); ax.set_yticklabels(["0", "1e-4"])
        ax.set_xlabel("learning rate"); ax.set_ylabel("weight decay (L2)")
        ax.set_title(f"{NAMES[m]}: best validation accuracy", color=COLORS[m])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.5, -0.02, "Red box = combination chosen on validation accuracy (60 epochs each, seed 42)",
             ha="center", fontsize=12, color=MUTED)
    fig.tight_layout()
    save(fig, "fig_tuning.png")


# ---------------------------------------------------------------- errors
def fig_lstm_errors():
    e = pd.read_csv(os.path.join(RESULTS_DIR, "lstm", "errors.csv"))
    diag = json.load(open(os.path.join(RESULTS_DIR, "diagnostics.json")))["lstm_error_structure"]
    ct = pd.crosstab(e.true_label, e.predicted_label)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    pal = {"understand": "#C0392B", "again": "#2980B9", "hearing_person": "#27AE60"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    left = np.zeros(len(ct))
    for col in ct.columns:
        a.barh(range(len(ct)), ct[col], left=left, color=pal.get(col, "#888"),
               label=f"predicted '{col}'", height=0.7)
        left += ct[col].values
    a.set_yticks(range(len(ct))); a.set_yticklabels([pretty(t) for t in ct.index])
    a.set_xlabel("mistaken test clips"); a.legend(frameon=False, loc="lower right", fontsize=12)
    a.set_title("LSTM mistakes: true sign vs what it said")
    b.hist(e.predicted_confidence, bins=14, color="#9AA5B1")
    chance = diag["chance_confidence"]
    b.axvline(chance, color=ACCENT, lw=2.5)
    b.text(chance - 0.008, b.get_ylim()[1] * 0.6, "chance\n0.09", color=ACCENT, fontsize=13,
           fontweight="bold", ha="right", va="center")
    b.set_xlabel("confidence the LSTM gave to its wrong answer"); b.set_ylabel("mistakes")
    b.set_title("It was guessing, not confident")
    b.set_xlim(0, 0.5)
    fig.tight_layout()
    save(fig, "fig_lstm_errors.png")


# ---------------------------------------------------------------- diagnostics
def fig_diagnostics():
    d = json.load(open(os.path.join(RESULTS_DIR, "diagnostics.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(3); w = 0.26
    keys = [("test_accuracy", "original order", 1.0), ("shuffled_frames_mean", "frames shuffled", 0.6),
            ("reversed_time", "time reversed", 0.35)]
    for k, (key, lab, alpha) in enumerate(keys):
        vals = [d["models"][m][key] for m in APPROACHES]
        a.bar(x + (k - 1) * w, vals, w, color=[COLORS[m] for m in APPROACHES], alpha=alpha)
        for xi, v in zip(x + (k - 1) * w, vals):
            a.text(xi, v + 0.015, f"{v*100:.0f}", ha="center", fontsize=11)
    a.set_xticks(x); a.set_xticklabels([NAMES[m] for m in APPROACHES]); a.set_ylim(0, 1.15)
    a.set_ylabel("test accuracy")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    a.set_title("Does frame order matter?")
    a.legend(handles=[Patch(fc="#555555", alpha=al, label=lab) for _, lab, al in keys],
             frameon=False, loc="upper right", fontsize=12)
    items = [("hand presence only\n(2 numbers, logistic reg.)", d["baselines"]["logreg_hand_presence_only"], "#9AA5B1"),
             ("clip average, no order\n(logistic regression)", d["baselines"]["logreg_clip_mean_no_order"], "#5D6D7E"),
             ("GRU", d["models"]["gru"]["test_accuracy"], COLORS["gru"]),
             ("Transformer", d["models"]["transformer"]["test_accuracy"], COLORS["transformer"]),
             ("LSTM", d["models"]["lstm"]["test_accuracy"], COLORS["lstm"])]
    b.barh(range(len(items))[::-1], [i[1] for i in items], color=[i[2] for i in items], height=0.66)
    for yy, it in zip(range(len(items))[::-1], items):
        b.text(it[1] + 0.015, yy, f"{it[1]*100:.0f}%", va="center", fontsize=13, fontweight="bold")
    b.set_yticks(range(len(items))[::-1]); b.set_yticklabels([i[0] for i in items], fontsize=12)
    b.set_xlim(0, 1.15); b.set_xlabel("test accuracy")
    b.set_title("How hard is this dataset?")
    fig.tight_layout()
    save(fig, "fig_diagnostics.png")


def main():
    labels = list(np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True))
    fig_pipeline()
    fig_dataset_split(labels)
    fig_hand_detection(labels)
    fig_architectures()
    fig_curves()
    fig_results()
    fig_accuracy_bars()
    fig_tuning()
    fig_lstm_errors()
    fig_diagnostics()


if __name__ == "__main__":
    main()
