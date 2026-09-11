# ============================================================
# Stage 2 — extract_landmarks.py
# Purpose : Walk through the data/ folder, load every .npy
#           sequence file, and compile ONE combined dataset:
#             X.npy  — shape (N, 30, 258)  ← inputs
#             y.npy  — shape (N,)           ← integer labels
#             labels.npy — the list of word names
#
# How to run:
#   venv\Scripts\python.exe src\extract_landmarks.py
# ============================================================

import numpy as np
import os

DATA_DIR = "data"    # folder created by collect_data.py
OUT_DIR  = "data"    # we save X.npy / y.npy in the same folder

def main():
    # ── Find which words exist on disk ──
    words = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and not d.endswith(".npy")   # skip .npy files
    ])

    if not words:
        print("No word folders found in data/. Run collect_data.py first.")
        return

    print(f"Found {len(words)} words: {words}")

    sequences = []   # list of arrays, each shape (30, 258)
    labels    = []   # matching integer label for each sequence

    for label_idx, word in enumerate(words):
        word_dir = os.path.join(DATA_DIR, word)
        seq_folders = sorted(os.listdir(word_dir), key=lambda x: int(x))

        for seq_folder in seq_folders:
            npy_path = os.path.join(word_dir, seq_folder, "landmarks.npy")
            if not os.path.exists(npy_path):
                continue

            seq_array = np.load(npy_path)          # shape (30, 258)

            # Basic validation
            if seq_array.shape != (30, 258):
                print(f"  Skipping {npy_path} — unexpected shape {seq_array.shape}")
                continue

            sequences.append(seq_array)
            labels.append(label_idx)

        print(f"  Loaded {len(seq_folders)} sequences for '{word}' (label={label_idx})")

    if not sequences:
        print("No valid sequences found. Check your data/ folder.")
        return

    X = np.array(sequences)   # shape: (N, 30, 258)
    y = np.array(labels)       # shape: (N,)

    np.save(os.path.join(OUT_DIR, "X.npy"),      X)
    np.save(os.path.join(OUT_DIR, "y.npy"),      y)
    np.save(os.path.join(OUT_DIR, "labels.npy"), np.array(words))

    print(f"\nSaved X.npy      : shape {X.shape}")
    print(f"Saved y.npy      : shape {y.shape}")
    print(f"Saved labels.npy : {words}")
    print("\nExtraction complete! Next step: run preprocess.py")


if __name__ == "__main__":
    main()
