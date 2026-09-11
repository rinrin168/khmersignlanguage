# ============================================================
# dataset.py — PyTorch Dataset wrapping the landmark sequences.
# Shared by train.py / evaluate.py / hyperparam_search.py so every
# approach loads data identically.
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset


class LandmarkSequenceDataset(Dataset):
    """Wraps (N, seq_len, features) landmark arrays and integer labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
