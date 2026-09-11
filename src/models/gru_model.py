# ============================================================
# gru_model.py — Approach 2: stacked GRU, trained from scratch.
#
# Same depth/width as the LSTM approach so the comparison isolates
# the recurrent cell type (GRU has no separate cell state, fewer
# gates/parameters than LSTM) — a distinct architecture, dimension
# (A) in the assignment brief.
# ============================================================

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int = 258, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_size, 64, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(64, 128, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.gru3 = nn.GRU(128, 64, batch_first=True)
        self.drop3 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.drop4 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru1(x)
        x = self.drop1(x)
        x, _ = self.gru2(x)
        x = self.drop2(x)
        x, h_n = self.gru3(x)
        x = h_n[-1]
        x = self.drop3(x)

        x = self.relu(self.fc1(x))
        x = self.drop4(x)
        return self.fc2(x)
