# ============================================================
# lstm_model.py — Approach 1: stacked LSTM, trained from scratch.
#
# Same shape as the original Keras architecture, ported to PyTorch:
#   LSTM(64)  -> Dropout -> LSTM(128) -> Dropout -> LSTM(64) -> Dropout
#   -> Dense(64, ReLU) -> Dropout -> Dense(num_classes)
# ============================================================

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int = 258, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.drop3 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.drop4 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x, (h_n, _) = self.lstm3(x)
        x = h_n[-1]                 # last layer's final hidden state -> (batch, 64)
        x = self.drop3(x)

        x = self.relu(self.fc1(x))
        x = self.drop4(x)
        return self.fc2(x)          # raw logits — CrossEntropyLoss applies softmax
