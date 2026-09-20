# ============================================================
# transformer_model.py — Approach 3: Transformer encoder, trained
# from scratch (no recurrence at all — the third distinct
# architecture family, dimension (A) in the assignment brief).
#
#   Linear projection (258 -> d_model) + sinusoidal positional encoding
#   -> TransformerEncoder (self-attention over the clip's frames)
#   -> mean-pool over time -> Dense(64, ReLU) -> Dropout -> Dense(num_classes)
# ============================================================

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_size: int = 258, num_classes: int = 10, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.2, max_len: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)          # (batch, seq_len, d_model)
        x = x.mean(dim=1)            # mean-pool over the time dimension

        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
