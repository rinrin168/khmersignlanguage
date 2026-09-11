# ============================================================
# utils.py — shared helpers used by every training/evaluation
# script (seeding, device selection, checkpointing, param count).
# ============================================================

import os
import random
import time
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fix random seeds across Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str, model, optimizer, epoch: int, best_val_acc: float):
    """Save a resumable checkpoint (guards against Colab disconnects)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


class Timer:
    """Context manager returning elapsed wall-clock seconds."""

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self._start
