from .lstm_model import LSTMClassifier
from .gru_model import GRUClassifier
from .transformer_model import TransformerClassifier

MODEL_REGISTRY = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "transformer": TransformerClassifier,
}


def build_model(name: str, input_size: int = 258, num_classes: int = 10, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](input_size=input_size, num_classes=num_classes, **kwargs)
