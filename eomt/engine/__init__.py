"""Training, validation and inference engines for EoMT."""

from .match import match
from .predict import predict, predict_batch, predict_image, predict_set, predict_subject
from .train import AuxConfig, ModelConfig, MultiviewConfig, OptimConfig, ValConfig, train
from .validate import evaluate, evaluate_detection, evaluate_sets, sweep

__all__ = [
    "train", "ModelConfig", "OptimConfig", "AuxConfig", "ValConfig", "MultiviewConfig",
    "evaluate", "evaluate_detection", "evaluate_sets", "sweep",
    "predict", "predict_batch", "predict_image", "predict_set", "predict_subject", "match",
]
