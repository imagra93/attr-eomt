"""Training, validation and inference engines for EoMT."""

from .match import match
from .predict import (group_owners, photo_similarity, predict, predict_batch, predict_image, predict_set,
                      predict_subject, similar_groups, subject_groups)
from .train import AuxConfig, ModelConfig, MultiviewConfig, OptimConfig, ValConfig, train
from .validate import evaluate, evaluate_detection, evaluate_sets, sweep

__all__ = [
    "train", "ModelConfig", "OptimConfig", "AuxConfig", "ValConfig", "MultiviewConfig",
    "evaluate", "evaluate_detection", "evaluate_sets", "sweep",
    "predict", "predict_batch", "predict_image", "predict_set", "predict_subject", "match",
    "subject_groups", "similar_groups", "photo_similarity", "group_owners",
]
