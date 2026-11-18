"""attr-eomt: EoMT instance segmentation with per-instance attributes, multi-view prediction and instance matching.

    >>> from eomt import EoMT
    >>> model = EoMT("l")                          # a new model (DINOv2 backbone)
    >>> model.train(data="run.yaml")               # the run config: data, attribute heads, settings
    >>> model = EoMT("runs/train/eomt-l")          # a trained run (everything read from its checkpoint)
    >>> results = model.predict("images/", plot=True)
    >>> linked = model.match("photos/subject_42/")

The settings sections of a training (:mod:`eomt.engine.train`) are exported too, for code that prefers them to plain
mappings: ``EoMT("l").train(data=..., optim=OptimConfig(lr0=5e-5))``.
"""

from __future__ import annotations

from .api import EoMT
from .config import EOMT_CONFIGS, SIZES, AuxHeadSpec
from .data.subjects import SubjectSetConfig
from .data.transforms import AugConfig
from .engine.train import AuxConfig, ModelConfig, MultiviewConfig, OptimConfig, ValConfig
from .model import EoMTEncoder, EoMTModel, build_model, load_dinov2_backbone
from .serialization import get_version, load_model, migrate_checkpoint, save_checkpoint, summarize_checkpoint

__all__ = [
    "EoMT",
    "AugConfig",
    "ModelConfig",
    "OptimConfig",
    "AuxConfig",
    "ValConfig",
    "MultiviewConfig",
    "SubjectSetConfig",
    "AuxHeadSpec",
    "EOMT_CONFIGS",
    "SIZES",
    "EoMTModel",
    "EoMTEncoder",
    "build_model",
    "load_dinov2_backbone",
    "load_model",
    "migrate_checkpoint",
    "save_checkpoint",
    "summarize_checkpoint",
]


def __getattr__(name: str):
    if name == "__version__":  # computed on first use: from a source checkout it asks git
        return get_version()
    raise AttributeError(f"module 'eomt' has no attribute {name!r}")
