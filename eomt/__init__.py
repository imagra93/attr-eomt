"""libre-eomt: standalone EoMT instance segmentation.

Public API:

    >>> from eomt import EoMT
    >>> model = EoMT("l")                      # fresh model (DINOv2 backbone)
    >>> model.train(data="coco", epochs=50)    # COCO auto-downloads if missing
    >>> model = EoMT("runs/train/eomt-l")      # reload a run (everything auto-detected)
    >>> results = model.predict("images/", plot=True)

Lower-level building blocks (``build_model`` / ``load_model`` / ``EoMTModel`` /
``EoMTEncoder``) remain available for advanced use.
"""

from __future__ import annotations

from .api import EoMT
from .config import EOMT_CONFIGS, SIZES, build_eomt_config
from .model import EoMTEncoder, EoMTModel, build_model, load_dinov2_backbone
from .postprocess import postprocess_instance
from .serialization import (
    load_model,
    load_raw,
    resolve_checkpoint,
    save_checkpoint,
    wrap_checkpoint,
)

__all__ = [
    "EoMT",
    "EOMT_CONFIGS",
    "SIZES",
    "build_eomt_config",
    "EoMTModel",
    "EoMTEncoder",
    "build_model",
    "load_dinov2_backbone",
    "postprocess_instance",
    "load_model",
    "load_raw",
    "resolve_checkpoint",
    "save_checkpoint",
    "wrap_checkpoint",
]

try:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("libre-eomt")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"
