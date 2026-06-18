"""libre-eomt: standalone EoMT instance segmentation.

Public API:

    >>> from eomt import build_model, load_model, load_dinov2_backbone
    >>> model = build_model("s", nc=80)        # build architecture
    >>> load_dinov2_backbone(model)            # init encoder from DINOv2
    >>> seg = load_model("best.pt")            # reload a trained checkpoint
"""

from __future__ import annotations

from .config import EOMT_CONFIGS, SIZES, build_eomt_config
from .model import EoMTModel, build_model, load_dinov2_backbone
from .postprocess import postprocess_instance
from .serialization import load_model, load_raw, save_checkpoint, wrap_checkpoint

__all__ = [
    "EOMT_CONFIGS",
    "SIZES",
    "build_eomt_config",
    "EoMTModel",
    "build_model",
    "load_dinov2_backbone",
    "postprocess_instance",
    "load_model",
    "load_raw",
    "save_checkpoint",
    "wrap_checkpoint",
]

try:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("libre-eomt")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"
