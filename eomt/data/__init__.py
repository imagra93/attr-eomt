"""COCO-format data loading, augmentation and autodownload for EoMT."""

from .coco import (
    CocoDetection,
    CocoInstanceSeg,
    CocoValImages,
    collate_train,
    collate_val,
)
from .config import load_data_config
from .download import ensure_coco
from .transforms import build_train_transform, build_val_transform

__all__ = [
    "CocoDetection",
    "CocoInstanceSeg",
    "CocoValImages",
    "collate_train",
    "collate_val",
    "load_data_config",
    "ensure_coco",
    "build_train_transform",
    "build_val_transform",
]
