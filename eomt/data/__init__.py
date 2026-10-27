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
from .sets import CocoViewSets, SetConfig, collate_sets
from .subjects import CocoSubjectSets, MixedSets, SubjectSetConfig, subjects_from_file_names
from .transforms import AugConfig, TrainAugment, build_train_transform, build_val_transform

__all__ = [
    "AugConfig",
    "TrainAugment",
    "CocoDetection",
    "CocoInstanceSeg",
    "CocoValImages",
    "CocoViewSets",
    "CocoSubjectSets",
    "MixedSets",
    "SetConfig",
    "SubjectSetConfig",
    "subjects_from_file_names",
    "collate_sets",
    "collate_train",
    "collate_val",
    "load_data_config",
    "ensure_coco",
    "build_train_transform",
    "build_val_transform",
]
