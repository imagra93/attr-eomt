"""Dataset config (YAML) resolution for COCO-format datasets.

A dataset YAML describes where the COCO images and annotation JSONs live and
whether to auto-download COCO 2017 if missing::

    path: datasets/coco
    train_images: train2017
    train_json: annotations/instances_train2017.json
    val_images: val2017
    val_json: annotations/instances_val2017.json
    download: true            # fetch COCO 2017 from cocodataset.org if absent

The same YAML is the **run config**: everything a training run needs besides the data, written out in one file, so that
the dataset holds only images, annotations and its attributes' names, types and categories, and the YAML says how a
model is trained on it and how it is to be run (``sample_data/run.yaml`` is a commented template). Every section is
optional::

    attributes:                    # the per-instance heads to train (an attribute of the data left out is not) and how
      laterality: {flip_aware: {left: right}}       # each one learns (eomt.data.coco.BEHAVIOR_KEYS); without the
      area: {log: true, range: [0, 50000], crop: scale}   # section, every attribute as the data declares it
      typology:
    aug:                           # AugConfig fields (``train_aug``, the older name, is the same)
      min_scale: 0.6
      instance_crop_prob: 0.8
    train:                         # train() keywords: epochs, imgsz, lr0, init_weights, project, name, hires_epochs...
      epochs: 20
      init_weights: ../runs/parts/weights/best.pt
    inference:                     # saved in every checkpoint; the defaults of predict() / track() with that model
      conf_thres: 0.25

Explicit arguments (code, command line) beat the YAML, which beats ``train()``'s defaults. Relative paths (``path``,
``train.init_weights``, ``train.project``) are relative to the YAML. The model size and the device are not part of it
(``EoMT("l")`` / ``--size``, ``--device``): a warm start takes the size from its checkpoint.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from .download import ensure_coco

#: Path-valued keys of the ``train`` section, resolved against the YAML.
_TRAIN_PATH_KEYS = ("init_weights", "project")
#: train() keywords the ``train`` section cannot set: the data paths are the YAML's top level, the size and the device
#: are chosen when the model is built.
_TRAIN_RESERVED = ("train_images", "train_json", "val_images", "val_json", "size", "device")
_SECTIONS = ("attributes", "aug", "train_aug", "train", "inference")
_KEYS = {"path", "train_images", "train_json", "val_images", "val_json", "download", *_SECTIONS}


def _read_config(yaml_path: Path) -> dict:
    """The YAML with its sections checked and its relative paths made absolute against its folder."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    here = yaml_path.parent
    for key in _SECTIONS:
        if cfg.get(key) is not None and not isinstance(cfg[key], dict):
            raise ValueError(f"`{key}` in {yaml_path} must be a mapping, got {type(cfg[key]).__name__}")
    unknown = set(cfg) - _KEYS
    if unknown:
        warnings.warn(f"{yaml_path}: unknown keys {sorted(unknown)} ignored", stacklevel=3)
    reserved = set(cfg.get("train") or {}) & set(_TRAIN_RESERVED)
    if reserved:
        raise ValueError(f"{yaml_path}: `train` cannot set {sorted(reserved)} (the data paths go at the top level; the size "
                         "and device when the model is built: EoMT(size) / --size, --device)")
    if cfg.get("path") is not None:
        cfg["path"] = str((here / cfg["path"]).resolve())
    for key in _TRAIN_PATH_KEYS:
        if isinstance((cfg.get("train") or {}).get(key), str):
            cfg["train"][key] = str((here / cfg["train"][key]).resolve())
    return cfg


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_data_config(yaml_path: str | Path, *, autodownload: bool = True) -> dict[str, Any]:
    """Resolve a dataset YAML into absolute train/val image and JSON paths.

    If ``download: true`` (and ``autodownload``) and the expected files are
    missing, COCO 2017 is downloaded into ``path``.
    """
    yaml_path = Path(yaml_path).resolve()
    cfg = _read_config(yaml_path)
    root = Path(cfg.get("path") or yaml_path.parent)

    train_images = _resolve(root, cfg.get("train_images"))
    train_json = _resolve(root, cfg.get("train_json"))
    val_images = _resolve(root, cfg.get("val_images"))
    val_json = _resolve(root, cfg.get("val_json"))

    want_train = train_json is not None
    want_val = val_json is not None
    missing = any(
        p is not None and not p.exists()
        for p in (train_images, train_json, val_images, val_json)
    )

    if cfg.get("download") and autodownload and missing:
        paths = ensure_coco(root, train=want_train, val=want_val)
        train_images = train_images or paths["train_images"]
        train_json = train_json or paths["train_json"]
        val_images = val_images or paths["val_images"]
        val_json = val_json or paths["val_json"]

    if cfg.get("aug") is not None and cfg.get("train_aug") is not None:
        raise ValueError(f"{yaml_path}: `aug` and `train_aug` are the same section; use one")
    train_aug = cfg.get("aug", cfg.get("train_aug"))

    return {
        "attributes": cfg.get("attributes"),
        "train": cfg.get("train") or {},
        "inference": cfg.get("inference"),
        "path": str(root),
        "train_images": str(train_images) if train_images else None,
        "train_json": str(train_json) if train_json else None,
        "val_images": str(val_images) if val_images else None,
        "val_json": str(val_json) if val_json else None,
        "train_aug": train_aug,
    }
