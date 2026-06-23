"""Dataset config (YAML) resolution for COCO-format datasets.

A dataset YAML describes where the COCO images and annotation JSONs live and
whether to auto-download COCO 2017 if missing::

    path: datasets/coco
    train_images: train2017
    train_json: annotations/instances_train2017.json
    val_images: val2017
    val_json: annotations/instances_val2017.json
    download: true            # fetch COCO 2017 from cocodataset.org if absent
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .download import ensure_coco


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
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    root = Path(cfg.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

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

    return {
        "path": str(root),
        "train_images": str(train_images) if train_images else None,
        "train_json": str(train_json) if train_json else None,
        "val_images": str(val_images) if val_images else None,
        "val_json": str(val_json) if val_json else None,
    }
