"""Augmentations for EoMT instance segmentation (torchvision transforms v2).

Transforms operate jointly on the image and the per-instance masks via
``tv_tensors``. The recipe (flip + scale/translation jitter + color jitter, then
square resize + ImageNet normalize) mirrors the Mask2Former/EoMT training style.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

from ..preprocess import IMAGENET_MEAN, IMAGENET_STD

_MEAN = IMAGENET_MEAN.tolist()
_STD = IMAGENET_STD.tolist()


def build_train_transform(
    imgsz: int,
    *,
    flip_prob: float = 0.5,
    min_scale: float = 0.5,
    max_scale: float = 1.0,
    color_jitter: bool = True,
) -> v2.Compose:
    """Training transform: flip + random-resized-crop (scale jitter) + color jitter.

    Output image is a normalized ``float32`` ``(3, imgsz, imgsz)`` tensor; masks
    are ``float32`` ``(N, imgsz, imgsz)`` in ``{0, 1}`` (nearest resampling).
    """
    tfs: list = [v2.RandomHorizontalFlip(p=flip_prob)]
    tfs.append(
        v2.RandomResizedCrop(
            size=(imgsz, imgsz),
            scale=(min_scale, max_scale),
            ratio=(0.75, 1.3333),
            antialias=True,
        )
    )
    if color_jitter:
        tfs.append(v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05))
    tfs.append(v2.ToDtype(torch.float32, scale=True))
    tfs.append(v2.Normalize(mean=_MEAN, std=_STD))
    return v2.Compose(tfs)


def build_val_transform(imgsz: int) -> v2.Compose:
    """Validation transform: square (stretch) resize + ImageNet normalize, no aug."""
    return v2.Compose(
        [
            v2.Resize(size=(imgsz, imgsz), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_MEAN, std=_STD),
        ]
    )
