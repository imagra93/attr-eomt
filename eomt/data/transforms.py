"""Augmentations for EoMT instance segmentation (torchvision transforms v2).

Transforms operate jointly on the image and the per-instance masks via
``tv_tensors``. Training uses the Mask2Former/EoMT recipe: horizontal flip +
**Large-Scale Jitter** (aspect-preserving resize over a wide scale range, then a
fixed-size crop that pads when the scaled image is smaller than the crop) + color
jitter, then ImageNet normalize. Validation/inference use an aspect-preserving
**letterbox** resize (resize the long side, pad to square) so object shapes are
not distorted — the inverse padding/scale is undone in postprocessing.
"""

from __future__ import annotations

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF  # noqa: N812

from ..preprocess import IMAGENET_MEAN, IMAGENET_STD

_MEAN = IMAGENET_MEAN.tolist()
_STD = IMAGENET_STD.tolist()
# Pad value for the image in *uint8* space ≈ the ImageNet mean, so the padded
# border is ~0 after normalization. Masks always pad with 0 (background).
_PAD_RGB = [int(round(m * 255)) for m in _MEAN]


def build_train_transform(
    imgsz: int,
    *,
    flip_prob: float = 0.5,
    min_scale: float = 0.1,
    max_scale: float = 2.0,
    color_jitter: bool = True,
) -> v2.Compose:
    """Training transform: flip + Large-Scale Jitter (LSJ) + color jitter + normalize.

    ``ScaleJitter`` resizes the image (aspect-ratio preserving) to a random factor
    in ``[min_scale, max_scale]`` of the target size; ``RandomCrop`` then crops to
    a fixed square, padding with the mean color (image) / 0 (masks) when the scaled
    image is smaller than the crop. Output image is a normalized ``float32``
    ``(3, imgsz, imgsz)`` tensor; masks stay ``uint8`` ``(N, imgsz, imgsz)`` in
    ``{0, 1}`` (nearest resampling) — the dataset drops cropped-away instances and
    casts to ``float32`` afterwards.
    """
    tfs: list = [v2.RandomHorizontalFlip(p=flip_prob)]
    tfs.append(
        v2.ScaleJitter(
            target_size=(imgsz, imgsz),
            scale_range=(min_scale, max_scale),
            antialias=True,
        )
    )
    tfs.append(
        v2.RandomCrop(
            size=(imgsz, imgsz),
            pad_if_needed=True,
            fill={tv_tensors.Image: _PAD_RGB, "others": 0},
            padding_mode="constant",
        )
    )
    if color_jitter:
        tfs.append(v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05))
    tfs.append(v2.ToDtype(torch.float32, scale=True))
    tfs.append(v2.Normalize(mean=_MEAN, std=_STD))
    return v2.Compose(tfs)


def letterbox_size(h: int, w: int, imgsz: int) -> tuple[int, int]:
    """Content ``(height, width)`` after an aspect-preserving resize of the long side to ``imgsz``."""
    scale = imgsz / max(h, w)
    return max(1, round(h * scale)), max(1, round(w * scale))


def build_val_transform(imgsz: int, *, letterbox: bool = True):
    """Deterministic eval transform (no augmentation), as a ``(image, masks) -> (image, masks)`` callable.

    With ``letterbox=True`` the image is resized aspect-preserving (long side to
    ``imgsz``) and padded bottom/right to a square — masks resized/padded to match.
    With ``letterbox=False`` it falls back to the legacy square stretch-resize.
    """
    norm = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=_MEAN, std=_STD)])

    if not letterbox:
        resize = v2.Resize(size=(imgsz, imgsz), antialias=True)

        def _stretch(img, masks):
            img, masks = resize(img, masks)
            return norm(img, masks)

        return _stretch

    def _letterbox(img, masks):
        # img: uint8 Image (C, H, W); masks: uint8 Mask (N, H, W) — pad bottom/right.
        h, w = img.shape[-2], img.shape[-1]
        nh, nw = letterbox_size(h, w, imgsz)
        img = TF.resize(img, [nh, nw], antialias=True)
        masks = TF.resize(masks, [nh, nw])  # nearest for masks
        pad = [0, 0, imgsz - nw, imgsz - nh]  # left, top, right, bottom
        img = TF.pad(img, pad, fill=_PAD_RGB)
        masks = TF.pad(masks, pad, fill=0)
        return norm(img, masks)

    return _letterbox
