"""Preprocessing helpers for EoMT.

EoMT / DINOv2 expect a square, ImageNet-normalized RGB tensor. We stretch-resize
to the square input (DETR-style, no letterbox) so the patch grid is exact.
"""

from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int):
    """Stretch-resize an RGB HWC uint8 image to ``(C, input_size, input_size)``.

    Returns ``(chw_float32, ratio)``. ``ratio`` is unused for stretch resize
    (kept as ``1.0`` for signature parity).
    """
    from PIL import Image

    img = Image.fromarray(img_rgb_hwc).resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
    return chw, 1.0
