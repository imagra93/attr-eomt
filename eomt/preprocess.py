"""Preprocessing helpers for EoMT.

EoMT / DINOv2 expect a square, ImageNet-normalized RGB tensor. Two modes:

* **letterbox** (default): resize the long side to ``input_size`` preserving
  aspect ratio, then pad bottom/right to a square — object shapes are undistorted.
* **stretch**: resize straight to a square (DETR-style, legacy) — faster but
  distorts aspect ratio.

Both return ``(chw_float32, meta)`` where ``meta`` records what postprocessing
needs to map masks back to the original image (see :func:`make_preprocess_meta`).
"""

from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_preprocess_meta(letterbox: bool, content_hw: tuple[int, int], input_size: int) -> dict:
    """Pack the info postprocessing needs to invert preprocessing.

    ``content_hw`` is the ``(height, width)`` the real image occupies inside the
    square canvas (equal to ``(input_size, input_size)`` for stretch).
    """
    return {"letterbox": bool(letterbox), "content_hw": tuple(content_hw), "input_size": int(input_size)}


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int, *, letterbox: bool = True):
    """Resize an RGB HWC uint8 image to ``(C, input_size, input_size)``, normalized.

    Returns ``(chw_float32, meta)``; ``meta`` is from :func:`make_preprocess_meta`.
    The padded border (letterbox) is filled with 0 in normalized space, i.e. the
    ImageNet mean colour.
    """
    from PIL import Image

    h, w = img_rgb_hwc.shape[:2]
    if not letterbox:
        img = Image.fromarray(img_rgb_hwc).resize((input_size, input_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
        return chw, make_preprocess_meta(False, (input_size, input_size), input_size)

    scale = input_size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = Image.fromarray(img_rgb_hwc).resize((nw, nh), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD  # (nh, nw, 3)
    canvas = np.zeros((input_size, input_size, 3), dtype=np.float32)  # 0 == mean post-normalize
    canvas[:nh, :nw] = arr
    chw = np.ascontiguousarray(canvas.transpose(2, 0, 1))
    return chw, make_preprocess_meta(True, (nh, nw), input_size)
