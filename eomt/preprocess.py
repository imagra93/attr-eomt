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


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int,
    *,
    letterbox: bool = True,
    mean=None,
    std=None,
):
    """Resize an RGB HWC uint8 image to ``(C, input_size, input_size)``, normalized.

    ``mean`` / ``std`` default to :data:`IMAGENET_MEAN` / :data:`IMAGENET_STD`; pass
    a model's stored ``pixel_mean`` / ``pixel_std`` to reproduce its exact training
    normalization. Returns ``(chw_float32, meta)``; ``meta`` is from
    :func:`make_preprocess_meta`. The padded border (letterbox) is filled with 0 in
    normalized space, i.e. the mean colour.
    """
    from PIL import Image

    mean = IMAGENET_MEAN if mean is None else np.asarray(mean, dtype=np.float32)
    std = IMAGENET_STD if std is None else np.asarray(std, dtype=np.float32)

    h, w = img_rgb_hwc.shape[:2]
    if not letterbox:
        img = Image.fromarray(img_rgb_hwc).resize((input_size, input_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
        return chw, make_preprocess_meta(False, (input_size, input_size), input_size)

    scale = input_size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = Image.fromarray(img_rgb_hwc).resize((nw, nh), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std  # (nh, nw, 3)
    canvas = np.zeros((input_size, input_size, 3), dtype=np.float32)  # 0 == mean post-normalize
    canvas[:nh, :nw] = arr
    chw = np.ascontiguousarray(canvas.transpose(2, 0, 1))
    return chw, make_preprocess_meta(True, (nh, nw), input_size)


def preprocess_torch(
    img_rgb_hwc: np.ndarray,
    input_size: int,
    *,
    letterbox: bool = True,
    mean=None,
    std=None,
    device="cpu",
):
    """:func:`preprocess_numpy` run with torch on ``device`` (e.g. the model's GPU).

    The resize is an antialiased bilinear one rounded to uint8, as PIL's, so the result matches
    :func:`preprocess_numpy` within one grey level. Returns ``(chw_float32_tensor, meta)``.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    h, w = img_rgb_hwc.shape[:2]
    if letterbox:
        scale = input_size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    else:
        nh, nw = input_size, input_size
    x = torch.from_numpy(np.ascontiguousarray(img_rgb_hwc)).to(device).permute(2, 0, 1)[None].float()
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False).round_()
    mean = torch.as_tensor(IMAGENET_MEAN if mean is None else mean, dtype=torch.float32, device=device).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD if std is None else std, dtype=torch.float32, device=device).view(3, 1, 1)
    chw = x.new_zeros((3, input_size, input_size))  # 0 == mean post-normalize
    chw[:, :nh, :nw] = (x[0] / 255.0 - mean) / std
    return chw, make_preprocess_meta(letterbox, (nh, nw), input_size)
