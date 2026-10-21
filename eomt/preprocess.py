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

import functools

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#: Fixed-point precision of PIL's 8-bit resampling (Pillow's Resample.c).
_PIL_PRECISION_BITS = 32 - 8 - 2


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


@functools.lru_cache(maxsize=64)
def _pil_bilinear_taps(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """PIL's fixed-point bilinear coefficients for resizing ``in_size`` px to ``out_size``.

    Mirrors ``precompute_coeffs`` and ``normalize_coeffs_8bpc`` of Pillow's Resample.c. Returns the input index and
    the integer weight of every tap, both ``(out_size, taps)``; padding taps have weight 0.
    """
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = filterscale  # the bilinear filter's support is 1
    center = (np.arange(out_size) + 0.5) * scale
    xmin = np.maximum((center - support + 0.5).astype(np.int64), 0)  # C's (int) truncation
    xmax = np.minimum((center + support + 0.5).astype(np.int64), in_size)
    x = xmin[:, None] + np.arange(int(np.ceil(support)) * 2 + 1)
    w = np.maximum(1.0 - np.abs((x - center[:, None] + 0.5) * (1.0 / filterscale)), 0.0) * (x < xmax[:, None])
    total = np.cumsum(w, axis=1)[:, -1:]  # summed in order, as PIL does
    w = np.divide(w, total, out=np.zeros_like(w), where=total != 0)
    return np.minimum(x, in_size - 1), np.floor(0.5 + w * (1 << _PIL_PRECISION_BITS)).astype(np.int32)


def _pil_resize(x, nh: int, nw: int):
    """Resize a ``(C, H, W)`` int32 image as PIL's ``Image.resize(..., BILINEAR)`` does an 8-bit one, bit for bit:
    a horizontal then a vertical pass, each rounded to uint8."""
    import torch

    half = 1 << (_PIL_PRECISION_BITS - 1)
    if nw != x.shape[2]:
        idx, k = (torch.from_numpy(a).to(x.device) for a in _pil_bilinear_taps(x.shape[2], nw))
        acc = torch.full((x.shape[0], x.shape[1], nw), half, dtype=torch.int32, device=x.device)
        for t in range(k.shape[1]):
            acc += x[:, :, idx[:, t]] * k[:, t]
        x = (acc >> _PIL_PRECISION_BITS).clamp_(0, 255)
    if nh != x.shape[1]:
        idx, k = (torch.from_numpy(a).to(x.device) for a in _pil_bilinear_taps(x.shape[1], nh))
        acc = torch.full((x.shape[0], nh, x.shape[2]), half, dtype=torch.int32, device=x.device)
        for t in range(k.shape[1]):
            acc += x[:, idx[:, t], :] * k[:, t, None]
        x = (acc >> _PIL_PRECISION_BITS).clamp_(0, 255)
    return x


def preprocess_torch(
    img_rgb_hwc: np.ndarray,
    input_size: int,
    *,
    letterbox: bool = True,
    mean=None,
    std=None,
    device="cpu",
):
    """:func:`preprocess_numpy` run with torch on ``device`` (e.g. the model's GPU), with the same result.

    The resize reproduces PIL's fixed-point bilinear one, so the output equals :func:`preprocess_numpy`'s.
    Returns ``(chw_float32_tensor, meta)``.
    """
    import torch

    h, w = img_rgb_hwc.shape[:2]
    if letterbox:
        scale = input_size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    else:
        nh, nw = input_size, input_size
    x = torch.from_numpy(np.ascontiguousarray(img_rgb_hwc)).to(device).permute(2, 0, 1).to(torch.int32)
    x = _pil_resize(x, nh, nw).float()
    mean = torch.as_tensor(IMAGENET_MEAN if mean is None else mean, dtype=torch.float32, device=device).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD if std is None else std, dtype=torch.float32, device=device).view(3, 1, 1)
    levels = torch.full((3, 1, 1), 255.0, device=device)  # not a scalar, which torch would divide by as x * (1 / 255)
    chw = x.new_zeros((3, input_size, input_size))  # 0 == mean post-normalize
    chw[:, :nh, :nw] = (x / levels - mean) / std
    return chw, make_preprocess_meta(letterbox, (nh, nw), input_size)
