"""Augmentations for EoMT training (instance masks or boxes) and the deterministic eval transform.

One pipeline, one config
------------------------
Everything is driven by :class:`AugConfig` (a flat set of probabilities / ranges) and executed by
:class:`TrainAugment`. The defaults are a *strong* general-purpose recipe: the original EoMT/Mask2Former
recipe (horizontal flip + Large-Scale Jitter + random crop + colour jitter) plus small affine/perspective
warps, gamma, grayscale, blur, sensor noise, JPEG recompression, glare and "safe" random erasing. The
multi-image augmentations (mosaic / mixup / copy-paste) and the instance-aware crop are **off by default**.

Pipeline order (per training sample)::

    flip / rot90                       (source resolution)
    blur, JPEG                         (source resolution: they emulate the camera, then get resized with the photo)
    scale jitter (LSJ)                 (antialiased image; masks AREA-averaged -> soft, mass preserving)
    choose crop window                 (random, or instance-aware)
    rotate / shear / perspective + crop  (ONE bilinear resampling of image and soft masks / box corners)
    [mosaic | mixup | copy-paste]      (optional, instance family only)
    colour jitter, gamma, grayscale, noise, glare, safe erasing   (image only)
    ImageNet normalise

Why masks stay *soft*: nearest-neighbour resizing keeps the area of a thin mask but breaks it into dots (a 5 px
crack shrunk 3x becomes a dashed 1-px line). The EoMT loss already reads targets through bilinear ``grid_sample``,
so area-averaged float targets in [0, 1] are the faithful ground truth. ``mask_resize="nearest"`` restores the old
hard masks.

Padding uses the ImageNet mean colour for the image (so it is ~0 after normalisation) and 0 for masks.
"""

from __future__ import annotations

import io
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF  # noqa: N812

from ..preprocess import IMAGENET_MEAN, IMAGENET_STD

_MEAN = IMAGENET_MEAN.tolist()
_STD = IMAGENET_STD.tolist()
# Pad value for the image in *uint8* space ≈ the ImageNet mean, so the padded
# border is ~0 after normalization. Masks always pad with 0 (background).
_PAD_RGB = [int(round(m * 255)) for m in _MEAN]

_RANGE_FIELDS = ("gamma_range", "blur_sigma", "noise_std", "jpeg_quality", "glare_strength", "erasing_scale")
_PROB_FIELDS = (
    "flip_prob", "vflip_prob", "rot90_prob", "rotate_prob", "shear_prob", "perspective_prob", "color_jitter_prob",
    "gamma_prob", "grayscale_prob", "blur_prob", "noise_prob", "jpeg_prob", "glare_prob", "erasing_prob",
    "mosaic_prob", "mixup_prob", "copy_paste_prob", "instance_crop_prob", "instance_crop_empty_prob",
)


# ---------------------------------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class AugConfig:
    """All training-augmentation knobs, with the defaults used by ``train()``.

    Every ``*_prob`` is the per-sample probability of applying that op (``0`` disables it). Ranges are
    ``(low, high)`` and sampled uniformly. Build one from a dict / YAML mapping with :meth:`from_dict`
    (unknown keys raise), start from the old recipe with ``AugConfig.from_dict({"preset": "legacy"})``.

    **Geometry**

    * ``flip_prob`` / ``vflip_prob`` / ``rot90_prob`` — horizontal flip, vertical flip, random 90° turns.
      Vertical flips and 90° turns are *off*: sky/ground orientation usually carries information, and any
      flip/rotation silently corrupts attributes that encode orientation.
    * ``min_scale`` / ``max_scale`` — Large-Scale Jitter: the image is resized aspect-preservingly so its long side
      is ``imgsz × s`` with ``s ~ U(min_scale, max_scale)``. The effective scale relative to the original photo is
      ``r = (imgsz / long_side) × s``. For datasets of thin or tiny objects keep ``r`` within ~[0.4, 1.0]
      (e.g. ``min_scale = 0.4·long_side/imgsz``, ``max_scale = 1.0·long_side/imgsz``): below that a hairline has no
      image evidence left, and above 1.0 there is no new information.
    * ``rotate_*`` (± degrees), ``shear_*`` (± degrees, x and y), ``perspective_*`` (corner displacement as a
      fraction of half the crop size) — warps applied together with the crop in one bilinear resampling.

    **Optics / sensor (image only)**

    * ``color_jitter_prob`` + ``brightness``/``contrast``/``saturation``/``hue`` — torchvision ``ColorJitter``.
    * ``gamma_*`` — random gamma. ``grayscale_prob`` — drop colour.
    * ``blur_*`` — Gaussian blur with ``sigma`` in **source pixels** (applied before resizing, so it shrinks with the
      photo exactly like real defocus/motion would).
    * ``noise_*`` — additive Gaussian noise, ``std`` in [0, 1] intensity units.
    * ``jpeg_*`` — JPEG recompression at the source resolution, ``quality`` in [1, 100].
    * ``glare_*`` — soft additive highlights / streaks (reflections on glass, flash).
    * ``erasing_*`` — random rectangles filled with a flat colour; with ``erasing_avoid_instances`` they never touch
      an instance (so thin objects are never erased while keeping their label).

    **Multi-image (instance family only; off by default)**

    * ``mosaic_prob`` — 2×2 mosaic of four samples (each cell is an independently augmented half-size crop).
    * ``mixup_prob`` / ``mixup_alpha`` — blend two samples, ``λ ~ Beta(α, α)``; instances of both are kept.
    * ``copy_paste_prob`` / ``copy_paste_max`` / ``copy_paste_context_px`` — paste up to ``copy_paste_max`` instances
      (pixels inside their mask, feathered by ``copy_paste_context_px``) from another sample; the pasted instances
      occlude the existing masks.

    **Crop**

    * ``instance_crop_prob`` — probability that the crop window is *placed* around a focus instance (kept fully inside
      when it fits) instead of at random. ``instance_crop_empty_prob`` — probability of searching an instance-free
      window instead (a hard-negative crop; only possible when the crop is much smaller than the image).
      ``instance_crop_rarity_power`` weights the focus choice by ``1/freq**power`` of the instance's class or of the
      attribute named by ``instance_crop_rarity_attr`` (e.g. ``"material"``); ``instance_crop_visibility_px`` raises
      the scale so the focus instance's band is at least this wide at the model input (``0`` = off).

    **Masks**

    * ``mask_resize`` — ``"area"`` (soft, mass-preserving, default) or ``"nearest"`` (legacy hard masks);
      ``min_mask_mass`` — an instance whose remaining (soft) mask has fewer pixels than this is dropped.
    """

    # geometry
    flip_prob: float = 0.5
    vflip_prob: float = 0.0
    rot90_prob: float = 0.0
    min_scale: float = 0.1
    max_scale: float = 2.0
    rotate_prob: float = 0.3
    rotate_deg: float = 10.0
    shear_prob: float = 0.2
    shear_deg: float = 5.0
    perspective_prob: float = 0.15
    perspective_scale: float = 0.08
    # optics / sensor
    color_jitter_prob: float = 1.0
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.05
    gamma_prob: float = 0.3
    gamma_range: tuple[float, float] = (0.7, 1.5)
    grayscale_prob: float = 0.05
    blur_prob: float = 0.2
    blur_sigma: tuple[float, float] = (0.3, 1.5)
    noise_prob: float = 0.2
    noise_std: tuple[float, float] = (0.005, 0.03)
    jpeg_prob: float = 0.3
    jpeg_quality: tuple[float, float] = (35, 95)
    glare_prob: float = 0.15
    glare_strength: tuple[float, float] = (0.2, 0.5)
    erasing_prob: float = 0.2
    erasing_scale: tuple[float, float] = (0.02, 0.12)
    erasing_avoid_instances: bool = True
    # multi-image
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    mixup_alpha: float = 8.0
    copy_paste_prob: float = 0.0
    copy_paste_max: int = 3
    copy_paste_context_px: float = 2.0
    # crop
    instance_crop_prob: float = 0.0
    instance_crop_empty_prob: float = 0.0
    instance_crop_rarity_power: float = 0.5
    instance_crop_rarity_attr: str | None = None
    instance_crop_visibility_px: float = 0.0
    # masks
    mask_resize: str = "area"
    min_mask_mass: float = 1.0

    def __post_init__(self) -> None:
        for name in _RANGE_FIELDS:
            v = getattr(self, name)
            if not isinstance(v, (tuple, list)) or len(v) != 2:
                raise ValueError(f"AugConfig.{name} must be a (low, high) pair, got {v!r}")
            lo, hi = float(v[0]), float(v[1])
            if lo > hi:
                raise ValueError(f"AugConfig.{name}: low {lo} > high {hi}")
            object.__setattr__(self, name, (lo, hi))
        for name in _PROB_FIELDS:
            p = float(getattr(self, name))
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"AugConfig.{name} must be a probability in [0, 1], got {p}")
        if not 0 < self.min_scale <= self.max_scale:
            raise ValueError(f"need 0 < min_scale <= max_scale, got {self.min_scale}, {self.max_scale}")
        if self.mask_resize not in ("area", "nearest"):
            raise ValueError(f"AugConfig.mask_resize must be 'area' or 'nearest', got {self.mask_resize!r}")
        if self.gamma_range[0] <= 0:
            raise ValueError("AugConfig.gamma_range must be positive")
        if not (1 <= self.jpeg_quality[0] and self.jpeg_quality[1] <= 100):
            raise ValueError("AugConfig.jpeg_quality must lie in [1, 100]")
        if self.rotate_deg < 0 or self.shear_deg < 0 or not 0 <= self.perspective_scale < 1:
            raise ValueError("rotate_deg / shear_deg must be >= 0 and perspective_scale in [0, 1)")
        if self.copy_paste_max < 1 or self.mixup_alpha <= 0 or self.copy_paste_context_px < 0:
            raise ValueError("copy_paste_max >= 1, mixup_alpha > 0, copy_paste_context_px >= 0 required")

    # -- construction ---------------------------------------------------------------------------------------------

    @classmethod
    def preset(cls, name: str) -> AugConfig:
        """``"default"`` (the strong recipe) or ``"legacy"`` (the original flip + LSJ + crop + colour jitter, hard masks)."""
        if name == "default":
            return cls()
        if name == "legacy":
            return cls(
                rotate_prob=0.0, shear_prob=0.0, perspective_prob=0.0, gamma_prob=0.0, grayscale_prob=0.0,
                blur_prob=0.0, noise_prob=0.0, jpeg_prob=0.0, glare_prob=0.0, erasing_prob=0.0, mask_resize="nearest",
            )
        raise ValueError(f"unknown augmentation preset {name!r}; choose 'default' or 'legacy'")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AugConfig:
        """Build from a mapping (YAML). A ``preset`` key selects the starting point; unknown keys raise."""
        d = dict(d)
        base = cls.preset(d.pop("preset")) if "preset" in d else cls()
        valid = {f.name for f in fields(cls)}
        unknown = set(d) - valid
        if unknown:
            raise ValueError(f"unknown augmentation keys {sorted(unknown)}; valid keys: {sorted(valid)} (+ 'preset')")
        for k in _RANGE_FIELDS:
            if k in d:
                d[k] = tuple(d[k])
        return replace(base, **d)

    @classmethod
    def from_any(cls, obj: AugConfig | Mapping[str, Any] | None) -> AugConfig:
        if obj is None:
            return cls()
        if isinstance(obj, AugConfig):
            return replace(obj)
        return cls.from_dict(obj)

    @classmethod
    def resolve(cls, aug: AugConfig | Mapping[str, Any] | None = None, **explicit: Any) -> AugConfig:
        """``aug`` (config / mapping / None) overridden by every *non-None* keyword (e.g. ``flip_prob=0``)."""
        cfg = cls.from_any(aug)
        over = {k: v for k, v in explicit.items() if v is not None}
        return replace(cfg, **over) if over else cfg

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (ranges as lists) for ``args.yaml``."""
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    def active(self) -> list[str]:
        """Names of the ops that can fire (probability > 0), for logging."""
        names = []
        for name in _PROB_FIELDS:
            if getattr(self, name) > 0:
                names.append(name.removesuffix("_prob"))
        return names


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------


def _rand() -> float:
    return float(torch.rand(1))


def _coin(p: float) -> bool:
    return p > 0.0 and _rand() < p


def _uniform(rng: tuple[float, float]) -> float:
    lo, hi = rng
    return lo + (hi - lo) * _rand()


def _randint(lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] (inclusive)."""
    return int(lo + int(torch.randint(0, hi - lo + 1, (1,))))


def _as_tensor(x: Any) -> torch.Tensor:
    return x.as_subclass(torch.Tensor) if isinstance(x, torch.Tensor) and type(x) is not torch.Tensor else x


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping the 4 ``src`` points onto the 4 ``dst`` points (both (4, 2))."""
    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i], b[2 * i + 1] = u, v
    h = np.linalg.solve(a, b)
    return np.append(h, 1.0).reshape(3, 3)


def _apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points."""
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ h.T
    return p[:, :2] / p[:, 2:3]


def _affine(theta_deg: float, shear_x_deg: float, shear_y_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    rot = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]], dtype=np.float64)
    shear = np.array([[1, math.tan(math.radians(shear_x_deg)), 0], [math.tan(math.radians(shear_y_deg)), 1, 0], [0, 0, 1]],
                     dtype=np.float64)
    return rot @ shear


def _clipped_aabb(quad: np.ndarray, w: float, h: float) -> tuple[float, float, float, float]:
    """Axis-aligned box of the convex polygon ``quad`` (N, 2) clipped to ``[0, w] x [0, h]`` (Sutherland-Hodgman).

    Clipping *before* taking the box matters for rotated objects that the window cuts: the box of the clipped shape is
    tighter than the clipped box of the whole shape. Returns zeros when nothing of the polygon is inside.
    """
    poly = [(float(x), float(y)) for x, y in quad]
    for axis, bound, keep_ge in ((0, 0.0, True), (0, w, False), (1, 0.0, True), (1, h, False)):
        if not poly:
            break
        out = []
        for i in range(len(poly)):
            a, b = poly[i - 1], poly[i]
            ia = a[axis] >= bound if keep_ge else a[axis] <= bound
            ib = b[axis] >= bound if keep_ge else b[axis] <= bound
            if ia != ib:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            if ib:
                out.append(b)
        poly = out
    if not poly:
        return 0.0, 0.0, 0.0, 0.0
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    # clamp: the intersection arithmetic can leave a ~1e-16 sliver outside the window
    return max(min(xs), 0.0), max(min(ys), 0.0), min(max(xs), w), min(max(ys), h)


def _translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def _origin_axis(lo: int, hi: int, length: int, out: int) -> int:
    """Window origin along one axis: keep ``[lo, hi)`` inside the window if it fits, else a window inside ``[lo, hi)``."""
    if length <= out:  # the image is smaller than the window: place it at random inside the canvas
        return _randint(length - out, 0) if length < out else 0
    if hi - lo <= out:
        a, b = hi - out, lo
    else:
        a, b = lo, hi - out
    a, b = max(a, 0), min(b, length - out)
    if b < a:
        a = b = min(max(a, 0), length - out)
    return _randint(a, b)


def _random_origin(length: int, out: int) -> int:
    if length >= out:
        return _randint(0, length - out)
    return _randint(length - out, 0)  # image smaller than the window: random placement inside the canvas


# ---------------------------------------------------------------------------------------------------------------------
# Image-only ops (float32 CHW in [0, 1] unless stated)
# ---------------------------------------------------------------------------------------------------------------------


def _blur_u8(img: torch.Tensor, sigma: float) -> torch.Tensor:
    k = 2 * int(math.ceil(3 * sigma)) + 1
    return TF.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])


def _jpeg_u8(img: torch.Tensor, quality: int) -> torch.Tensor:
    buf = io.BytesIO()
    Image.fromarray(img.permute(1, 2, 0).contiguous().numpy()).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return torch.from_numpy(np.asarray(Image.open(buf).convert("RGB")).copy()).permute(2, 0, 1).contiguous()


def _glare(img: torch.Tensor, strength: tuple[float, float]) -> torch.Tensor:
    """1-2 soft highlights (some elongated into streaks) blended with the 'screen' operator, so they only brighten."""
    _, h, w = img.shape
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    out = img
    for _ in range(_randint(1, 2)):
        cx, cy = _rand() * w, _rand() * h
        sx = (0.08 + 0.27 * _rand()) * max(h, w)
        sy = sx * (1.0 if _rand() < 0.7 else 0.12 + 0.2 * _rand())  # 30% streaks
        t = _rand() * math.pi
        dx, dy = xx - cx, yy - cy
        u = dx * math.cos(t) + dy * math.sin(t)
        v = -dx * math.sin(t) + dy * math.cos(t)
        g = torch.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
        tint = torch.tensor([1.0, 0.98 + 0.02 * _rand(), 0.94 + 0.06 * _rand()]).view(3, 1, 1)
        out = out + _uniform(strength) * g[None] * tint * (1.0 - out)
    return out.clamp_(0.0, 1.0)


def _erase(img: torch.Tensor, occupied: torch.Tensor | None, scale: tuple[float, float]) -> torch.Tensor:
    """Fill one random rectangle with a flat colour; if ``occupied`` (bool HxW) is given, never overlap it."""
    img = img.contiguous()  # rgb_to_grayscale returns an expanded (aliased) view; in-place writes need real memory
    _, h, w = img.shape
    for _ in range(12):
        area = _uniform(scale) * h * w
        ar = math.exp(_uniform((math.log(0.3), math.log(3.3))))
        eh, ew = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
        if not (1 <= eh < h and 1 <= ew < w):
            continue
        y0, x0 = _randint(0, h - eh), _randint(0, w - ew)
        if occupied is not None and bool(occupied[y0:y0 + eh, x0:x0 + ew].any()):
            continue
        img[:, y0:y0 + eh, x0:x0 + ew] = torch.rand(3, 1, 1)
        break
    return img


# ---------------------------------------------------------------------------------------------------------------------
# The training pipeline
# ---------------------------------------------------------------------------------------------------------------------


class TrainAugment:
    """Training transform for the instance (masks) and detect (boxes) families, driven by an :class:`AugConfig`.

    Two call forms:

    * **legacy** ``tf(image, target) -> (image, target)`` — ``image`` uint8 ``(3, H, W)`` (``tv_tensors.Image`` ok),
      ``target`` either instance masks ``(N, H, W)`` (``tv_tensors.Mask`` ok) or ``tv_tensors.BoundingBoxes`` (XYXY).
      Returns the normalised float image ``(3, S, S)`` and float masks ``(N, S, S)`` (soft, in [0, 1]) or XYXY boxes
      ``(N, 4)`` in output pixels. Instances are never dropped here, so labels stay aligned by position.
      Multi-image augmentations are skipped (they would change the instance list).
    * **full** ``tf(image, masks, labels, attrs, weights=…, sampler=…) -> (image, masks, labels, attrs)`` — also
      carries per-instance ``labels`` (LongTensor) and ``attrs`` (``{name: LongTensor}``) through mosaic / mixup /
      copy-paste, which add instances. ``sampler()`` must return another raw sample as a dict with keys
      ``image``, ``target``, ``labels``, ``attrs`` (and optionally ``weights``).

    ``weights`` (per instance, any positive scale) drive the instance-aware crop's choice of focus instance.
    """

    #: the dataset checks this to decide whether to use the full call form
    is_train_augment = True

    def __init__(self, imgsz: int, cfg: AugConfig | Mapping[str, Any] | None = None):
        self.imgsz = int(imgsz)
        self.cfg = AugConfig.from_any(cfg)
        c = self.cfg
        self._cj = (
            v2.ColorJitter(brightness=c.brightness, contrast=c.contrast, saturation=c.saturation, hue=c.hue)
            if c.color_jitter_prob > 0 and (c.brightness or c.contrast or c.saturation or c.hue)
            else None
        )
        self._mean = torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(_STD, dtype=torch.float32).view(3, 1, 1)
        self.last_info: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"TrainAugment(imgsz={self.imgsz}, active={self.cfg.active()}, mask_resize={self.cfg.mask_resize!r})"

    # ------------------------------------------------------------------------------------------------ entry point

    def __call__(self, image, target, labels=None, attrs=None, *, weights=None, sampler=None):
        c = self.cfg
        is_boxes = isinstance(target, tv_tensors.BoundingBoxes)
        if is_boxes and target.format != tv_tensors.BoundingBoxFormat.XYXY:
            raise ValueError(f"TrainAugment expects XYXY boxes, got {target.format}; convert first.")
        img = _as_tensor(image)
        tgt = _as_tensor(target)
        if not is_boxes:
            tgt = tgt.to(torch.float32) if tgt.dtype != torch.float32 else tgt
        full = labels is not None
        labels_o, attrs_o = labels, attrs
        w = None if weights is None else np.asarray(weights, dtype=np.float64)

        multi = None
        if full and not is_boxes and sampler is not None:
            for name, p in (("mosaic", c.mosaic_prob), ("copy_paste", c.copy_paste_prob), ("mixup", c.mixup_prob)):
                if _coin(p):
                    multi = name
                    break
        self.last_info = {"multi": multi}

        S = self.imgsz
        if multi == "mosaic":
            img_f, tgt, labels_o, attrs_o = self._mosaic(img, tgt, labels, attrs, w, sampler)
        else:
            img_f, tgt, info = self._geometry(img, tgt, (S, S), w, is_boxes)
            self.last_info.update(info)
            if multi == "copy_paste":
                img_f, tgt, labels_o, attrs_o = self._copy_paste(img_f, tgt, labels, attrs, sampler)
            elif multi == "mixup":
                img_f, tgt, labels_o, attrs_o = self._mixup(img_f, tgt, labels, attrs, sampler)

        occupied = None
        if c.erasing_prob > 0 and c.erasing_avoid_instances:
            occupied = self._occupied(tgt, is_boxes, img_f.shape[-2:])
        img_out = self._finalize(img_f, occupied)
        if not full:
            return img_out, tgt
        return img_out, tgt, labels_o, attrs_o

    # ------------------------------------------------------------------------------------------------ geometry core

    def _orient(self, img, tgt, is_boxes):
        """Flips and 90° turns at source resolution. Boxes follow the same maps."""
        c = self.cfg
        H, W = img.shape[-2:]
        if _coin(c.flip_prob):
            img = img.flip(-1)
            if is_boxes:
                tgt = torch.stack([W - tgt[:, 2], tgt[:, 1], W - tgt[:, 0], tgt[:, 3]], 1)
            else:
                tgt = tgt.flip(-1)
        if _coin(c.vflip_prob):
            img = img.flip(-2)
            if is_boxes:
                tgt = torch.stack([tgt[:, 0], H - tgt[:, 3], tgt[:, 2], H - tgt[:, 1]], 1)
            else:
                tgt = tgt.flip(-2)
        if _coin(c.rot90_prob):
            k = _randint(1, 3)
            img = torch.rot90(img, k, dims=(-2, -1))
            if is_boxes:
                for _ in range(k):  # one counter-clockwise quarter turn: (x, y) -> (y, W - x), W <- H, H <- W
                    x0, y0, x1, y1 = tgt.unbind(1)
                    tgt = torch.stack([y0, W - x1, y1, W - x0], 1)
                    H, W = W, H
            else:
                tgt = torch.rot90(tgt, k, dims=(-2, -1))
        return img, tgt

    def _resize_target(self, tgt, is_boxes, hw_src, hw_new):
        (H, W), (Hs, Ws) = hw_src, hw_new
        if is_boxes:
            return tgt * torch.tensor([Ws / W, Hs / H, Ws / W, Hs / H], dtype=tgt.dtype)
        if tgt.shape[0] == 0 or (Hs, Ws) == (H, W):
            return tgt
        mode = self.cfg.mask_resize
        if mode == "nearest":
            return F.interpolate(tgt[None], size=(Hs, Ws), mode="nearest")[0]
        if Hs * Ws < H * W:  # shrinking: area averaging (mass preserving)
            return F.interpolate(tgt[None], size=(Hs, Ws), mode="area")[0].clamp_(0, 1)
        return F.interpolate(tgt[None], size=(Hs, Ws), mode="bilinear", align_corners=False)[0].clamp_(0, 1)

    @staticmethod
    def _extents(tgt, is_boxes, i) -> tuple[int, int, int, int] | None:
        """(x0, y0, x1, y1) of instance ``i`` in the current (resized) frame, or None if it has vanished."""
        if is_boxes:
            x0, y0, x1, y1 = (float(v) for v in tgt[i])
            return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))
        ys, xs = torch.nonzero(tgt[i] > 0.05, as_tuple=True)
        if ys.numel() == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _band_width(mask: torch.Tensor) -> float:
        """~2*area/perimeter of a binary-ish mask (px): the width of a thin band; used by the visibility floor."""
        m = (mask > 0.5).float()[None, None]
        area = float(m.sum())
        if area < 4:
            return 0.0
        edge = float(F.max_pool2d(m, 3, 1, 1).sum() - (-F.max_pool2d(-m, 3, 1, 1)).sum())  # ~ 2 x perimeter
        return 4.0 * area / max(edge, 1.0)

    def _geometry(self, img, tgt, out_hw, weights, is_boxes, *, with_focus=True):
        """Flip/turn, source optics, LSJ, window choice, rotate/shear/perspective + crop. Returns (img [0,1], tgt, info)."""
        c = self.cfg
        S = self.imgsz
        Ho, Wo = out_hw
        img, tgt = self._orient(img, tgt, is_boxes)
        H, W = img.shape[-2:]
        n = tgt.shape[0]

        # source-resolution camera effects (they are resized together with the photo)
        if _coin(c.blur_prob):
            sigma = _uniform(c.blur_sigma)
            if sigma >= 0.25:
                img = _blur_u8(img, sigma)
        if _coin(c.jpeg_prob):
            img = _jpeg_u8(img, int(round(_uniform(c.jpeg_quality))))

        # focus instance (instance-aware crop) and the scale it implies
        focus = None
        if with_focus and n > 0 and (c.instance_crop_prob > 0 or c.instance_crop_empty_prob > 0):
            u = _rand()
            if u < c.instance_crop_prob:
                focus = self._pick_focus(n, weights)
        s = _uniform((c.min_scale, c.max_scale))
        fit = min(S / H, S / W)
        if focus is not None and c.instance_crop_visibility_px > 0 and not is_boxes:
            w0 = self._band_width(tgt[focus])
            if w0 > 0:
                s = min(max(s, c.instance_crop_visibility_px / (w0 * fit)), c.max_scale * 1.5)
        r = fit * s
        Hs, Ws = max(1, int(H * r)), max(1, int(W * r))
        if (Hs, Ws) != (H, W):
            img = TF.resize(img, [Hs, Ws], antialias=True)
        tgt = self._resize_target(tgt, is_boxes, (H, W), (Hs, Ws))

        # window origin
        mode = "random"
        if focus is not None:
            ext = self._extents(tgt, is_boxes, focus)
            if ext is None:
                ext = (Ws // 2, Hs // 2, Ws // 2 + 1, Hs // 2 + 1)
            x0 = _origin_axis(ext[0], ext[2], Ws, Wo)
            y0 = _origin_axis(ext[1], ext[3], Hs, Ho)
            mode = "focus"
        elif n > 0 and c.instance_crop_empty_prob > 0 and _coin(c.instance_crop_empty_prob):
            found = self._empty_window(tgt, is_boxes, (Hs, Ws), (Ho, Wo))
            if found is not None:
                y0, x0 = found
                mode = "empty"
            else:
                y0, x0 = _random_origin(Hs, Ho), _random_origin(Ws, Wo)
        else:
            y0, x0 = _random_origin(Hs, Ho), _random_origin(Ws, Wo)

        # warp (rotate / shear / perspective); the focus instance must stay in view when it fits
        fwd = None
        for _ in range(6 if focus is not None else 1):
            cand = self._sample_warp((Ho, Wo), (x0 + Wo / 2.0, y0 + Ho / 2.0))
            if cand is None:
                break
            if focus is None or self._focus_visible(cand, tgt, is_boxes, focus, (Ho, Wo)):
                fwd = cand
                break
        info = {"scale": s, "eff_scale": r, "mode": mode, "focus": focus, "warp": fwd is not None, "origin": (y0, x0)}
        out_img, out_tgt = self._crop(img, tgt, is_boxes, (Hs, Ws), (Ho, Wo), (y0, x0), fwd)
        return out_img, out_tgt, info

    def _pick_focus(self, n, weights) -> int:
        if weights is None or len(weights) != n or float(np.sum(weights)) <= 0:
            return _randint(0, n - 1)
        p = np.asarray(weights, dtype=np.float64)
        cdf = np.cumsum(p / p.sum())
        return int(min(np.searchsorted(cdf, _rand()), n - 1))

    def _empty_window(self, tgt, is_boxes, hw, out_hw):
        Hs, Ws = hw
        Ho, Wo = out_hw
        occ = self._occupied(tgt, is_boxes, (Hs, Ws))
        for _ in range(30):
            y0, x0 = _random_origin(Hs, Ho), _random_origin(Ws, Wo)
            ya, xa = max(y0, 0), max(x0, 0)
            if not bool(occ[ya:max(y0 + Ho, 0), xa:max(x0 + Wo, 0)].any()):
                return y0, x0
        return None

    @staticmethod
    def _occupied(tgt, is_boxes, hw) -> torch.Tensor:
        """Bool (H, W) union of the instances (mask > 0.05, or filled boxes)."""
        H, W = int(hw[0]), int(hw[1])
        if tgt.shape[0] == 0:
            return torch.zeros(H, W, dtype=torch.bool)
        if not is_boxes:
            return (tgt > 0.05).any(0)
        occ = torch.zeros(H, W, dtype=torch.bool)
        for x0, y0, x1, y1 in tgt.tolist():
            occ[max(int(y0), 0):max(int(math.ceil(y1)), 0), max(int(x0), 0):max(int(math.ceil(x1)), 0)] = True
        return occ

    def _sample_warp(self, out_hw, center_src):
        """Forward homography (resized-image px -> output px) for this draw, or None when no warp fires."""
        c = self.cfg
        Ho, Wo = out_hw
        rot = _uniform((-c.rotate_deg, c.rotate_deg)) if _coin(c.rotate_prob) else 0.0
        shx = _uniform((-c.shear_deg, c.shear_deg)) if _coin(c.shear_prob) else 0.0
        shy = _uniform((-c.shear_deg, c.shear_deg)) if shx != 0.0 else 0.0
        persp = _coin(c.perspective_prob) and c.perspective_scale > 0
        if rot == 0.0 and shx == 0.0 and shy == 0.0 and not persp:
            return None
        cx, cy = center_src
        m = _translate(Wo / 2.0, Ho / 2.0) @ _affine(rot, shx, shy) @ _translate(-cx, -cy)
        if persp:
            d = c.perspective_scale
            dx, dy = d * Wo / 2.0, d * Ho / 2.0
            start = np.array([[0, 0], [Wo, 0], [Wo, Ho], [0, Ho]], dtype=np.float64)
            end = start + np.array(
                [[_rand() * dx, _rand() * dy], [-_rand() * dx, _rand() * dy], [-_rand() * dx, -_rand() * dy], [_rand() * dx, -_rand() * dy]]
            )
            m = _homography(start, end) @ m
        return m

    def _focus_visible(self, fwd, tgt, is_boxes, focus, out_hw) -> bool:
        ext = self._extents(tgt, is_boxes, focus)
        if ext is None:
            return True
        Ho, Wo = out_hw
        if ext[2] - ext[0] > Wo or ext[3] - ext[1] > Ho:  # cannot fit anyway: do not insist
            return True
        pts = _apply_h(fwd, np.array([[ext[0], ext[1]], [ext[2], ext[1]], [ext[2], ext[3]], [ext[0], ext[3]]], dtype=np.float64))
        return bool((pts[:, 0] >= 0).all() and (pts[:, 1] >= 0).all() and (pts[:, 0] <= Wo).all() and (pts[:, 1] <= Ho).all())

    def _crop(self, img, tgt, is_boxes, hw, out_hw, origin, fwd):
        """Crop (and warp) image + target to ``out_hw``. Returns float image in [0, 1] and the target."""
        Hs, Ws = hw
        Ho, Wo = out_hw
        y0, x0 = origin
        mean = self._mean
        imgf = img.to(torch.float32) / 255.0
        if fwd is None:  # pure crop / pad: exact, no resampling
            out = mean.expand(3, Ho, Wo).clone()
            ya, xa = max(y0, 0), max(x0, 0)
            yb, xb = min(y0 + Ho, Hs), min(x0 + Wo, Ws)
            if yb > ya and xb > xa:
                out[:, ya - y0:yb - y0, xa - x0:xb - x0] = imgf[:, ya:yb, xa:xb]
            if is_boxes:
                b = tgt - torch.tensor([x0, y0, x0, y0], dtype=tgt.dtype)
                b[:, [0, 2]] = b[:, [0, 2]].clamp(0, Wo)
                b[:, [1, 3]] = b[:, [1, 3]].clamp(0, Ho)
                return out, b
            m = torch.zeros(tgt.shape[0], Ho, Wo, dtype=torch.float32)
            if yb > ya and xb > xa and tgt.shape[0]:
                m[:, ya - y0:yb - y0, xa - x0:xb - x0] = tgt[:, ya:yb, xa:xb]
            return out, m

        inv = np.linalg.inv(fwd)
        jj, ii = np.meshgrid(np.arange(Wo, dtype=np.float64) + 0.5, np.arange(Ho, dtype=np.float64) + 0.5)
        pts = np.stack([jj.ravel(), ii.ravel(), np.ones(jj.size)], 0)
        src = inv @ pts
        gx = (2.0 * (src[0] / src[2]) / Ws - 1.0).reshape(Ho, Wo)
        gy = (2.0 * (src[1] / src[2]) / Hs - 1.0).reshape(Ho, Wo)
        grid = torch.from_numpy(np.stack([gx, gy], -1)).to(torch.float32)[None]
        out = F.grid_sample((imgf - mean)[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0] + mean
        out = out.clamp_(0.0, 1.0)
        if is_boxes:
            corners = np.stack(
                [np.array([[x0_, y0_], [x1_, y0_], [x1_, y1_], [x0_, y1_]], dtype=np.float64) for x0_, y0_, x1_, y1_ in tgt.tolist()]
            ) if tgt.shape[0] else np.zeros((0, 4, 2))
            res = np.zeros((len(corners), 4), dtype=np.float32)
            for k, quad in enumerate(corners):
                q = _apply_h(fwd, quad)
                res[k] = _clipped_aabb(q, Wo, Ho)
            return out, torch.from_numpy(res)
        if tgt.shape[0] == 0:
            return out, torch.zeros(0, Ho, Wo)
        m = F.grid_sample(tgt[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        return out, m.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------------------------------------ image-only finish

    def _finalize(self, img: torch.Tensor, occupied: torch.Tensor | None) -> torch.Tensor:
        """Colour / sensor ops on the float crop, then ImageNet normalisation."""
        c = self.cfg
        if self._cj is not None and _coin(c.color_jitter_prob):
            img = self._cj(img).clamp_(0.0, 1.0)
        if _coin(c.gamma_prob):
            img = img.clamp(0.0, 1.0) ** _uniform(c.gamma_range)
        if _coin(c.grayscale_prob):
            img = TF.rgb_to_grayscale(img, num_output_channels=3).contiguous()
        if _coin(c.noise_prob):
            img = (img + torch.randn_like(img) * _uniform(c.noise_std)).clamp_(0.0, 1.0)
        if _coin(c.glare_prob):
            img = _glare(img, c.glare_strength)
        if _coin(c.erasing_prob):
            img = _erase(img, occupied, c.erasing_scale)
        return (img - self._mean) / self._std

    # ------------------------------------------------------------------------------------------------ multi-image ops

    def _raw(self, sampler):
        s = sampler()
        w = s.get("weights")
        return _as_tensor(s["image"]), _as_tensor(s["target"]).to(torch.float32), s["labels"], s["attrs"], (None if w is None else np.asarray(w, dtype=np.float64))

    def _mosaic(self, img, tgt, labels, attrs, weights, sampler):
        """2x2 mosaic: the current sample + 3 others, each an independently augmented half-size crop."""
        S = self.imgsz
        hs, ws = (S // 2, S - S // 2), (S // 2, S - S // 2)
        samples = [(img, tgt, labels, attrs, weights)] + [self._raw(sampler) for _ in range(3)]
        order = torch.randperm(4).tolist()
        canvas = self._mean.expand(3, S, S).clone()
        masks, labs, atts = [], [], []
        for k, cell in enumerate(order):
            ri, ci = divmod(cell, 2)
            im_k, t_k, l_k, a_k, w_k = samples[k]
            Ho, Wo = hs[ri], ws[ci]
            out_i, out_m, _ = self._geometry(im_k, t_k, (Ho, Wo), w_k, False)
            y_off, x_off = ri * hs[0], ci * ws[0]
            canvas[:, y_off:y_off + Ho, x_off:x_off + Wo] = out_i
            full = torch.zeros(out_m.shape[0], S, S)
            full[:, y_off:y_off + Ho, x_off:x_off + Wo] = out_m
            masks.append(full)
            labs.append(l_k)
            atts.append(a_k)
        return canvas, torch.cat(masks, 0), torch.cat(labs, 0), {k: torch.cat([a[k] for a in atts], 0) for k in atts[0]}

    def _mixup(self, img_f, tgt, labels, attrs, sampler):
        """Blend with a second augmented sample (lambda ~ Beta(a, a)); the instances of both are kept."""
        im2, t2, l2, a2, w2 = self._raw(sampler)
        o2, m2, _ = self._geometry(im2, t2, (self.imgsz, self.imgsz), w2, False)
        lam = float(torch.distributions.Beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha).sample())
        return lam * img_f + (1.0 - lam) * o2, torch.cat([tgt, m2], 0), torch.cat([labels, l2], 0), {k: torch.cat([attrs[k], a2[k]], 0) for k in attrs}

    def _copy_paste(self, img_f, tgt, labels, attrs, sampler):
        """Paste up to ``copy_paste_max`` instances (their pixels, feathered) from another sample; they occlude existing masks."""
        c = self.cfg
        im2, t2, l2, a2, w2 = self._raw(sampler)
        o2, m2, _ = self._geometry(im2, t2, (self.imgsz, self.imgsz), w2, False)
        mass = m2.flatten(1).sum(1)
        cand = [i for i in range(m2.shape[0]) if float(mass[i]) >= max(c.min_mask_mass, 1.0)]
        if not cand:
            return img_f, tgt, labels, attrs
        perm = torch.randperm(len(cand)).tolist()
        pick = [cand[i] for i in perm[: _randint(1, min(c.copy_paste_max, len(cand)))]]
        sel = m2[pick]
        union = sel.amax(0, keepdim=True)  # (1, H, W)
        alpha = union
        if c.copy_paste_context_px > 0:  # feather: grow by the context, then blur so the seam is soft
            k = 2 * int(math.ceil(c.copy_paste_context_px)) + 1
            alpha = F.max_pool2d(union[None], k, 1, k // 2)[0]
            alpha = TF.gaussian_blur(alpha, kernel_size=[k, k], sigma=[max(c.copy_paste_context_px / 2.0, 0.3)] * 2).clamp_(0, 1)
            alpha = torch.maximum(alpha, union)
        img_f = img_f * (1.0 - alpha) + o2 * alpha
        tgt = tgt * (1.0 - union)  # occlusion: pasted pixels replace the old instances there
        return (
            img_f,
            torch.cat([tgt, sel], 0),
            torch.cat([labels, l2[pick]], 0),
            {k: torch.cat([attrs[k], a2[k][pick]], 0) for k in attrs},
        )


def build_train_transform(
    imgsz: int,
    *,
    flip_prob: float | None = None,
    min_scale: float | None = None,
    max_scale: float | None = None,
    color_jitter: bool | None = None,
    aug: AugConfig | Mapping[str, Any] | None = None,
) -> TrainAugment:
    """Training transform. ``aug`` is an :class:`AugConfig` (or mapping / ``None`` for the defaults); the explicit
    keywords override it when not ``None`` (``color_jitter=False`` disables colour jitter).

    The returned :class:`TrainAugment` is callable as ``tf(image, target) -> (image, target)``: ``image`` uint8
    ``(3, H, W)``, ``target`` instance masks ``(N, H, W)`` or ``tv_tensors.BoundingBoxes``; the output image is a
    normalised ``float32`` ``(3, imgsz, imgsz)`` tensor and the masks are ``float32`` ``(N, imgsz, imgsz)`` in
    [0, 1] (soft; see the module docstring). Instances that were cropped away have an all-zero mask: the dataset drops
    them.
    """
    cfg = AugConfig.resolve(aug, flip_prob=flip_prob, min_scale=min_scale, max_scale=max_scale)
    if color_jitter is False:
        cfg = replace(cfg, color_jitter_prob=0.0)
    return TrainAugment(imgsz, cfg)


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
