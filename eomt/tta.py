"""Test-time augmentation and tiled inference for EoMT instance segmentation (``augment=`` / ``tiles=``).

Both run the model on several *views* of one image and merge the instances the views find:

* ``augment=True``: the whole image at ``imgsz`` and zoomed to 1.5x ``imgsz`` (the zoom is what helps small and thin
  objects; on the windscreen damages it gave +5 % mask mAP and +23 % small-object AP, where the mirror gave ~0).
  ``augment={...}`` overrides ``scales`` (whole-image input sizes as multiples of ``imgsz``, default ``[1.0, 1.5]``)
  and ``flip`` (also run every view mirrored, default False). Keep ``flip`` off for a model whose classes or
  attributes depend on orientation (left / right): a mirrored view predicts them swapped and the merge averages it in.
* ``tiles=True``: overlapping ``size`` x ``size`` px crops (default ``size`` = ``imgsz``: native resolution), each run
  at ``imgsz``, ``overlap`` 0.25, plus the whole image (``full``), so the small objects of a large image are seen at a
  useful scale. ``tiles=1024`` sets the size; ``tiles={...}`` overrides ``size`` / ``overlap`` / ``full``. With
  ``augment`` too, the tiles are mirrored as well (the scales apply to the whole image only).

Either dict also takes the merge settings ``iou`` (0.5), ``score`` (``"mean"`` / ``"max"``, default ``"mean"``) and
``batch`` (views per forward pass, 8).

**Merging.** Each view's queries with score >= ``conf_thres`` (at most ``max_det``) are decoded onto the image exactly
as a single prediction is (onto a copy downscaled to ``_MAX_SIDE`` px for a larger image). Two instances from different
views are one object when they have the same class and a mask IoU >= ``iou`` measured *where both views can see* (the
intersection of their footprints): for whole-image views that is the plain IoU; for neighbouring tiles it is the IoU of
the two pieces inside their overlap, which is how an object cut by a tile border is put back together. The IoU is
measured on a coarse grid of about one mask-logit cell of the finest view, masks dilated by one cell, so a thin
structure that shifts by a cell between views still matches. Greedy by score, single linkage, never two instances of
one view in a cluster.

A cluster's mask is the mean of its members' mask *logits*, each pixel averaged over the members whose view covers it,
decoded as a single prediction is (thresholded at ``logit(mask_thresh)``): averaging probabilities instead moves the
boundary of a sharp mask by a fraction of a cell, which reshapes thin ones. Its score is ``"mean"``: the members'
summed scores over the number of views that fully contain the object, so a view that saw it and missed it votes 0 (as
in weighted boxes fusion); or ``"max"``, the best member's, which keeps every single-view detection at its own score
(with tiles on the windscreen damages it lost 4 % mask mAP to ``"mean"``, its single-tile false positives flooding the
repair / replace rules). Attribute probabilities and embeddings are score-weighted means. Instance family only.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from .postprocess import _masks_to_original, aux_result, boxes_from_masks, query_scores
from .preprocess import preprocess_numpy

_AUGMENT_KEYS = {"flip", "scales"}
_TILES_KEYS = {"size", "overlap", "full"}
_MERGE_KEYS = {"iou", "score", "batch"}


@dataclass(frozen=True)
class TTAConfig:
    """Views and merge settings of :func:`predict_views`, built from ``augment=`` / ``tiles=`` by :meth:`resolve`."""

    flip: bool = False  # add the mirrored view of every view
    scales: tuple[float, ...] = (1.0,)  # whole-image input sizes, as multiples of imgsz
    tiles: bool = False
    tile_size: int | None = None  # tile side in image px; None = imgsz (native resolution)
    overlap: float = 0.25  # minimum overlap of neighbouring tiles, as a fraction of the tile
    full: bool = True  # with tiles, also run the whole-image views
    iou: float = 0.5  # merge threshold
    score: str = "mean"  # "mean" | "max"
    batch: int = 8  # views per forward pass

    def __post_init__(self) -> None:
        if not self.scales or any(s <= 0 for s in self.scales):
            raise ValueError(f"scales must be positive, got {self.scales}")
        if self.tile_size is not None and self.tile_size < 1:
            raise ValueError(f"tile size must be >= 1 px, got {self.tile_size}")
        if not 0 <= self.overlap < 1:
            raise ValueError(f"tile overlap must be in [0, 1), got {self.overlap}")
        if not 0 < self.iou <= 1:
            raise ValueError(f"iou must be in (0, 1], got {self.iou}")
        if self.score not in ("mean", "max"):
            raise ValueError(f"score must be 'mean' or 'max', got {self.score!r}")
        if self.batch < 1:
            raise ValueError(f"batch must be >= 1, got {self.batch}")

    @classmethod
    def resolve(cls, augment=False, tiles=False) -> TTAConfig | None:
        """The config of ``augment`` / ``tiles`` (see the module docstring), None when both are off."""
        aug = _as_dict(augment, "augment", _AUGMENT_KEYS)
        til = _as_dict(tiles, "tiles", _TILES_KEYS)
        if aug is None and til is None:
            return None
        fields: dict = {}
        for d in (aug or {}), (til or {}):
            for k in _MERGE_KEYS & set(d):
                if k in fields and fields[k] != d[k]:
                    raise ValueError(f"augment and tiles set {k!r} differently: {fields[k]!r} vs {d[k]!r}")
                fields[k] = d[k]
        if aug is not None:
            fields["flip"] = bool(aug.get("flip", False))
            fields["scales"] = tuple(float(s) for s in aug.get("scales", (1.0, 1.5)))
        if til is not None:
            fields.update(tiles=True, tile_size=til.get("size"), overlap=float(til.get("overlap", 0.25)),
                          full=bool(til.get("full", True)))
        return cls(**fields)


def _as_dict(value, name: str, keys: set) -> dict | None:
    """``False`` / ``None`` -> None, ``True`` -> {}, an int (tiles) -> {"size": n}, a mapping -> checked copy."""
    if value is None or value is False:
        return None
    if value is True:
        return {}
    if name == "tiles" and isinstance(value, int):
        return {"size": value}
    d = dict(value)
    unknown = set(d) - keys - _MERGE_KEYS
    if unknown:
        raise ValueError(f"unknown {name} options {sorted(unknown)}; valid: {sorted(keys | _MERGE_KEYS)}")
    return d


@dataclass(frozen=True)
class View:
    """One model input: the ``rect`` ``(x0, y0, x1, y1)`` of the image it sees, its input ``size``, mirrored or not."""

    rect: tuple[int, int, int, int]
    size: int
    flip: bool = False


def _tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    """Evenly spread tile origins covering ``[0, length)``, neighbours overlapping by >= ``overlap`` of a tile."""
    if length <= tile:
        return [0]
    n = math.ceil((length - tile) / (tile * (1 - overlap))) + 1
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


def make_views(h: int, w: int, imgsz: int, cfg: TTAConfig, patch: int = 14) -> list[View]:
    """The views of an ``h`` x ``w`` image, duplicates removed (e.g. a single tile that is the whole image)."""
    flips = (False, True) if cfg.flip else (False,)
    views = []
    if not cfg.tiles or cfg.full:
        for s in cfg.scales:
            size = max(patch, round(imgsz * s / patch) * patch)
            views += [View((0, 0, w, h), size, f) for f in flips]
    if cfg.tiles:
        t = cfg.tile_size or imgsz
        for y in _tile_starts(h, t, cfg.overlap):
            for x in _tile_starts(w, t, cfg.overlap):
                views += [View((x, y, min(x + t, w), min(y + t, h)), imgsz, f) for f in flips]
    return list(dict.fromkeys(views))


#: Mask logit of a pixel that no member of a cluster can see.
_BACKGROUND = -20.0
#: Views are decoded and merged on the image itself up to this long side (px), on a downscaled copy above it.
_MAX_SIDE = 2048


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _grid_cells(rect, g: float, hg: int, wg: int) -> tuple[int, int, int, int]:
    """``rect`` (image px) as grid cells ``(y0, x0, y1, x1)``, at least one cell."""
    x0, y0, x1, y1 = rect
    gy0, gx0 = min(round(y0 * g), hg - 1), min(round(x0 * g), wg - 1)
    return gy0, gx0, max(gy0 + 1, min(round(y1 * g), hg)), max(gx0 + 1, min(round(x1 * g), wg))


def _crop(image: np.ndarray, view: View) -> np.ndarray:
    x0, y0, x1, y1 = view.rect
    crop = image[y0:y1, x0:x1]
    return np.ascontiguousarray(crop[:, ::-1] if view.flip else crop)


def _view_instances(out, b, view, meta, cells, conf_thres, max_det, mask_thresh, embed) -> dict | None:
    """One view's kept queries, their mask logits resampled onto the view's grid ``cells`` (None when it kept none)."""
    mask_logits = out["masks_queries_logits"][b].float()
    quality = out.get("quality_logits")
    scores, classes = query_scores(
        out["class_queries_logits"][b], mask_logits, None if quality is None else quality[b], mask_thresh
    )
    sel = (scores >= conf_thres).nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        return None
    if sel.numel() > max_det:
        sel = sel[scores[sel].topk(max_det).indices]
    y0, x0, y1, x1 = cells
    logits = _masks_to_original(mask_logits[sel], y1 - y0, x1 - x0, meta)
    if view.flip:
        logits = logits.flip(-1)
    gbox = boxes_from_masks(logits > _logit(mask_thresh)) + torch.tensor([x0, y0, x0, y0], device=logits.device)
    return {
        "cells": cells,
        "logits": logits.half() if logits.is_cuda else logits,  # (n, H, W) per view: halved on the GPU
        "gbox": gbox.cpu().numpy(),  # mask extent in grid cells (x0, y0, x1, y1); x0 == x1 for an empty mask
        "scores": scores[sel],
        "classes": classes[sel],
        "aux": {n: lg[b].float().softmax(-1)[sel] for n, lg in (out.get("aux_queries_logits") or {}).items()},
        "embed": out["query_embed"][b, sel].float() if embed else None,
    }


def _coarse(f: dict, mask_thresh: float, pool: int):
    """``f``'s masks max-pooled onto the global grid of ``pool`` x ``pool`` px cells (padding aligns every view to it)
    and dilated by one cell, with their cells ``(y0, x0, y1, x1)`` on that grid."""
    y0, x0, y1, x1 = f["cells"]
    out = []
    for chunk in f["logits"].split(32):  # bounded memory: (32, H, W) floats at a time
        m = F.pad((chunk > _logit(mask_thresh)).float()[:, None], (x0 % pool, 0, y0 % pool, 0))
        out.append(F.max_pool2d(F.max_pool2d(m, pool, pool, ceil_mode=True), 3, 1, 1)[:, 0])
    return torch.cat(out), (y0 // pool, x0 // pool, -(-y1 // pool), -(-x1 // pool))


def _match_iou(found: list[dict], mask_thresh: float, pool: int) -> np.ndarray:
    """IoU of every pair of instances from different views, inside the two views' common footprint, on the coarse
    dilated masks of :func:`_coarse`; 0 for different classes, disjoint footprints and instances of the same view."""
    offs = np.cumsum([0] + [len(f["scores"]) for f in found])
    iou = np.zeros((offs[-1], offs[-1]), dtype=np.float32)
    coarse = [_coarse(f, mask_thresh, pool) for f in found]
    for u, v in itertools.combinations(range(len(found)), 2):
        (bu, (ay0, ax0, ay1, ax1)), (bv, (by0, bx0, by1, bx1)) = coarse[u], coarse[v]
        y0, x0, y1, x1 = max(ay0, by0), max(ax0, bx0), min(ay1, by1), min(ax1, bx1)
        if y0 >= y1 or x0 >= x1:
            continue
        a = bu[:, y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0].flatten(1)
        b = bv[:, y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0].flatten(1)
        inter = a @ b.T
        union = a.sum(1)[:, None] + b.sum(1)[None] - inter
        same = found[u]["classes"][:, None] == found[v]["classes"][None]
        m = (inter / union.clamp(min=1) * same).cpu().numpy()
        iou[offs[u] : offs[u + 1], offs[v] : offs[v + 1]] = m
        iou[offs[v] : offs[v + 1], offs[u] : offs[u + 1]] = m.T
    return iou


def _clusters(iou: np.ndarray, scores: np.ndarray, views: np.ndarray, thr: float) -> list[list[int]]:
    """Greedy by score, single linkage: an instance joins every cluster it matches (IoU >= ``thr`` with any member),
    best match first, as long as none of them holds an instance of its view and no two of them share a view; the
    clusters it joins are merged (an object cut by several tiles is reassembled whatever order its pieces come in)."""
    owner = np.full(len(scores), -1)
    members: dict[int, list[int]] = {}
    seen: dict[int, set] = {}
    for i in np.argsort(-scores, kind="stable"):
        js = np.nonzero(iou[i] >= thr)[0]
        k = -1
        for j in js[np.argsort(-iou[i, js], kind="stable")]:
            c = owner[j]
            if c < 0 or c == k or views[i] in seen[c] or (k >= 0 and seen[k] & seen[c]):
                continue
            if k < 0:
                k = c
                continue
            for m in members[c]:
                owner[m] = k
            members[k] += members.pop(c)
            seen[k] |= seen.pop(c)
        if k < 0:
            k = int(i)
            members[k], seen[k] = [], set()
        owner[i] = k
        members[k].append(int(i))
        seen[k].add(views[i])
    return list(members.values())


@torch.no_grad()
def predict_views(
    model,
    image: np.ndarray,
    cfg: TTAConfig,
    *,
    imgsz: int,
    conf_thres: float = 0.0,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    min_mask_area: float = 0.0,
    letterbox: bool = True,
    mean=None,
    std=None,
    amp_dtype: torch.dtype | None = None,
    aux_scopes: dict | None = None,
    embed: bool = False,
) -> dict:
    """Instances of the RGB ``image`` ``(H, W, 3)`` uint8, merged over the views of ``cfg``.

    Returns the :func:`~eomt.postprocess.postprocess_instance` dict (``boxes`` / ``scores`` / ``classes`` / ``masks``
    and, with attribute heads, ``aux``), plus ``embed`` ``(N, hidden)`` with ``embed=True``. There is no ``query_idx``:
    a merged instance comes from several queries.
    """
    if getattr(model, "family", "instance") != "instance":
        raise NotImplementedError("TTA / tiled inference supports the instance family only.")
    h, w = image.shape[:2]
    dev = next(model.parameters()).device
    views = make_views(h, w, imgsz, cfg, model.patch_size)
    # The merge grid is the image (downscaled above _MAX_SIDE); matching pools it to ~one mask-logit cell of the
    # finest view (letterbox fits a view's long side to its input).
    cells_per_px = 2**model.num_upscale_blocks / model.patch_size
    fit = max if letterbox else min
    density = max(cells_per_px * v.size / fit(v.rect[2] - v.rect[0], v.rect[3] - v.rect[1]) for v in views)
    g = min(1.0, _MAX_SIDE / max(h, w))
    hg, wg = max(1, round(h * g)), max(1, round(w * g))
    pool = max(1, round(g / density))
    view_cells = [_grid_cells(v.rect, g, hg, wg) for v in views]

    found: list[dict] = []
    for size in dict.fromkeys(v.size for v in views):
        idx = [i for i, v in enumerate(views) if v.size == size]
        for start in range(0, len(idx), cfg.batch):
            chunk = idx[start : start + cfg.batch]
            pre = [
                preprocess_numpy(_crop(image, views[i]), size, letterbox=letterbox, mean=mean, std=std) for i in chunk
            ]
            x = torch.from_numpy(np.stack([chw for chw, _ in pre])).to(dev)
            ctx = torch.autocast(dev.type, dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
            with ctx:
                out = model(x)
            for b, (i, (_, meta)) in enumerate(zip(chunk, pre)):
                inst = _view_instances(out, b, views[i], meta, view_cells[i], conf_thres, max_det, mask_thresh, embed)
                if inst is not None:
                    inst["view"] = i
                    found.append(inst)

    if not found:
        aux_sizes = {s.name: s.num_classes for s in getattr(model, "aux_specs", [])}
        return _result(torch.zeros((0, h, w), dtype=torch.bool), torch.zeros(0), torch.zeros(0, dtype=torch.long),
                       {n: torch.zeros(0, k) for n, k in aux_sizes.items()},
                       torch.zeros(0, model.config.hidden_size) if embed else None, aux_scopes)

    scores = torch.cat([f["scores"] for f in found]).float()
    classes = torch.cat([f["classes"] for f in found])
    views_of = np.concatenate([[f["view"]] * len(f["scores"]) for f in found])
    offs = np.cumsum([0] + [len(f["scores"]) for f in found])
    gboxes = np.concatenate([f["gbox"] for f in found])
    s_np = scores.cpu().numpy()
    clusters = _clusters(_match_iou(found, mask_thresh, pool), s_np, views_of, cfg.iou)

    fused = np.empty(len(clusters), dtype=np.float32)
    for k, m in enumerate(clusters):
        if cfg.score == "max":
            fused[k] = s_np[m].max()
            continue
        boxes = gboxes[m][gboxes[m][:, 2] > gboxes[m][:, 0]]  # non-empty members
        seeing = 0
        if len(boxes):
            bx0, by0 = boxes[:, :2].min(0)
            bx1, by1 = boxes[:, 2:].max(0)
            seeing = sum(y0 <= by0 and x0 <= bx0 and by1 <= y1 and bx1 <= x1 for y0, x0, y1, x1 in view_cells)
        fused[k] = s_np[m].sum() / max(len(m), seeing)
    keep = [k for k in np.argsort(-fused, kind="stable") if fused[k] >= conf_thres][:max_det]

    # Membership (kept cluster x instance), and the same normalised by score for the weighted means below. Everything
    # after the clustering is a few matrix products per view, not a GPU op per instance.
    member = np.zeros((len(keep), len(s_np)), dtype=np.float32)
    for c, k in enumerate(keep):
        member[c, clusters[k]] = 1.0
    weight = member * s_np[None]
    weight /= np.maximum(weight.sum(1, keepdims=True), 1e-12)
    member_t = torch.from_numpy(member).to(scores.device)
    weight_t = torch.from_numpy(weight).to(scores.device)

    # Each kept cluster's mean mask logit (each pixel over the members whose view covers it; one none of them covers is
    # background), thresholded like a single prediction; 16 clusters at a time ((16, H, W) floats).
    masks = torch.zeros((len(keep), h, w), dtype=torch.bool, device=scores.device)
    for start in range(0, len(keep), 16):
        mt = member_t[start : start + 16]
        acc = torch.zeros(len(mt), hg, wg, device=scores.device)
        cnt = torch.zeros_like(acc)
        for u, f in enumerate(found):
            y0, x0, y1, x1 = f["cells"]
            mu = mt[:, offs[u] : offs[u + 1]]
            summed = mu.to(f["logits"].dtype) @ f["logits"].flatten(1)
            acc[:, y0:y1, x0:x1] += summed.float().view(len(mt), y1 - y0, x1 - x0)
            cnt[:, y0:y1, x0:x1] += mu.sum(1)[:, None, None]
        logit = torch.where(cnt > 0, acc / cnt.clamp(min=1), _BACKGROUND)
        if (hg, wg) != (h, w):
            logit = F.interpolate(logit[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        masks[start : start + 16] = logit > _logit(mask_thresh)

    # Score-weighted means of the members' attribute probabilities and embeddings.
    aux_probs = {n: weight_t @ torch.cat([f["aux"][n] for f in found]) for n in found[0]["aux"]}
    embeds = weight_t @ torch.cat([f["embed"] for f in found]) if embed else None
    out_scores = torch.as_tensor(fused[keep], device=scores.device)
    out_classes = classes[[clusters[k][0] for k in keep]] if keep else classes[:0]
    if min_mask_area > 0:
        ok = masks.flatten(1).sum(1) >= min_mask_area
        masks, out_scores, out_classes = masks[ok], out_scores[ok], out_classes[ok]
        aux_probs = {n: p[ok] for n, p in aux_probs.items()}
        embeds = embeds[ok] if embeds is not None else None
    return _result(masks, out_scores, out_classes, aux_probs, embeds, aux_scopes)


def _result(masks, scores, classes, aux_probs: dict, embeds, aux_scopes) -> dict:
    res = {
        "num_detections": int(scores.numel()),
        "boxes": boxes_from_masks(masks),
        "scores": scores,
        "classes": classes.long(),
        "masks": masks,
    }
    if aux_probs:
        res["aux"] = aux_result(aux_probs, classes, aux_scopes)
    if embeds is not None:
        res["embed"] = embeds.cpu()
    return res
