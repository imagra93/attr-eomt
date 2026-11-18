"""COCO instance-segmentation validation (pycocotools ``COCOeval``).

Runs the model over a :class:`~eomt.data.coco.CocoValImages` set, converts each
prediction to a COCO result (RLE mask + score + original category id) and scores
it with ``COCOeval(iouType='segm')``. Optionally also reports bbox mAP using the
mask-extent boxes.

On several GPUs (:mod:`eomt.engine.distributed`) each process runs its share of the
images and the main process scores (or counts) them all; the others get ``{}``.

``group_by`` (an image field, or several) adds the main metrics per value of that field (e.g. the use case of each
photo): ``<field>=<value>/segm/mAP``, and for sets ``<prefix>/<field>=<value>/...``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ..data import collate_val
from ..postprocess import postprocess_detection, postprocess_instance, slice_batch
from ..progress import progress
from ..tta import TTAConfig, predict_views
from .distributed import broadcast, gather, shard, shard_groups, sum_over_processes

#: Seconds a validation data loader may wait for one batch before it raises (a stalled read, e.g. on a network share,
#: then fails with a message instead of hanging every process until the distributed watchdog kills the run).
LOADER_TIMEOUT = 900


def _loader(ds, batch_size: int, num_workers: int, collate_fn) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=torch.cuda.is_available(), timeout=LOADER_TIMEOUT if num_workers > 0 else 0)


def amp_dtype(device, amp: bool):
    """The autocast dtype of an evaluation forward: bf16 where the GPU has it (as training and inference run), else
    fp16; ``None`` (full precision) on the CPU or with ``amp=False``."""
    if not amp or torch.device(device).type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _autocast(device, amp: bool):
    dtype = amp_dtype(device, amp)
    return contextlib.nullcontext() if dtype is None else torch.autocast("cuda", dtype=dtype)


def binarize_gt(masks: torch.Tensor) -> torch.Tensor:
    """Soft (area-resized) GT masks ``(N, H, W)`` as boolean masks, each at half its own peak: a thin or tiny instance
    whose resized mask never reaches 0.5 keeps its pixels (the rule of the loss's IoU and the attribute gate)."""
    if not len(masks):
        return masks.bool()
    m = masks.float()
    return m >= (0.5 * m.flatten(1).amax(1)).clamp(min=1e-6)[:, None, None]


#: COCOeval.stats index -> metric name.
_SEGM_KEYS = [
    "mAP",
    "mAP50",
    "mAP75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _encode_mask(mask: np.ndarray):
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle



def _bin_labels(bins) -> list[str]:
    """``[10, 35]`` -> ``["<10", "10-35", ">=35"]``: the GT-value buckets of a regression head's ``bins``."""
    b = [f"{float(x):g}" for x in bins]
    return [f"<{b[0]}"] + [f"{lo}-{hi}" for lo, hi in zip(b, b[1:])] + [f">={b[-1]}"]


def attr_metric_names(spec) -> list[str]:
    """The validation metrics of one attribute head: ``aux_acc/<head>`` for a categorical head; for a regression head
    ``aux_mae/<head>`` (mean absolute error, real units), plus for a ``log`` head ``aux_logmae`` (the MAE of the
    ``log1p`` values) and ``aux_ratio`` (the median of prediction / GT), and all of them again per GT-value bucket of
    the head's ``bins`` (``aux_mae/<head>@10-35``)."""
    if not spec.regression:
        return [f"aux_acc/{spec.name}"]
    kinds = ["aux_mae"] + (["aux_logmae", "aux_ratio"] if spec.log else [])
    names = [f"{k}/{spec.name}" for k in kinds]
    for label in _bin_labels(spec.bins) if spec.bins else []:
        names += [f"{k}/{spec.name}@{label}" for k in kinds]
    return names


def regression_metrics(spec, pred, gt) -> dict[str, float]:
    """The :func:`attr_metric_names` of a regression head from its predictions and GT values (real units). A large value
    dominates the MAE; the log-space MAE and the median ratio weigh every instance alike, and the buckets show where the
    errors are."""
    pred = torch.as_tensor(pred, dtype=torch.float64).flatten()
    gt = torch.as_tensor(gt, dtype=torch.float64).flatten()

    def stats(sel: torch.Tensor) -> dict[str, float]:
        p, g = pred[sel], gt[sel]
        out = {"aux_mae": float((p - g).abs().mean()) if len(p) else math.nan}
        if spec.log:
            out["aux_logmae"] = float((p.clamp(min=-1 + 1e-9).log1p() - g.log1p()).abs().mean()) if len(p) else math.nan
            pos = g > 0
            out["aux_ratio"] = float((p[pos] / g[pos]).median()) if bool(pos.any()) else math.nan
        return out

    res = {f"{k}/{spec.name}": v for k, v in stats(torch.ones_like(gt, dtype=torch.bool)).items()}
    if spec.bins:
        edges = [-math.inf, *[float(b) for b in spec.bins], math.inf]
        for label, lo, hi in zip(_bin_labels(spec.bins), edges, edges[1:]):
            res.update({f"{k}/{spec.name}@{label}": v for k, v in stats((gt >= lo) & (gt < hi)).items()})
    return res


def _fields(group_by) -> tuple[str, ...]:
    return () if not group_by else (group_by,) if isinstance(group_by, str) else tuple(group_by)


def _field_value(coco, image_id, field: str) -> str | None:
    """An image's ``field`` as text (``None`` when it has none)."""
    v = coco.imgs.get(image_id, {}).get(field)
    return None if v is None else str(v)


def group_values(coco, image_ids, group_by) -> dict[str, list[str]]:
    """``{field: [value, ...]}``: the values each ``group_by`` field takes over ``image_ids``, sorted."""
    return {f: sorted({v for i in image_ids if (v := _field_value(coco, i, f)) is not None}) for f in _fields(group_by)}


def _check_attr_vocab(model, val_ds) -> list:
    """The val dataset's head specs that the model also has, after checking both use the same categories (a different
    contiguous-id order would score every prediction against the wrong label)."""
    model_specs = {s.name: s for s in getattr(model, "aux_specs", [])}
    specs = [s for s in val_ds.aux_specs if s.name in model_specs]
    for s in specs:
        if not s.regression and s.names != model_specs[s.name].names:
            raise ValueError(
                f"attribute {s.name!r}: the val categories {s.names} differ from the model's {model_specs[s.name].names}"
            )
    return specs


def _adopt_flip_aware(model, val_ds) -> None:
    """Mirrored views need each categorical head's flip mapping. A checkpoint saved before it was stored has none, so
    take the val dataset's (the definition the trainer swapped by), with a warning."""
    declared = {s.name: s for s in val_ds.aux_specs}
    for s in getattr(model, "aux_specs", []):
        d = declared.get(s.name)
        if not s.regression and not s.flip_aware and d is not None and d.flip_aware:
            pairs = {s.names[a]: s.names[b] for a, b in d.flip_aware.items() if a < b}
            warnings.warn(f"attribute {s.name!r}: the checkpoint stores no flip_aware mapping (saved by an older version); "
                          f"mirrored views swap {pairs} as the val dataset declares", stacklevel=3)
            s.flip_aware = dict(d.flip_aware)


def _greedy_pairs(iou: np.ndarray, gate: float) -> list[tuple[int, int]]:
    """One-to-one ``(row, col)`` pairs of an IoU matrix, highest IoU first, down to ``gate``."""
    pairs: list[tuple[int, int]] = []
    taken_r: set[int] = set()
    taken_c: set[int] = set()
    for flat in np.argsort(-iou, axis=None):
        r, c = divmod(int(flat), iou.shape[1])
        if iou[r, c] < gate:
            break
        if r not in taken_r and c not in taken_c:
            taken_r.add(r)
            taken_c.add(c)
            pairs.append((r, c))
    return pairs


def _count_attrs(res, rles, val_ds, img_id, specs, iou_gate, hits, tot, reg) -> None:
    """Add one image's attributes to ``hits`` / ``tot`` (categorical heads) and ``reg`` (``{head: (predictions, GT
    values)}``, regression heads): each GT instance is matched to the kept detection of its class with the highest mask
    IoU (one to one, IoU >= ``iou_gate``) and every head is scored on it. Labels follow the trainer's (``_attr_label``:
    missing, out-of-vocabulary and out-of-scope values are ignored)."""
    from pycocotools import mask as mask_utils

    from ..data.coco import _attr_label

    gts = [a for a in val_ds.coco.imgToAnns.get(img_id, []) if not a.get("iscrowd")]
    if not gts or not rles:
        return
    iou = np.asarray(mask_utils.iou(rles, [val_ds.coco.annToRLE(a) for a in gts], [0] * len(gts)))
    gt_cls = np.array([val_ds.cat2contig[a["category_id"]] for a in gts])
    iou[res["classes"].cpu().numpy()[:, None] != gt_cls[None, :]] = 0.0
    for d, g in _greedy_pairs(iou, iou_gate):
        for s in specs:
            label = _attr_label(s, val_ds._attr_id_maps, gts[g].get("attributes") or {}, int(gt_cls[g]))
            if s.regression:
                if label == label:  # not NaN
                    reg[s.name][0].append(float(res["aux_values"][s.name][d]))
                    reg[s.name][1].append(float(label))
            elif label != -100:
                hits[s.name] += int(res["aux"][s.name]["ids"][d]) == label
                tot[s.name] += 1


def _cocoeval(coco_gt, results, img_ids, iou_type: str, verbose: bool) -> dict:
    from pycocotools.cocoeval import COCOeval

    if not results:
        return {f"{iou_type}/{k}": 0.0 for k in _SEGM_KEYS}

    buf = io.StringIO()
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(buf)
    with ctx:
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {f"{iou_type}/{k}": float(v) for k, v in zip(_SEGM_KEYS, ev.stats)}


def _records(img_id, res: dict, contig2cat, *, segm: bool = True, bbox: bool = True) -> tuple[list, list, list]:
    """COCO result records of one image's detections ``(segm, bbox, rles)``, with one copy to the host per image."""
    classes, scores, boxes = res["classes"].tolist(), res["scores"].tolist(), res["boxes"].tolist()
    masks = res["masks"].cpu().numpy() if segm else None
    seg, box, rles = [], [], []
    for i in range(res["num_detections"]):
        common = {"image_id": int(img_id), "category_id": int(contig2cat[int(classes[i])]), "score": float(scores[i])}
        if segm:
            rles.append(_encode_mask(masks[i]))
            seg.append({**common, "segmentation": rles[-1]})
        if bbox:
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            box.append({**common, "bbox": [x1, y1, x2 - x1, y2 - y1]})
    return seg, box, rles


def _group_metrics(coco, ids, group_by, results: dict[str, list[dict]]) -> dict[str, float]:
    """``<field>=<value>/<type>/mAP`` for every value of the ``group_by`` image fields, from the results per type."""
    out = {}
    for field, values in group_values(coco, ids, group_by).items():
        for v in values:
            sub = [i for i in ids if _field_value(coco, i, field) == v]
            wanted = set(sub)
            for t, res in results.items():  # copies: loadRes writes into the result dicts
                out[f"{field}={v}/{t}/mAP"] = _cocoeval(
                    coco, [dict(r) for r in res if r["image_id"] in wanted], sub, t, False)[f"{t}/mAP"]
    return out


@torch.no_grad()
def evaluate(
    model,
    val_ds,
    *,
    device,
    batch_size: int = 4,
    num_workers: int = 4,
    conf_thres: float = 0.0,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    min_mask_area: float = 0.0,
    amp: bool = False,
    also_bbox: bool = True,
    verbose: bool = True,
    augment: bool | dict = False,
    tiles: bool | int | dict = False,
    attributes: bool = False,
    attr_iou: float = 0.5,
    group_by: str | list[str] | None = None,
) -> dict[str, float]:
    """Evaluate ``model`` on ``val_ds`` and return a COCO metrics dict.

    ``augment`` / ``tiles`` (see :mod:`eomt.tta`) evaluate with test-time augmentation / tiled inference: each image is
    predicted alone from its views, at ``val_ds.imgsz`` with its letterbox and normalisation.

    ``attributes`` adds the secondary heads' metrics, scored on the same predictions as the mAP (so also under
    ``augment`` / ``tiles``): ``aux_acc/<head>`` accuracy, or for a regression head ``aux_mae/<head>`` mean absolute
    error in real units (and the others of :func:`attr_metric_names`), over the GT instances matched to a same-class
    detection with mask IoU >= ``attr_iou``. That
    population is the deployed one (kept detections), not the trainer's matched queries, so the numbers are close to the
    ``val/aux_*`` columns of ``metrics.csv`` but not equal. ``val_ds`` needs the attribute definitions (it builds them
    from its JSON) and the model the same heads and categories. ``group_by`` adds ``<field>=<value>/segm/mAP`` and
    ``.../bbox/mAP`` per value of an image field.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    contig2cat = val_ds.contig2cat
    cfg = TTAConfig.resolve(augment, tiles)
    if cfg is not None and not getattr(val_ds, "raw", False):  # the views are cut from the original image
        val_ds = copy.copy(val_ds)
        val_ds.raw = True
    loader = _loader(shard(val_ds), batch_size, num_workers, collate_val)

    segm_results: list[dict] = []
    bbox_results: list[dict] = []
    attr_specs = _check_attr_vocab(model, val_ds) if attributes else []
    if attr_specs and cfg is not None and cfg.flip:
        _adopt_flip_aware(model, val_ds)
    attr_hits = {s.name: 0 for s in attr_specs if not s.regression}
    attr_tot = {s.name: 0 for s in attr_specs if not s.regression}
    attr_reg = {s.name: ([], []) for s in attr_specs if s.regression}

    for pixel_values, image_ids, sizes, metas in progress(
        loader, desc="val", unit="batch", leave=False, disable=not verbose
    ):
        if cfg is not None:
            batch_results = [
                predict_views(
                    model, image, cfg, imgsz=val_ds.imgsz, conf_thres=conf_thres, max_det=max_det,
                    mask_thresh=mask_thresh, min_mask_area=min_mask_area, letterbox=val_ds.letterbox,
                    mean=val_ds.mean, std=val_ds.std, amp_dtype=amp_dtype(device, amp),
                )
                for image in pixel_values
            ]
        else:
            pixel_values = pixel_values.to(device, non_blocking=True)
            with _autocast(device, amp):
                out = model(pixel_values)
            mql = out["masks_queries_logits"]
            cql = out["class_queries_logits"]
            batch_results = [
                postprocess_instance(
                    {
                        "masks_queries_logits": mql[b : b + 1],
                        "class_queries_logits": cql[b : b + 1],
                        **({"quality_logits": out["quality_logits"][b : b + 1]} if "quality_logits" in out else {}),
                        **({k: {n: v[b : b + 1] for n, v in out[k].items()}
                            for k in ("aux_queries_logits", "aux_queries_values") if k in out} if attr_specs else {}),
                    },
                    conf_thres,
                    (orig_w, orig_h),
                    max_det=max_det,
                    mask_thresh=mask_thresh,
                    min_mask_area=min_mask_area,
                    preprocess_meta=meta,
                )
                for b, ((orig_w, orig_h), meta) in enumerate(zip(sizes, metas))
            ]

        for img_id, res in zip(image_ids, batch_results):
            seg, box, rles = _records(img_id, res, contig2cat, bbox=also_bbox)
            segm_results += seg
            bbox_results += box
            if attr_specs:
                _count_attrs(res, rles, val_ds, int(img_id), attr_specs, attr_iou, attr_hits, attr_tot, attr_reg)

    parts = gather((segm_results, bbox_results, attr_hits, attr_tot, attr_reg))
    if parts is None:  # several GPUs: the main process scores the results of all
        return {}
    segm_results = [r for p in parts for r in p[0]]
    bbox_results = [r for p in parts for r in p[1]]
    # copies: ``loadRes`` writes ``bbox`` / ``area`` into the result dicts, which breaks a second evaluation of them
    metrics = _cocoeval(val_ds.coco, [dict(r) for r in segm_results], val_ds.ids, "segm", verbose)
    if also_bbox:
        metrics.update(_cocoeval(val_ds.coco, [dict(r) for r in bbox_results], val_ds.ids, "bbox", verbose))
    for s in attr_specs:
        if s.regression:
            metrics.update(regression_metrics(s, [x for p in parts for x in p[4][s.name][0]],
                                              [x for p in parts for x in p[4][s.name][1]]))
        else:
            hit, tot = sum(p[2][s.name] for p in parts), sum(p[3][s.name] for p in parts)
            metrics[f"aux_acc/{s.name}"] = hit / tot if tot else float("nan")
    metrics.update(_group_metrics(val_ds.coco, val_ds.ids, group_by,
                                  {"segm": segm_results, **({"bbox": bbox_results} if also_bbox else {})}))
    return metrics


@torch.no_grad()
def evaluate_detection(
    model,
    val_ds,
    *,
    device,
    batch_size: int = 4,
    num_workers: int = 4,
    conf_thres: float = 0.0,
    max_det: int = 100,
    amp: bool = False,
    verbose: bool = True,
    augment: bool | dict = False,
    tiles: bool | int | dict = False,
    group_by: str | list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a ``family="detect"`` model on ``val_ds`` -> COCO **bbox** metrics.

    Mirrors :func:`evaluate` but reads the box head (``pred_boxes``) via
    :func:`~eomt.postprocess.postprocess_detection` and scores only ``bbox`` mAP
    (there are no masks). Returns a metrics dict keyed ``bbox/<metric>`` (and ``<field>=<value>/bbox/mAP`` per value of
    a ``group_by`` image field).
    """
    if TTAConfig.resolve(augment, tiles) is not None:
        raise NotImplementedError("augment / tiles support the instance family only.")
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    contig2cat = val_ds.contig2cat
    loader = _loader(shard(val_ds), batch_size, num_workers, collate_val)

    bbox_results: list[dict] = []
    for pixel_values, image_ids, sizes, metas in progress(
        loader, desc="val", unit="batch", leave=False, disable=not verbose
    ):
        pixel_values = pixel_values.to(device, non_blocking=True)
        with _autocast(device, amp):
            out = model(pixel_values)
        pb = out["pred_boxes"]
        cql = out["class_queries_logits"]

        for b, (img_id, (orig_w, orig_h), meta) in enumerate(zip(image_ids, sizes, metas)):
            res = postprocess_detection(
                {"pred_boxes": pb[b : b + 1], "class_queries_logits": cql[b : b + 1]},
                conf_thres,
                (orig_w, orig_h),
                max_det=max_det,
                preprocess_meta=meta,
            )
            bbox_results += _records(img_id, res, contig2cat, segm=False)[1]

    parts = gather(bbox_results)
    if parts is None:  # several GPUs: the main process scores the results of all
        return {}
    results = [r for part in parts for r in part]
    metrics = _cocoeval(val_ds.coco, [dict(r) for r in results], val_ds.ids, "bbox", verbose)
    metrics.update(_group_metrics(val_ds.coco, val_ds.ids, group_by, {"bbox": results}))
    return metrics


@torch.no_grad()
def sweep(
    model,
    val_ds,
    *,
    device,
    grid: list[dict],
    batch_size: int = 4,
    num_workers: int = 4,
    amp: bool = False,
    also_bbox: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Sweep postprocessing knobs over ``val_ds``, sharing one forward per batch.

    ``grid`` is a list of knob dicts, each with any of ``conf_thres`` / ``mask_thresh``
    / ``max_det`` / ``min_mask_area`` (missing keys take the postprocess defaults).
    The model runs **once per batch**; every grid point re-runs only the (cheap)
    postprocess + COCO encoding on the shared logits, so the expensive forward is
    not repeated. Returns one row per grid point: ``{**knobs, **segm_metrics}``
    (plus ``bbox/*`` when ``also_bbox``), in the same order as ``grid``.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    contig2cat = val_ds.contig2cat
    loader = _loader(val_ds, batch_size, num_workers, collate_val)

    # One result accumulator per grid point.
    segm_results: list[list[dict]] = [[] for _ in grid]
    bbox_results: list[list[dict]] = [[] for _ in grid]

    for pixel_values, image_ids, sizes, metas in progress(
        loader, desc="sweep", unit="batch", leave=False, disable=not verbose
    ):
        pixel_values = pixel_values.to(device, non_blocking=True)
        with _autocast(device, amp):
            out = model(pixel_values)
        mql = out["masks_queries_logits"]
        cql = out["class_queries_logits"]

        for b, (img_id, (orig_w, orig_h), meta) in enumerate(zip(image_ids, sizes, metas)):
            single = {
                "masks_queries_logits": mql[b : b + 1],
                "class_queries_logits": cql[b : b + 1],
                **({"quality_logits": out["quality_logits"][b : b + 1]} if "quality_logits" in out else {}),
            }
            for gi, knobs in enumerate(grid):
                res = postprocess_instance(
                    single,
                    knobs.get("conf_thres", 0.0),
                    (orig_w, orig_h),
                    max_det=int(knobs.get("max_det", 100)),
                    mask_thresh=float(knobs.get("mask_thresh", 0.5)),
                    min_mask_area=float(knobs.get("min_mask_area", 0.0)),
                    preprocess_meta=meta,
                )
                seg, box, _ = _records(img_id, res, contig2cat, bbox=also_bbox)
                segm_results[gi] += seg
                bbox_results[gi] += box

    rows: list[dict] = []
    for gi, knobs in enumerate(grid):
        row = dict(knobs)
        row.update(_cocoeval(val_ds.coco, segm_results[gi], val_ds.ids, "segm", verbose=False))
        if also_bbox:
            row.update(_cocoeval(val_ds.coco, bbox_results[gi], val_ds.ids, "bbox", verbose=False))
        rows.append(row)
    return rows


@torch.no_grad()
def aux_evaluate(
    model,
    val_seg_ds,
    *,
    device,
    batch_size: int = 4,
    num_workers: int = 4,
    amp: bool = False,
    verbose: bool = True,
    iou_gate: float = 0.5,
    class_gate: bool = True,
) -> tuple[dict[str, float], dict[str, dict[int, tuple[int, int]]]]:
    """Held-out matched-query accuracy per secondary head, plus a per-primary breakdown.

    ``val_seg_ds`` is a :class:`~eomt.data.coco.CocoInstanceSeg` over the val split
    (deterministic val transform, sharing train's attribute id map). Runs the model
    as given (the trainer sets its masked-attention regime), matches predictions to GT with EoMT's matcher, and reports top-1
    accuracy on matched queries (ignoring ``-100`` labels); for a regression head the mean absolute error in real
    units instead (ignoring NaN) and the other :func:`attr_metric_names`, and the per-primary pairs hold its summed
    absolute error in place of ``correct``.

    The scalar accuracy uses the same gate as training (``iou_gate`` + ``class_gate``)
    so it reflects the trained population. The per-primary diagnostic uses an
    IoU-only gate (no class gate) so primary classes the model often *misclassifies*
    still show up. Returns ``({metric: value}, {name: {primary_cls: (correct, total)}})``, the metrics keyed as
    :func:`attr_metric_names` names them (``aux_acc/<head>``, ``aux_mae/<head>``, ...).
    """
    from ..aux_cls import aux_accuracy, aux_accuracy_by_primary, aux_regression_pairs, gate_indices, match_queries
    from ..data import collate_train

    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    loader = _loader(shard(val_seg_ds), batch_size, num_workers, collate_train)
    hits = {s.name: 0 for s in val_seg_ds.aux_specs}
    tot = {s.name: 0 for s in val_seg_ds.aux_specs}
    per_class: dict[str, dict[int, list[int]]] = {s.name: {} for s in val_seg_ds.aux_specs}
    reg: dict[str, tuple[list, list]] = {s.name: ([], []) for s in val_seg_ds.aux_specs if s.regression}

    for pixel_values, mask_labels, class_labels, aux_labels in progress(
        loader, desc="val-aux", unit="batch", leave=False, disable=not verbose
    ):
        pixel_values = pixel_values.to(device, non_blocking=True)
        mask_labels = [m.to(device) for m in mask_labels]
        class_labels = [c.to(device) for c in class_labels]
        aux_labels = {k: [t.to(device) for t in v] for k, v in aux_labels.items()}
        with _autocast(device, amp):
            out = model(pixel_values)
        indices = match_queries(model, out, mask_labels, class_labels)
        gated = gate_indices(
            out, indices, mask_labels, class_labels,
            iou_thr=iou_gate, require_class=class_gate,
        )
        for name, (hit, t) in aux_accuracy(
            model, out, mask_labels, class_labels, aux_labels, indices=gated
        ).items():
            hits[name] += hit
            tot[name] += t
        for name, (pred, gt) in aux_regression_pairs(model, out, aux_labels, gated).items():
            reg[name][0].extend(pred.tolist())
            reg[name][1].extend(gt.tolist())
        # per-primary diagnostic: IoU-gated only (keep misclassified primaries)
        iou_only = gate_indices(
            out, indices, mask_labels, class_labels,
            iou_thr=iou_gate, require_class=False,
        )
        for name, buckets in aux_accuracy_by_primary(
            model, out, mask_labels, class_labels, aux_labels, indices=iou_only
        ).items():
            for cls_id, (c, t) in buckets.items():
                acc = per_class[name].setdefault(cls_id, [0, 0])
                acc[0] += c
                acc[1] += t

    counts = sum_over_processes((hits, tot, per_class))
    regs = gather(reg)
    if counts is None:  # several GPUs: the main process reports the counts of all
        return {}, {}
    hits, tot, per_class = counts
    scalar = {}
    for s in val_seg_ds.aux_specs:
        if s.regression:
            scalar.update(regression_metrics(s, [x for r in regs for x in r[s.name][0]],
                                             [x for r in regs for x in r[s.name][1]]))
        else:
            scalar[f"aux_acc/{s.name}"] = hits[s.name] / tot[s.name] if tot[s.name] else float("nan")
    pc = {n: {c: (v[0], v[1]) for c, v in buckets.items()} for n, buckets in per_class.items()}
    return scalar, pc


class _Indexed(torch.utils.data.Dataset):
    """``(i, ds[i])``: keeps each set's index (its image id) through the sharding over several GPUs."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return i, self.ds[i]


def _only(batch):
    return batch[0]


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """IoU of two stacks of boolean masks ``(n, H, W)`` x ``(m, H, W)`` -> ``(n, m)``."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    b = b.to(a.device)
    a, b = a.flatten(1).float(), b.flatten(1).float()
    inter = a @ b.T
    return (inter / (a.sum(1)[:, None] + b.sum(1)[None] - inter).clamp_min(1)).cpu().numpy()


def _in_ignore(masks: torch.Tensor, ignore: torch.Tensor | None) -> torch.Tensor:
    """Per detection (``masks`` ``(N, H, W)`` bool), whether at least half of it lies in the ``ignore`` map
    (1 = left out), on the CPU."""
    if ignore is None or not len(masks) or not bool(ignore.any()):
        return torch.zeros(len(masks), dtype=torch.bool)
    ign = ignore.to(masks.device) > 0.5
    inside = (masks & ign).flatten(1).sum(1).float() / masks.flatten(1).sum(1).clamp(min=1)
    return (inside >= 0.5).cpu()


#: The embeddings the links between the photos of a set are scored with (:func:`evaluate_sets`): ``match``, the one
#: inference links with (the matching head's, else the per-query embedding) in the forward of the set; and ``reid``, the
#: per-query embedding of the photos predicted alone (the baseline: re-identification with no training for it).
LINK_SOURCES = ("match", "reid")
_MODES = ("alone", "set")
_LINK_STATS = ("f1", "precision", "recall", "thres")


def _det_ids(r: dict, sel: torch.Tensor, gt_masks: torch.Tensor, gt_ids: torch.Tensor, gate: float,
             unsure: torch.Tensor | None = None) -> np.ndarray:
    """Which instance each detection ``sel`` of a photo is: the id of the annotated instance its mask overlaps (IoU >=
    ``gate``, one to one), -1 for none, -2 when unknown (``unsure``: it may be an instance nobody annotated)."""
    d_ids = np.full(len(sel), -1)
    if unsure is not None and len(sel):
        d_ids[unsure[sel.cpu()].numpy()] = -2
    for d, g in _greedy_pairs(_mask_iou(r["masks"][sel.to(r["masks"].device)], gt_masks), gate):
        d_ids[d] = int(gt_ids[g])
    return d_ids


def _link_record(r: dict, embed: torch.Tensor, sel: torch.Tensor, d_ids: np.ndarray, gate_keys: tuple) -> dict:
    """What linking needs of one photo's detections ``sel``: their L2-normalized embeddings (``embed`` ``(Q, d)``, on the
    CPU), instance ids (:func:`_det_ids`) and gate keys (the class and categorical attributes named by ``gate_keys``)."""
    q = r["query_idx"][sel.to(r["query_idx"].device)]
    emb = F.normalize(embed[q.to(embed.device)].float(), dim=-1).cpu()
    classes = r["classes"].tolist()
    aux_ids = {k: r["aux"][k]["ids"].tolist() for k in gate_keys if k != "class"}
    keys = [tuple(classes[i] if k == "class" else aux_ids[k][i] for k in gate_keys) for i in sel.tolist()]
    return {"emb": emb, "ids": d_ids, "keys": keys}


def _flatten_records(records: list[dict], gated: bool):
    """``(similarity (M, M), photo (M,), keys, ids (M,))`` of the records of several photos."""
    emb = torch.cat([r["emb"] for r in records]) if records else torch.zeros(0, 0)
    photo = torch.tensor([i for i, r in enumerate(records) for _ in r["ids"]], dtype=torch.long)
    keys = [k if gated else () for r in records for k in r["keys"]]
    ids = np.concatenate([r["ids"] for r in records]) if records else np.zeros(0, dtype=int)
    return emb @ emb.T, photo, keys, ids


def _record_links(records: list[dict], gated: bool) -> list[tuple[float, bool]]:
    """Every link the linker proposes between two photos of a set, as at inference (:func:`eomt.reid.pairwise_matches`:
    Hungarian per photo pair, within each gate group when ``gated``), as ``(similarity, same instance)``; a link with a
    detection of unknown id is left out."""
    from ..reid import pairwise_matches

    sim, photo, keys, ids = _flatten_records(records, gated)
    if len(ids) < 2:
        return []
    out = []
    for p in pairwise_matches(sim, photo, keys, sim_thres=math.inf):
        a, b = int(ids[p["a"]]), int(ids[p["b"]])
        if min(a, b) >= -1:  # -2: unknown whether the two are one instance
            out.append((p["similarity"], a >= 0 and a == b))
    return out


def _prf(tp: int, n: int, n_true: int) -> tuple[float, float, float]:
    """Precision, recall and F1 of ``n`` kept links of which ``tp`` are right, ``n_true`` to find."""
    p, r = (tp / n if n else 0.0), tp / n_true
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _link_f1(pairs: list[tuple[float, bool]], n_true: int) -> dict[str, float]:
    """Precision / recall / F1 of the links at the similarity threshold of the best F1 (``thres``): a link is kept
    when its similarity reaches the threshold; ``n_true`` counts the instances shown by two photos (the links to find)."""
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "thres": float("nan")}
    if not len(pairs) or not n_true:
        return best
    sim = np.array([p[0] for p in pairs], dtype=np.float64)
    ok = np.array([p[1] for p in pairs], dtype=bool)
    for t in np.linspace(-1.0, 1.0, 201):
        sel = sim >= t
        p, r, f = _prf(int((sel & ok).sum()), int(sel.sum()), n_true)
        if f > best["f1"]:
            best = {"f1": f, "precision": p, "recall": r, "thres": float(t)}
    return best


def _f1_at(pairs, n_true: int, thres) -> float:
    """F1 of the links kept at a given similarity threshold (NaN without a threshold or links to find)."""
    if thres is None or not math.isfinite(thres) or not n_true:
        return float("nan")
    sim = np.array([p[0] for p in pairs], dtype=np.float64)
    ok = np.array([p[1] for p in pairs], dtype=bool)
    sel = sim >= thres
    return _prf(int((sel & ok).sum()), int(sel.sum()), n_true)[2]


def _bcubed(identity: np.ndarray, ids: np.ndarray) -> tuple[float, float, int]:
    """B-cubed precision and recall, summed over the detections of annotated instances (``ids >= 0``): a detection's
    identity counts the detections of known status (``ids >= -1``) it holds, its instance the detections of that
    instance. A false detection merged into an identity costs precision; an instance split in two costs recall."""
    known = ids >= -1
    p_sum = r_sum = 0.0
    items = np.nonzero(ids >= 0)[0]
    for d in items:
        c = known & (identity == identity[d])
        g = ids == ids[d]
        both = float((c & g).sum())
        p_sum += both / float(c.sum())
        r_sum += both / float(g.sum())
    return p_sum, r_sum, len(items)


def _cluster_counts(records: list[dict], thres: float, gated: bool) -> tuple[float, float, int]:
    """Link the detections of every photo of one subject as :func:`eomt.engine.predict.predict_subject` does
    (``identities``: Hungarian per photo pair at ``thres``, union-find with its merge guard) and count the B-cubed
    sums (:func:`_bcubed`) of the identities."""
    from ..reid import cluster, gate_mask, pairwise_matches

    sim, photo, keys, ids = _flatten_records(records, gated)
    if not len(ids) or thres is None or not math.isfinite(thres):
        return 0.0, 0.0, 0
    pairs = pairwise_matches(sim, photo, keys, sim_thres=thres)
    identity, _ = cluster(pairs, sim, photo, num_instances=len(ids), sim_thres=thres, gate=gate_mask(photo, keys))
    return _bcubed(identity.numpy(), ids)


def _coco_gt(images: list[dict], anns: list[dict], names: dict, ignore=None):
    """A COCO ground truth of ``images`` / ``anns``; the annotations ``ignore(a)`` holds for become crowd regions
    (detections on them are neither true nor false), to score a subset of the instances."""
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = {"images": images,
                    "annotations": [{**{k: v for k, v in a.items() if k != "linked"}, "id": i + 1,
                                     "iscrowd": int(bool(ignore and ignore(a)))} for i, a in enumerate(anns)],
                    "categories": [{"id": c + 1, "name": n} for c, n in names.items()]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def set_metric_keys(prefix: str, specs, *, gated: bool = False, groups: dict | None = None) -> list[str]:
    """The keys :func:`evaluate_sets` returns (the ``metrics.csv`` columns of a set validation, without ``val/``)."""
    keys = [f"segm/{k}_{m}" for k in ("mAP", "mAP50", "mAP_small", "AR100") for m in _MODES]
    keys += [f"segm/{k}_{sub}_{m}" for sub in ("linked", "unlinked") for k in ("mAP", "AR100") for m in _MODES]
    keys += [f"{n}_{m}" for s in specs for n in attr_metric_names(s) for m in _MODES]
    keys += [f"{src}{v}/{n}" for src in LINK_SOURCES for v in ("", "_gated")[: 1 + gated] for n in _LINK_STATS]
    keys += ["match/f1_stored"]
    keys += [f"{src}/cluster_{n}" for src in LINK_SOURCES for n in ("f1", "precision", "recall")]
    for field, values in (groups or {}).items():
        for v in values:
            keys += [f"{field}={v}/segm/mAP_{m}" for m in _MODES] + [f"{field}={v}/{src}/f1" for src in LINK_SOURCES]
    return [f"{prefix}/{k}" for k in keys]


def set_group_values(sets_ds, group_by) -> dict[str, list[str]]:
    """The values each ``group_by`` image field takes over the first photos of the sets of ``sets_ds``."""
    return group_values(sets_ds.coco, sets_ds.first_photos(), group_by)


@torch.no_grad()
def evaluate_sets(
    model,
    sets_ds,
    *,
    device,
    num_workers: int = 4,
    amp: bool = False,
    max_det: int = 100,
    link_conf: float = 0.3,
    iou_gate: float = 0.5,
    verbose: bool = True,
    prefix: str = "sets",
    link_group_by=None,
    thres: float | None = None,
    group_by: str | list[str] | None = None,
) -> dict[str, float]:
    """Score a model on fixed sets of photos of one subject (:class:`~eomt.data.subjects.CocoSubjectSets`, unaugmented),
    each photo predicted alone and with its set.

    Every fully annotated photo of a set is scored (those whose extras' ``neg`` is below 1 are not), and the detections
    lying mostly in a photo's ignore region are dropped first; a detection that is no annotated instance in a photo not
    fully annotated, or in an ignore region, takes no part in the links (whether it is an instance is unknown). The GT is
    each photo's own masks at the input resolution (so the size buckets are those of the input size), binarized as the
    loss does (:func:`binarize_gt`). The keys start with ``prefix`` (:func:`set_metric_keys` lists them):

    * ``segm/<metric>_{alone,set}``: COCO segm of the scored photos, predicted alone and together with the rest of the
      set. A model without cross-view sublayers predicts each photo alone either way.
    * ``segm/{mAP,AR100}_linked_{alone,set}``: the same restricted to the **linked** instances, those with a known id
      that another photo of the set shows (where the other photos can help), the others left out as crowd regions;
      ``_unlinked_`` the rest.
    * ``aux_acc/<head>_{alone,set}`` (``aux_mae`` and the others of :func:`attr_metric_names` for a regression head): the
      scored photos' attributes on their GT instances matched to a same-class detection (mask IoU >= ``iou_gate``).
    * ``<source>/{f1,precision,recall,thres}`` for each of :data:`LINK_SOURCES`: the links between the photos of each set
      (Hungarian per photo pair, as at inference) of the detections scoring at least ``link_conf``, at the similarity
      threshold of the best F1 (``thres``). With ``link_group_by`` (gate keys: ``"class"`` and categorical heads) also
      ``<source>_gated/*``: the links only within each gate group.
    * ``match/f1_stored``: F1 at the given ``thres`` (e.g. the checkpoint's stored one) instead of the best one of this
      validation (an optimistic number); gated when ``link_group_by`` is set, as the links are then made.
    * ``<source>/cluster_{f1,precision,recall}``: the identities over each whole subject: the first photo of every set
      (each predicted with its own set) linked with all the others of its subject and clustered as
      ``predict_subject(identities=True)`` does, at ``thres`` (else this validation's best), scored B-cubed
      (:func:`_bcubed`): a transitive error costs more than a pair.
    * ``group_by`` (image fields): ``<field>=<value>/segm/mAP_{alone,set}`` and ``<field>=<value>/<source>/f1`` (at the
      overall best threshold) per value of the field of each set's first photo.
    """
    from pycocotools import mask as mask_utils

    from ..reid import validate_group_by

    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    size = sets_ds.imgsz
    specs = _check_attr_vocab(model, sets_ds)
    gate_keys = validate_group_by(link_group_by, getattr(model, "aux_specs", []))
    variants = (False, True) if gate_keys else (False,)
    deployed = bool(gate_keys)  # the variant the stored threshold, the clusters and the groups use
    # a class-scoped head says -1 ("not applicable") outside its classes, as predict_set does: the gate groups of
    # the links are those of inference
    aux_scopes = {s.name: s.applies_to for s in getattr(model, "aux_specs", []) if s.applies_to is not None}
    cross_view = getattr(getattr(model, "eomt", None), "xview", None) is not None
    fields = _fields(group_by)
    indexed = _Indexed(sets_ds)
    mine = shard_groups(indexed, [sets_ds.subject_of_set(i) for i in range(len(sets_ds))])
    loader = DataLoader(mine, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=_only,
                        timeout=LOADER_TIMEOUT if num_workers > 0 else 0)
    max_views = sets_ds.sets.max_views
    images, anns, img_group = [], [], {}
    dets: dict[str, list] = {m: [] for m in _MODES}
    hits = {m: {s.name: 0 for s in specs if not s.regression} for m in _MODES}
    tot = {m: {s.name: 0 for s in specs if not s.regression} for m in _MODES}
    reg = {m: {s.name: ([], []) for s in specs if s.regression} for m in _MODES}
    links = {(src, g): [] for src in LINK_SOURCES for g in variants}  # (similarity, same instance, set's groups)
    n_true: dict = {}  # set's groups -> instances shown by two photos (the links to find)
    anchors: dict = {}  # subject -> {source: [record of the first photo of each of its sets]}
    for idx, item in progress(loader, desc="val-sets", unit="set", leave=False, disable=not verbose):
        views = item["views"]
        pv = torch.stack([v[0] for v in views]).to(device)
        gt_masks = [binarize_gt(v[1].to(device)) for v in views]
        gt_ids = [v[4] for v in views]
        extras = item["extras"]
        first = item["photos"][0]
        grp = tuple((f, _field_value(sets_ds.coco, first, f)) for f in fields)
        shown = [{i for i in ids.tolist() if i >= 0} for ids in gt_ids]  # linked instances only
        n_true[grp] = n_true.get(grp, 0) + sum(len(shown[a] & shown[b]) for a in range(len(views))
                                               for b in range(a + 1, len(views)))
        # the scored views (every fully annotated one) and their GT
        gt_of = {}
        for k in (k for k in range(len(views)) if extras[k].get("neg", 1.0) >= 1.0):
            image_id = idx * max_views + k
            keep = gt_masks[k].flatten(1).any(1).cpu()
            others = set().union(*(shown[j] for j in range(len(views)) if j != k))
            a_ids = gt_ids[k][keep]
            gt_of[k] = (image_id, gt_masks[k][keep.to(device)], views[k][2][keep],
                        {n: t[keep] for n, t in views[k][3].items()})
            images.append({"id": image_id, "width": size, "height": size})
            img_group[image_id] = grp
            for g, rle in enumerate(_encode_mask(m) for m in gt_of[k][1].cpu().numpy()):
                anns.append({"image_id": image_id, "category_id": int(gt_of[k][2][g]) + 1, "segmentation": rle,
                             "area": float(mask_utils.area(rle)), "bbox": [float(x) for x in mask_utils.toBbox(rle)],
                             "linked": int(a_ids[g]) >= 0 and int(a_ids[g]) in others})
        records = {src: [] for src in LINK_SOURCES}
        out = None
        for m in _MODES:  # one forward at a time, each photo's masks dropped once used (validation memory)
            if out is None or cross_view:  # without cross-view sublayers a photo's prediction is the same in its set
                with _autocast(device, amp):
                    out = model(pv) if m == "alone" else model(pv, set_ids=torch.zeros(len(views), dtype=torch.long))
            embeds = ({"reid": out["query_embed"]} if m == "alone"
                      else {"match": out.get("match_embed", out["query_embed"])})
            for k in range(len(views)):
                r = postprocess_instance(slice_batch(out, k), 0.0, (size, size), max_det=max_det, aux_scopes=aux_scopes)
                # the detections that may be an instance nobody annotated: all of a photo not fully annotated, else
                # those lying in its ignore region
                unsure = (torch.ones(r["num_detections"], dtype=torch.bool) if extras[k].get("neg", 1.0) < 1.0
                          else _in_ignore(r["masks"], extras[k].get("ignore")))
                if k in gt_of:
                    image_id, a_masks, a_cls, a_attrs = gt_of[k]
                    classes, scores = r["classes"].tolist(), r["scores"].tolist()
                    masks = r["masks"].cpu().numpy()
                    for i in (~unsure).nonzero().flatten().tolist():  # detections in an ignore region do not count
                        dets[m].append({"image_id": image_id, "category_id": classes[i] + 1, "score": scores[i],
                                        "segmentation": _encode_mask(masks[i])})
                    iou = _mask_iou(r["masks"], a_masks)
                    iou[np.array(classes)[:, None] != a_cls.numpy()[None]] = 0.0
                    aux_ids = {s.name: r["aux"][s.name]["ids"].tolist() for s in specs if not s.regression}
                    aux_vals = {s.name: r["aux_values"][s.name].tolist() for s in specs if s.regression}
                    for d, g in _greedy_pairs(iou, iou_gate):
                        for s in specs:
                            label = a_attrs[s.name][g]
                            if s.regression and not torch.isnan(label):
                                reg[m][s.name][0].append(float(aux_vals[s.name][d]))
                                reg[m][s.name][1].append(float(label))
                            elif not s.regression and int(label) != -100:
                                hits[m][s.name] += aux_ids[s.name][d] == int(label)
                                tot[m][s.name] += 1
                sel = (r["scores"] >= link_conf).nonzero().flatten()
                d_ids = _det_ids(r, sel, gt_masks[k], gt_ids[k], iou_gate, unsure)
                for src, e in embeds.items():
                    records[src].append(_link_record(r, e[k], sel, d_ids, gate_keys))
                del r
            del embeds
        del out
        for src, recs in records.items():
            for g in variants:
                links[(src, g)] += [(sim, ok, grp) for sim, ok in _record_links(recs, g)]
            anchors.setdefault(sets_ds.subject_of_set(idx), {}).setdefault(src, []).append(recs[0])

    parts = gather((images, anns, dets, hits, tot, reg, links, n_true, img_group))
    metrics: dict[str, float] = {}
    cluster_thres = None
    if parts is not None:  # the main process scores the sets of all
        metrics, cluster_thres = _score_sets(sets_ds, parts, specs, prefix, variants, deployed, thres, fields)
    cluster_thres = broadcast(cluster_thres)  # every process clusters its own subjects, at the main process's thresholds
    counts = {src: [0.0, 0.0, 0] for src in LINK_SOURCES}
    for by_src in anchors.values():
        for src, recs in by_src.items():
            p, r, n = _cluster_counts(recs, cluster_thres[src], deployed)
            counts[src] = [counts[src][0] + p, counts[src][1] + r, counts[src][2] + n]
    counts = sum_over_processes(counts)
    if counts is not None:
        for src, (p_sum, r_sum, n) in counts.items():
            p, r = (p_sum / n, r_sum / n) if n else (math.nan, math.nan)
            f = 2 * p * r / (p + r) if n and p + r else (math.nan if not n else 0.0)
            metrics.update({f"{prefix}/{src}/cluster_f1": f, f"{prefix}/{src}/cluster_precision": p,
                            f"{prefix}/{src}/cluster_recall": r})
    return metrics if parts is not None else {}


def _score_sets(sets_ds, parts, specs, prefix, variants, deployed, thres, fields):
    """The metrics of :func:`evaluate_sets` from the gathered results of every process, and the thresholds the subjects
    are clustered at (the given ``thres`` for ``match``, else this validation's best)."""
    images = [x for p in parts for x in p[0]]
    anns = [x for p in parts for x in p[1]]
    dets = {m: [x for p in parts for x in p[2][m]] for m in _MODES}
    links = {key: [x for p in parts for x in p[6][key]] for key in parts[0][6]}
    n_true: dict = {}
    for p in parts:
        for grp, n in p[7].items():
            n_true[grp] = n_true.get(grp, 0) + n
    n_all = sum(n_true.values())
    group_of = {im_id: grp for p in parts for im_id, grp in p[8].items()}
    ids = [i["id"] for i in images]
    names = sets_ds.names if isinstance(sets_ds.names, dict) else dict(enumerate(sets_ds.names))
    coco = _coco_gt(images, anns, names)
    subsets = {"linked": _coco_gt(images, anns, names, ignore=lambda a: not a["linked"]),
               "unlinked": _coco_gt(images, anns, names, ignore=lambda a: a["linked"])}
    metrics: dict[str, float] = {}
    for m in _MODES:
        # copies: ``loadRes`` writes ``bbox`` / ``area`` into the result dicts, which breaks a second evaluation of them
        ev = _cocoeval(coco, [dict(d) for d in dets[m]], ids, "segm", False)
        metrics.update({f"{prefix}/segm/{k}_{m}": ev[f"segm/{k}"] for k in ("mAP", "mAP50", "mAP_small", "AR100")})
        for sub, sub_coco in subsets.items():
            ev = _cocoeval(sub_coco, [dict(d) for d in dets[m]], ids, "segm", False)
            metrics.update({f"{prefix}/segm/{k}_{sub}_{m}": ev[f"segm/{k}"] if ev[f"segm/{k}"] >= 0 else math.nan
                            for k in ("mAP", "AR100")})
        for s in specs:
            if s.regression:
                got = regression_metrics(s, [x for p in parts for x in p[5][m][s.name][0]],
                                         [x for p in parts for x in p[5][m][s.name][1]])
                metrics.update({f"{prefix}/{k}_{m}": v for k, v in got.items()})
            else:
                hit, tot = sum(p[3][m][s.name] for p in parts), sum(p[4][m][s.name] for p in parts)
                metrics[f"{prefix}/aux_acc/{s.name}_{m}"] = hit / tot if tot else float("nan")
    best = {}
    for (src, gated), pairs in links.items():
        res = _link_f1(pairs, n_all)
        metrics.update({f"{prefix}/{src}{'_gated' if gated else ''}/{n}": v for n, v in res.items()})
        if gated == deployed:
            best[src] = res["thres"]
    metrics[f"{prefix}/match/f1_stored"] = _f1_at(links[("match", deployed)], n_all, thres)
    for f in fields:  # per value of each field of the sets' first photos
        for v in sorted({dict(g).get(f) for g in n_true} - {None}):
            in_group = lambda g: dict(g).get(f) == v  # noqa: E731
            img_ids = [i for i in ids if in_group(group_of.get(i, ()))]
            wanted = set(img_ids)
            for m in _MODES:
                got = _cocoeval(coco, [dict(d) for d in dets[m] if d["image_id"] in wanted], img_ids, "segm", False)
                metrics[f"{prefix}/{f}={v}/segm/mAP_{m}"] = got["segm/mAP"]
            n = sum(c for g, c in n_true.items() if in_group(g))
            for src in LINK_SOURCES:
                pairs = [x for x in links[(src, deployed)] if in_group(x[2])]
                metrics[f"{prefix}/{f}={v}/{src}/f1"] = _f1_at(pairs, n, best[src])
    stored = thres if thres is not None and math.isfinite(thres) else None
    cluster_thres = {"match": stored if stored is not None else best["match"], "reid": best["reid"]}
    return metrics, cluster_thres
