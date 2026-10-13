"""COCO instance-segmentation validation (pycocotools ``COCOeval``).

Runs the model over a :class:`~eomt.data.coco.CocoValImages` set, converts each
prediction to a COCO result (RLE mask + score + original category id) and scores
it with ``COCOeval(iouType='segm')``. Optionally also reports bbox mAP using the
mask-extent boxes.

On several GPUs (:mod:`eomt.engine.distributed`) each process runs its share of the
images and the main process scores (or counts) them all; the others get ``{}``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import collate_val
from ..postprocess import postprocess_detection, postprocess_instance
from ..tta import TTAConfig, predict_views
from .distributed import gather, shard, sum_over_processes

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



def _attr_metric(spec) -> str:
    """Key prefix of a head's metric (as the trainer logs it): accuracy, or mean absolute error in real units."""
    return "aux_mae" if spec.regression else "aux_acc"


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


def _count_attrs(res, rles, val_ds, img_id, specs, iou_gate, hits, tot) -> None:
    """Add one image's attribute hits to ``hits`` / ``tot``: each GT instance is matched to the kept detection of its
    class with the highest mask IoU (one to one, IoU >= ``iou_gate``) and every head is scored on it (accuracy; for a
    regression head the absolute error in real units). Labels follow the trainer's (``_attr_label``: missing,
    out-of-vocabulary and out-of-scope values are ignored)."""
    from pycocotools import mask as mask_utils

    from ..data.coco import _attr_label

    gts = [a for a in val_ds.coco.imgToAnns.get(img_id, []) if not a.get("iscrowd")]
    if not gts or not rles:
        return
    iou = np.asarray(mask_utils.iou(rles, [val_ds.coco.annToRLE(a) for a in gts], [0] * len(gts)))
    gt_cls = np.array([val_ds.cat2contig[a["category_id"]] for a in gts])
    iou[res["classes"].cpu().numpy()[:, None] != gt_cls[None, :]] = 0.0
    taken_d: set[int] = set()
    taken_g: set[int] = set()
    for flat in np.argsort(-iou, axis=None):
        d, g = divmod(int(flat), len(gts))
        if iou[d, g] < iou_gate:
            break
        if d in taken_d or g in taken_g:
            continue
        taken_d.add(d)
        taken_g.add(g)
        for s in specs:
            label = _attr_label(s, val_ds._attr_id_maps, gts[g].get("attributes") or {}, int(gt_cls[g]))
            if s.regression:
                if label == label:  # not NaN
                    hits[s.name] += abs(float(res["aux_values"][s.name][d]) - label)
                    tot[s.name] += 1
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
) -> dict[str, float]:
    """Evaluate ``model`` on ``val_ds`` and return a COCO metrics dict.

    ``augment`` / ``tiles`` (see :mod:`eomt.tta`) evaluate with test-time augmentation / tiled inference: each image is
    predicted alone from its views, at ``val_ds.imgsz`` with its letterbox and normalisation.

    ``attributes`` adds the secondary heads' metrics, scored on the same predictions as the mAP (so also under
    ``augment`` / ``tiles``): ``aux_acc/<head>`` accuracy, or ``aux_mae/<head>`` mean absolute error in real units for a
    regression head, over the GT instances matched to a same-class detection with mask IoU >= ``attr_iou``. That
    population is the deployed one (kept detections), not the trainer's matched queries, so the numbers are close to the
    ``val/aux_*`` columns of ``metrics.csv`` but not equal. ``val_ds`` needs the attribute definitions (it builds them
    from its JSON) and the model the same heads and categories.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    use_amp = amp and device.type == "cuda"
    contig2cat = val_ds.contig2cat
    cfg = TTAConfig.resolve(augment, tiles)
    if cfg is not None and not getattr(val_ds, "raw", False):  # the views are cut from the original image
        val_ds = copy.copy(val_ds)
        val_ds.raw = True

    loader = DataLoader(
        shard(val_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_val,
        pin_memory=True,
    )

    segm_results: list[dict] = []
    bbox_results: list[dict] = []
    attr_specs = _check_attr_vocab(model, val_ds) if attributes else []
    if attr_specs and cfg is not None and cfg.flip:
        _adopt_flip_aware(model, val_ds)
    attr_hits = {s.name: 0.0 for s in attr_specs}
    attr_tot = {s.name: 0 for s in attr_specs}

    for pixel_values, image_ids, sizes, metas in tqdm(
        loader, desc="val", unit="batch", leave=False, disable=not verbose
    ):
        if cfg is not None:
            batch_results = [
                predict_views(
                    model, image, cfg, imgsz=val_ds.imgsz, conf_thres=conf_thres, max_det=max_det,
                    mask_thresh=mask_thresh, min_mask_area=min_mask_area, letterbox=val_ds.letterbox,
                    mean=val_ds.mean, std=val_ds.std, amp_dtype=torch.float16 if use_amp else None,
                )
                for image in pixel_values
            ]
        else:
            pixel_values = pixel_values.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
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
            n = res["num_detections"]
            rles = []
            for i in range(n):
                cat_id = int(contig2cat[int(res["classes"][i])])
                score = float(res["scores"][i])
                mask_np = res["masks"][i].cpu().numpy()
                rles.append(_encode_mask(mask_np))
                segm_results.append(
                    {
                        "image_id": int(img_id),
                        "category_id": cat_id,
                        "segmentation": rles[-1],
                        "score": score,
                    }
                )
                if also_bbox:
                    x1, y1, x2, y2 = [float(v) for v in res["boxes"][i].tolist()]
                    bbox_results.append(
                        {
                            "image_id": int(img_id),
                            "category_id": cat_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        }
                    )
            if attr_specs:
                _count_attrs(res, rles, val_ds, int(img_id), attr_specs, attr_iou, attr_hits, attr_tot)

    attr_counts = sum_over_processes((attr_hits, attr_tot)) if attr_specs else None
    parts = gather((segm_results, bbox_results))
    if parts is None:  # several GPUs: the main process scores the results of all
        return {}
    segm_results = [r for segm, _ in parts for r in segm]
    bbox_results = [r for _, bbox in parts for r in bbox]
    metrics = _cocoeval(val_ds.coco, segm_results, val_ds.ids, "segm", verbose)
    if also_bbox:
        metrics.update(_cocoeval(val_ds.coco, bbox_results, val_ds.ids, "bbox", verbose))
    if attr_specs:
        hits, tot = attr_counts
        metrics.update({f"{_attr_metric(s)}/{s.name}": hits[s.name] / tot[s.name] if tot[s.name] else float("nan")
                        for s in attr_specs})
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
) -> dict[str, float]:
    """Evaluate a ``family="detect"`` model on ``val_ds`` -> COCO **bbox** metrics.

    Mirrors :func:`evaluate` but reads the box head (``pred_boxes``) via
    :func:`~eomt.postprocess.postprocess_detection` and scores only ``bbox`` mAP
    (there are no masks). Returns a metrics dict keyed ``bbox/<metric>``.
    """
    if TTAConfig.resolve(augment, tiles) is not None:
        raise NotImplementedError("augment / tiles support the instance family only.")
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    use_amp = amp and device.type == "cuda"
    contig2cat = val_ds.contig2cat

    loader = DataLoader(
        shard(val_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_val,
        pin_memory=True,
    )

    bbox_results: list[dict] = []
    for pixel_values, image_ids, sizes, metas in tqdm(
        loader, desc="val", unit="batch", leave=False, disable=not verbose
    ):
        pixel_values = pixel_values.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
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
            for i in range(res["num_detections"]):
                cat_id = int(contig2cat[int(res["classes"][i])])
                x1, y1, x2, y2 = [float(v) for v in res["boxes"][i].tolist()]
                bbox_results.append(
                    {
                        "image_id": int(img_id),
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(res["scores"][i]),
                    }
                )

    parts = gather(bbox_results)
    if parts is None:  # several GPUs: the main process scores the results of all
        return {}
    return _cocoeval(val_ds.coco, [r for part in parts for r in part], val_ds.ids, "bbox", verbose)


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
    use_amp = amp and device.type == "cuda"
    contig2cat = val_ds.contig2cat

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_val,
        pin_memory=True,
    )

    # One result accumulator per grid point.
    segm_results: list[list[dict]] = [[] for _ in grid]
    bbox_results: list[list[dict]] = [[] for _ in grid]

    for pixel_values, image_ids, sizes, metas in tqdm(
        loader, desc="sweep", unit="batch", leave=False, disable=not verbose
    ):
        pixel_values = pixel_values.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
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
                for i in range(res["num_detections"]):
                    cat_id = int(contig2cat[int(res["classes"][i])])
                    score = float(res["scores"][i])
                    segm_results[gi].append(
                        {
                            "image_id": int(img_id),
                            "category_id": cat_id,
                            "segmentation": _encode_mask(res["masks"][i].cpu().numpy()),
                            "score": score,
                        }
                    )
                    if also_bbox:
                        x1, y1, x2, y2 = [float(v) for v in res["boxes"][i].tolist()]
                        bbox_results[gi].append(
                            {
                                "image_id": int(img_id),
                                "category_id": cat_id,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": score,
                            }
                        )

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
    units instead (ignoring NaN), and the per-primary pairs hold its summed absolute error in place of ``correct``.

    The scalar accuracy uses the same gate as training (``iou_gate`` + ``class_gate``)
    so it reflects the trained population. The per-primary diagnostic uses an
    IoU-only gate (no class gate) so primary classes the model often *misclassifies*
    still show up. Returns ``({name: accuracy}, {name: {primary_cls: (correct, total)}})``.
    """
    from ..aux_cls import aux_accuracy, aux_accuracy_by_primary, gate_indices, match_queries
    from ..data import collate_train

    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    loader = DataLoader(
        shard(val_seg_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_train,
        pin_memory=True,
    )
    hits = {s.name: 0 for s in val_seg_ds.aux_specs}
    tot = {s.name: 0 for s in val_seg_ds.aux_specs}
    per_class: dict[str, dict[int, list[int]]] = {s.name: {} for s in val_seg_ds.aux_specs}
    use_amp = amp and device.type == "cuda"

    for pixel_values, mask_labels, class_labels, aux_labels in tqdm(
        loader, desc="val-aux", unit="batch", leave=False, disable=not verbose
    ):
        pixel_values = pixel_values.to(device)
        mask_labels = [m.to(device) for m in mask_labels]
        class_labels = [c.to(device) for c in class_labels]
        aux_labels = {k: [t.to(device) for t in v] for k, v in aux_labels.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
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
    if counts is None:  # several GPUs: the main process reports the counts of all
        return {}, {}
    hits, tot, per_class = counts
    scalar = {n: (hits[n] / tot[n] if tot[n] else float("nan")) for n in hits}
    pc = {n: {c: (v[0], v[1]) for c, v in buckets.items()} for n, buckets in per_class.items()}
    return scalar, pc
