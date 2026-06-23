"""COCO instance-segmentation validation (pycocotools ``COCOeval``).

Runs the model over a :class:`~eomt.data.coco.CocoValImages` set, converts each
prediction to a COCO result (RLE mask + score + original category id) and scores
it with ``COCOeval(iouType='segm')``. Optionally also reports bbox mAP using the
mask-extent boxes.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import collate_val
from ..postprocess import postprocess_instance

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
) -> dict[str, float]:
    """Evaluate ``model`` on ``val_ds`` and return a COCO metrics dict."""
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

    segm_results: list[dict] = []
    bbox_results: list[dict] = []

    for pixel_values, image_ids, sizes, metas in tqdm(
        loader, desc="val", unit="batch", leave=False, disable=not verbose
    ):
        pixel_values = pixel_values.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(pixel_values)
        mql = out["masks_queries_logits"]
        cql = out["class_queries_logits"]

        for b, (img_id, (orig_w, orig_h), meta) in enumerate(zip(image_ids, sizes, metas)):
            res = postprocess_instance(
                {
                    "masks_queries_logits": mql[b : b + 1],
                    "class_queries_logits": cql[b : b + 1],
                },
                conf_thres,
                (orig_w, orig_h),
                max_det=max_det,
                mask_thresh=mask_thresh,
                min_mask_area=min_mask_area,
                preprocess_meta=meta,
            )
            n = res["num_detections"]
            for i in range(n):
                cat_id = int(contig2cat[int(res["classes"][i])])
                score = float(res["scores"][i])
                mask_np = res["masks"][i].cpu().numpy()
                segm_results.append(
                    {
                        "image_id": int(img_id),
                        "category_id": cat_id,
                        "segmentation": _encode_mask(mask_np),
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

    metrics = _cocoeval(val_ds.coco, segm_results, val_ds.ids, "segm", verbose)
    if also_bbox:
        metrics.update(_cocoeval(val_ds.coco, bbox_results, val_ds.ids, "bbox", verbose))
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
    mask-free, matches predictions to GT with EoMT's matcher, and reports top-1
    accuracy on matched queries (ignoring ``-100`` labels).

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
        val_seg_ds,
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

    scalar = {n: (hits[n] / tot[n] if tot[n] else float("nan")) for n in hits}
    pc = {n: {c: (v[0], v[1]) for c, v in buckets.items()} for n, buckets in per_class.items()}
    return scalar, pc
