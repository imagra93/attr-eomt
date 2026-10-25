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
from ..postprocess import postprocess_detection, postprocess_instance, slice_batch
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
    for d, g in _greedy_pairs(iou, iou_gate):
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
    a, b = a.flatten(1).float(), b.flatten(1).float()
    inter = a @ b.T
    return (inter / (a.sum(1)[:, None] + b.sum(1)[None] - inter).clamp_min(1)).cpu().numpy()


def _set_links(results, embeds, gt_masks, gt_ids, conf: float, gate: float) -> list[tuple[float, bool]]:
    """The links a set's photos get: between every two photos, the detections (score >= ``conf``) matched by the
    Hungarian algorithm on the cosine similarity of ``embeds``, as ``(similarity, same instance)``; a detection is an
    instance when its mask overlaps that instance's (IoU >= ``gate``, one to one)."""
    from scipy.optimize import linear_sum_assignment

    ids, feats = [], []
    for r, e, gm, gi in zip(results, embeds, gt_masks, gt_ids):
        sel = (r["scores"] >= conf).nonzero().flatten()
        d_ids = np.full(len(sel), -1)
        for d, g in _greedy_pairs(_mask_iou(r["masks"][sel.to(r["masks"].device)], gm), gate):
            d_ids[d] = int(gi[g])
        ids.append(d_ids)
        feats.append(torch.nn.functional.normalize(e[r["query_idx"][sel]].float(), dim=-1))
    pairs = []
    for a in range(len(results)):
        for b in range(a + 1, len(results)):
            if len(ids[a]) and len(ids[b]):
                sim = (feats[a] @ feats[b].T).cpu().numpy()
                for i, j in zip(*linear_sum_assignment(-sim)):
                    pairs.append((float(sim[i, j]), bool(ids[a][i] >= 0 and ids[a][i] == ids[b][j])))
    return pairs


def _link_f1(pairs: list[tuple[float, bool]], n_true: int) -> dict[str, float]:
    """Precision / recall / F1 of the links at the similarity threshold of the best F1 (``thres``): a link is kept
    when its similarity reaches the threshold; ``n_true`` counts the instances shown by two photos (the links to find)."""
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "thres": float("nan")}
    if not pairs or not n_true:
        return best
    sim = np.array([p[0] for p in pairs])
    ok = np.array([p[1] for p in pairs])
    for t in np.linspace(-1.0, 1.0, 201):
        sel = sim >= t
        tp, n = int((sel & ok).sum()), int(sel.sum())
        p, r = (tp / n if n else 0.0), tp / n_true
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best["f1"]:
            best = {"f1": f, "precision": p, "recall": r, "thres": float(t)}
    return best


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
) -> dict[str, float]:
    """Score a cross-view model on fixed synthetic sets (:class:`~eomt.data.sets.CocoViewSets` with a ``seed``): there
    are no real multi-photo annotations, so this measures what the sets were built to teach, not real viewpoints.

    * ``sets/segm/<metric>_{alone,set}`` and ``sets/<case>/mAP_{alone,set}``: COCO segm of each set's hard view (the
      first photo), predicted alone and together with the rest of its set, over all sets and per case; the GT is the
      view's own masks at the input resolution (so the size buckets are those of 644 px images).
    * ``sets/aux_acc/<head>_{alone,set}`` (``aux_mae`` for a regression head): the hard view's attributes on its GT
      instances matched to a same-class detection (mask IoU >= ``iou_gate``).
    * ``sets/match/{f1,precision,recall,thres}``: links between the photos of each set from the matching head (the
      model's ``match_embed``, else ``query_embed``, in the joint forward), against ``sets/reid/*``: the same from the
      per-query embedding of the photos predicted alone (:mod:`eomt.reid`'s fingerprint). Detections scoring at least
      ``link_conf`` are linked; ``thres`` is the similarity threshold of the best F1.
    """
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO

    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    use_amp = amp and device.type == "cuda"
    size = sets_ds.imgsz
    specs = _check_attr_vocab(model, sets_ds)
    loader = DataLoader(shard(_Indexed(sets_ds)), batch_size=1, shuffle=False, num_workers=num_workers,
                        collate_fn=_only)
    images, anns, cases = [], [], {}
    dets: dict[str, list] = {"alone": [], "set": []}
    hits = {m: {s.name: 0.0 for s in specs} for m in dets}
    tot = {m: {s.name: 0 for s in specs} for m in dets}
    links: dict[str, list] = {"match": [], "reid": []}
    n_true = 0
    for idx, item in tqdm(loader, desc="val-sets", unit="set", leave=False, disable=not verbose):
        views = item["views"]
        pv = torch.stack([v[0] for v in views]).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outs = {"alone": model(pv), "set": model(pv, set_ids=torch.zeros(len(views), dtype=torch.long))}
        res = {m: [postprocess_instance(slice_batch(o, b), 0.0, (size, size), max_det=max_det)
                   for b in range(len(views))] for m, o in outs.items()}
        gt_masks = [v[1].to(device) > 0.5 for v in views]
        gt_ids = [v[4] for v in views]

        # the hard view: COCO results and attributes, alone and with its set
        keep = gt_masks[0].flatten(1).any(1).cpu()
        a_masks, a_cls = gt_masks[0][keep.to(device)], views[0][2][keep]
        a_attrs = {k: t[keep] for k, t in views[0][3].items()}
        images.append({"id": idx, "width": size, "height": size})
        cases.setdefault(item["case"], []).append(idx)
        for g in range(len(a_masks)):
            rle = _encode_mask(a_masks[g].cpu().numpy())
            anns.append({"image_id": idx, "category_id": int(a_cls[g]) + 1, "segmentation": rle,
                         "area": float(mask_utils.area(rle)), "bbox": [float(x) for x in mask_utils.toBbox(rle)],
                         "iscrowd": 0})
        for m in dets:
            r = res[m][0]
            for i in range(r["num_detections"]):
                dets[m].append({"image_id": idx, "category_id": int(r["classes"][i]) + 1, "score": float(r["scores"][i]),
                                "segmentation": _encode_mask(r["masks"][i].cpu().numpy())})
            iou = _mask_iou(r["masks"], a_masks)
            iou[r["classes"].cpu().numpy()[:, None] != a_cls.numpy()[None]] = 0.0
            for d, g in _greedy_pairs(iou, iou_gate):
                for s in specs:
                    label = a_attrs[s.name][g]
                    if s.regression and not torch.isnan(label):
                        hits[m][s.name] += abs(float(r["aux_values"][s.name][d]) - float(label))
                        tot[m][s.name] += 1
                    elif not s.regression and int(label) != -100:
                        hits[m][s.name] += int(r["aux"][s.name]["ids"][d]) == int(label)
                        tot[m][s.name] += 1

        # cross-photo links over every photo of the set
        shown = [set(i.tolist()) for i in gt_ids]
        n_true += sum(len(shown[a] & shown[b]) for a in range(len(views)) for b in range(a + 1, len(views)))
        set_embed = outs["set"].get("match_embed", outs["set"]["query_embed"])
        links["match"] += _set_links(res["set"], set_embed, gt_masks, gt_ids, link_conf, iou_gate)
        links["reid"] += _set_links(res["alone"], outs["alone"]["query_embed"], gt_masks, gt_ids, link_conf, iou_gate)

    parts = gather((images, anns, cases, dets, hits, tot, links, n_true))
    if parts is None:  # several GPUs: the main process scores the sets of all
        return {}
    images = [x for p in parts for x in p[0]]
    anns = [{**a, "id": i + 1} for i, a in enumerate(x for p in parts for x in p[1])]
    cases = {c: [i for p in parts for i in p[2].get(c, [])] for c in {c for p in parts for c in p[2]}}
    dets = {m: [x for p in parts for x in p[3][m]] for m in dets}
    hits = {m: {n: sum(p[4][m][n] for p in parts) for n in hits[m]} for m in hits}
    tot = {m: {n: sum(p[5][m][n] for p in parts) for n in tot[m]} for m in tot}
    links = {k: [x for p in parts for x in p[6][k]] for k in links}
    n_true = sum(p[7] for p in parts)

    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns,
                    "categories": [{"id": c + 1, "name": n} for c, n in sets_ds.names.items()]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    ids = [i["id"] for i in images]
    metrics: dict[str, float] = {}
    for m in dets:
        # copies: ``loadRes`` writes ``bbox`` / ``area`` into the result dicts, which breaks a second evaluation of them
        ev = _cocoeval(coco, [dict(d) for d in dets[m]], ids, "segm", False)
        metrics.update({f"sets/segm/{k}_{m}": ev[f"segm/{k}"] for k in ("mAP", "mAP50", "mAP_small", "AR100")})
        for case, case_ids in sorted(cases.items()):
            wanted = set(case_ids)
            case_dets = [dict(d) for d in dets[m] if d["image_id"] in wanted]
            metrics[f"sets/{case}/mAP_{m}"] = _cocoeval(coco, case_dets, case_ids, "segm", False)["segm/mAP"]
        for s in specs:
            metrics[f"sets/{_attr_metric(s)}/{s.name}_{m}"] = hits[m][s.name] / tot[m][s.name] if tot[m][s.name] else float("nan")
    for k, pairs in links.items():
        metrics.update({f"sets/{k}/{n}": v for n, v in _link_f1(pairs, n_true).items()})
    return metrics
