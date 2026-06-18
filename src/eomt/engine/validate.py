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
    amp: bool = False,
    also_bbox: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate ``model`` on ``val_ds`` and return a COCO metrics dict."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_val,
        pin_memory=True,
    )
    contig2cat = val_ds.contig2cat

    segm_results: list[dict] = []
    bbox_results: list[dict] = []
    use_amp = amp and device.type == "cuda"

    for pixel_values, image_ids, sizes in tqdm(
        loader, desc="val", unit="batch", leave=False, disable=not verbose
    ):
        pixel_values = pixel_values.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(pixel_values)
        mql = out["masks_queries_logits"]
        cql = out["class_queries_logits"]

        for b, (img_id, (orig_w, orig_h)) in enumerate(zip(image_ids, sizes)):
            res = postprocess_instance(
                {
                    "masks_queries_logits": mql[b : b + 1],
                    "class_queries_logits": cql[b : b + 1],
                },
                conf_thres,
                (orig_w, orig_h),
                max_det=max_det,
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
