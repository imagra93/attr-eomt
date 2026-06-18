"""Postprocessing for EoMT instance segmentation.

EoMT emits mask-classification output (Mask2Former-style): per-query class logits
``(Q, C+1)`` and per-query mask logits ``(Q, h, w)``. We convert that to the
instance contract — ``boxes`` / ``scores`` / ``classes`` / ``masks`` — by taking
the best non-background class per query, thresholding, upsampling the masks to the
original image, and deriving each box from its mask's extent (EoMT has no
box-regression head).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """Derive ``xyxy`` boxes (pixel coords) from boolean masks ``(N, H, W)``."""
    n = masks.shape[0]
    boxes = masks.new_zeros((n, 4), dtype=torch.float32)
    for i in range(n):
        ys, xs = torch.where(masks[i])
        if ys.numel() == 0:
            continue
        boxes[i, 0] = xs.min()
        boxes[i, 1] = ys.min()
        boxes[i, 2] = xs.max() + 1
        boxes[i, 3] = ys.max() + 1
    return boxes


def postprocess_instance(
    output: dict,
    conf_thres: float,
    original_size: tuple[int, int],
    *,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    **_: object,
) -> dict:
    """Convert raw EoMT output to the canonical instance-seg dict.

    Args:
        output: ``{"masks_queries_logits": (1, Q, h, w),
                   "class_queries_logits": (1, Q, C+1)}``.
        conf_thres: minimum class confidence to keep a query.
        original_size: ``(width, height)`` of the source image.
        max_det: cap on returned instances (highest scoring first).
        mask_thresh: sigmoid threshold for binarizing masks.

    Returns:
        ``{"num_detections", "boxes": (N,4), "scores": (N,), "classes": (N,),
        "masks": (N,H,W)}``.
    """
    mask_logits = output["masks_queries_logits"][0].float()  # (Q, h, w)
    class_logits = output["class_queries_logits"][0].float()  # (Q, C+1)

    # Last column is the "no-object" class — drop it.
    scores_all = class_logits.softmax(dim=-1)[:, :-1]  # (Q, C)
    scores, classes = scores_all.max(dim=-1)  # (Q,)

    keep = scores >= conf_thres
    orig_w, orig_h = original_size
    if keep.sum() == 0:
        return {
            "num_detections": 0,
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "classes": torch.zeros((0,), dtype=torch.long),
            "masks": torch.zeros((0, orig_h, orig_w), dtype=torch.bool),
        }

    scores, classes = scores[keep], classes[keep]
    mask_logits = mask_logits[keep]

    if scores.numel() > max_det:
        topk = scores.topk(max_det)
        scores, classes = topk.values, classes[topk.indices]
        mask_logits = mask_logits[topk.indices]

    masks = F.interpolate(
        mask_logits.unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )[0]
    masks = masks.sigmoid() > mask_thresh  # (N, H, W) bool

    boxes = boxes_from_masks(masks)
    return {
        "num_detections": int(scores.numel()),
        "boxes": boxes,
        "scores": scores,
        "classes": classes.long(),
        "masks": masks,
    }
