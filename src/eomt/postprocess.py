"""Postprocessing for EoMT instance segmentation.

EoMT emits mask-classification output (Mask2Former-style): per-query class logits
``(Q, C+1)`` and per-query mask logits ``(Q, h, w)``. We convert that to the
instance contract — ``boxes`` / ``scores`` / ``classes`` / ``masks`` — by taking
the best non-background class per query, weighting its confidence by the mask's
"objectness" (Mask2Former-style), thresholding, upsampling the masks to the
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
        "masks": (N,H,W)}``. If the model has secondary heads, ``output`` also
        carries ``"aux_queries_logits": {name: (1, Q, ns)}`` and the result gains
        ``"aux": {name: {"ids": (N,), "probs": (N, ns)}}`` for the same kept queries.
    """
    mask_logits = output["masks_queries_logits"][0].float()  # (Q, h, w)
    class_logits = output["class_queries_logits"][0].float()  # (Q, C+1)
    aux_logits = output.get("aux_queries_logits")  # {name: (1, Q, ns)} or None

    # Last column is the "no-object" class — drop it.
    scores_all = class_logits.softmax(dim=-1)[:, :-1]  # (Q, C)
    cls_scores, classes = scores_all.max(dim=-1)  # (Q,)

    # Mask2Former-style score: weight the class confidence by mask "objectness"
    # (mean sigmoid over the binarized region), computed on the low-res logits.
    # This ranks crisp, confident masks above diffuse ones and lifts mAP.
    mask_prob = mask_logits.sigmoid()  # (Q, h, w)
    binar = mask_prob > mask_thresh
    mask_scores = (mask_prob * binar).flatten(1).sum(1) / (binar.flatten(1).sum(1) + 1e-6)
    scores = cls_scores * mask_scores  # (Q,)

    orig_w, orig_h = original_size
    sel = (scores >= conf_thres).nonzero(as_tuple=True)[0]  # indices into Q
    if sel.numel() == 0:
        empty = {
            "num_detections": 0,
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "classes": torch.zeros((0,), dtype=torch.long),
            "masks": torch.zeros((0, orig_h, orig_w), dtype=torch.bool),
        }
        if aux_logits is not None:
            empty["aux"] = {
                name: {
                    "ids": torch.zeros((0,), dtype=torch.long),
                    "probs": torch.zeros((0, lg.shape[-1])),
                }
                for name, lg in aux_logits.items()
            }
        return empty

    if sel.numel() > max_det:
        sel = sel[scores[sel].topk(max_det).indices]

    scores, classes = scores[sel], classes[sel]
    mask_logits = mask_logits[sel]

    masks = F.interpolate(
        mask_logits.unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )[0]
    masks = masks.sigmoid() > mask_thresh  # (N, H, W) bool

    boxes = boxes_from_masks(masks)
    result = {
        "num_detections": int(scores.numel()),
        "boxes": boxes,
        "scores": scores,
        "classes": classes.long(),
        "masks": masks,
    }
    if aux_logits is not None:
        result["aux"] = {}
        for name, lg in aux_logits.items():
            probs = lg[0].float().softmax(dim=-1)[sel]  # (N, ns)
            result["aux"][name] = {"ids": probs.argmax(dim=-1), "probs": probs}
    return result
