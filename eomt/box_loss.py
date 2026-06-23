"""Pure-PyTorch DETR-style detection loss: Hungarian matcher + L1/GIoU + class CE.

The detection counterpart to :mod:`eomt.loss` (the mask-classification stack). It
keeps the **same** query→GT assignment machinery — a Hungarian matcher exposed as
``DetectionLoss.matcher`` with a ``(geometry, class_queries_logits, geom_labels,
class_labels)`` signature and the same list-of-``(src, tgt)`` return — so
:func:`eomt.aux_cls.match_queries` can drive attribute supervision identically for
the box head. The only difference from :mod:`eomt.loss` is the geometry term: per-query
mask BCE + dice is replaced by per-query box **L1 + GIoU** (DETR's box loss).

All boxes here are normalized ``cxcywh`` in ``[0, 1]`` (the box head emits sigmoid
``cxcywh``; the detection dataset stores GT the same way), so no image size is needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """``(..., 4)`` boxes from center form ``(cx, cy, w, h)`` to corner form ``(x1, y1, x2, y2)``."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """Pairwise IoU and union between two sets of ``xyxy`` boxes ``([N,4], [M,4])``."""
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise GIoU between two sets of ``xyxy`` boxes (DETR's enclosing-box variant)."""
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area.clamp(min=1e-6)


class DetectionHungarianMatcher(nn.Module):
    """1-to-1 assignment between queries and GT boxes via class + L1 + GIoU cost."""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 1.0, cost_giou: float = 1.0):
        super().__init__()
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("All costs can't be 0")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_boxes: Tensor,
        class_queries_logits: Tensor,
        box_labels: list[Tensor],
        class_labels: list[Tensor],
    ) -> list[tuple[Tensor, Tensor]]:
        indices: list = []
        batch_size = pred_boxes.shape[0]
        for i in range(batch_size):
            # Cost math in fp32: cdist/GIoU are not implemented for fp16 on CUDA, and
            # the matcher is called both inside autocast (training) and outside it
            # (eval/aux), where the tensors stay half.
            tgt_boxes = box_labels[i].to(device=pred_boxes.device, dtype=torch.float32)
            tgt_cls = class_labels[i]
            if tgt_boxes.numel() == 0:
                indices.append(
                    (
                        torch.as_tensor([], dtype=torch.int64),
                        torch.as_tensor([], dtype=torch.int64),
                    )
                )
                continue
            pred_probs = class_queries_logits[i].float().softmax(-1)  # [Q, C+1]
            out_boxes = pred_boxes[i].float()  # [Q, 4]

            cost_class = -pred_probs[:, tgt_cls]  # [Q, num_tgt]
            cost_bbox = torch.cdist(out_boxes, tgt_boxes, p=1)  # [Q, num_tgt]
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_boxes), box_cxcywh_to_xyxy(tgt_boxes)
            )
            cost_matrix = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            indices.append(linear_sum_assignment(cost_matrix.cpu()))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class DetectionLoss(nn.Module):
    """DETR-style detection loss (class CE + box L1 + GIoU), matching EoMTLoss's API.

    ``weight_dict`` carries ``{"loss_cross_entropy", "loss_bbox", "loss_giou"}``; the
    matcher reuses ``class_weight`` (from the config) and the L1/GIoU weights from
    ``weight_dict``. ``forward`` returns the three *unweighted* loss terms — the caller
    (``EoMTEncoder.get_loss_dict``) applies ``weight_dict``, exactly as for the mask loss.
    """

    def __init__(self, config, weight_dict: dict[str, float]):
        super().__init__()
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = DetectionHungarianMatcher(
            cost_class=config.class_weight,
            cost_bbox=weight_dict["loss_bbox"],
            cost_giou=weight_dict["loss_giou"],
        )

    def _get_predictions_permutation_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def loss_labels(self, class_queries_logits, class_labels, indices) -> dict[str, Tensor]:
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        ).to(pred_logits.device)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64,
            device=pred_logits.device,
        )
        target_classes[idx] = target_classes_o
        loss_ce = criterion(pred_logits.transpose(1, 2), target_classes)
        return {"loss_cross_entropy": loss_ce}

    def loss_boxes(self, pred_boxes, box_labels, indices, num_boxes) -> dict[str, Tensor]:
        idx = self._get_predictions_permutation_indices(indices)
        src_boxes = pred_boxes[idx]  # [N, 4] cxcywh
        tgt_boxes = torch.cat(
            [t[j] for t, (_, j) in zip(box_labels, indices)]
        ).to(src_boxes)  # [N, 4] cxcywh
        if src_boxes.numel() == 0:  # no matched query this batch — keep graph alive
            zero = pred_boxes.sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)
        ).diagonal()
        loss_giou = (1 - giou).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def get_num_boxes(self, class_labels, device) -> Tensor:
        num = sum(len(c) for c in class_labels)
        return torch.clamp(torch.as_tensor(num, dtype=torch.float, device=device), min=1)

    def forward(
        self,
        masks_queries_logits: Tensor,  # unused; kept for signature symmetry with EoMTLoss
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],  # actually box_labels (cxcywh) for the detect family
        class_labels: list[Tensor],
        auxiliary_predictions=None,
    ) -> dict[str, Tensor]:
        pred_boxes, box_labels = masks_queries_logits, mask_labels
        indices = self.matcher(pred_boxes, class_queries_logits, box_labels, class_labels)
        num_boxes = self.get_num_boxes(class_labels, device=class_queries_logits.device)
        return {
            **self.loss_boxes(pred_boxes, box_labels, indices, num_boxes),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
