"""Pure-PyTorch EoMT loss: Hungarian matcher + mask BCE/dice + class CE.

A reimplementation of ``transformers.models.eomt.modeling_eomt``'s loss stack (``EomtHungarianMatcher`` /
``EomtLoss``) with no dependency on the transformers *model code* — only ``scipy`` (already a project dep) and
``torch``. Deliberate deviations:

* **Mask terms.** Matching is dense and deterministic: every query is scored on every cell of the mask-logit grid
  against the GT area-averaged onto it (:func:`grid_targets`). The reference scores 12,544 random points instead, one
  per ~33 px^2 at a 644 px input, so an instance of a few dozen pixels often got no point and its assigned query changed
  from one draw to the next. The mask loss of the matched queries has two parts with different jobs: dice on every GT
  pixel against the logits bilinearly upsampled to the GT resolution (:func:`upsample_logits`, as inference reads
  them), so every instance is supervised however small; and the reference's BCE on 12,544 points, three quarters of
  them where the prediction is least certain (:func:`uncertain_points`), which concentrates it on the boundaries. A
  dense BCE spread that weight over every pixel and gave coarser masks (AP75 halved on thin structures).
* ``get_num_masks`` drops the ``accelerate`` all-reduce branch — training here is single-process (no
  ``PartialState`` is initialised). Re-add a reduce if DDP is introduced.

The matcher is exposed as ``EoMTLoss.matcher`` with the exact
``(masks_queries_logits, class_queries_logits, mask_labels, class_labels)`` -> list-of-(src, tgt) signature that
:func:`eomt.aux_cls.match_queries` calls.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .box_loss import box_cxcywh_to_xyxy, generalized_box_iou, masks_to_norm_boxes


def upsample_logits(logits: Tensor, size) -> Tensor:
    """Mask logits ``(N, h, w)`` bilinearly upsampled to the GT resolution ``size``, as inference reads them."""
    if tuple(logits.shape[-2:]) == tuple(size):
        return logits
    return F.interpolate(logits[:, None], size=tuple(size), mode="bilinear", align_corners=False)[:, 0]


def sample_point(features: Tensor, coords: Tensor) -> Tensor:
    """Bilinear samples of ``features`` ``(N, C, H, W)`` at ``coords`` ``(N, P, 2)`` in [0, 1] -> ``(N, C, P)``."""
    return F.grid_sample(features, 2.0 * coords[:, :, None].to(features.dtype) - 1.0, align_corners=False)[..., 0]


@torch.no_grad()
def uncertain_points(logits: Tensor, num_points: int = 12544, oversample: float = 3.0, importance: float = 0.75) -> Tensor:
    """PointRend importance sampling: ``importance`` of the points where ``|logit|`` is smallest among ``oversample`` x
    as many random candidates, the rest uniform. ``logits`` ``(N, 1, h, w)`` -> coords ``(N, num_points, 2)``."""
    n = logits.shape[0]
    cand = torch.rand(n, int(num_points * oversample), 2, device=logits.device)
    k = int(importance * num_points)
    idx = (-sample_point(logits, cand)[:, 0].abs()).topk(k, dim=1).indices
    picked = torch.gather(cand, 1, idx[..., None].expand(-1, -1, 2))
    return torch.cat([picked, torch.rand(n, num_points - k, 2, device=logits.device)], dim=1)


def grid_targets(masks: Tensor, size) -> Tensor:
    """GT masks ``(N, H, W)`` area-averaged onto the ``size`` mask-logit grid: the share of each cell an instance covers.

    Every instance keeps its whole mass however small (a 15 px chip at a 644 px input is ~1.2 cells of a 184 x 184 grid).
    """
    size = tuple(size)
    if masks.shape[0] == 0:
        return masks.new_zeros((0, *size), dtype=torch.float32)
    return F.interpolate(masks[:, None].float(), size=size, mode="area")[:, 0]


def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def pair_wise_sigmoid_cross_entropy_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    height_and_width = inputs.shape[1]
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    ce_pos = criterion(inputs, torch.ones_like(inputs))
    ce_neg = criterion(inputs, torch.zeros_like(inputs))
    loss_pos = torch.matmul(ce_pos / height_and_width, labels.T)
    loss_neg = torch.matmul(ce_neg / height_and_width, (1 - labels).T)
    return loss_pos + loss_neg


def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_cross_entropy_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)
    return cross_entropy_loss.mean(1).sum() / num_masks


class HungarianMatcher(nn.Module):
    """1-to-1 assignment between queries and GT masks via class + mask + dice cost (dense, on the logit grid).

    ponytail: an instance smaller than a grid cell has a near-flat mask cost on the grid, so its class cost picks its
    query. Scoring the masks at the GT resolution fixes that but nearly doubles the step (all queries x every pixel).
    """

    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0,
        cost_bbox: float = 0.0, cost_giou: float = 0.0,
    ):
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs can't be 0")
        # Optional box terms (L1 + GIoU on normalized cxcywh): 0 = the original mask-only matcher.
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
        pred_boxes: Tensor | None = None,
        box_labels: list[Tensor] | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """``pred_boxes`` ``(B, Q, 4)`` (normalized cxcywh) adds an L1 + GIoU term to the cost; the GT boxes are
        ``box_labels`` or, when omitted, the tight boxes of ``mask_labels``."""
        indices: list[tuple[np.ndarray, np.ndarray]] = []
        batch_size = masks_queries_logits.shape[0]
        grid = masks_queries_logits.shape[-2:]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            cost_class = -pred_probs[:, class_labels[i]]
            pred_mask = masks_queries_logits[i].flatten(1).float()  # [Q, cells]
            target_mask = grid_targets(mask_labels[i], grid).flatten(1).to(pred_mask)  # [G, cells]

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            if pred_boxes is not None and (self.cost_bbox or self.cost_giou) and mask_labels[i].shape[0] > 0:
                tgt_boxes = (box_labels[i] if box_labels is not None else masks_to_norm_boxes(mask_labels[i])).to(
                    device=pred_boxes.device, dtype=torch.float32
                )
                out_boxes = pred_boxes[i].float()
                cost_matrix = (
                    cost_matrix
                    + self.cost_bbox * torch.cdist(out_boxes, tgt_boxes, p=1)
                    - self.cost_giou * generalized_box_iou(box_cxcywh_to_xyxy(out_boxes), box_cxcywh_to_xyxy(tgt_boxes))
                )
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            indices.append(linear_sum_assignment(cost_matrix.cpu()))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class EoMTLoss(nn.Module):
    """EoMT mask-classification loss (class CE + mask BCE + dice on the logit grid)."""

    def __init__(self, config, weight_dict: dict[str, float]):
        super().__init__()
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # Auxiliary box head (instance family): L1 + GIoU on the matched queries, and the same two terms in
        # the matching cost. Off unless ``config.aux_box_head`` is set.
        self.use_boxes = bool(getattr(config, "aux_box_head", False))
        self.l1_weight = float(getattr(config, "l1_weight", 5.0)) if self.use_boxes else 0.0
        self.giou_weight = float(getattr(config, "giou_weight", 2.0)) if self.use_boxes else 0.0

        self.matcher = HungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            cost_bbox=self.l1_weight,
            cost_giou=self.giou_weight,
        )

    def loss_labels(self, class_queries_logits, class_labels, indices) -> dict[str, Tensor]:
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        loss_ce = criterion(pred_logits.transpose(1, 2), target_classes)
        return {"loss_cross_entropy": loss_ce}

    def loss_masks(self, masks_queries_logits, mask_labels, indices, num_masks) -> dict[str, Tensor]:
        """Mask loss of the matched queries: dice on every GT pixel (see :func:`upsample_logits`), so every instance is
        supervised however small; BCE on 12,544 points, three quarters of them where the prediction is least certain
        (:func:`uncertain_points`), which concentrates it on the boundaries."""
        targets = [t[j.to(t.device)] for t, (_, j) in zip(mask_labels, indices) if len(j)]
        if not targets:  # no matched query this batch: keep the graph alive
            zero = masks_queries_logits.sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}
        src_idx = self._get_predictions_permutation_indices(indices)
        target_masks = torch.cat(targets).float()
        logits = masks_queries_logits[src_idx].float()
        pred_masks = upsample_logits(logits, target_masks.shape[-2:]).flatten(1)
        coords = uncertain_points(logits[:, None])
        point_labels = sample_point(target_masks[:, None], coords)[:, 0]
        point_logits = sample_point(logits[:, None], coords)[:, 0]
        return {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks.flatten(1).to(pred_masks), num_masks),
        }

    def loss_boxes(self, pred_boxes, box_labels, indices, num_boxes) -> dict[str, Tensor]:
        """Box L1 + GIoU on the matched queries (normalized cxcywh)."""
        idx = self._get_predictions_permutation_indices(indices)
        src = pred_boxes[idx].float()
        tgt = torch.cat([t[j.to(t.device)] for t, (_, j) in zip(box_labels, indices)]).to(src)
        if src.numel() == 0:  # no matched query this batch: keep the graph alive
            zero = pred_boxes.sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}
        giou = generalized_box_iou(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt)).diagonal()
        return {
            "loss_bbox": F.l1_loss(src, tgt, reduction="sum") / num_boxes,
            "loss_giou": (1 - giou).sum() / num_boxes,
        }

    def _get_predictions_permutation_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
        auxiliary_predictions: dict[str, Tensor] | None = None,
        pred_boxes: Tensor | None = None,
        box_labels: list[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """``pred_boxes`` (with the model built with ``aux_box_head``) adds the box terms to the matching and the loss."""
        use_boxes = self.use_boxes and pred_boxes is not None
        if use_boxes and box_labels is None:
            box_labels = [masks_to_norm_boxes(m) for m in mask_labels]
        indices = self.matcher(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels,
            pred_boxes=pred_boxes if use_boxes else None, box_labels=box_labels if use_boxes else None,
        )
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        losses: dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        if use_boxes:
            losses.update(self.loss_boxes(pred_boxes, box_labels, indices, num_masks))
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                loss_dict = self.forward(
                    aux_outputs["masks_queries_logits"],
                    aux_outputs["class_queries_logits"],
                    mask_labels,
                    class_labels,
                )
                losses.update({f"{k}_{idx}": v for k, v in loss_dict.items()})
        return losses

    def get_num_masks(self, class_labels, device) -> Tensor:
        num_masks = sum(len(classes) for classes in class_labels)
        num_masks = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        return torch.clamp(num_masks, min=1)
