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
  them where the prediction is least certain (:func:`uncertain_points`), which concentrates it on the boundaries.
* ``get_num_masks`` counts the instances of this process's batch: on several GPUs each process normalises by its own
  count (the reference all-reduces it), and DistributedDataParallel averages the gradients.
* **Partial labels** (:mod:`eomt.data.subjects`). ``ignore_masks`` (one map per photo, 1 = leave out) spare the
  unmatched queries whose mask lies mostly inside an ignore region the "no object" class loss, and ``neg_weight`` (one
  value per photo) scales that loss for every unmatched query of the photo: 0 for a photo whose annotations list only
  some of its instances, so nothing unannotated is taught as background. Matched queries are untouched.

The matcher is exposed as ``EoMTLoss.matcher`` with the ``(masks_queries_logits, class_queries_logits, mask_labels,
class_labels)`` -> list-of-(src, tgt) signature that :func:`eomt.aux_cls.match_queries` calls; the loss also returns the
matching it used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn



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


def dice_loss(inputs: Tensor, labels: Tensor, num_masks) -> Tensor:
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks


def sigmoid_cross_entropy_loss(inputs: Tensor, labels: Tensor, num_masks) -> Tensor:
    return F.binary_cross_entropy_with_logits(inputs, labels, reduction="none").mean(1).sum() / num_masks


def solve_assignments(costs: list[Tensor]) -> list[tuple[Tensor, Tensor]]:
    """Hungarian assignment of each cost matrix, with ONE device-to-host copy for all of them (one per image would wait
    for the GPU once per image)."""
    costs = [torch.nan_to_num(c.float().clamp(-1e10, 1e10), 0) for c in costs]
    flat = torch.cat([c.flatten() for c in costs]).cpu().numpy() if costs else np.zeros(0)
    out, at = [], 0
    for c in costs:
        n = c.numel()
        i, j = linear_sum_assignment(flat[at:at + n].reshape(c.shape))
        at += n
        out.append((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)))
    return out


class HungarianMatcher(nn.Module):
    """1-to-1 assignment between queries and GT masks via class + mask + dice cost (dense, on the logit grid).

    ponytail: an instance smaller than a grid cell has a near-flat mask cost on the grid, so its class cost picks its
    query. Scoring the masks at the GT resolution fixes that but nearly doubles the step (all queries x every pixel).
    """

    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs can't be 0")
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
    ) -> list[tuple[Tensor, Tensor]]:
        grid = masks_queries_logits.shape[-2:]
        costs = []
        for i in range(masks_queries_logits.shape[0]):
            pred_probs = class_queries_logits[i].float().softmax(-1)
            cost_class = -pred_probs[:, class_labels[i]]
            pred_mask = masks_queries_logits[i].flatten(1).float()  # [Q, cells]
            target_mask = grid_targets(mask_labels[i], grid).flatten(1).to(pred_mask)  # [G, cells]
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            costs.append(self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice)
        return solve_assignments(costs)


class EoMTLoss(nn.Module):
    """EoMT mask-classification loss (class CE + mask BCE + dice on the logit grid)."""

    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.matcher = HungarianMatcher(
            cost_class=config.class_weight, cost_dice=config.dice_weight, cost_mask=config.mask_weight
        )
        # IoU-aware class targets: a matched query's class target is (1 + IoU) / 2 of its mask instead of 1, so the score
        # ranks masks by quality; above 0.5, so a matched query stays classified as its GT class (the attribute gate).
        self.iou_aware_cls = bool(getattr(config, "iou_aware_cls", False))

    def loss_labels(self, class_queries_logits, class_labels, indices, quality: Tensor | None = None,
                    negatives: Tensor | None = None) -> dict[str, Tensor]:
        """Class CE. ``quality`` (IoU of each matched query, in permutation order) softens the matched targets to
        ``(1 + IoU) / 2`` for the GT class and the rest for "no object"; every query keeps its weight (1 matched,
        ``no_object_weight`` unmatched), so hard targets give exactly the weighted CE. ``negatives`` ``(B, Q)`` scales
        the "no object" term of each unmatched query (:meth:`negative_weights`); the normalisation stays that of the
        full loss, so a left-out query only removes its term."""
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        if quality is None and negatives is None:
            criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
            return {"loss_cross_entropy": criterion(pred_logits.transpose(1, 2), target_classes)}
        target = F.one_hot(target_classes, self.num_labels + 1).float()
        if quality is not None:
            q = 0.5 + 0.5 * quality.to(device=pred_logits.device, dtype=torch.float32)
            target[idx] = target[idx] * q[:, None]
            target[idx[0], idx[1], -1] = 1 - q
        weight = self.empty_weight[target_classes]
        kept = weight
        if negatives is not None:
            unmatched = torch.ones_like(weight, dtype=torch.bool)
            unmatched[idx] = False
            kept = torch.where(unmatched, weight * negatives.to(weight), weight)
        loss = -(target * F.log_softmax(pred_logits.float(), -1)).sum(-1)
        return {"loss_cross_entropy": (kept * loss).sum() / weight.sum()}

    @torch.no_grad()
    def negative_weights(self, masks_queries_logits: Tensor, ignore_masks: list[Tensor | None] | None,
                         neg_weight: Tensor | None, inside: float = 0.5) -> Tensor | None:
        """``(B, Q)`` weights of the "no object" term of every query: ``neg_weight[b]`` for photo ``b`` (1 when
        omitted), and 0 for a query whose predicted mask has at least ``inside`` of its (soft) mass in the photo's
        ignore region. ``None`` when there is nothing to leave out (the plain class loss)."""
        if ignore_masks is None and neg_weight is None:
            return None
        b, q = masks_queries_logits.shape[:2]
        w = torch.ones((b, q), device=masks_queries_logits.device)
        if neg_weight is not None:
            w = w * neg_weight.to(w).view(-1, 1)
        if ignore_masks is not None:
            grid = masks_queries_logits.shape[-2:]
            for i, ign in enumerate(ignore_masks):
                if ign is None or not bool((ign > 0).any()):
                    continue
                cells = grid_targets(ign[None].to(masks_queries_logits.device), grid)[0]  # share of each cell ignored
                prob = masks_queries_logits[i].float().sigmoid()
                share = (prob * cells).flatten(1).sum(1) / prob.flatten(1).sum(1).clamp(min=1e-6)
                w[i] = w[i] * (share < inside).to(w)
        return w

    @torch.no_grad()
    def matched_iou(self, masks_queries_logits, mask_labels, indices) -> Tensor:
        """Mask IoU of each matched query with its GT at the GT resolution, in permutation order (prediction > 0 after
        :func:`upsample_logits`; a soft GT binarised at half its peak, as the attribute gate does)."""
        targets = [t[j.to(t.device)] for t, (_, j) in zip(mask_labels, indices) if len(j)]
        if not targets:
            return masks_queries_logits.new_zeros(0, dtype=torch.float32)
        gt = torch.cat(targets).float()
        gt = gt >= (0.5 * gt.flatten(1).amax(1)).clamp(min=1e-6)[:, None, None]
        logits = masks_queries_logits[self._get_predictions_permutation_indices(indices)].float()
        pred = upsample_logits(logits, gt.shape[-2:]) > 0
        inter = (pred & gt).flatten(1).sum(1).float()
        return inter / (pred | gt).flatten(1).sum(1).float().clamp(min=1)

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
        pred_quality: Tensor | None = None,
        ignore_masks: list[Tensor | None] | None = None,
        neg_weight: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], list[tuple[Tensor, Tensor]]]:
        """``(losses, indices)``: the unweighted loss terms and the query->target matching they used.
        ``pred_quality`` ``(B, Q)`` (mask-quality logits) adds ``loss_quality``: BCE of the matched queries' predicted
        against their measured mask IoU (Mask Scoring R-CNN); it does not enter the matching. ``ignore_masks`` and
        ``neg_weight`` leave unannotated regions and partly annotated photos out of the "no object" loss (see the module
        docstring)."""
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        quality = self.matched_iou(masks_queries_logits, mask_labels, indices) if self.iou_aware_cls else None
        negatives = self.negative_weights(masks_queries_logits, ignore_masks, neg_weight)
        losses: dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices, quality, negatives),
        }
        if pred_quality is not None:
            iou = quality if quality is not None else self.matched_iou(masks_queries_logits, mask_labels, indices)
            pred = pred_quality[self._get_predictions_permutation_indices(indices)].float()
            losses["loss_quality"] = (
                F.binary_cross_entropy_with_logits(pred, iou.to(pred), reduction="sum") / num_masks
                if pred.numel() else pred_quality.sum() * 0.0
            )
        return losses, indices

    def get_num_masks(self, class_labels, device) -> Tensor:
        num_masks = sum(len(classes) for classes in class_labels)
        num_masks = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        return torch.clamp(num_masks, min=1)
