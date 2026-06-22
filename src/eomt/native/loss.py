"""Pure-PyTorch EoMT loss: Hungarian matcher + point-sampled mask/dice + class CE.

A faithful reimplementation of ``transformers.models.eomt.modeling_eomt``'s loss
stack (``EomtHungarianMatcher`` / ``EomtLoss`` and the PointRend helpers), with no
dependency on the transformers *model code* — only ``scipy`` (already a project
dep) and ``torch``. Numerics match the upstream loss; the only intentional
deviations are:

* ``sample_point`` casts the sampling grid to the feature dtype, so CUDA+AMP
  training works without the external monkeypatch (``grid_sample`` requires the
  grid and input to share a dtype).
* ``get_num_masks`` drops the ``accelerate`` all-reduce branch — training here is
  single-process (no ``PartialState`` is initialised). Re-add a reduce if DDP is
  introduced.

The matcher is exposed as ``NativeEomtLoss.matcher`` with the exact
``(masks_queries_logits, class_queries_logits, mask_labels, class_labels)`` ->
list-of-(src, tgt) signature that :func:`eomt.aux_cls.match_queries` calls.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim: bool = False, **kwargs
) -> torch.Tensor:
    """``grid_sample`` wrapper supporting 3D point coords; AMP-safe (casts grid dtype)."""
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)
    if point_coordinates.dtype != input_features.dtype:
        point_coordinates = point_coordinates.to(input_features.dtype)
    point_features = F.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)
    return point_features


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


class NativeHungarianMatcher(nn.Module):
    """1-to-1 assignment between queries and GT masks via class + mask + dice cost."""

    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544
    ):
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs can't be 0")
        self.num_points = num_points
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
        indices: list[tuple[np.ndarray, np.ndarray]] = []
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]

            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)
            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            indices.append(linear_sum_assignment(cost_matrix.cpu()))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class NativeEomtLoss(nn.Module):
    """EoMT mask-classification loss (class CE + point-sampled mask BCE + dice)."""

    def __init__(self, config, weight_dict: dict[str, float]):
        super().__init__()
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio

        self.matcher = NativeHungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=self.num_points,
        )

    def _max_by_axis(self, sizes: list[list[int]]) -> list[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: list[Tensor]) -> tuple[Tensor, Tensor]:
        max_size = self._max_by_axis([list(t.shape) for t in tensors])
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype, device = tensors[0].dtype, tensors[0].device
        padded = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for tensor, pad, pmask in zip(tensors, padded, padding_masks):
            pad[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            pmask[: tensor.shape[1], : tensor.shape[2]] = False
        return padded, padding_masks

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
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }
        del pred_masks, target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def calculate_uncertainty(self, logits: Tensor) -> Tensor:
        return -(torch.abs(logits))

    def sample_points_using_uncertainty(
        self, logits, uncertainty_function, num_points, oversample_ratio, importance_sample_ratio
    ) -> Tensor:
        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(
            num_boxes, num_uncertain_points, 2
        )
        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)], dim=1
            )
        return point_coordinates

    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
        auxiliary_predictions: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        losses: dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
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
