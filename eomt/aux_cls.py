"""Secondary per-instance classification heads ("attributes") for EoMT.

The primary task (instance segmentation over ``nc`` classes) is unchanged. Each
aux head predicts an extra attribute per *detected instance* — typology,
laterality, severity, … — read from the matched query's embedding.

The supervision reuses EoMT's own Hungarian matcher
(``model.eomt.criterion.matcher``) so each attribute is trained on the **same**
query→GT assignment the detection loss used. Several specs ⇒ several independent
heads, summed (optionally weighted) into one scalar.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


@torch.no_grad()
def match_queries(model, out: dict, geom_labels, class_labels):
    """Hungarian query→GT indices from EoMT's matcher (list of ``(src, tgt)``).

    ``geom_labels`` is the GT geometry for the model's family — instance masks for the
    ``"instance"`` family, normalized ``cxcywh`` boxes for ``"detect"``. The right
    matcher (mask-cost or box-cost) is selected by what the model emitted in ``out``.
    """
    geom = out["pred_boxes"] if "pred_boxes" in out else out["masks_queries_logits"]
    return model.eomt.criterion.matcher(
        geom, out["class_queries_logits"], geom_labels, class_labels
    )


def _gather_matched(out: dict, indices):
    """Return ``(matched_feats [N, hidden], [(b, tgt_idx)...])`` for matched queries."""
    q = out["query_embed"]  # [B, Q, hidden]
    feats, batch_tgt = [], []
    for b, (src, tgt) in enumerate(indices):
        if src.numel() == 0:
            continue
        feats.append(q[b, src])
        batch_tgt.append((b, tgt))
    if not feats:
        return None, batch_tgt
    return torch.cat(feats), batch_tgt


@torch.no_grad()
def gate_indices(
    out: dict,
    indices,
    geom_labels,
    class_labels,
    *,
    iou_thr: float = 0.5,
    require_class: bool = False,
):
    """Keep only well-localized (and optionally correctly-classified) matched pairs.

    The Hungarian matcher assigns *every* GT a query, even one whose prediction barely
    overlaps it (common early in training). For attribute supervision we want queries
    that actually localize the instance, so we drop matched ``(src, tgt)`` pairs whose
    predicted↔GT IoU is ``< iou_thr`` (mask IoU for the instance family, box IoU for
    detect). With ``require_class`` we also drop pairs whose predicted primary class ≠
    the GT class. Returns the same ``[(src, tgt), ...]`` structure with each pair
    filtered (possibly to empty).

    A no-op (returns ``indices`` unchanged) when ``iou_thr <= 0`` and not
    ``require_class`` — i.e. the pre-gate behaviour.
    """
    if iou_thr <= 0 and not require_class:
        return indices
    is_detect = "pred_boxes" in out
    cls = out["class_queries_logits"]  # [B, Q, C+1]
    dev = cls.device
    gated = []
    for b, (src, tgt) in enumerate(indices):
        src = src.to(dev)
        tgt = tgt.to(dev)
        if src.numel() == 0:
            gated.append((src, tgt))
            continue
        keep = torch.ones(src.numel(), dtype=torch.bool, device=dev)
        if iou_thr > 0:
            keep &= _localization_iou(out, b, src, tgt, geom_labels, is_detect, dev) >= iou_thr
        if require_class:
            pred_cls = cls[b, src].argmax(-1)
            keep &= pred_cls == class_labels[b][tgt].to(dev)
        gated.append((src[keep], tgt[keep]))
    return gated


def _localization_iou(out, b, src, tgt, geom_labels, is_detect, dev):
    """Per-pair predicted↔GT IoU for the matched queries (box IoU or mask IoU)."""
    if is_detect:
        from .box_loss import box_cxcywh_to_xyxy, box_iou

        pred = box_cxcywh_to_xyxy(out["pred_boxes"][b, src].to(dev))  # [n, 4]
        gt = box_cxcywh_to_xyxy(geom_labels[b][tgt].to(dev).float())  # [n, 4]
        return box_iou(pred, gt)[0].diagonal()  # per-matched-pair IoU
    masks = out["masks_queries_logits"]  # [B, Q, h, w]
    pm = masks[b, src].sigmoid() > 0.5  # [n, h, w]
    gm = geom_labels[b][tgt].to(dev).float()  # [n, H, W] in {0, 1}
    if gm.shape[-2:] != pm.shape[-2:]:
        gm = F.interpolate(gm.unsqueeze(1), size=pm.shape[-2:], mode="nearest").squeeze(1)
    gm = gm > 0.5
    inter = (pm & gm).flatten(1).sum(1).float()
    union = (pm | gm).flatten(1).sum(1).float().clamp(min=1.0)
    return inter / union


def aux_loss(
    model,
    out: dict,
    mask_labels,
    class_labels,
    aux_labels: dict[str, list[torch.Tensor]],
    weights: dict[str, float] | None = None,
    *,
    indices=None,
    class_weights: dict[str, torch.Tensor] | None = None,
    ignore_index: int = -100,
):
    """Weighted sum of per-attribute CE over matched queries.

    ``aux_labels`` maps head name → per-image label tensors (line-aligned with
    ``class_labels``). Labels equal to ``ignore_index`` (missing / out-of-vocab
    attributes) are skipped — they contribute no loss instead of being trained as
    class 0. ``weights`` scales each head; ``class_weights`` optionally re-weights
    classes within a head (imbalance). Pass ``indices`` to reuse a matching already
    computed this step (avoids re-running the Hungarian matcher). Returns
    ``(total_loss, {name: per_head_loss})``.
    """
    if indices is None:
        indices = match_queries(model, out, mask_labels, class_labels)
    matched, batch_tgt = _gather_matched(out, indices)
    if matched is None:  # no query matched in this batch — keep the graph alive
        zero = out["query_embed"].sum() * 0.0
        return zero, {name: zero.detach() for name in model.aux_heads}

    weights = weights or {}
    class_weights = class_weights or {}
    total: torch.Tensor | None = None
    per_head: dict[str, torch.Tensor] = {}
    for name, head in model.aux_heads.items():
        logits = head(matched)  # [N, ns]
        gt = torch.cat([aux_labels[name][b][tgt] for (b, tgt) in batch_tgt]).to(logits.device)
        cw = class_weights.get(name)
        if cw is not None:
            cw = cw.to(logits.device, logits.dtype)
        if (gt != ignore_index).any():
            loss = F.cross_entropy(logits, gt, weight=cw, ignore_index=ignore_index)
        else:  # every matched label ignored — keep the graph alive with a zero
            loss = logits.sum() * 0.0
        per_head[name] = loss
        w = float(weights.get(name, 1.0))
        total = w * loss if total is None else total + w * loss
    return total, per_head


@torch.no_grad()
def aux_accuracy(
    model,
    out: dict,
    mask_labels,
    class_labels,
    aux_labels: dict[str, list[torch.Tensor]],
    *,
    indices=None,
    ignore_index: int = -100,
) -> dict[str, tuple[int, int]]:
    """Top-1 ``{name: (correct, total)}`` on matched queries (the ``typ_acc`` analogue).

    Ignored labels (``ignore_index``) are excluded from both correct and total.
    Pass ``indices`` to reuse a matching already computed this step.
    """
    if indices is None:
        indices = match_queries(model, out, mask_labels, class_labels)
    matched, batch_tgt = _gather_matched(out, indices)
    if matched is None:
        return {name: (0, 0) for name in model.aux_heads}
    res: dict[str, tuple[int, int]] = {}
    for name, head in model.aux_heads.items():
        pred = head(matched).argmax(-1)
        gt = torch.cat([aux_labels[name][b][tgt] for (b, tgt) in batch_tgt]).to(pred.device)
        valid = gt != ignore_index
        res[name] = (int(((pred == gt) & valid).sum()), int(valid.sum()))
    return res


@torch.no_grad()
def aux_accuracy_by_primary(
    model,
    out: dict,
    mask_labels,
    class_labels,
    aux_labels: dict[str, list[torch.Tensor]],
    *,
    indices=None,
    ignore_index: int = -100,
) -> dict[str, dict[int, tuple[int, int]]]:
    """Per-head accuracy bucketed by GT **primary** class.

    Returns ``{head: {primary_class_id: (correct, total)}}`` over matched queries —
    a diagnostic for *which primary classes* the attribute is (in)accurate on. Pass
    ``indices`` (e.g. IoU-gated, but not class-gated, so weak primary classes still
    appear) to control the population. Ignored labels are excluded.
    """
    if indices is None:
        indices = match_queries(model, out, mask_labels, class_labels)
    matched, batch_tgt = _gather_matched(out, indices)
    res: dict[str, dict[int, tuple[int, int]]] = {name: {} for name in model.aux_heads}
    if matched is None:
        return res
    prim = torch.cat([class_labels[b][tgt] for (b, tgt) in batch_tgt]).to(matched.device)
    for name, head in model.aux_heads.items():
        pred = head(matched).argmax(-1)
        gt = torch.cat([aux_labels[name][b][tgt] for (b, tgt) in batch_tgt]).to(pred.device)
        valid = gt != ignore_index
        correct = (pred == gt) & valid
        d: dict[int, tuple[int, int]] = {}
        for c in prim[valid].unique().tolist():
            sel = (prim == c) & valid
            d[int(c)] = (int(correct[sel].sum()), int(sel.sum()))
        res[name] = d
    return res
