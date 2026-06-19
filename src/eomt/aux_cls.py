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
def match_queries(model, out: dict, mask_labels, class_labels):
    """Hungarian query→GT indices from EoMT's matcher (list of ``(src, tgt)``)."""
    return model.eomt.criterion.matcher(
        out["masks_queries_logits"], out["class_queries_logits"], mask_labels, class_labels
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
