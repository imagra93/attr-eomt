"""Secondary per-instance classification heads ("attributes") for EoMT.

The primary task (instance segmentation over ``nc`` classes) is unchanged. Each
aux head predicts an extra attribute per *detected instance* — color, material,
size, … — read from the matched query's embedding.

The supervision reuses EoMT's own Hungarian matcher
(``model.eomt.criterion.matcher``) so each attribute is trained on the **same**
query→GT assignment the detection loss used. Several specs ⇒ several independent
heads, summed (optionally weighted) into one scalar. A regression head
(``AuxHeadSpec.regression``) regresses one real number per instance instead: a
Huber loss on its standardized value (O(1), like the cross-entropy of the
categorical heads), evaluated as the mean absolute error in real units.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812


@torch.no_grad()
def match_queries(model, out: dict, geom_labels, class_labels):
    """Hungarian query→GT indices from EoMT's matcher (list of ``(src, tgt)``).

    ``geom_labels`` is the GT geometry for the model's family — instance masks for the
    ``"instance"`` family, normalized ``cxcywh`` boxes for ``"detect"``. The right
    matcher (mask-cost or box-cost) is selected by what the model emitted in ``out``.
    """
    if "pred_boxes" in out:  # detect family
        return model.eomt.criterion.matcher(out["pred_boxes"], out["class_queries_logits"], geom_labels, class_labels)
    return model.eomt.criterion.matcher(out["masks_queries_logits"], out["class_queries_logits"], geom_labels,
                                        class_labels)


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
    # Compared at the GT's resolution: shrinking the GT to the logit grid with "nearest" erases many instances of a
    # few pixels (IoU 0, so they never trained the attribute heads). A thin or tiny soft GT can have no pixel above
    # 0.5, so each GT is binarised at half its own peak.
    logits = out["masks_queries_logits"][b, src].float()  # [n, h, w]
    gm = geom_labels[b][tgt].to(dev).float()  # [n, H, W], soft or binary
    if logits.shape[-2:] != gm.shape[-2:]:
        logits = F.interpolate(logits.unsqueeze(1), size=gm.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    pm = logits > 0
    gm = gm >= (0.5 * gm.flatten(1).amax(1)).clamp(min=1e-6)[:, None, None]
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
    class_weights: dict | None = None,
    ignore_index: int = -100,
    stats: dict | None = None,
):
    """Weighted sum of per-attribute losses over matched queries: CE, or Huber for a regression head.

    ``aux_labels`` maps head name → per-image label tensors (line-aligned with ``class_labels``). Labels equal to
    ``ignore_index`` (missing / out-of-vocab attributes; NaN for a regression head) are skipped — they contribute no
    loss instead of being trained as class 0. ``weights`` scales each head; ``class_weights`` optionally re-weights the
    classes within a categorical head (imbalance). Pass ``indices`` to reuse a matching already computed this step.
    ``stats`` (a dict) receives per head ``(correct, counted)`` as device tensors (no host sync; for a regression head
    the summed absolute error in real units instead of ``correct``), from the same logits.
    Returns ``(total_loss, {name: per_head_loss})``, the per-head losses unweighted.
    """
    if indices is None:
        indices = match_queries(model, out, mask_labels, class_labels)
    matched, batch_tgt = _gather_matched(out, indices)
    if matched is None:  # no query matched in this batch — keep the graph alive
        zero = out["query_embed"].sum() * 0.0
        return zero, {name: zero.detach() for name in model.aux_heads}

    weights = weights or {}
    class_weights = class_weights or {}
    specs = {s.name: s for s in model.aux_specs}
    total: torch.Tensor | None = None
    per_head: dict[str, torch.Tensor] = {}
    for name, head in model.aux_heads.items():
        logits = head(matched)  # [N, ns]
        gt = torch.cat([aux_labels[name][b][tgt] for (b, tgt) in batch_tgt]).to(logits.device)
        if specs[name].regression:
            valid = ~gt.isnan()
            if valid.any():
                z = specs[name].encode(gt[valid].float())
                loss = F.huber_loss(logits[valid, 0].float(), z, delta=1.0)
            else:  # every matched value ignored — keep the graph alive with a zero
                loss = logits.sum() * 0.0
        elif (gt != ignore_index).any():
            cw = class_weights.get(name)
            loss = F.cross_entropy(logits, gt, weight=None if cw is None else cw.to(logits.device, logits.dtype),
                                   ignore_index=ignore_index)
        else:  # every matched label ignored — keep the graph alive with a zero
            loss = logits.sum() * 0.0
        per_head[name] = loss
        if stats is not None:
            with torch.no_grad():
                hit, valid = _hits(specs[name], logits, aux_labels[name], batch_tgt, ignore_index)
                stats[name] = (hit.float().sum(), valid.sum())
        w = float(weights.get(name, 1.0))
        total = w * loss if total is None else total + w * loss
    return total, per_head


def match_loss_params() -> torch.nn.ParameterDict:
    """The learned scalars of :func:`match_loss`: its log-temperature ``t`` and bias ``b`` (SigLIP's start, 10 and
    -10)."""
    return torch.nn.ParameterDict({"t": torch.nn.Parameter(torch.tensor(math.log(10.0))),
                                   "b": torch.nn.Parameter(torch.tensor(-10.0))})


def match_loss(out: dict, indices, inst_ids: list[torch.Tensor], set_ids: torch.Tensor, params: torch.nn.ParameterDict,
               *, negatives: torch.Tensor | None = None, stats: dict | None = None):
    """Loss of the matching head (``out["match_embed"]``) over the matched queries of each set of photos: a binary "same
    instance?" for every two queries of two different photos of a set, with logit ``exp(t) * cos + b`` (SigLIP,
    ``params`` from :func:`match_loss_params`), summed per query. It calibrates the absolute similarity, so that one
    global threshold separates the links, as inference makes them.

    Two matched queries are the same instance when their targets carry the same instance id (``inst_ids``) in two
    different photos of a set. ``indices`` is the query->target matching (gated as for the attribute heads). Targets
    with a negative id (an instance whose links are unknown) take no part. ``negatives`` ``(B, Q)`` bool marks more
    queries that are no instance (the confident unmatched queries of fully annotated photos: the false detections and
    duplicates that make false links at inference); they are negatives of the queries of the OTHER photos of their
    set. Computed in fp32 even under autocast: its absolute scale is what it calibrates.

    ``stats`` (a dict) receives, as device tensors (NaN when undefined), ``pos_cos``, the mean cosine of the pairs of one
    instance; ``hardneg_cos``, the mean over the queries with a partner of their highest cosine with a query of another
    photo that is not it; and ``unpaired``, the share of the queries of known instances in sets of several photos
    without a partner in their set. Returns 0 when nothing pairs up.
    """
    emb = out["match_embed"]
    dev = emb.device
    feats, ids, photo, sets, extra = [], [], [], [], []
    for b, (src, tgt) in enumerate(indices):
        if src.numel():
            src, tgt = src.to(dev), tgt.to(dev)
            feats.append(emb[b, src])
            ids.append(inst_ids[b].to(dev)[tgt])
            photo.append(torch.full_like(src, b))
            sets.append(torch.full_like(src, int(set_ids[b])))
            extra.append(torch.zeros_like(src, dtype=torch.bool))
        if negatives is not None:
            q = negatives[b].nonzero().flatten().to(dev)
            if q.numel():
                feats.append(emb[b, q])
                ids.append(torch.full_like(q, -1))
                photo.append(torch.full_like(q, b))
                sets.append(torch.full_like(q, int(set_ids[b])))
                extra.append(torch.ones_like(q, dtype=torch.bool))
    zero = emb.sum() * 0.0
    if not feats:
        return zero
    f, ids, photo, sets, extra = (torch.cat(t) for t in (feats, ids, photo, sets, extra))
    keep = (ids >= 0) | extra
    f, ids, photo, sets, extra = f[keep], ids[keep], photo[keep], sets[keep], extra[keep]
    known = ids >= 0
    if not known.any():
        return zero
    other = (sets[:, None] == sets[None]) & (photo[:, None] != photo[None])
    pos = other & known[:, None] & known[None] & (ids[:, None] == ids[None])
    with torch.autocast(dev.type, enabled=False):  # fp32: autocast would make the cosines bf16
        cos = f.float() @ f.float().T
        if stats is not None:
            with torch.no_grad():
                has = pos.any(1)
                negs = other & ~pos & (known[None] | extra[None])
                hn = cos.masked_fill(~negs, float("-inf")).amax(1)[has]
                hn = hn[hn.isfinite()]
                multi = torch.bincount(set_ids.cpu().long()).to(dev)[sets] > 1  # an instance of a set of several photos
                stats.update(pos_cos=cos[pos].mean(), hardneg_cos=hn.mean(),  # the mean of nothing is NaN
                             unpaired=(known & ~has & multi).sum() / (known & multi).sum())
        pairs = other & ~(extra[:, None] & extra[None]) & torch.ones_like(other).triu(1)
        if not pairs.any():
            return zero
        logit = params["t"].float().exp() * cos + params["b"].float()
        terms = F.softplus(-(pos.float() * 2 - 1) * logit)[pairs]
        return terms.sum() / (pairs.any(1) | pairs.any(0)).sum()


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

    For a regression head the pair is ``(sum of absolute errors in real units, total)``, so its ratio is the MAE.
    Ignored labels (``ignore_index``, NaN) are excluded from both correct and total.
    Pass ``indices`` to reuse a matching already computed this step.
    """
    if indices is None:
        indices = match_queries(model, out, mask_labels, class_labels)
    matched, batch_tgt = _gather_matched(out, indices)
    if matched is None:
        return {name: (0, 0) for name in model.aux_heads}
    specs = {s.name: s for s in model.aux_specs}
    res: dict[str, tuple] = {}
    for name, head in model.aux_heads.items():
        hit, valid = _hits(specs[name], head(matched), aux_labels[name], batch_tgt, ignore_index)
        res[name] = (hit.sum().item() if specs[name].regression else int(hit.sum()), int(valid.sum()))
    return res


@torch.no_grad()
def aux_regression_pairs(model, out: dict, aux_labels: dict[str, list[torch.Tensor]], indices):
    """``{head: (predictions, GT values)}`` of every regression head over the matched queries ``indices`` with a valid
    GT value, in real units, on the CPU (for :func:`eomt.engine.validate.regression_metrics`)."""
    matched, batch_tgt = _gather_matched(out, indices)
    res = {}
    for spec in model.aux_specs:
        if not spec.regression:
            continue
        if matched is None:
            res[spec.name] = (torch.zeros(0), torch.zeros(0))
            continue
        gt = torch.cat([aux_labels[spec.name][b][tgt] for (b, tgt) in batch_tgt]).to(matched.device).float()
        valid = ~gt.isnan()
        pred = spec.decode(model.aux_heads[spec.name](matched)[:, 0].float())
        res[spec.name] = (pred[valid].cpu(), gt[valid].cpu())
    return res


def _hits(spec, logits, labels, batch_tgt, ignore_index):
    """Per matched query: correct (bool), or for a regression head the absolute error in real units; and validity."""
    gt = torch.cat([labels[b][tgt] for (b, tgt) in batch_tgt]).to(logits.device)
    if spec.regression:
        valid = ~gt.isnan()
        return (spec.decode(logits[:, 0].float()) - gt).abs().nan_to_num(0.0), valid
    valid = gt != ignore_index
    return (logits.argmax(-1) == gt) & valid, valid


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
    a diagnostic for *which primary classes* the attribute is (in)accurate on (a regression head: the summed absolute
    error instead of ``correct``, as in :func:`aux_accuracy`). Pass
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
    specs = {s.name: s for s in model.aux_specs}
    for name, head in model.aux_heads.items():
        hit, valid = _hits(specs[name], head(matched), aux_labels[name], batch_tgt, ignore_index)
        d: dict[int, tuple] = {}
        for c in prim[valid].unique().tolist():
            sel = (prim == c) & valid
            d[int(c)] = (hit[sel].sum().item() if specs[name].regression else int(hit[sel].sum()), int(sel.sum()))
        res[name] = d
    return res
