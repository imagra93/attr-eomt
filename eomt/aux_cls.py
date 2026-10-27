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
    # Instance family; with the auxiliary box head the matcher also uses the box cost, as the loss did.
    boxes = {"pred_boxes": out["aux_boxes"]} if out.get("aux_boxes") is not None else {}
    return model.eomt.criterion.matcher(
        out["masks_queries_logits"], out["class_queries_logits"], geom_labels, class_labels, **boxes
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


def _density_weights(bins, z: torch.Tensor) -> torch.Tensor:
    """Per-sample weight of standardized regression targets ``z`` from ``bins = (edges, weights)``."""
    edges, w = (t.to(z.device, z.dtype) for t in bins)
    return w[torch.bucketize(z, edges[1:-1])]


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
    log_vars=None,
    ignore_index: int = -100,
):
    """Weighted sum of per-attribute losses over matched queries: CE, or Huber for a regression head.

    ``aux_labels`` maps head name → per-image label tensors (line-aligned with
    ``class_labels``). Labels equal to ``ignore_index`` (missing / out-of-vocab
    attributes; NaN for a regression head) are skipped — they contribute no loss instead of being trained as
    class 0. ``weights`` scales each head; ``class_weights`` optionally re-weights
    classes within a head (imbalance), or, for a regression head, its values by density (``(edges, weights)``
    over the standardized values). ``log_vars`` (``{name: scalar parameter s}``) turns on uncertainty weighting
    (Kendall et al., 2018): each head contributes ``exp(-s) * loss + s``, so ``exp(-s)`` is a learned weight.
    Pass ``indices`` to reuse a matching already
    computed this step (avoids re-running the Hungarian matcher). Returns
    ``(total_loss, {name: per_head_loss})``, the per-head losses unweighted.
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
        cw = class_weights.get(name)
        if specs[name].regression:
            valid = ~gt.isnan()
            if valid.any():
                z = specs[name].encode(gt[valid].float())
                err = F.huber_loss(logits[valid, 0].float(), z, delta=1.0, reduction="none")
                sw = _density_weights(cw, z) if cw is not None else None
                loss = (err * sw).sum() / sw.sum() if sw is not None else err.mean()
            else:  # every matched value ignored — keep the graph alive with a zero
                loss = logits.sum() * 0.0
        elif (gt != ignore_index).any():
            if cw is not None:
                cw = cw.to(logits.device, logits.dtype)
            loss = F.cross_entropy(logits, gt, weight=cw, ignore_index=ignore_index)
        else:  # every matched label ignored — keep the graph alive with a zero
            loss = logits.sum() * 0.0
        per_head[name] = loss
        w = float(weights.get(name, 1.0))
        if log_vars is not None:
            loss = torch.exp(-log_vars[name]) * loss + log_vars[name]
        total = w * loss if total is None else total + w * loss
    return total, per_head


def match_loss(out: dict, indices, inst_ids: list[torch.Tensor], set_ids: torch.Tensor, tau: float = 0.1):
    """Contrastive loss of the matching head (``out["match_embed"]``) over the matched queries of each set of photos.

    Two matched queries are positives when their targets carry the same instance id in two different photos of a set;
    every other matched query of the set (in any of its photos) is a negative. Each query with a positive is an anchor:
    ``-log`` of the softmax mass (temperature ``tau``) its positives take among all its candidates (supervised InfoNCE).
    ``indices`` is the query->target matching (gated as for the attribute heads); returns 0 when nothing pairs up.
    Targets with a negative id (another photo of the subject, which of its instances are which unknown,
    :data:`eomt.data.sets.UNKNOWN_IDS`) are neither anchors nor candidates.
    """
    emb = out["match_embed"]
    feats, ids, photo, sets = [], [], [], []
    for b, (src, tgt) in enumerate(indices):
        if src.numel():
            feats.append(emb[b, src])
            ids.append(inst_ids[b].to(emb.device)[tgt])
            photo.append(torch.full_like(src, b, device=emb.device))
            sets.append(torch.full_like(src, int(set_ids[b]), device=emb.device))
    if not feats:
        return emb.sum() * 0.0
    f, ids, photo, sets = torch.cat(feats), torch.cat(ids), torch.cat(photo), torch.cat(sets)
    known = ids >= 0
    if not known.any():
        return emb.sum() * 0.0
    f, ids, photo, sets = f[known], ids[known], photo[known], sets[known]
    sim = (f @ f.T) / tau
    cand = (sets[:, None] == sets[None]) & ~torch.eye(len(f), dtype=torch.bool, device=f.device)
    pos = cand & (ids[:, None] == ids[None]) & (photo[:, None] != photo[None])
    anchors = pos.any(1)
    if not anchors.any():
        return emb.sum() * 0.0
    sim, cand, pos = sim[anchors], cand[anchors], pos[anchors]
    neg_inf = torch.finfo(sim.dtype).min
    return (sim.masked_fill(~cand, neg_inf).logsumexp(1) - sim.masked_fill(~pos, neg_inf).logsumexp(1)).mean()


def harm_loss(out: dict, alone: dict, indices, class_labels: list[torch.Tensor], set_ids: torch.Tensor):
    """Do-no-harm loss of the cross-view post-training: how much predicting a photo with the other photos of its set
    lowers the probability of a true instance's class below predicting it alone, hinged at 0 (the others may raise it,
    never lower it). ``alone`` is the forward of the same photos each in its own set (with the checkpoint frozen, its
    own prediction); ``indices`` the query->target matching of ``out``. Mean over the matched queries of the photos in
    sets of several; 0 when there are none.

    NOTE: what is lost to a lowered true instance downstream (an FP filter can drop false detections, not bring back a
    true one) is not what is gained by a lowered false one, so only the true instances are held to their score alone.
    """
    sizes = torch.bincount(set_ids.cpu())
    drops = []
    for b, (src, tgt) in enumerate(indices):
        if src.numel() and sizes[int(set_ids[b])] > 1:
            cls = class_labels[b][tgt.to(class_labels[b].device)].to(out["class_queries_logits"].device)[:, None]
            p = out["class_queries_logits"][b, src].float().softmax(-1).gather(1, cls)[:, 0]
            p_alone = alone["class_queries_logits"][b, src].float().softmax(-1).gather(1, cls)[:, 0]
            drops.append((p_alone - p).clamp(min=0))
    if not drops:
        return out["class_queries_logits"].sum() * 0.0
    return torch.cat(drops).mean()


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
