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
    target_weights: list[torch.Tensor] | None = None,
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
    computed this step (avoids re-running the Hungarian matcher). ``target_weights`` (one weight per target of each
    image) makes each head's loss a weighted mean over its matched queries. Returns
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
    tw = None
    if target_weights is not None:
        tw = torch.cat([target_weights[b].to(matched.device)[tgt.to(matched.device)] for (b, tgt) in batch_tgt]).float()
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
                if tw is not None:
                    sw = tw[valid] if sw is None else sw * tw[valid]
                loss = (err * sw).sum() / sw.sum().clamp(min=1e-12) if sw is not None else err.mean()
            else:  # every matched value ignored — keep the graph alive with a zero
                loss = logits.sum() * 0.0
        elif (gt != ignore_index).any():
            if cw is not None:
                cw = cw.to(logits.device, logits.dtype)
            if tw is None:
                loss = F.cross_entropy(logits, gt, weight=cw, ignore_index=ignore_index)
            else:  # the weighted mean F.cross_entropy takes, with each query's weight too
                valid = gt != ignore_index
                per = F.cross_entropy(logits, gt, weight=cw, ignore_index=ignore_index, reduction="none")
                denom = tw[valid] * (cw[gt[valid]] if cw is not None else 1.0)
                loss = (per[valid] * tw[valid]).sum() / denom.sum().clamp(min=1e-12)
        else:  # every matched label ignored — keep the graph alive with a zero
            loss = logits.sum() * 0.0
        per_head[name] = loss
        w = float(weights.get(name, 1.0))
        if log_vars is not None:
            loss = torch.exp(-log_vars[name]) * loss + log_vars[name]
        total = w * loss if total is None else total + w * loss
    return total, per_head


#: The matching head's losses (``multiview["match_loss"]``, see :func:`match_loss`).
MATCH_LOSSES = ("infonce", "sigmoid", "dustbin")
#: Sinkhorn iterations of the ``"dustbin"`` matching loss.
SINKHORN_ITERS = 50


def match_loss_params(kind: str, tau: float = 0.1) -> torch.nn.ParameterDict:
    """The learned scalars of a matching loss: ``"sigmoid"`` its log-temperature ``t`` and bias ``b`` (SigLIP's start:
    10 and -10), ``"dustbin"`` its "no partner" score (in units of ``cos / tau``, starting at a cosine of 0.5);
    ``"infonce"`` none."""
    if kind not in MATCH_LOSSES:
        raise ValueError(f"match_loss {kind!r}: choose from {MATCH_LOSSES}")
    init = {"sigmoid": {"t": math.log(10.0), "b": -10.0}, "dustbin": {"dustbin": 0.5 / tau}}.get(kind, {})
    return torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(float(v))) for k, v in init.items()})


def _log_sinkhorn(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """Log of the optimal-transport assignment of an ``(m, n)`` score matrix augmented with a dustbin row and column of
    score ``alpha`` (SuperGlue): every row and column sends one unit, a dustbin takes what the others leave. ``(m + 1, n
    + 1)``, scaled so that a row's (a column's) probabilities sum to 1."""
    m, n = scores.shape
    a = alpha.to(scores)
    couplings = torch.cat([torch.cat([scores, a.expand(m, 1)], 1), torch.cat([a.expand(1, n), a.view(1, 1)], 1)], 0)
    norm = -math.log(m + n)
    log_mu = torch.cat([scores.new_full((m,), norm), scores.new_full((1,), math.log(n) + norm)])
    log_nu = torch.cat([scores.new_full((n,), norm), scores.new_full((1,), math.log(m) + norm)])
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(couplings + v[None], 1)
        v = log_nu - torch.logsumexp(couplings + u[:, None], 0)
    return couplings + u[:, None] + v[None] - norm


def match_loss(out: dict, indices, inst_ids: list[torch.Tensor], set_ids: torch.Tensor, tau: float = 0.1, *,
               kind: str = "infonce", negatives: torch.Tensor | None = None,
               hard_keys: list[torch.Tensor] | None = None, hard_w: float = 1.0,
               params: torch.nn.ParameterDict | None = None, stats: dict | None = None,
               incomplete: torch.Tensor | None = None):
    """Loss of the matching head (``out["match_embed"]``) over the matched queries of each set of photos.

    Two matched queries are the same instance when their targets carry the same instance id in two different photos of
    a set. ``indices`` is the query->target matching (gated as for the attribute heads). Targets with a negative id
    (another photo of the subject, which of its instances are which unknown, :data:`eomt.data.sets.UNKNOWN_IDS`) take
    no part. ``negatives`` ``(B, Q)`` bool marks more queries that are no instance (e.g. the confident unmatched queries
    of fully annotated photos): they join as negatives of the queries of the OTHER photos of their set — the false
    detections and duplicates that make the false links at inference.

    ``kind`` (:data:`MATCH_LOSSES`):

    * ``"infonce"`` (supervised InfoNCE, temperature ``tau``): each query with a partner in another photo is an anchor:
      ``-log`` of the softmax mass its partners take among its candidates (the other matched queries of its set, in any
      photo, and the extra negatives of the other photos). Relative per anchor: it trains no absolute scale.
    * ``"sigmoid"`` (SigLIP): every two queries of two different photos of a set make a binary "same instance?" with
      logit ``exp(t) * cos + b`` (``params``, :func:`match_loss_params`), summed per query: it calibrates the absolute
      similarity, so that one global threshold separates the links.
    * ``"dustbin"`` (SuperGlue): for every two photos of a set, the optimal-transport assignment (log-domain Sinkhorn) of
      their queries' similarities (``cos / tau``) with a learned "no partner" score (``params["dustbin"]``); every query
      is an anchor and its target is its partner, or the dustbin when the other photo does not show it: what the
      Hungarian matching plus a threshold decides at inference. "Does not show it" must be known: an instance whose
      partner is among the other photo's targets (``inst_ids``) but not among its matched queries (gated out), or an
      instance facing a photo that may show it unannotated (a target of unknown id, or ``incomplete`` ``(B,)`` bool:
      partly annotated, ignore regions), takes no target; the extra negatives always take the dustbin.

    The sigmoid and dustbin losses compute in fp32 even under autocast: their absolute scale is what they calibrate.

    ``hard_keys`` (one int key per target of each photo, e.g. its class and attributes; negative = none) with
    ``hard_w``: a negative pair whose targets share a key weighs ``hard_w`` ("infonce": its logit plus ``log hard_w``;
    "sigmoid": its term times ``hard_w``). ``stats`` (a dict) receives ``pos_cos``, the mean cosine of the pairs of one
    instance; ``hardneg_cos``, the mean over the queries with a partner of their highest cosine with a query of another
    photo that is not it; and ``unpaired``, the share of the queries of known instances in sets of several photos
    without a partner in their set. Returns 0 when nothing pairs up.
    """
    if kind not in MATCH_LOSSES:
        raise ValueError(f"match_loss {kind!r}: choose from {MATCH_LOSSES}")
    emb = out["match_embed"]
    dev = emb.device
    feats, ids, photo, sets, keys, extra = [], [], [], [], [], []
    for b, (src, tgt) in enumerate(indices):
        if src.numel():
            src, tgt = src.to(dev), tgt.to(dev)
            feats.append(emb[b, src])
            ids.append(inst_ids[b].to(dev)[tgt])
            photo.append(torch.full_like(src, b))
            sets.append(torch.full_like(src, int(set_ids[b])))
            keys.append(hard_keys[b].to(dev)[tgt] if hard_keys is not None else torch.full_like(src, -1))
            extra.append(torch.zeros_like(src, dtype=torch.bool))
        if negatives is not None and bool(negatives[b].any()):
            q = negatives[b].nonzero().flatten().to(dev)
            feats.append(emb[b, q])
            ids.append(torch.full_like(q, -1))
            photo.append(torch.full_like(q, b))
            sets.append(torch.full_like(q, int(set_ids[b])))
            keys.append(torch.full_like(q, -1))
            extra.append(torch.ones_like(q, dtype=torch.bool))
    zero = emb.sum() * 0.0
    if not feats:
        return zero
    f, ids, photo, sets, keys, extra = (torch.cat(t) for t in (feats, ids, photo, sets, keys, extra))
    keep = (ids >= 0) | extra
    if not (ids >= 0).any():
        return zero
    f, ids, photo, sets, keys, extra = f[keep], ids[keep], photo[keep], sets[keep], keys[keep], extra[keep]
    known = ids >= 0
    same_set = sets[:, None] == sets[None]
    other = same_set & (photo[:, None] != photo[None])
    pos = other & known[:, None] & known[None] & (ids[:, None] == ids[None])
    hard = (keys[:, None] == keys[None]) & (keys[:, None] >= 0) & ~pos
    cos = f.float() @ f.float().T
    if stats is not None:
        with torch.no_grad():
            has = pos.any(1)
            negs = other & ~pos & (known[None] | extra[None])
            hn = cos.masked_fill(~negs, float("-inf")).amax(1)[has]
            multi = torch.bincount(set_ids.cpu().long()).to(dev)[sets] > 1  # an instance of a set of several photos
            stats.update(pos_cos=float(cos[pos].mean()) if pos.any() else float("nan"),
                         hardneg_cos=float(hn[hn.isfinite()].mean()) if bool(hn.isfinite().any()) else float("nan"),
                         unpaired=float((known & ~has & multi).sum() / (known & multi).sum())
                         if bool((known & multi).any()) else float("nan"))
    if kind == "infonce":
        cand = same_set & ~torch.eye(len(f), dtype=torch.bool, device=dev) & (~extra[None] | other)
        anchors = pos.any(1)
        if not anchors.any():
            return zero
        sim = cos / tau
        if hard_w != 1.0:
            sim = sim + hard.float() * math.log(hard_w)
        sim, cand, p = sim[anchors], cand[anchors], pos[anchors]
        neg_inf = torch.finfo(sim.dtype).min
        return (sim.masked_fill(~cand, neg_inf).logsumexp(1) - sim.masked_fill(~p, neg_inf).logsumexp(1)).mean()
    with torch.autocast(dev.type, enabled=False):  # the calibrated losses in fp32 (autocast would make cos bf16)
        cos = f.float() @ f.float().T
        if kind == "sigmoid":
            pairs = other & ~(extra[:, None] & extra[None]) & torch.ones_like(other).triu(1)
            if not pairs.any():
                return zero
            logit = params["t"].float().exp() * cos + params["b"].float()
            sign = pos.float() * 2 - 1
            w = torch.where(hard, torch.full_like(cos, float(hard_w)), torch.ones_like(cos))
            terms = (F.softplus(-sign * logit) * w)[pairs]
            involved = (pairs.any(1) | pairs.any(0)).sum()
            return terms.sum() / involved
        # dustbin: every two photos of a set. "No partner" is a target only where it is known: against a photo whose
        # every instance has a known id and is annotated, for an instance it does not show (and for a false detection).
        shows = [set(t[t >= 0].tolist()) for t in inst_ids]
        complete = [not bool((t < 0).any()) and not (incomplete is not None and bool(incomplete[b]))
                    for b, t in enumerate(inst_ids)]

        def no_partner(rows: torch.Tensor, other_photo: int) -> torch.Tensor:
            if not complete[other_photo]:
                return extra[rows]
            absent = torch.tensor([int(i) not in shows[other_photo] for i in ids[rows].tolist()], dtype=torch.bool,
                                  device=dev)
            return extra[rows] | (known[rows] & absent)

        terms = []
        alpha = params["dustbin"].float()
        for a in photo.unique().tolist():
            for b in photo.unique().tolist():
                if b <= a:
                    continue
                ra, cb = (photo == a).nonzero().flatten(), (photo == b).nonzero().flatten()
                if not len(ra) or not len(cb) or sets[ra[0]] != sets[cb[0]]:
                    continue
                z = _log_sinkhorn(cos[ra][:, cb] / tau, alpha, SINKHORN_ITERS)
                p = pos[ra][:, cb]
                terms += [z[:-1, :-1][p], z[:-1, -1][~p.any(1) & no_partner(ra, b)],  # photo a's queries facing b
                          z[-1, :-1][~p.any(0) & no_partner(cb, a)]]  # and b's facing a
        if not terms:
            return zero
        return -torch.cat(terms).mean()


def consistency_loss(model, out: dict, indices, inst_ids: list[torch.Tensor], set_ids: torch.Tensor,
                     heads: list[str], class_labels: list[torch.Tensor]):
    """Agreement of the attribute heads across the photos of a set: for every two matched queries of two different
    photos whose targets are the same instance (same known id), the symmetric KL divergence of a categorical head's
    distributions, or the Huber distance of a regression head's standardized outputs; the mean over the pairs, then the
    heads. A class-scoped head takes only the pairs of a class it applies to. No label is needed: an instance that one
    photo's annotation leaves without a value is still pulled towards what its other views predict."""
    q = out["query_embed"]
    dev = q.device
    feats, ids, photo, sets, cls = [], [], [], [], []
    for b, (src, tgt) in enumerate(indices):
        if src.numel():
            src, tgt = src.to(dev), tgt.to(dev)
            feats.append(q[b, src])
            ids.append(inst_ids[b].to(dev)[tgt])
            photo.append(torch.full_like(src, b))
            sets.append(torch.full_like(src, int(set_ids[b])))
            cls.append(class_labels[b].to(dev)[tgt])
    zero = q.sum() * 0.0
    if not feats:
        return zero
    f, ids, photo, sets, cls = (torch.cat(t) for t in (feats, ids, photo, sets, cls))
    same = (ids[:, None] == ids[None]) & (ids[:, None] >= 0) & (sets[:, None] == sets[None])
    i, j = (same & (photo[:, None] != photo[None])).triu(1).nonzero(as_tuple=True)
    if not len(i):
        return zero
    specs = {s.name: s for s in model.aux_specs}
    terms = []
    for name in heads:
        spec, head = specs[name], model.aux_heads[name]
        sel = torch.ones_like(i, dtype=torch.bool)
        if spec.applies_to is not None:
            sel = torch.tensor([int(c) in spec.applies_to for c in cls[i].tolist()], dtype=torch.bool, device=dev)
        if not sel.any():
            continue
        oi, oj = head(f[i[sel]]).float(), head(f[j[sel]]).float()
        if spec.regression:
            terms.append(F.huber_loss(oi[:, 0], oj[:, 0], delta=1.0))
        else:
            li, lj = oi.log_softmax(-1), oj.log_softmax(-1)
            terms.append(0.5 * ((li.exp() * (li - lj)).sum(-1) + (lj.exp() * (lj - li)).sum(-1)).mean())
    return torch.stack(terms).mean() if terms else zero


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
