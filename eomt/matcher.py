"""Linking the instances of several photos of one subject: a learned set matcher (SuperGlue / LightGlue style).

Each instance of a photo enters as its query embedding (what the class and attribute heads read) plus the box of its
mask on the input canvas. A few layers of attention let it look at the other instances of its own photo (self) and at
those of the other photos of the set (cross), so a match is decided in context: of three similar scratches on one
bumper, the matcher can tell which is which from where each sits relative to the others. The scores of two instances
of two photos, with a "no partner in that photo" bin, go through a softmax over each photo's instances in both
directions (dual softmax): ``P(i <-> j) = softmax_j(S_i.) * softmax_i(S_.j)`` in ``[0, 1]``, the similarity the
linker (Hungarian per photo pair, union-find; :mod:`eomt.reid`) and its calibrated threshold work with.

Training (:func:`matcher_loss`) reads the ``instance_id`` links: two instances of two photos with the same id are a
pair, an instance with a known id and no partner in a photo goes to that photo's bin, and so do the confident queries
no target matched (the false detections and duplicates that make false links at inference).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#: The matcher of a multi-view post-training (``EoMTModel(matcher=...)``, saved in the checkpoint).
DEFAULT_MATCHER: dict = {"dim": 256, "layers": 3, "heads": 4}


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        m, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(m, 3, self.heads, d // self.heads).permute(1, 2, 0, 3)  # [h, M, dh]
        y = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=allowed[None, None])
        return x + self.out(y[0].transpose(0, 1).reshape(m, d))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.self_attn = _Attention(dim, heads)
        self.cross_attn = _Attention(dim, heads)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, same, cross):
        x = self.cross_attn(self.self_attn(x, same), cross)
        return x + self.ffn(self.norm(x))


class SetMatcher(nn.Module):
    """Scores every two instances of two photos of a set (see the module docstring). ``hidden``: the query embedding's
    width; ``dim`` / ``layers`` / ``heads``: the matcher's own."""

    def __init__(self, hidden: int, dim: int = 256, layers: int = 3, heads: int = 4):
        super().__init__()
        self.dim = dim
        self.inp = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim))
        self.pos = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.bin = nn.Parameter(torch.tensor(1.0))  # the score of "no partner in that photo"

    def forward(self, feats: torch.Tensor, boxes: torch.Tensor, photo: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """``feats`` ``(M, hidden)``, ``boxes`` ``(M, 4)`` (``x0, y0, x1, y1`` in ``[0, 1]``), ``photo`` / ``group``
        ``(M,)`` ids. Returns the log-probabilities ``(M, M)`` of every two instances of two photos of one group being
        the same (``-inf`` for any other pair) and ``(M, P)`` of each instance having no partner in photo ``p``."""
        photo, group = photo.to(feats.device), group.to(feats.device)
        same_group = group[:, None] == group[None]
        same = (photo[:, None] == photo[None]) & same_group
        cross = (same_group & ~same) | torch.eye(len(photo), dtype=torch.bool, device=feats.device)  # never empty
        x = self.inp(feats.float()) + self.pos(boxes.float())
        for blk in self.blocks:
            x = blk(x, same, cross)
        d = self.proj(self.norm(x))
        scores = (d @ d.T) / math.sqrt(self.dim)
        return dual_log_softmax(scores, photo, same_group & ~same, self.bin)


def dual_log_softmax(scores: torch.Tensor, photo: torch.Tensor, valid: torch.Tensor, bin_score: torch.Tensor):
    """``(log P (M, M), log P_bin (M, P))``: for instance ``i`` and another photo ``b`` of its group, a softmax over
    ``b``'s instances and the bin; ``log P(i <-> j)`` adds the two directions, ``log P_bin[i, b]`` is ``i``'s bin
    toward ``b``. ``photo`` holds ids ``0..P-1``; ``valid`` marks the pairs of two photos of one group."""
    m = scores.shape[0]
    n_photos = int(photo.max()) + 1 if m else 0
    s = scores.masked_fill(~valid, float("-inf"))
    onehot = F.one_hot(photo, n_photos).to(s.dtype)  # [M, P]
    top = torch.maximum(s.max(1, keepdim=True).values.clamp_min(-1e4), bin_score)  # [M, 1], finite
    per_photo = torch.exp(s - top) @ onehot  # [M, P]: sum over each photo's instances
    lse = top + torch.log(per_photo + torch.exp(bin_score - top))  # [M, P]
    row = s - lse.gather(1, photo[None].expand(m, -1))  # [M, M]: log softmax of j among photo(j) and the bin, for i
    log_p = row + row.T
    log_bin = bin_score - lse
    return log_p.masked_fill(~valid, float("-inf")), log_bin


def mask_boxes(mask_logits: torch.Tensor) -> torch.Tensor:
    """``(M, 4)`` boxes ``x0, y0, x1, y1`` in ``[0, 1]`` of the masks ``(M, h, w)`` (logits > 0) on the canvas they are
    predicted on; an empty mask gives zeros."""
    m = mask_logits > 0
    if not m.numel():
        return mask_logits.new_zeros(mask_logits.shape[0], 4)
    h, w = m.shape[-2:]
    rows, cols = m.any(2), m.any(1)
    ar_h = torch.arange(h, device=m.device)
    ar_w = torch.arange(w, device=m.device)
    big = torch.iinfo(torch.long).max
    y0 = torch.where(rows, ar_h, big).amin(1)
    y1 = torch.where(rows, ar_h, -1).amax(1) + 1
    x0 = torch.where(cols, ar_w, big).amin(1)
    x1 = torch.where(cols, ar_w, -1).amax(1) + 1
    box = torch.stack((x0 / w, y0 / h, x1 / w, y1 / h), 1).float()
    return torch.where(rows.any(1, keepdim=True), box, torch.zeros_like(box))


def matcher_loss(log_p: torch.Tensor, log_bin: torch.Tensor, ids: torch.Tensor, photo: torch.Tensor,
                 group: torch.Tensor, known: torch.Tensor, stats: dict | None = None) -> torch.Tensor:
    """The matcher's loss on one batch of instances: ``-log P(i <-> j)`` for every pair of two photos with the same
    known id, ``-log P_bin[i, b]`` for an instance of known id (``known``; a negative id is "no instance") with no
    partner in a photo ``b`` of its group; the mean of all. Instances whose links are unknown (not ``known``) take
    part only as context. ``stats`` receives the mean probability of the true pairs (``pos_p``) and the share of the
    instances of a set of several photos with a partner (``paired``)."""
    dev = log_p.device
    ids, photo, group, known = ids.to(dev), photo.to(dev), group.to(dev), known.to(dev)
    pair_ok = (group[:, None] == group[None]) & (photo[:, None] != photo[None])
    pos = pair_ok & known[:, None] & known[None] & (ids[:, None] == ids[None]) & (ids[:, None] >= 0)
    pos = pos & torch.ones_like(pos).triu(1)
    n_photos = log_bin.shape[1]
    # the photos of each instance's group, other than its own: where it has a partner or not
    photo_group = torch.full((n_photos,), -1, dtype=group.dtype, device=dev).scatter(0, photo, group)
    in_group = (photo_group[None] == group[:, None]) & (torch.arange(n_photos, device=dev)[None] != photo[:, None])
    onehot = F.one_hot(photo, n_photos).bool()
    pos_sym = pos | pos.T
    has_partner = (pos_sym.float() @ onehot.float()) > 0  # [M, P]
    unpaired = in_group & ~has_partner & known[:, None]
    terms = torch.cat((-log_p[pos], -log_bin[unpaired]))
    if stats is not None:
        with torch.no_grad():
            multi = in_group.any(1) & known & (ids >= 0)
            stats.update(pos_p=log_p[pos].exp().mean() if pos.any() else torch.tensor(float("nan"), device=dev),
                         paired=(pos_sym.any(1) & multi).sum() / multi.sum().clamp_min(1))
    if not terms.numel():
        return log_p.new_zeros(()) + log_bin.sum() * 0.0
    return terms.mean()


@torch.no_grad()
def link_probabilities(matcher: SetMatcher, feats: list[torch.Tensor], boxes: list[torch.Tensor],
                       groups: list[list[int]] | None = None) -> torch.Tensor:
    """``(M, M)`` probabilities that two instances of two photos are the same, for the instances of every photo
    (``feats[p]`` ``(n_p, hidden)``, ``boxes[p]`` ``(n_p, 4)``), flattened in photo order. The photos of one of
    ``groups`` (lists of photo indices, e.g. the groups they were predicted in) are matched together, in context;
    every other two photos as a pair. Pairs within a photo are 0."""
    dev = next(matcher.parameters()).device
    sizes = [len(f) for f in feats]
    offsets = [0]
    for n in sizes:
        offsets.append(offsets[-1] + n)
    out = torch.zeros(offsets[-1], offsets[-1])
    n = len(feats)
    done = torch.eye(n, dtype=torch.bool)
    runs = [list(g) for g in (groups or []) if len(g) > 1]
    for g in runs:
        for a in g:
            for b in g:
                done[a, b] = True
    runs += [[a, b] for a in range(n) for b in range(a + 1, n) if not done[a, b]]
    for g in runs:
        g = [p for p in g if sizes[p]]
        if len(g) < 2:
            continue
        f = torch.cat([feats[p] for p in g]).to(dev)
        bx = torch.cat([boxes[p] for p in g]).to(dev)
        ph = torch.cat([torch.full((sizes[p],), k, dtype=torch.long) for k, p in enumerate(g)]).to(dev)
        log_p, _ = matcher(f, bx, ph, torch.zeros_like(ph))
        prob = log_p.exp().float().cpu()
        idx = torch.cat([torch.arange(offsets[p], offsets[p + 1]) for p in g])
        out[idx[:, None], idx[None]] = torch.maximum(out[idx[:, None], idx[None]], prob)
    return out


def set_match_loss(model, out: dict, indices, inst_ids: list[torch.Tensor], set_ids: torch.Tensor,
                   negatives: torch.Tensor | None = None, stats: dict | None = None) -> torch.Tensor:
    """The matcher's loss on a training batch: its instances are the queries matched to a target (``indices``; the
    target's ``inst_ids`` entry is its id, negative when its links are unknown) and the confident unmatched queries
    (``negatives`` ``(B, Q)``: no instance, so no partner anywhere), each with its query embedding and the box of its
    final mask; the photos of a set (``set_ids``) are matched together. Computed in fp32 (its probabilities are what
    the links' threshold is calibrated on). See :func:`matcher_loss`."""
    q_embed, masks = out["query_embed"], out["masks_queries_logits"]
    dev = q_embed.device
    sel, ids, known, photo, group = [], [], [], [], []
    for b, (src, tgt) in enumerate(indices):
        src, tgt = src.to(dev), tgt.to(dev)
        t_ids = inst_ids[b].to(dev)[tgt]
        neg = negatives[b].nonzero().flatten() if negatives is not None else src.new_zeros(0)
        q = torch.cat((src, neg))
        sel.append(torch.stack((torch.full_like(q, b), q), 1))
        ids.append(torch.cat((t_ids, torch.full_like(neg, -1))))
        known.append(torch.cat((t_ids >= 0, torch.ones_like(neg, dtype=torch.bool))))
        photo.append(torch.full_like(q, b))
        group.append(torch.full_like(q, int(set_ids[b])))
    sel = torch.cat(sel)
    if not len(sel):
        return sum(p.sum() for p in model.matcher.parameters()) * 0.0
    feats = q_embed[sel[:, 0], sel[:, 1]]
    boxes = mask_boxes(masks[sel[:, 0], sel[:, 1]].detach())
    photo = torch.cat(photo)
    with torch.autocast(dev.type, enabled=False):
        log_p, log_bin = model.matcher(feats.float(), boxes, photo, torch.cat(group))
        return matcher_loss(log_p, log_bin, torch.cat(ids), photo, torch.cat(group), torch.cat(known), stats=stats)
