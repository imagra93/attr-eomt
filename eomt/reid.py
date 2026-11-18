"""Cross-photo instance re-identification from EoMT's own per-query embeddings.

Given several photos of the *same subject* from different viewpoints, decide which
detected instances across those photos are the same physical instance. This is pure
inference: the appearance fingerprint is ``query_embed`` — the per-query embedding the
detector already computes on its way to the class head — so there is no second model
and nothing to retrain.

The recipe is DeepSORT's, applied between photos instead of video frames: detect,
describe, associate with a Hungarian matcher, then link the pairwise associations into
identities. Three properties of EoMT make it work:

* the model is **NMS-free**, so two overlapping instances stay two distinct queries;
* each query "owns" one instance, so its embedding describes that instance alone;
* embeddings are L2-normalized here, so a dot product **is** the cosine similarity.

Everything in this module is pure: tensors in, tensors and plain dicts out. No model,
no file I/O, no PIL — :mod:`eomt.engine.match` supplies those. All instances across all
photos are flattened into one table of ``M`` rows with one ``(M, M)`` similarity matrix;
every function below indexes into that.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

#: Gate key that always resolves, on any checkpoint: the primary class index.
CLASS_KEY = "class"


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------
def validate_group_by(group_by, aux_specs) -> tuple[str, ...]:
    """Normalize and validate the gate keys against a checkpoint's aux heads.

    Two instances are candidates for matching only if they agree on every gate key.
    ``"class"`` is always available; any aux head name may be added to tighten the
    gate. That matters when the primary class is coarser than the distinction you
    care about: if one class covers instances sitting at different places on the
    subject, gating on ``("class", "position")`` stops an instance in one location
    from ever matching one in another, however alike the two look.

    Args:
        group_by: keys to gate on, or ``None`` / ``()`` to disable gating entirely.
        aux_specs: the model's ``aux_specs`` (list of ``AuxHeadSpec``).

    Returns:
        The normalized key tuple (possibly empty).

    Raises:
        ValueError: on an unknown key — a typo must fail loudly, and it must fail
            *before* the first forward pass rather than after N of them.
    """
    if group_by is None:
        return ()
    if isinstance(group_by, str):
        group_by = (group_by,)
    keys = tuple(str(k) for k in group_by)
    if not keys:
        return ()

    by_name = {s.name: s for s in (aux_specs or [])}
    valid = [CLASS_KEY, *by_name]
    unknown = [k for k in keys if k != CLASS_KEY and k not in by_name]
    if unknown:
        raise ValueError(
            f"Unknown group_by key(s) {unknown}. This checkpoint supports: {valid}."
        )
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate group_by key(s) in {keys}.")
    regression = [k for k in keys if k != CLASS_KEY and getattr(by_name[k], "regression", False)]
    if regression:
        raise ValueError(f"group_by cannot gate on regression attribute(s) {regression}: they have no categories.")

    # A class-scoped head reports ``-1`` ("not applicable") for out-of-scope
    # detections. Gating on it *without* "class" would make every out-of-scope
    # instance compare equal, silently merging across classes.
    scoped = [k for k in keys if k != CLASS_KEY and by_name[k].applies_to is not None]
    if scoped and CLASS_KEY not in keys:
        warnings.warn(
            f"group_by={keys} gates on class-scoped head(s) {scoped} without "
            f"{CLASS_KEY!r}: out-of-scope instances all share the -1 sentinel and "
            "will gate together across classes. Add 'class' to group_by.",
            stacklevel=2,
        )
    return keys


def flatten_instances(results: list[dict], *, group_by: tuple[str, ...] = (CLASS_KEY,), dim: int | None = None):
    """Flatten per-photo results into one instance table.

    Args:
        results: per-photo dicts from :func:`eomt.engine.predict.predict_image`
            called with ``embed=True``.
        group_by: validated gate keys (see :func:`validate_group_by`).
        dim: embedding width, used only when *every* photo is empty and there is no
            row to infer it from. Pass ``model.config.hidden_size``.

    Returns:
        ``(embed (M, D), photo (M,), keys, origin)`` where ``keys[m]`` is the gate
        tuple for instance ``m`` (``()`` when gating is off) and ``origin[m]`` is its
        ``(image_index, detection_index)``.
    """
    embeds, photo, keys, origin = [], [], [], []
    for i, res in enumerate(results):
        n = int(res.get("num_detections", 0))
        if n:
            embeds.append(res["embed"].float())
        for j in range(n):
            photo.append(i)
            origin.append((i, j))
            key = []
            for k in group_by:
                if k == CLASS_KEY:
                    key.append(int(res["classes"][j]))
                else:
                    key.append(int(res["aux"][k]["ids"][j]))
            keys.append(tuple(key))

    if embeds:
        embed = torch.cat(embeds)
    else:
        # No detections anywhere: the width cannot be read off the data.
        embed = torch.zeros((0, int(dim or 0)))
    return embed, torch.tensor(photo, dtype=torch.long), keys, origin


def gate_mask(photo: torch.Tensor, keys: list[tuple[int, ...]]) -> torch.Tensor:
    """``(M, M)`` bool: True where a pair is a legal match candidate.

    Legal means *different photo* (one physical instance appears once per photo) and
    *equal gate keys*.

    Vectorized: the key tuples are interned to ints so both conditions become
    broadcast comparisons. The pairwise Python loop this replaces cost ~2.6 s at
    ``M=900`` (30 photos x ``max_det=30``) — more than every other matching step
    combined.
    """
    m = len(keys)
    if m == 0:
        return torch.zeros((0, 0), dtype=torch.bool)
    interned = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    kid = torch.tensor([interned[k] for k in keys], dtype=torch.long)
    out = (photo[:, None] != photo[None, :]) & (kid[:, None] == kid[None, :])
    out.fill_diagonal_(False)  # a pair is two distinct instances
    return out


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def l2_normalize(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise unit norm. A zero row stays zero rather than becoming NaN."""
    if x.numel() == 0:
        return x
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def similarity_matrix(embed: torch.Tensor, *, center: bool = False) -> torch.Tensor:
    """``(M, M)`` cosine similarity between L2-normalized embeddings.

    ``center`` subtracts the mean embedding over all instances in the set before
    normalizing. That removes the common-mode component shared by every query and
    slightly widens the gap between matches and non-matches, but it is *transductive*
    — the similarities, and therefore the meaning of ``sim_thres``, then depend on
    which photos happen to be in the set, so it is off by default.
    """
    if embed.numel() == 0:
        return torch.zeros((embed.shape[0], embed.shape[0]))
    x = embed.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    z = l2_normalize(x)
    return z @ z.T


# ---------------------------------------------------------------------------
# Pairwise matching
# ---------------------------------------------------------------------------
def pairwise_matches(
    sim: torch.Tensor,
    photo: torch.Tensor,
    keys: list[tuple[int, ...]],
    *,
    sim_thres: float = 0.6,
) -> list[dict]:
    """Hungarian-match every pair of photos, one gate group at a time.

    The matcher is run per ``(photo_a < photo_b)`` and, within that, per gate key
    present in both — never on one masked matrix. That is not an optimization, it is
    a correctness requirement: ``linear_sum_assignment`` rejects ``-inf`` outright and
    raises ``cost matrix is infeasible`` on ``+inf`` whenever the smaller side cannot
    be fully assigned, which is the *common* case once pairs are gated out. Because
    gating is equality on a key tuple, the gate matrix is exactly block-diagonal over
    gate groups, so solving each block is equivalent and no infinity ever reaches
    scipy.

    Hungarian always returns a full assignment on the smaller side, so ``sim_thres``
    is what turns "best available partner" into "no partner — this is a new instance".

    Returns:
        Every candidate pair, accepted or not, as
        ``{"a", "b", "similarity", "accepted", "reason"}`` with global indices
        ``a < b``, sorted by descending similarity. Rejections are kept so a run's
        threshold can be retuned from its JSON without redoing inference.
    """
    s = sim.detach().cpu().numpy()
    photos = photo.tolist()
    by_photo: dict[int, list[int]] = {}
    for idx, p in enumerate(photos):
        by_photo.setdefault(int(p), []).append(idx)

    pairs: list[dict] = []
    for pa in sorted(by_photo):
        for pb in sorted(x for x in by_photo if x > pa):
            # Bucket each photo's instances by gate key, then solve per shared key.
            ga: dict[tuple, list[int]] = {}
            for idx in by_photo[pa]:
                ga.setdefault(keys[idx], []).append(idx)
            gb: dict[tuple, list[int]] = {}
            for idx in by_photo[pb]:
                gb.setdefault(keys[idx], []).append(idx)

            for key in sorted(set(ga) & set(gb)):
                rows, cols = ga[key], gb[key]
                block = s[np.ix_(rows, cols)]
                # float64: float32 ties resolve differently inside the solver.
                ri, ci = linear_sum_assignment(-block.astype(np.float64))
                for r, c in zip(ri.tolist(), ci.tolist()):
                    a, b = rows[r], cols[c]
                    value = float(block[r, c])
                    ok = value >= sim_thres
                    pairs.append({
                        "a": a, "b": b, "similarity": value,
                        "accepted": ok, "reason": "" if ok else "below_thres",
                    })
    # Explicit tie-break: never rely on sort stability over an unordered source.
    pairs.sort(key=lambda p: (-p["similarity"], p["a"], p["b"]))
    return pairs


# ---------------------------------------------------------------------------
# Linking pairwise matches into identities
# ---------------------------------------------------------------------------
class _DisjointSet:
    """Union-find carrying, per root, the member list and a photo-index bitmask."""

    def __init__(self, n: int, photo: list[int]):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.members = [[i] for i in range(n)]
        self.photos = [1 << int(p) for p in photo]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, ra: int, rb: int) -> int:
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.members[ra] = self.members[ra] + self.members[rb]
        self.photos[ra] |= self.photos[rb]
        return ra


def cluster(
    pairs: list[dict],
    sim: torch.Tensor,
    photo: torch.Tensor,
    *,
    num_instances: int,
    sim_thres: float = 0.6,
    guard: str = "mean",
    guard_factor: float = 1.0,
    gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Link accepted pairs into cross-photo identities with union-find.

    Accepted pairs are consumed in descending similarity, so the strongest evidence
    forms clusters first and a weak pair arrives to be tested against an
    already-formed cluster. The merge guard's whole meaning rests on that order, so
    this function re-sorts rather than trusting the caller to have done it — with the
    same explicit tie-break :func:`pairwise_matches` uses, so both agree. Two checks
    run before each union:

    **Photo-uniqueness (always on).** One physical instance appears at most once per
    photo, so two instances of the *same* photo must never land in one identity. Plain
    union-find happily chains them via a third photo; the per-root photo bitmask makes
    that structurally impossible (``reason="photo_conflict"``).

    **Merge guard** (``guard="mean" | "min" | "off"``). Union-find is transitive: A-B
    and B-C chain into one identity even when A and C are nothing alike. Before
    merging clusters A and B, require the similarity across the A x B cut to be at
    least ``guard_factor * sim_thres`` (``reason="guard"``). Merging two singletons
    skips the check — their cut is exactly the pair that already cleared the
    threshold. ``"mean"`` tolerates one genuinely hard viewpoint pair inside an
    otherwise strong cluster and is the default; ``"min"`` is the paranoid setting and
    will block almost everything at 30 photos.

    Returns:
        ``(identity (M,) long, pairs)`` — identity ids are assigned in *first
        appearance* order, never by cluster size, so they stay stable (and so do the
        render colors) when one detection changes between runs. ``pairs`` is a copy
        with ``accepted`` / ``reason`` updated to reflect what the linker did.
    """
    pairs = [dict(p) for p in pairs]
    pairs.sort(key=lambda p: (-p["similarity"], p["a"], p["b"]))
    ds = _DisjointSet(num_instances, photo.tolist())
    s = sim.detach().cpu().numpy()
    g = gate.detach().cpu().numpy() if gate is not None else None
    floor = guard_factor * sim_thres

    for p in pairs:
        if not p["accepted"]:
            continue
        ra, rb = ds.find(p["a"]), ds.find(p["b"])
        if ra == rb:
            continue  # already one identity via a stronger path
        if ds.photos[ra] & ds.photos[rb]:
            p["accepted"], p["reason"] = False, "photo_conflict"
            continue
        ma, mb = ds.members[ra], ds.members[rb]
        if guard != "off" and (len(ma) > 1 or len(mb) > 1):
            block = s[np.ix_(ma, mb)]
            if g is not None:
                # Gating is transitive, so under the default gate every cut pair is
                # available and this is a no-op. It stops being one the moment the
                # gate becomes a non-transitive predicate (IoU, score, a learned
                # compatibility) — an un-evaluable cut is not evidence for merging.
                block = block[g[np.ix_(ma, mb)]]
            if block.size == 0:
                p["accepted"], p["reason"] = False, "guard"
                continue
            stat = float(block.mean()) if guard == "mean" else float(block.min())
            if stat < floor:
                p["accepted"], p["reason"] = False, "guard"
                continue
        ds.union(ra, rb)

    # Number identities by first appearance.
    identity = torch.full((num_instances,), -1, dtype=torch.long)
    seen: dict[int, int] = {}
    for m in range(num_instances):
        root = ds.find(m)
        if root not in seen:
            seen[root] = len(seen)
        identity[m] = seen[root]
    for p in pairs:
        p["identity_id"] = int(identity[p["a"]]) if p["accepted"] else None
    return identity, pairs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize_identities(
    identity: torch.Tensor,
    origin: list[tuple[int, int]],
    results: list[dict],
    sim: torch.Tensor,
    *,
    names: dict[int, str] | None = None,
    aux_specs=None,
    paths: list[str] | None = None,
) -> list[dict]:
    """One JSON-able record per identity, sorted by how many photos it appears in.

    Attributes are smoothed across the identity's members by **mean probability**,
    then argmaxed — the same running-mean idea the video tracker uses per track.
    Members whose id is ``-1`` (a class-scoped head that does not apply) are excluded
    from both the sum and the count rather than summed as zeros, and a head that no
    member has an opinion on is omitted entirely rather than reported as class 0.

    Regression heads (``aux_values``) are fused into ``values``: the score-weighted mean of the members' values, in the
    head's own space (of ``log1p`` for a ``log`` head, i.e. a weighted geometric mean), skipping the members it does not
    apply to. One physical instance has one value, so the identity's is better than any single photo's.
    """
    s = sim.detach().cpu().numpy()
    aux_names = {sp.name: sp.names for sp in (aux_specs or [])}
    log_heads = {sp.name for sp in (aux_specs or []) if getattr(sp, "log", False)}
    by_identity: dict[int, list[int]] = {}
    for m, cid in enumerate(identity.tolist()):
        by_identity.setdefault(int(cid), []).append(m)

    records = []
    for cid, members in by_identity.items():
        classes, photos, entries = [], [], []
        acc: dict[str, list] = {}
        reg: dict[str, list] = {}
        for m in members:
            img, det = origin[m]
            res = results[img]
            classes.append(int(res["classes"][det]))
            photos.append(img)
            entries.append({
                "image_index": img,
                "path": (paths[img] if paths else res.get("path", "")),
                "det_index": det,
                "score": float(res["scores"][det]),
                "box": [round(float(v), 2) for v in res["boxes"][det].tolist()],
            })
            for head, pred in (res.get("aux") or {}).items():
                hid = int(pred["ids"][det])
                if hid < 0:
                    continue  # scoped head that does not apply: no opinion, not a vote
                probs = pred["probs"][det].detach().float().cpu().numpy()
                slot = acc.setdefault(head, [np.zeros_like(probs), 0])
                slot[0] += probs
                slot[1] += 1
            for head, vals in (res.get("aux_values") or {}).items():
                v = float(vals[det])
                if v == v:  # NaN: a class-scoped head that does not apply
                    reg.setdefault(head, []).append((v, float(res["scores"][det])))

        attribute_ids, attributes = {}, {}
        for head, (total, count) in acc.items():
            if not count:
                continue
            hid = int(total.argmax())
            attribute_ids[head] = hid
            attributes[head] = str(aux_names.get(head, {}).get(hid, hid))

        values = {}
        for head, pairs in reg.items():
            x, w = np.array([v for v, _ in pairs]), np.array([sc for _, sc in pairs])
            t = np.log1p(np.clip(x, -1 + 1e-9, None)) if head in log_heads else x
            mean = float((w * t).sum() / w.sum()) if w.sum() > 0 else float(t.mean())
            values[head] = round(float(np.expm1(mean)) if head in log_heads else mean, 6)

        cls = max(set(classes), key=classes.count)
        cuts = [s[a, b] for i, a in enumerate(members) for b in members[i + 1:]]
        records.append({
            "identity_id": int(cid),
            "class": cls,
            "class_name": str(names.get(cls, cls)) if names else str(cls),
            "attributes": attributes,
            "attribute_ids": attribute_ids,
            "values": values,
            "num_photos": len(set(photos)),
            "num_instances": len(members),
            "members": entries,
            "mean_similarity": round(float(np.mean(cuts)), 4) if cuts else None,
            "min_similarity": round(float(np.min(cuts)), 4) if cuts else None,
            "max_similarity": round(float(np.max(cuts)), 4) if cuts else None,
        })
    records.sort(key=lambda r: (-r["num_photos"], r["identity_id"]))
    return records


def link_identities(
    results: list[dict],
    *,
    sim_thres: float,
    group_by: tuple[str, ...] = (CLASS_KEY,),
    dim: int | None = None,
    center: bool = False,
    guard: str = "mean",
    guard_factor: float = 1.0,
    names: dict[int, str] | None = None,
    aux_specs=None,
    paths: list[str] | None = None,
) -> dict:
    """The whole linking of a set of photos' results (each with ``"embed"``): flatten, similarity, Hungarian per photo
    pair within the gate groups, union-find with the merge guard, one record per identity. Stamps every result with
    ``identity_ids`` ``(n,)`` and returns ``{"identity", "identities", "pairs", "origin", "sim", "diagnostics"}``
    (:func:`flatten_instances`, :func:`pairwise_matches`, :func:`cluster`, :func:`summarize_identities`,
    :func:`similarity_diagnostics`). ``group_by`` must be validated (:func:`validate_group_by`)."""
    embed, photo, keys, origin = flatten_instances(results, group_by=group_by, dim=dim)
    sim = similarity_matrix(embed, center=center)
    gate = gate_mask(photo, keys)
    pairs = pairwise_matches(sim, photo, keys, sim_thres=sim_thres)
    identity, pairs = cluster(pairs, sim, photo, num_instances=len(origin), sim_thres=sim_thres, guard=guard,
                              guard_factor=guard_factor, gate=gate)
    identities = summarize_identities(identity, origin, results, sim, names=names, aux_specs=aux_specs, paths=paths)
    flat_of = {o: m for m, o in enumerate(origin)}
    for i, res in enumerate(results):
        n = int(res["num_detections"])
        res["identity_ids"] = torch.tensor([int(identity[flat_of[(i, j)]]) for j in range(n)], dtype=torch.long)
    return {"identity": identity, "identities": identities, "pairs": pairs, "origin": origin, "sim": sim,
            "diagnostics": similarity_diagnostics(sim, gate, identity)}


def similarity_diagnostics(
    sim: torch.Tensor, gate: torch.Tensor, identity: torch.Tensor
) -> dict:
    """Percentiles of gated similarity, split within- vs across-identity.

    This is what tells you whether ``sim_thres`` is anywhere near the right value for
    a dataset. If the two distributions overlap completely, the fingerprint is not
    separating instances and no threshold will save the run — better to read that off
    the JSON than to squint at the grid.
    """
    s = sim.detach().cpu().numpy()
    g = gate.detach().cpu().numpy()
    ident = identity.tolist()
    within, across = [], []
    for a in range(len(ident)):
        for b in range(a + 1, len(ident)):
            if not g[a, b]:
                continue
            (within if ident[a] == ident[b] else across).append(float(s[a, b]))

    def stats(v):
        if not v:
            return {"n": 0}
        arr = np.asarray(v)
        return {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 4),
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }

    return {"within_identity": stats(within), "across_identity": stats(across)}
