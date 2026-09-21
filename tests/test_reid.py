"""CPU, no-model unit tests for cross-photo re-identification (:mod:`eomt.reid`).

Everything here is driven by hand-built embeddings and synthetic result dicts, so the
matching/linking logic is tested without a forward pass anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eomt.reid import (
    cluster,
    flatten_instances,
    gate_mask,
    l2_normalize,
    pairwise_matches,
    similarity_diagnostics,
    similarity_matrix,
    summarize_identities,
    validate_group_by,
)


def _unit(*vals) -> torch.Tensor:
    """A 2-D embedding on the unit circle, from angles in radians."""
    return torch.tensor([[np.cos(a), np.sin(a)] for a in vals], dtype=torch.float32)


def _from_sim(sim) -> torch.Tensor:
    """Embeddings whose pairwise cosines reproduce ``sim`` (via its Cholesky root)."""
    s = np.asarray(sim, dtype=np.float64)
    w, v = np.linalg.eigh(s)
    x = v @ np.diag(np.sqrt(np.clip(w, 1e-9, None)))
    return torch.tensor(x, dtype=torch.float32)


# ---------------------------------------------------------------- similarity
def test_l2_normalize_and_similarity():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    z = l2_normalize(x)
    assert torch.allclose(z[0].norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.equal(z[1], torch.zeros(2))  # zero row stays zero, never NaN
    assert not torch.isnan(z).any()

    sim = similarity_matrix(x)
    assert sim.shape == (3, 3)
    assert torch.allclose(sim, sim.T, atol=1e-6)
    assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert sim.min() >= -1.0001 and sim.max() <= 1.0001


def test_similarity_matrix_handles_empty():
    assert similarity_matrix(torch.zeros((0, 8))).shape == (0, 0)


# --------------------------------------------------------------------- gating
def test_validate_group_by():
    class Spec:
        def __init__(self, name, applies_to=None):
            self.name = name
            self.applies_to = applies_to

    specs = [Spec("color"), Spec("posture", applies_to=frozenset({0}))]
    assert validate_group_by(None, specs) == ()
    assert validate_group_by((), specs) == ()
    assert validate_group_by("class", specs) == ("class",)
    assert validate_group_by(("class", "color"), specs) == ("class", "color")
    # "class" alone is valid even with no aux heads at all.
    assert validate_group_by(("class",), []) == ("class",)

    with pytest.raises(ValueError, match="Unknown group_by key"):
        validate_group_by(("class", "frontalty"), specs)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_group_by(("class", "class"), specs)
    # A class-scoped head gated on without "class": every out-of-scope instance
    # shares the -1 sentinel and would gate together across classes.
    with pytest.warns(UserWarning, match="sentinel"):
        validate_group_by(("posture",), specs)


def test_gate_mask_blocks_same_photo_and_key_mismatch():
    photo = torch.tensor([0, 0, 1, 2])
    keys = [(1,), (2,), (1,), (1,)]
    g = gate_mask(photo, keys)
    assert not g[0, 1]  # same photo
    assert not g[0, 0]
    assert g[0, 2] and g[2, 0]  # different photo, same key
    assert not g[1, 2]  # key mismatch
    assert g[0, 3] and g[2, 3]
    assert torch.equal(g, g.T)


def test_flatten_instances_builds_keys_and_origin():
    results = [
        {"num_detections": 2, "embed": torch.eye(2),
         "classes": torch.tensor([0, 1]),
         "aux": {"side": {"ids": torch.tensor([1, 2])}}},
        {"num_detections": 1, "embed": torch.ones(1, 2),
         "classes": torch.tensor([0]),
         "aux": {"side": {"ids": torch.tensor([1])}}},
    ]
    embed, photo, keys, origin = flatten_instances(results, group_by=("class", "side"))
    assert embed.shape == (3, 2)
    assert photo.tolist() == [0, 0, 1]
    assert keys == [(0, 1), (1, 2), (0, 1)]
    assert origin == [(0, 0), (0, 1), (1, 0)]
    # Gating off -> every key is the empty tuple, so everything is a candidate.
    _, _, keys_off, _ = flatten_instances(results, group_by=())
    assert keys_off == [(), (), ()]


def test_flatten_instances_all_empty_uses_declared_dim():
    embed, photo, keys, origin = flatten_instances(
        [{"num_detections": 0}, {"num_detections": 0}], group_by=("class",), dim=384
    )
    assert embed.shape == (0, 384)  # width cannot be read off the data
    assert photo.numel() == 0 and keys == [] and origin == []


# ------------------------------------------------------------------ matching
def test_hungarian_picks_optimal_not_greedy():
    # Greedy-by-best-cell takes (a0,b0)=.95 then is stuck with (a1,b1)=.10 -> 1.05.
    # The optimal assignment is (a0,b1)+(a1,b0) = .80+.94 = 1.74.
    sim = torch.tensor([
        [1.00, 0.00, 0.95, 0.80],
        [0.00, 1.00, 0.94, 0.10],
        [0.95, 0.94, 1.00, 0.00],
        [0.80, 0.10, 0.00, 1.00],
    ])
    photo = torch.tensor([0, 0, 1, 1])
    keys = [(0,)] * 4
    pairs = pairwise_matches(sim, photo, keys, sim_thres=0.0)
    got = {(p["a"], p["b"]) for p in pairs}
    assert got == {(0, 3), (1, 2)}
    assert sum(p["similarity"] for p in pairs) == pytest.approx(1.74, abs=1e-5)


def test_threshold_turns_best_partner_into_new_instance():
    sim = torch.tensor([[1.0, 0.55], [0.55, 1.0]])
    photo = torch.tensor([0, 1])
    keys = [(0,), (0,)]
    (pair,) = pairwise_matches(sim, photo, keys, sim_thres=0.6)
    assert pair["accepted"] is False and pair["reason"] == "below_thres"
    identity, _ = cluster([pair], sim, photo, num_instances=2, sim_thres=0.6)
    assert identity.tolist() == [0, 1]  # two separate instances


def test_gate_groups_never_reach_an_infeasible_hungarian():
    # Identical embeddings, different gate keys. This is the regression test for the
    # -inf masking trap: scipy rejects -inf outright and raises "cost matrix is
    # infeasible" on +inf when the smaller side cannot be fully assigned. Solving per
    # gate group means no infinity is ever constructed.
    sim = torch.ones((2, 2))
    photo = torch.tensor([0, 1])
    keys = [(0,), (1,)]
    assert pairwise_matches(sim, photo, keys, sim_thres=0.0) == []


def test_matches_are_sorted_and_keep_rejections():
    sim = torch.tensor([
        [1.0, 0.0, 0.9, 0.2],
        [0.0, 1.0, 0.2, 0.3],
        [0.9, 0.2, 1.0, 0.0],
        [0.2, 0.3, 0.0, 1.0],
    ])
    photo = torch.tensor([0, 0, 1, 1])
    pairs = pairwise_matches(sim, photo, [(0,)] * 4, sim_thres=0.6)
    sims = [p["similarity"] for p in pairs]
    assert sims == sorted(sims, reverse=True)
    assert any(not p["accepted"] for p in pairs)  # rejections retained for retuning


# ----------------------------------------------------------------- clustering
def test_guard_blocks_transitive_chain():
    # a-b and b-c both clear .6, but a and c are nothing alike. Union-find would
    # happily chain all three into one identity.
    sim = torch.tensor([[1.0, 0.7, 0.1], [0.7, 1.0, 0.7], [0.1, 0.7, 1.0]])
    photo = torch.tensor([0, 1, 2])
    pairs = pairwise_matches(sim, photo, [(0,)] * 3, sim_thres=0.6)

    ident_guard, out = cluster(pairs, sim, photo, num_instances=3, sim_thres=0.6,
                               guard="mean", guard_factor=1.0)
    assert len(set(ident_guard.tolist())) == 2  # the chain is broken
    assert any(p["reason"] == "guard" for p in out)

    ident_off, _ = cluster(pairs, sim, photo, num_instances=3, sim_thres=0.6, guard="off")
    assert len(set(ident_off.tolist())) == 1  # and restored when the guard is off


def test_guard_allows_one_hard_pair_inside_a_strong_cluster():
    # Cut mean over {a,b} x {c} is (0.95 + 0.62)/2 = 0.785 >= 0.6: a genuinely hard
    # viewpoint should not veto an otherwise strong identity.
    sim = torch.tensor([[1.0, 0.95, 0.95], [0.95, 1.0, 0.62], [0.95, 0.62, 1.0]])
    photo = torch.tensor([0, 1, 2])
    pairs = pairwise_matches(sim, photo, [(0,)] * 3, sim_thres=0.6)
    identity, _ = cluster(pairs, sim, photo, num_instances=3, sim_thres=0.6, guard="mean")
    assert len(set(identity.tolist())) == 1
    # "min" is the paranoid setting and does veto it at guard_factor=1.1.
    identity_min, _ = cluster(pairs, sim, photo, num_instances=3, sim_thres=0.6,
                              guard="min", guard_factor=1.1)
    assert len(set(identity_min.tolist())) == 2


def test_no_identity_holds_two_instances_of_one_photo():
    # a0 and a1 are both in photo 0; b is in photo 1 and matches both strongly.
    # Plain union-find chains a0-b-a1 into one identity, which is physically
    # impossible: one object appears at most once per photo.
    sim = torch.tensor([[1.0, 0.99, 0.95], [0.99, 1.0, 0.96], [0.95, 0.96, 1.0]])
    photo = torch.tensor([0, 0, 1])
    pairs = [
        {"a": 0, "b": 2, "similarity": 0.95, "accepted": True, "reason": ""},
        {"a": 1, "b": 2, "similarity": 0.96, "accepted": True, "reason": ""},
    ]
    identity, out = cluster(pairs, sim, photo, num_instances=3, sim_thres=0.6)
    assert identity[0] != identity[1]
    assert any(p["reason"] == "photo_conflict" for p in out)
    # The invariant, stated globally.
    for cid in set(identity.tolist()):
        photos = [int(photo[m]) for m in range(3) if identity[m] == cid]
        assert len(photos) == len(set(photos))


def test_identity_ids_are_first_appearance_ordered():
    # Instance 0 is a singleton; 1..3 form the big cluster. Numbering by size would
    # give the big cluster id 0 and flip every render color; numbering by appearance
    # keeps ids stable.
    sim = torch.full((4, 4), 0.9)
    sim[0, :] = sim[:, 0] = 0.0
    sim[0, 0] = 1.0
    photo = torch.tensor([0, 1, 2, 3])
    pairs = pairwise_matches(sim, photo, [(0,)] * 4, sim_thres=0.6)
    identity, _ = cluster(pairs, sim, photo, num_instances=4, sim_thres=0.6, guard="off")
    assert identity[0].item() == 0
    assert len(set(identity[1:].tolist())) == 1 and identity[1].item() == 1


def test_clustering_is_deterministic_under_ties():
    sim = torch.full((6, 6), 0.8)
    sim.fill_diagonal_(1.0)
    photo = torch.tensor([0, 1, 2, 3, 4, 5])
    pairs = pairwise_matches(sim, photo, [(0,)] * 6, sim_thres=0.6)
    a, _ = cluster(pairs, sim, photo, num_instances=6, sim_thres=0.6)
    b, _ = cluster(pairs, sim, photo, num_instances=6, sim_thres=0.6)
    shuffled = list(reversed(pairs))
    c, _ = cluster(shuffled, sim, photo, num_instances=6, sim_thres=0.6)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_cluster_sorts_defensively_so_strongest_evidence_wins():
    # a0 and a1 share photo 0, so only one of them can join b — and it must be the
    # stronger pair (a1-b, 0.96), whichever order the caller hands the pairs in.
    # ``cluster``'s guard is meaningless unless it sees pairs strongest-first, so it
    # re-sorts rather than trusting the caller to have done it.
    sim = torch.tensor([[1.0, 0.99, 0.95], [0.99, 1.0, 0.96], [0.95, 0.96, 1.0]])
    photo = torch.tensor([0, 0, 1])
    weak = {"a": 0, "b": 2, "similarity": 0.95, "accepted": True, "reason": ""}
    strong = {"a": 1, "b": 2, "similarity": 0.96, "accepted": True, "reason": ""}
    for order in ([weak, strong], [strong, weak]):
        identity, out = cluster(order, sim, photo, num_instances=3, sim_thres=0.6)
        assert identity[1] == identity[2]  # the 0.96 pair linked
        assert identity[0] != identity[2]  # the 0.95 pair lost the conflict
        rejected = [p for p in out if p["reason"] == "photo_conflict"]
        assert [(p["a"], p["b"]) for p in rejected] == [(0, 2)]


def test_degenerate_inputs():
    empty = torch.zeros((0, 0))
    ident, pairs = cluster([], empty, torch.zeros(0, dtype=torch.long), num_instances=0)
    assert ident.numel() == 0 and pairs == []

    # One photo: nothing to match across, one identity per detection.
    sim = torch.eye(2)
    photo = torch.tensor([0, 0])
    assert pairwise_matches(sim, photo, [(0,), (0,)], sim_thres=0.0) == []
    ident, _ = cluster([], sim, photo, num_instances=2)
    assert sorted(ident.tolist()) == [0, 1]


# ----------------------------------------------------------------- reporting
def _result(classes, aux=None, n=None):
    n = len(classes) if n is None else n
    return {
        "num_detections": n,
        "classes": torch.tensor(classes, dtype=torch.long),
        "scores": torch.ones(n),
        "boxes": torch.zeros((n, 4)),
        "aux": aux or {},
        "path": "p.jpg",
    }


def test_summarize_smooths_by_mean_prob():
    # Per-member argmax is [0, 1], but the mean probability favors class 1.
    aux_a = {"typ": {"ids": torch.tensor([0]), "probs": torch.tensor([[0.51, 0.49]])}}
    aux_b = {"typ": {"ids": torch.tensor([1]), "probs": torch.tensor([[0.10, 0.90]])}}
    results = [_result([0], aux_a), _result([0], aux_b)]
    identity = torch.tensor([0, 0])
    recs = summarize_identities(
        identity, [(0, 0), (1, 0)], results, torch.full((2, 2), 0.9),
        names={0: "widget"}, aux_specs=[], paths=["a.jpg", "b.jpg"],
    )
    assert len(recs) == 1
    assert recs[0]["attribute_ids"]["typ"] == 1
    assert recs[0]["num_photos"] == 2 and recs[0]["class_name"] == "widget"


def test_summarize_excludes_minus_one_members():
    # A scoped head that does not apply reports -1 with zeroed probs. Summing those
    # zeros into the mean (as a naive running average would) drags the result toward
    # class 0; they must be excluded from both the sum and the count.
    aux_a = {"typ": {"ids": torch.tensor([1]), "probs": torch.tensor([[0.10, 0.90]])}}
    aux_na = {"typ": {"ids": torch.tensor([-1]), "probs": torch.tensor([[0.0, 0.0]])}}
    base = summarize_identities(
        torch.tensor([0]), [(0, 0)], [_result([0], aux_a)], torch.ones((1, 1)),
        names=None, aux_specs=[], paths=["a.jpg"],
    )
    with_na = summarize_identities(
        torch.tensor([0, 0]), [(0, 0), (1, 0)],
        [_result([0], aux_a), _result([0], aux_na)], torch.full((2, 2), 0.9),
        names=None, aux_specs=[], paths=["a.jpg", "b.jpg"],
    )
    assert base[0]["attribute_ids"] == with_na[0]["attribute_ids"] == {"typ": 1}

    # A head no member has an opinion on is omitted, not reported as class 0.
    only_na = summarize_identities(
        torch.tensor([0]), [(0, 0)], [_result([0], aux_na)], torch.ones((1, 1)),
        names=None, aux_specs=[], paths=["a.jpg"],
    )
    assert "typ" not in only_na[0]["attributes"]


def test_similarity_diagnostics_splits_within_and_across():
    sim = torch.tensor([[1.0, 0.9, 0.2], [0.9, 1.0, 0.3], [0.2, 0.3, 1.0]])
    gate = torch.ones((3, 3), dtype=torch.bool)
    diag = similarity_diagnostics(sim, gate, torch.tensor([0, 0, 1]))
    assert diag["within_identity"]["n"] == 1
    assert diag["within_identity"]["mean"] == pytest.approx(0.9, abs=1e-4)
    assert diag["across_identity"]["n"] == 2
    assert diag["across_identity"]["max"] == pytest.approx(0.3, abs=1e-4)
    # An empty split reports n=0 rather than raising on an empty percentile.
    assert similarity_diagnostics(
        sim, torch.zeros((3, 3), dtype=torch.bool), torch.tensor([0, 1, 2])
    )["within_identity"] == {"n": 0}
