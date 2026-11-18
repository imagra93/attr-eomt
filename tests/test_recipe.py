"""The optimisation recipe (reference EoMT): layer-wise LR groups, two-stage warmup + polynomial decay, staggered
per-block masked-attention annealing, and that ``train()`` wires them into the optimizer (CPU, no network)."""

from __future__ import annotations

import pytest
import torch
import yaml

from eomt import build_model
from eomt.engine.train import (
    EOMT_MASK_ANNEAL_END,
    EOMT_MASK_ANNEAL_START,
    _attn_mask_probs,
    _poly_two_stage_factors,
    build_optimizer,
)

from test_augment import _mini_coco  # noqa: E402  (tests/ is on sys.path)

LR = 1e-4


def _lr_by_param(model, **kw):
    opt = build_optimizer(model, LR, 0.05, **kw)
    lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    return {n: lr_of[id(p)] for n, p in model.named_parameters()}


def test_layerwise_lr_follows_the_reference_recipe():
    model = build_model("s", nc=3, imgsz=140)
    n = model.eomt.config.num_hidden_layers
    lrs = _lr_by_param(model, backbone_lr_mult=1.0, llrd=0.8)
    for name, lr in lrs.items():
        if name.startswith("eomt.layers."):
            i = int(name.split("eomt.layers.")[1].split(".")[0])
            assert lr == pytest.approx(LR * 0.8 ** (n - 1 - i)), name   # the top block is exactly the base LR
        elif name.startswith("eomt.embeddings"):
            assert lr == pytest.approx(LR * 0.8 ** (n - 1)), name       # same as the first block
        elif name.startswith("eomt.layernorm"):
            assert lr == pytest.approx(LR), name                        # final norm: base LR
        else:
            assert lr == pytest.approx(LR), name                        # queries, mask/class heads, aux heads


def test_head_and_encoder_never_share_an_optimizer_group():
    """The schedule gives encoder groups the ViT warmup/decay and the others the head's, so a group must not mix them.

    With the reference recipe the top ViT block and the final norm have exactly the head's LR (and weight decay), so a
    group keyed by (lr, wd) alone silently put head parameters on the ViT schedule (LR held at 0 during the head warmup).
    """
    model = build_model("s", nc=3, imgsz=140)
    opt = build_optimizer(model, LR, 0.05, backbone_lr_mult=1.0, llrd=0.8)
    group_of = {id(p): g for g in opt.param_groups for p in g["params"]}
    assert len(group_of) == sum(1 for _ in model.parameters())
    for name, p in model.named_parameters():
        is_encoder = name.startswith(("eomt.embeddings", "eomt.layers", "eomt.layernorm"))
        assert group_of[id(p)]["is_backbone"] == is_encoder, name


def test_the_last_query_blocks_are_not_starved():
    """Regression for the old defaults (0.1x multiplier): the blocks that process the queries must learn at head speed."""
    model = build_model("s", nc=3, imgsz=140)
    n, q = model.eomt.config.num_hidden_layers, model.eomt.config.num_blocks
    lrs = _lr_by_param(model, backbone_lr_mult=1.0, llrd=0.8)
    top = [v for k, v in lrs.items() if k.startswith(f"eomt.layers.{n - 1}.")]
    first_query_block = [v for k, v in lrs.items() if k.startswith(f"eomt.layers.{n - q}.")]
    assert min(top) == pytest.approx(LR)
    assert min(first_query_block) == pytest.approx(LR * 0.8 ** (q - 1))   # 0.51x, not 0.085x
    # the multiplier and a flat encoder remain available as explicit options
    flat = _lr_by_param(model, backbone_lr_mult=0.5, llrd=1.0)
    assert all(v == pytest.approx(LR * 0.5) for k, v in flat.items() if k.startswith(("eomt.layers.", "eomt.embeddings")))


def test_two_stage_warmup_then_polynomial_decay():
    total, warm, power = 6000, (500, 1000), 0.9
    f = lambda s: _poly_two_stage_factors(s, total, warm, power)  # noqa: E731
    # head warms up linearly while the ViT is held at 0
    assert f(0) == (0.0, 0.0) and f(250) == (0.5, 0.0) and f(499)[1] == 0.0
    # head reaches 1 and starts decaying; the ViT then ramps over its own warmup
    assert f(500) == (1.0, 0.0)
    assert f(1000)[1] == pytest.approx(0.5) and f(1500)[1] == pytest.approx(1.0)
    # each group decays as (1 - progress) ** power from the end of its own warmup, to exactly 0
    assert f(3500)[0] == pytest.approx((1 - 3000 / 5500) ** power)
    assert f(3500)[1] == pytest.approx((1 - 2000 / 4500) ** power)
    assert f(total) == (0.0, 0.0) and f(total + 100) == (0.0, 0.0)
    head = [f(s)[0] for s in range(500, total)]
    vit = [f(s)[1] for s in range(1500, total)]
    assert all(a >= b for a, b in zip(head, head[1:])) and all(a >= b for a, b in zip(vit, vit[1:]))


def test_mask_annealing_is_staggered_per_block_and_polynomial():
    total = 60_000
    probs = lambda frac: _attn_mask_probs(int(frac * total), total, 0.9)  # noqa: E731
    assert probs(0.0) == [1.0] * 4 and probs(0.10) == [1.0] * 4
    p = probs(0.25)                       # halfway through block 0's window (1/6 .. 2/6), later blocks untouched
    assert p[0] == pytest.approx(0.5 ** 0.9) and p[1:] == [1.0, 1.0, 1.0]
    assert probs(0.5) == [0.0, 0.0, 1.0, 1.0]                      # blocks anneal one after the other
    assert probs(0.8)[:3] == [0.0, 0.0, 0.0] and 0 < probs(0.8)[3] < 1
    assert probs(5 / 6) == [0.0] * 4 and probs(0.95) == [0.0] * 4  # the last sixth is mask-free
    seqs = list(zip(*[_attn_mask_probs(i, total, 0.9) for i in range(0, total, 97)]))
    assert len(seqs) == len(EOMT_MASK_ANNEAL_START) == len(EOMT_MASK_ANNEAL_END)
    for seq in seqs:
        assert all(x >= y for x, y in zip(seq, seq[1:]))


def test_train_applies_the_recipe_to_the_optimizer(tmp_path):
    from eomt.engine.train import train

    img_dir, jf = _mini_coco(tmp_path)
    kw = dict(train_images=str(img_dir), train_json=str(jf), size="s", imgsz=112, epochs=1, batch=1, nominal_batch=1,
              workers=0, device="cpu", amp=False, pretrained=False, seed=0, project=str(tmp_path / "runs"))
    res = train(name="r", optim={"ema": False, "warmup_steps": (1, 1)}, **kw)
    args = yaml.safe_load((tmp_path / "runs" / "r" / "args.yaml").read_text())
    o = args["optim"]
    assert o["llrd"] == 0.8 and o["backbone_lr_mult"] == 1.0 and o["poly_power"] == 0.9
    assert o["warmup_steps"] == [1, 1] and o["mask_anneal"] is True
    # 4 images, batch 1: the last micro-step is it = 3 of 4 optimizer steps
    head_f, vit_f = _poly_two_stage_factors(3, 4, (1, 1), 0.9)
    ck = torch.load(res["last"], weights_only=True)
    groups = ck["optimizer"]["param_groups"]
    assert any(not g["is_backbone"] for g in groups) and any(g["is_backbone"] for g in groups)
    for g in groups:
        want = g["initial_lr"] * (vit_f if g["is_backbone"] else head_f)
        assert g["lr"] == pytest.approx(want, rel=1e-6)
    assert head_f > 0 and vit_f > 0   # the head group really is on the head schedule (not held at 0)
    assert max(g["initial_lr"] for g in groups) == pytest.approx(LR)
    top_backbone = max(g["initial_lr"] for g in groups if g["is_backbone"])
    assert top_backbone == pytest.approx(LR)                        # the top ViT block learns at the head's LR
    with pytest.raises(ValueError, match="warmup_steps"):
        train(name="bad", optim={"warmup_steps": (1,)}, **kw)
    with pytest.raises(ValueError, match="unknown keys"):
        train(name="bad", optim={"lr": 1e-4}, **kw)


def test_validation_follows_the_annealing_and_best_waits_until_mask_free(tmp_path, monkeypatch):
    """Validation runs at the epoch-end annealing (official EoMT); best.pt is only chosen once that is mask-free (its
    deployment regime), and every checkpoint is saved mask-free."""
    import sys

    from eomt.engine.train import train

    seen = []
    monkeypatch.setattr(sys.modules["eomt.engine.train"], "evaluate",  # the package re-exports train() under that name
                        lambda model, *a, **k: seen.append(list(model.eomt.mask_probs)) or {"segm/mAP": 0.5})
    img_dir, jf = _mini_coco(tmp_path)
    kw = dict(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s", imgsz=112,
              epochs=2, val={"interval": 1}, batch=1, nominal_batch=1, workers=0, device="cpu", amp=False,
              pretrained=False, optim={"warmup_steps": (1, 1)}, seed=0, project=str(tmp_path / "runs"), name="v")
    res = train(stop_after_epochs=1, **kw)
    assert seen == [[0.0, 0.0, 1.0, 1.0]] and res["best"] is None    # half-way: the last two blocks still mask
    ck = torch.load(res["last"], weights_only=True)
    assert "eomt.attn_mask_probs" not in ck["model"]   # no buffer: checkpoints are mask-free by construction
    res = train(resume=res["last"], **kw)
    assert seen[-1] == [0.0] * 4 and res["best"] is not None
