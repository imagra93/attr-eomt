"""CPU, no-network smoke tests for the libre-eomt package."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eomt import build_eomt_config, build_model, postprocess_instance
from eomt.config import EOMT_CONFIGS

IMGSZ = 140  # 14 * 10 -> tiny patch grid keeps the test fast
NC = 3


@pytest.mark.parametrize("size", ["s", "b", "l"])
def test_config_matches_size_preset(size):
    cfg = build_eomt_config(size, nc=NC, image_size=IMGSZ)
    preset = EOMT_CONFIGS[size]
    assert cfg.hidden_size == preset.hidden_size
    assert cfg.num_hidden_layers == preset.num_hidden_layers
    assert cfg.num_attention_heads == preset.num_attention_heads
    assert cfg.num_queries == preset.num_queries


def test_config_rejects_non_patch_aligned_imgsz():
    with pytest.raises(ValueError):
        build_eomt_config("s", nc=NC, image_size=100)  # 100 % 14 != 0


def test_build_forward_shapes():
    torch.manual_seed(0)
    model = build_model("s", nc=NC, imgsz=IMGSZ).eval()
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        out = model(x)
    assert set(out) == {"masks_queries_logits", "class_queries_logits"}
    q = EOMT_CONFIGS["s"].num_queries
    assert out["class_queries_logits"].shape == (2, q, NC + 1)
    assert out["masks_queries_logits"].shape[:2] == (2, q)


def test_train_step_backward():
    torch.manual_seed(0)
    model = build_model("s", nc=NC, imgsz=IMGSZ).train()
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    mask_labels = [
        (torch.rand(2, IMGSZ, IMGSZ) > 0.5).float(),
        (torch.rand(1, IMGSZ, IMGSZ) > 0.5).float(),
    ]
    class_labels = [torch.tensor([0, 1]), torch.tensor([2])]
    out = model(x, mask_labels=mask_labels, class_labels=class_labels)
    loss = out["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"


def test_aux_heads_train_and_infer():
    from eomt.aux_cls import aux_accuracy, aux_loss
    from eomt.config import AuxHeadSpec

    torch.manual_seed(0)
    specs = [AuxHeadSpec("typology", 4, {0: "a", 1: "b", 2: "c", 3: "d"}),
             AuxHeadSpec("severity", 3, {0: "lo", 1: "mid", 2: "hi"})]
    model = build_model("s", nc=NC, imgsz=IMGSZ, aux_heads=specs).train()
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    mask_labels = [
        (torch.rand(2, IMGSZ, IMGSZ) > 0.5).float(),
        (torch.rand(1, IMGSZ, IMGSZ) > 0.5).float(),
    ]
    class_labels = [torch.tensor([0, 1]), torch.tensor([2])]
    aux_labels = {
        "typology": [torch.tensor([1, 3]), torch.tensor([0])],
        "severity": [torch.tensor([2, 0]), torch.tensor([1])],
    }
    out = model(x, mask_labels=mask_labels, class_labels=class_labels)
    total = out["loss"]
    a_loss, per_head = aux_loss(model, out, mask_labels, class_labels, aux_labels)
    assert set(per_head) == {"typology", "severity"}
    (total + a_loss).backward()
    # default head is a small MLP -> assert grads flow through *some* head param
    assert any(p.grad is not None for p in model.aux_heads["typology"].parameters())

    acc = aux_accuracy(model, out, mask_labels, class_labels, aux_labels)
    assert set(acc) == {"typology", "severity"}

    # inference forward exposes per-head logits; postprocess attaches them
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out["aux_queries_logits"]["typology"].shape == (2, EOMT_CONFIGS["s"].num_queries, 4)
    res = postprocess_instance(
        {k: v[:1] if torch.is_tensor(v) else {n: t[:1] for n, t in v.items()}
         for k, v in out.items() if k != "query_embed"},
        conf_thres=0.0, original_size=(20, 15), max_det=5,
    )
    assert set(res["aux"]) == {"typology", "severity"}
    assert res["aux"]["typology"]["probs"].shape[1] == 4


def test_aux_ignore_index():
    """Missing/OOV attributes (label -100) contribute no loss and are not counted."""
    from eomt.aux_cls import aux_accuracy, aux_loss
    from eomt.config import AuxHeadSpec

    torch.manual_seed(0)
    model = build_model("s", nc=NC, imgsz=IMGSZ, aux_heads=[AuxHeadSpec("typ", 4)]).train()
    hidden = model.config.hidden_size
    out = {"query_embed": torch.randn(1, 5, hidden, requires_grad=True)}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]  # 2 matched queries

    # One valid label, one ignored -> finite loss, accuracy denominator counts only 1.
    aux_labels = {"typ": [torch.tensor([2, -100])]}
    a_loss, per_head = aux_loss(model, out, None, None, aux_labels, indices=indices)
    assert torch.isfinite(per_head["typ"])
    a_loss.backward()
    assert aux_accuracy(model, out, None, None, aux_labels, indices=indices)["typ"][1] == 1

    # All ignored -> exactly-zero, finite, graph-preserving loss; no counted samples.
    out2 = {"query_embed": torch.randn(1, 5, hidden, requires_grad=True)}
    all_ignored = {"typ": [torch.tensor([-100, -100])]}
    loss2, _ = aux_loss(model, out2, None, None, all_ignored, indices=indices)
    assert torch.isfinite(loss2) and float(loss2.detach()) == 0.0
    loss2.backward()  # must not raise (graph kept alive)
    assert aux_accuracy(model, out2, None, None, all_ignored, indices=indices)["typ"] == (0, 0)


def test_resolve_checkpoint_folder(tmp_path):
    """A run/weights folder resolves to best.pt (infer) or last.pt (resume)."""
    from eomt.serialization import resolve_checkpoint

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "best.pt").write_bytes(b"x")
    (weights / "last.pt").write_bytes(b"x")

    # run folder -> weights/{best,last}.pt by preference
    assert resolve_checkpoint(tmp_path, prefer="best").name == "best.pt"
    assert resolve_checkpoint(tmp_path, prefer="last").name == "last.pt"
    # weights folder directly
    assert resolve_checkpoint(weights, prefer="best").name == "best.pt"
    # an explicit file passes through unchanged
    assert resolve_checkpoint(weights / "last.pt", prefer="best").name == "last.pt"

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "nope")


def test_postprocess_instance_contract():
    q = 100
    out = {
        "masks_queries_logits": torch.randn(1, q, 10, 10),
        "class_queries_logits": torch.randn(1, q, NC + 1),
    }
    res = postprocess_instance(out, conf_thres=0.0, original_size=(20, 15), max_det=10)
    assert {"num_detections", "boxes", "scores", "classes", "masks"} <= set(res)
    assert res["masks"].shape[1:] == (15, 20)  # (orig_h, orig_w)
    assert res["boxes"].shape[1] == 4
    assert res["num_detections"] <= 10


def test_aux_head_mlp_checkpoint_roundtrip(tmp_path):
    """An MLP aux head must rebuild identically on reload (arch saved in metadata)."""
    import warnings

    import torch.nn as nn

    from eomt.config import AuxHeadSpec
    from eomt.serialization import load_model, save_checkpoint, wrap_checkpoint

    torch.manual_seed(0)
    specs = [AuxHeadSpec("typ", 4, {0: "a", 1: "b", 2: "c", 3: "d"})]
    arch = {"layers": 2, "hidden": 64, "dropout": 0.0}
    model = build_model("s", nc=NC, imgsz=IMGSZ, aux_heads=specs, aux_head_arch=arch).eval()
    assert isinstance(model.aux_heads["typ"], nn.Sequential)  # MLP, not bare Linear

    x = torch.randn(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        ref = model(x)["aux_queries_logits"]["typ"]

    ckpt = wrap_checkpoint(
        model.state_dict(), size="s", nc=NC, imgsz=IMGSZ, aux_heads=specs, aux_head_arch=arch
    )
    assert ckpt["aux_head_arch"] == arch
    path = tmp_path / "m.pt"
    save_checkpoint(ckpt, path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_model(path, device="cpu")
    msgs = [str(w.message) for w in caught]
    assert not any("missing" in m or "unexpected" in m for m in msgs), msgs
    assert isinstance(loaded.aux_heads["typ"], nn.Sequential)
    with torch.no_grad():
        got = loaded(x)["aux_queries_logits"]["typ"]
    assert torch.allclose(ref, got, atol=1e-5)


def test_gate_indices_iou_and_class():
    """gate_indices drops low-IoU and (optionally) wrong-class matched pairs."""
    from eomt.aux_cls import gate_indices

    # 1 image, 3 queries, 4x4 masks, NC+1=4 class logits.
    masks = torch.full((1, 3, 4, 4), -10.0)
    masks[0, 0, :2, :2] = 10.0  # query0 -> top-left block
    masks[0, 1, 2:, 2:] = 10.0  # query1 -> bottom-right block
    # query2 stays all-negative (empty mask -> IoU 0)
    cls = torch.full((1, 3, 4), -10.0)
    cls[0, 0, 0] = 10.0  # query0 predicts class 0
    cls[0, 1, 2] = 10.0  # query1 predicts class 2 (wrong for gt1)
    cls[0, 2, 1] = 10.0  # query2 predicts class 1
    out = {"masks_queries_logits": masks, "class_queries_logits": cls}

    gt0 = torch.zeros(4, 4); gt0[:2, :2] = 1.0
    gt1 = torch.zeros(4, 4); gt1[2:, 2:] = 1.0
    mask_labels = [torch.stack([gt0, gt1])]
    class_labels = [torch.tensor([0, 1])]

    # IoU-only: q0->gt0 (IoU 1) kept, q2->gt1 (IoU 0) dropped.
    idx = [(torch.tensor([0, 2]), torch.tensor([0, 1]))]
    src, tgt = gate_indices(out, idx, mask_labels, class_labels, iou_thr=0.5, require_class=False)[0]
    assert src.tolist() == [0] and tgt.tolist() == [0]

    # IoU + class: q1 localizes gt1 (IoU 1) but predicts class 2 != 1 -> dropped.
    idx = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    src, tgt = gate_indices(out, idx, mask_labels, class_labels, iou_thr=0.5, require_class=True)[0]
    assert src.tolist() == [0] and tgt.tolist() == [0]

    # Disabled gate is a no-op (returns the same object).
    assert gate_indices(out, idx, mask_labels, class_labels, iou_thr=0.0, require_class=False) is idx


def test_build_model_rejects_unimplemented_family():
    with pytest.raises(NotImplementedError):
        build_model("s", nc=NC, family="detect")
