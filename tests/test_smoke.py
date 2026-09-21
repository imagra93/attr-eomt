"""CPU, no-network smoke tests for the attr-eomt package."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eomt import build_eomt_config, build_model, postprocess_instance
from eomt.postprocess import postprocess_detection as _postprocess_detection
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
    # ``query_embed`` is emitted unconditionally — including for models with no aux
    # heads, which is what lets cross-photo re-id run on any checkpoint.
    assert set(out) == {"masks_queries_logits", "class_queries_logits", "query_embed"}
    q = EOMT_CONFIGS["s"].num_queries
    assert out["class_queries_logits"].shape == (2, q, NC + 1)
    assert out["masks_queries_logits"].shape[:2] == (2, q)
    assert out["query_embed"].shape == (2, q, EOMT_CONFIGS["s"].hidden_size)


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
    specs = [AuxHeadSpec("color", 4, {0: "a", 1: "b", 2: "c", 3: "d"}),
             AuxHeadSpec("material", 3, {0: "lo", 1: "mid", 2: "hi"})]
    model = build_model("s", nc=NC, imgsz=IMGSZ, aux_heads=specs).train()
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    mask_labels = [
        (torch.rand(2, IMGSZ, IMGSZ) > 0.5).float(),
        (torch.rand(1, IMGSZ, IMGSZ) > 0.5).float(),
    ]
    class_labels = [torch.tensor([0, 1]), torch.tensor([2])]
    aux_labels = {
        "color": [torch.tensor([1, 3]), torch.tensor([0])],
        "material": [torch.tensor([2, 0]), torch.tensor([1])],
    }
    out = model(x, mask_labels=mask_labels, class_labels=class_labels)
    total = out["loss"]
    a_loss, per_head = aux_loss(model, out, mask_labels, class_labels, aux_labels)
    assert set(per_head) == {"color", "material"}
    (total + a_loss).backward()
    # default head is a small MLP -> assert grads flow through *some* head param
    assert any(p.grad is not None for p in model.aux_heads["color"].parameters())

    acc = aux_accuracy(model, out, mask_labels, class_labels, aux_labels)
    assert set(acc) == {"color", "material"}

    # inference forward exposes per-head logits; postprocess attaches them
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out["aux_queries_logits"]["color"].shape == (2, EOMT_CONFIGS["s"].num_queries, 4)
    res = postprocess_instance(
        {k: v[:1] if torch.is_tensor(v) else {n: t[:1] for n, t in v.items()}
         for k, v in out.items() if k != "query_embed"},
        conf_thres=0.0, original_size=(20, 15), max_det=5,
    )
    assert set(res["aux"]) == {"color", "material"}
    assert res["aux"]["color"]["probs"].shape[1] == 4


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


def test_aux_applies_to_roundtrip():
    """A class-scoped head's applies_to survives checkpoint (de)serialization."""
    from eomt.config import AuxHeadSpec, aux_specs_from_meta, aux_specs_to_meta

    specs = [
        AuxHeadSpec("posture", 2, {0: "sit", 1: "stand"}, frozenset({1, 3})),
        AuxHeadSpec("coat", 3, {0: "short", 1: "long", 2: "curly"}, None),
    ]
    meta = aux_specs_to_meta(specs)
    assert meta[0]["applies_to"] == [1, 3]
    assert "applies_to" not in meta[1]  # unscoped -> key omitted
    back = aux_specs_from_meta(meta)
    assert back[0].applies_to == frozenset({1, 3})
    assert back[1].applies_to is None


def _write_mini_coco(tmp_path, with_applies_to: bool):
    """Write a 2-instance COCO (cat + dog) + one image; return (img_dir, json_path)."""
    import json

    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(img_dir / "im1.png")

    def sq(x0, y0, x1, y1):  # polygon segmentation for a rectangle
        return [[x0, y0, x1, y0, x1, y1, x0, y1]]

    attr_def = {
        "name": "posture",
        "categories": [{"id": 0, "name": "sit"}, {"id": 1, "name": "stand"}, {"id": 2, "name": "lie"}],
    }
    if with_applies_to:
        attr_def["applies_to"] = ["cat"]  # head applies to 'cat' only
    coco = {
        "images": [{"id": 1, "file_name": "im1.png", "width": 20, "height": 20}],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "attributes": [attr_def],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "iscrowd": 0,
             "bbox": [1, 1, 8, 8], "area": 64, "segmentation": sq(1, 1, 9, 9),
             "attributes": {"posture": 1}},   # cat, tagged 'stand'
            {"id": 11, "image_id": 1, "category_id": 2, "iscrowd": 0,
             "bbox": [10, 10, 8, 8], "area": 64, "segmentation": sq(10, 10, 18, 18),
             "attributes": {"posture": 2}},   # dog, tagged 'lie' (out of scope)
        ],
    }
    jf = tmp_path / "instances_train.json"
    jf.write_text(json.dumps(coco))
    return img_dir, jf


def test_aux_class_routing_in_dataset(tmp_path):
    """applies_to routes out-of-scope instances to -100 at label construction."""
    from eomt.data.coco import CocoInstanceSeg
    from eomt.data.transforms import build_val_transform

    # cat ids sorted -> cat=contig 0, dog=contig 1; head 'posture' applies to cat.
    img_dir, jf = _write_mini_coco(tmp_path, with_applies_to=True)
    ds = CocoInstanceSeg(img_dir, jf, imgsz=IMGSZ, transform=build_val_transform(IMGSZ))
    assert ds.aux_specs[0].applies_to == frozenset({0})  # 'cat' resolved to contig 0
    _, _, classes, attrs = ds[0]
    posture = attrs["posture"]
    # cat instance keeps its tagged id (1); dog instance is routed to -100.
    cat_pos = (classes == 0).nonzero(as_tuple=True)[0]
    dog_pos = (classes == 1).nonzero(as_tuple=True)[0]
    assert posture[cat_pos].item() == 1
    assert posture[dog_pos].item() == -100


def test_aux_no_applies_to_trains_all_classes(tmp_path):
    """Without applies_to (regression), the head is supervised on every class."""
    from eomt.data.coco import CocoInstanceSeg
    from eomt.data.transforms import build_val_transform

    img_dir, jf = _write_mini_coco(tmp_path, with_applies_to=False)
    ds = CocoInstanceSeg(img_dir, jf, imgsz=IMGSZ, transform=build_val_transform(IMGSZ))
    assert ds.aux_specs[0].applies_to is None
    _, _, classes, attrs = ds[0]
    # both instances keep their tagged posture (no routing to -100)
    assert set(attrs["posture"].tolist()) == {1, 2}


def test_postprocess_gates_aux_by_class():
    """Class-scoped aux heads emit ids=-1 for detections whose class is out of scope."""
    from eomt.postprocess import _build_aux_result

    aux_logits = {"posture": torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])}  # (1, Q=2, ns=2)
    sel = torch.tensor([0, 1])
    classes = torch.tensor([0, 1])  # det0 -> class 0 (in scope), det1 -> class 1 (out)
    res = _build_aux_result(aux_logits, sel, classes, {"posture": frozenset({0})})
    assert res["posture"]["ids"].tolist() == [0, -1]
    assert float(res["posture"]["probs"][1].sum()) == 0.0
    # unscoped -> unchanged
    res2 = _build_aux_result(aux_logits, sel, classes, {})
    assert res2["posture"]["ids"].tolist() == [0, 1]


def test_attribute_sidecar_merge(tmp_path):
    """A plain COCO + attributes.yaml + attributes/<split>.json is merged in memory."""
    import json

    from eomt.data.coco import _merge_attribute_sidecar

    (tmp_path / "annotations").mkdir()
    (tmp_path / "attributes").mkdir()
    (tmp_path / "attributes.yaml").write_text(
        "attributes:\n  - name: posture\n    categories: [sit, stand, lie]\n    applies_to: [cat]\n"
    )
    (tmp_path / "attributes" / "train.json").write_text(
        json.dumps({"10": {"posture": "stand"}, "11": {"posture": 2}})
    )

    class FakeCoco:
        dataset = {"annotations": [{"id": 10, "category_id": 1}, {"id": 11, "category_id": 2}]}

    coco = FakeCoco()
    _merge_attribute_sidecar(coco, tmp_path / "annotations" / "instances_train.json")
    assert coco.dataset["attributes"][0]["name"] == "posture"
    by_id = {a["id"]: a for a in coco.dataset["annotations"]}
    assert by_id[10]["attributes"] == {"posture": 1}  # label 'stand' -> raw id 1
    assert by_id[11]["attributes"] == {"posture": 2}  # raw id passthrough


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


def test_build_model_rejects_unknown_family():
    with pytest.raises(ValueError):
        build_model("s", nc=NC, family="banana")


def test_detect_family_forward_and_loss():
    """The detect family emits per-query boxes in [0,1] and a finite training loss."""
    model = build_model("s", nc=NC, imgsz=224, family="detect")
    assert model.family == "detect"
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
    boxes = out["pred_boxes"]
    assert boxes.shape == (2, model.config.num_queries, 4)
    assert float(boxes.min()) >= 0.0 and float(boxes.max()) <= 1.0
    assert "masks_queries_logits" not in out

    box_labels = [torch.rand(2, 4) * 0.5 + 0.25, torch.rand(1, 4) * 0.5 + 0.25]
    class_labels = [torch.tensor([0, 1]), torch.tensor([2])]
    model.train()
    out = model(x, box_labels=box_labels, class_labels=class_labels)
    assert torch.isfinite(out["loss"])


# --- training-recipe improvements -------------------------------------------


def test_grad_accumulation_matches_large_batch():
    """Accumulating ``loss/accum`` over micro-batches == one full-batch step (mean loss)."""
    import torch.nn as nn

    torch.manual_seed(0)
    lin = nn.Linear(4, 1)
    x, y = torch.randn(4, 4), torch.randn(4, 1)

    lin.zero_grad()
    ((lin(x) - y) ** 2).mean().backward()
    big = [p.grad.clone() for p in lin.parameters()]

    accum = 2
    lin.zero_grad()  # zero once at the window start
    for i in range(accum):
        xb, yb = x[i * 2 : (i + 1) * 2], y[i * 2 : (i + 1) * 2]
        (((lin(xb) - yb) ** 2).mean() / accum).backward()  # divide, accumulate
    acc = [p.grad.clone() for p in lin.parameters()]

    for a, b in zip(acc, big):
        assert torch.allclose(a, b, atol=1e-6)


def test_model_ema_tracks_params_and_copies_buffers():
    from eomt.ema import ModelEMA

    torch.manual_seed(0)
    model = build_model("s", nc=NC, imgsz=IMGSZ).train()
    ema = ModelEMA(model, decay=0.5, tau=0.0)  # tau<=0 -> constant decay 0.5

    name = next(iter(dict(model.named_parameters())))
    before = ema.module.state_dict()[name].clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    live = dict(model.named_parameters())[name]
    after = ema.module.state_dict()[name]
    assert torch.allclose(after, 0.5 * before + 0.5 * live, atol=1e-5)

    # The annealed attn_mask_probs buffer is copied verbatim, never averaged.
    model.eomt.attn_mask_probs.fill_(0.7)
    ema.update(model)
    assert torch.allclose(ema.module.eomt.attn_mask_probs, model.eomt.attn_mask_probs)


def test_build_optimizer_no_decay_and_llrd():
    from eomt.engine.train import _llrd_scale, build_optimizer

    model = build_model("s", nc=NC, imgsz=IMGSZ)
    opt = build_optimizer(model, lr=1e-4, weight_decay=0.05, backbone_lr_mult=0.1, llrd=1.0)

    # Both a weight-decay and a no-weight-decay group exist.
    wds = {g["weight_decay"] for g in opt.param_groups}
    assert 0.0 in wds and 0.05 in wds
    # llrd=1.0 -> exactly the legacy two LR levels (backbone*mult and head).
    lrs = sorted(round(g["lr"], 10) for g in opt.param_groups)
    assert set(lrs) == {1e-5, 1e-4}
    # A LayerNorm weight (1-D) lands in a no-decay group.
    pid2wd = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    ln = dict(model.named_parameters())["eomt.layernorm.weight"]
    assert pid2wd[id(ln)] == 0.0

    # llrd<1 -> deeper layers get higher LR than embeddings, and more groups.
    assert _llrd_scale("eomt.embeddings.x", 12, 0.85) < _llrd_scale("eomt.layers.11.x", 12, 0.85)
    opt2 = build_optimizer(model, 1e-4, 0.05, 0.1, llrd=0.85)
    assert len(opt2.param_groups) > len(opt.param_groups)


def test_lsj_train_transform_shapes_and_masks():
    from torchvision import tv_tensors

    from eomt.data.transforms import build_train_transform

    torch.manual_seed(0)
    tf = build_train_transform(IMGSZ, min_scale=0.1, max_scale=2.0)
    img = tv_tensors.Image(torch.randint(0, 255, (3, 80, 120), dtype=torch.uint8))
    masks = tv_tensors.Mask(torch.ones((2, 80, 120), dtype=torch.uint8))  # full -> always survive
    out_img, out_masks = tf(img, masks)
    assert out_img.shape == (3, IMGSZ, IMGSZ) and out_img.dtype == torch.float32
    assert out_masks.shape == (2, IMGSZ, IMGSZ)
    assert (out_masks.flatten(1).sum(1) > 0).all()  # both instances kept


def test_preprocess_letterbox_meta_and_padding():
    import numpy as np

    from eomt.preprocess import preprocess_numpy

    img = np.zeros((10, 40, 3), dtype=np.uint8)  # h=10, w=40 (wide)
    chw, meta = preprocess_numpy(img, 20, letterbox=True)
    assert chw.shape == (3, 20, 20)
    assert meta["letterbox"] and meta["content_hw"] == (5, 20) and meta["input_size"] == 20
    assert np.allclose(chw[:, 5:, :], 0.0)  # bottom padding == 0 (mean) in normalized space

    _, meta2 = preprocess_numpy(img, 20, letterbox=False)
    assert not meta2["letterbox"] and meta2["content_hw"] == (20, 20)


def test_letterbox_inverse_crops_content_not_padding():
    from eomt.postprocess import _masks_to_original
    from eomt.preprocess import make_preprocess_meta

    S = 20
    meta = make_preprocess_meta(True, (5, 20), S)  # content = top 5 rows of the canvas
    pos_content = torch.full((1, S, S), -10.0)
    pos_content[:, :5, :] = 10.0  # active only in the real-content region
    out = _masks_to_original(pos_content, 10, 40, meta)
    assert out.shape == (1, 10, 40)
    assert (out.sigmoid() > 0.5).all()  # content fills the whole original image

    pos_padding = torch.full((1, S, S), -10.0)
    pos_padding[:, 5:, :] = 10.0  # active only in the padding region
    out2 = _masks_to_original(pos_padding, 10, 40, meta)
    assert not (out2.sigmoid() > 0.5).any()  # padding is cropped away


def test_loss_weights_thread_into_criterion():
    """Tuned loss weights / num_upscale_blocks reach the HF criterion and mask head."""
    lw = {"no_object_weight": 0.05, "dice_weight": 8.0, "train_num_points": 24576}
    model = build_model("s", nc=NC, imgsz=IMGSZ, loss_weights=lw, num_upscale_blocks=3)
    crit = model.eomt.criterion
    assert float(crit.eos_coef) == 0.05
    assert float(crit.empty_weight[-1]) == pytest.approx(0.05)
    assert crit.num_points == 24576
    assert crit.matcher.cost_dice == 8.0
    assert model.eomt.weight_dict["loss_dice"] == 8.0
    assert model.num_upscale_blocks == 3
    n_blocks = len([k for k in model.state_dict()
                    if k.startswith("eomt.upscale_block.block.") and k.endswith(".conv1.weight")])
    assert n_blocks == 3
    # Unknown keys are rejected early.
    with pytest.raises(ValueError):
        build_model("s", nc=NC, imgsz=IMGSZ, loss_weights={"bogus": 1.0})


def test_loss_weights_checkpoint_roundtrip(tmp_path):
    """A tuned objective + non-default mask-head depth rebuilds identically on reload."""
    import warnings

    from eomt.serialization import load_model, save_checkpoint, wrap_checkpoint

    lw = {"no_object_weight": 0.05, "dice_weight": 8.0, "train_num_points": 24576}
    model = build_model("s", nc=NC, imgsz=IMGSZ, loss_weights=lw, num_upscale_blocks=3).eval()
    ckpt = wrap_checkpoint(
        model.state_dict(), size="s", nc=NC, imgsz=IMGSZ,
        loss_weights=model.loss_weights, num_upscale_blocks=model.num_upscale_blocks,
    )
    assert ckpt["loss_weights"]["no_object_weight"] == 0.05
    assert ckpt["num_upscale_blocks"] == 3
    path = tmp_path / "m.pt"
    save_checkpoint(ckpt, path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_model(path, device="cpu")
    msgs = [str(w.message) for w in caught]
    assert not any("missing" in m or "unexpected" in m for m in msgs), msgs
    crit = loaded.eomt.criterion
    assert float(crit.eos_coef) == 0.05 and crit.num_points == 24576
    assert crit.matcher.cost_dice == 8.0 and loaded.num_upscale_blocks == 3


def test_old_checkpoint_without_loss_metadata_loads(tmp_path):
    """A checkpoint predating loss_weights/num_upscale_blocks loads cleanly (defaults + inferred)."""
    import warnings

    from eomt.serialization import _infer_num_upscale_blocks, load_model, save_checkpoint

    model = build_model("s", nc=NC, imgsz=IMGSZ).eval()
    sd = model.state_dict()
    assert _infer_num_upscale_blocks(sd) == 2
    save_checkpoint({"model": sd, "size": "s", "nc": NC, "imgsz": IMGSZ}, tmp_path / "old.pt")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_model(tmp_path / "old.pt", device="cpu")
    msgs = [str(w.message) for w in caught]
    assert not any("missing" in m or "unexpected" in m for m in msgs), msgs
    assert float(loaded.eomt.criterion.eos_coef) == 0.1  # default restored
    assert loaded.num_upscale_blocks == 2


def test_checkpoint_stores_letterbox_mode(tmp_path):
    from eomt.serialization import load_model, save_checkpoint, wrap_checkpoint

    model = build_model("s", nc=NC, imgsz=IMGSZ).eval()
    ckpt = wrap_checkpoint(model.state_dict(), size="s", nc=NC, imgsz=IMGSZ, letterbox=True)
    assert ckpt["letterbox"] is True
    path = tmp_path / "m.pt"
    save_checkpoint(ckpt, path)
    loaded = load_model(path, device="cpu")
    assert loaded.preprocess_letterbox is True


def test_query_embed_matches_class_predictor_input():
    """``query_embed`` is the tensor the class head consumes — for both families.

    It is produced by slicing the encoder's final hidden state rather than by a
    forward hook; this pins that the slice is the same tensor the hook used to
    capture, which is what makes the hook removable.
    """
    from eomt.config import EOMT_CONFIGS

    torch.manual_seed(0)
    q = EOMT_CONFIGS["s"].num_queries
    for family in ("instance", "detect"):
        model = build_model("s", nc=NC, imgsz=IMGSZ, family=family).eval()
        captured = {}
        handle = model.eomt.class_predictor.register_forward_hook(
            lambda _m, inputs, _o: captured.__setitem__("q", inputs[0])
        )
        with torch.no_grad():
            out = model(torch.randn(1, 3, IMGSZ, IMGSZ))
        handle.remove()
        assert out["query_embed"].shape == (1, q, EOMT_CONFIGS["s"].hidden_size)
        assert torch.equal(out["query_embed"], captured["q"]), family


def test_postprocess_returns_aligned_query_idx():
    """``query_idx`` is the detection -> query mapping the whole re-id feature rests on."""
    torch.manual_seed(0)
    q, cls = 40, 3
    logits = torch.randn(1, q, cls + 1)
    out = {
        "masks_queries_logits": torch.randn(1, q, 8, 8),
        "class_queries_logits": logits,
    }
    res = postprocess_instance(out, 0.0, (16, 12), max_det=10, mask_thresh=0.5)
    sel = res["query_idx"]
    assert sel.dtype is torch.int64
    assert sel.shape == (10,) == res["classes"].shape  # the topk path fired
    assert len(set(sel.tolist())) == 10
    assert sel.min() >= 0 and sel.max() < q
    # The alignment guarantee: recomputing the class from the raw logits at the
    # kept query indices must reproduce exactly what postprocess returned.
    recomputed = logits[0].softmax(-1)[:, :-1].max(-1).indices[sel]
    assert torch.equal(recomputed, res["classes"])


def test_query_idx_empty_branches():
    torch.manual_seed(0)
    q = 8
    out = {
        "masks_queries_logits": torch.randn(1, q, 8, 8),
        "class_queries_logits": torch.randn(1, q, NC + 1),
        "pred_boxes": torch.rand(1, q, 4),
    }
    for fn, kwargs in (
        (postprocess_instance, {"mask_thresh": 0.5}),
        (_postprocess_detection, {}),
    ):
        res = fn(out, 1.1, (16, 12), max_det=5, **kwargs)  # nothing clears conf=1.1
        assert res["num_detections"] == 0
        assert res["query_idx"].shape == (0,) and res["query_idx"].dtype is torch.int64


def test_query_idx_survives_min_mask_area():
    torch.manual_seed(0)
    q = 20
    out = {
        "masks_queries_logits": torch.randn(1, q, 8, 8) - 0.5,
        "class_queries_logits": torch.randn(1, q, NC + 1),
    }
    res = postprocess_instance(
        out, 0.0, (16, 12), max_det=q, mask_thresh=0.5, min_mask_area=20.0
    )
    n = res["num_detections"]
    assert len(res["query_idx"]) == n == len(res["classes"]) == len(res["masks"])


def test_predict_image_embed_is_aligned_and_unit_norm():
    from PIL import Image

    from eomt.config import EOMT_CONFIGS
    from eomt.engine.predict import predict_image

    torch.manual_seed(0)
    model = build_model("s", nc=NC, imgsz=IMGSZ).eval()
    image = Image.fromarray(
        (torch.rand(40, 60, 3).numpy() * 255).astype("uint8"), mode="RGB"
    )
    res = predict_image(
        model, image, device=torch.device("cpu"), imgsz=IMGSZ,
        conf_thres=0.0, max_det=5, embed=True,
    )
    assert res["embed"].shape == (res["num_detections"], EOMT_CONFIGS["s"].hidden_size)
    assert len(res["embed"]) == len(res["query_idx"])
    assert res["embed"].dtype is torch.float32


def test_draw_instances_color_ids_and_label_prefix_are_additive():
    from PIL import Image

    from eomt.visualize import draw_instances

    torch.manual_seed(0)
    image = Image.new("RGB", (40, 30), (10, 10, 10))
    result = {
        "num_detections": 1,
        "boxes": torch.tensor([[4.0, 4.0, 20.0, 18.0]]),
        "scores": torch.tensor([0.9]),
        "classes": torch.tensor([1]),
        "masks": (torch.rand(1, 30, 40) > 0.5),
    }
    base = draw_instances(image, result, {1: "widget"})
    # Defaults left alone -> byte-identical to the pre-change path.
    assert np.array_equal(np.asarray(base), np.asarray(draw_instances(image, result, {1: "widget"})))
    # A different color index paints differently.
    a = draw_instances(image, result, {1: "widget"}, color_ids=[5])
    b = draw_instances(image, result, {1: "widget"}, color_ids=[6])
    assert not np.array_equal(np.asarray(a), np.asarray(b))
    # A label prefix changes the rendering too.
    p = draw_instances(image, result, {1: "widget"}, label_prefix=["#7"])
    assert not np.array_equal(np.asarray(p), np.asarray(base))


def test_draw_identity_grid_geometry_and_degenerates():
    from PIL import Image

    from eomt.visualize import draw_identity_grid

    torch.manual_seed(0)

    def panel(n):
        return {
            "image": Image.new("RGB", (60, 40), (30, 30, 30)),
            "result": {
                "num_detections": n,
                "boxes": torch.tensor([[2.0, 2.0, 30.0, 25.0]] * n).reshape(n, 4),
                "scores": torch.ones(n),
                "classes": torch.zeros(n, dtype=torch.long),
                "masks": (torch.rand(n, 40, 60) > 0.5),
            },
            "identity_ids": list(range(n)),
            "caption": "0. photo",
        }

    identities = [{"identity_id": 0, "class_name": "widget", "attributes": {},
                   "num_photos": 2}]
    arrows = [{"a_panel": 0, "a_det": 0, "b_panel": 1, "b_det": 0,
               "similarity": 0.8, "identity_id": 0}]
    grid = draw_identity_grid(
        [panel(1), panel(1)], names={0: "widget"}, arrows=arrows,
        identities=identities, cols=2, panel_size=64, title="t",
    )
    assert grid.mode == "RGB"
    assert grid.width == 2 * (64 + 12) + 12
    assert np.asarray(grid).std() > 0  # something was actually drawn

    # A photo with no detections still gets its panel, and no arrows is fine.
    grid2 = draw_identity_grid([panel(0), panel(1)], cols=2, panel_size=64, arrows=[])
    assert grid2.size[0] == 2 * (64 + 12) + 12


def test_draw_identity_grid_legend_attr_is_head_agnostic():
    """The legend must not hardcode any one checkpoint's attribute head name."""
    from PIL import Image

    from eomt.visualize import draw_identity_grid

    panels = [{
        "image": Image.new("RGB", (40, 40), (30, 30, 30)),
        "result": {
            "num_detections": 0,
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "classes": torch.zeros(0, dtype=torch.long),
        },
        "identity_ids": [],
        "caption": "0. photo",
    }]
    identities = [{
        "identity_id": 0, "class_name": "widget", "num_photos": 1,
        "attributes": {"alpha": "first", "beta": "second"},
    }]

    # panel_size drives the canvas width; too narrow and every legend string clips
    # at the same character, making distinct labels compare equal.
    def render(**kw):
        return np.asarray(
            draw_identity_grid(panels, identities=identities, cols=1, panel_size=320, **kw)
        )

    default = render()                          # -> first head ("alpha")
    explicit_first = render(legend_attr="alpha")
    explicit_second = render(legend_attr="beta")
    assert np.array_equal(default, explicit_first)
    assert not np.array_equal(explicit_first, explicit_second)

    # A head no identity carries just omits the attribute instead of raising.
    missing = render(legend_attr="not_a_head")
    assert not np.array_equal(missing, explicit_first)
    # ...and matches an identity that has no attributes at all.
    bare = np.asarray(draw_identity_grid(
        panels, cols=1, panel_size=320,
        identities=[{"identity_id": 0, "class_name": "widget", "num_photos": 1,
                     "attributes": {}}],
    ))
    assert np.array_equal(missing, bare)


def test_draw_identity_grid_arrow_width_tracks_similarity():
    """A confident link must draw thicker than a marginal one, on an absolute scale."""
    from PIL import Image

    from eomt.visualize import draw_identity_grid

    torch.manual_seed(0)

    def panel():
        m = torch.zeros(1, 40, 60, dtype=torch.bool)
        m[0, 18:22, 28:32] = True  # a small, centered blob -> a stable centroid
        return {
            "image": Image.new("RGB", (60, 40), (0, 0, 0)),
            "result": {
                "num_detections": 1,
                "boxes": torch.tensor([[28.0, 18.0, 32.0, 22.0]]),
                "scores": torch.ones(1),
                "classes": torch.zeros(1, dtype=torch.long),
                "masks": m,
            },
            "identity_ids": [0],
            "caption": "",
        }

    def ink(sim, **kw):
        """Pixels the connector overlay painted, i.e. how heavy the stroke is."""
        grid = draw_identity_grid(
            [panel(), panel()], cols=2, panel_size=240, legend=False,
            draw_boxes=False, alpha=0.0,
            arrows=[{"a_panel": 0, "a_det": 0, "b_panel": 1, "b_det": 0,
                     "similarity": sim, "identity_id": 0}],
            **kw,
        )
        return int((np.asarray(grid).sum(axis=2) > 0).sum())

    band = {"sim_range": (0.6, 1.0)}
    assert ink(0.98, **band) > ink(0.80, **band) > ink(0.61, **band)

    # Absolute, not relative: a lone arrow's width depends only on its similarity,
    # so a near-perfect match never renders hairline just because it is alone.
    assert ink(0.99, **band) > ink(0.62, **band)

    # Out-of-band similarities clamp instead of overshooting the width range.
    assert ink(2.0, **band) == ink(1.0, **band)
    assert ink(-1.0, **band) == ink(0.6, **band)

    # An explicit width range is honored.
    assert ink(0.9, sim_range=(0.6, 1.0), arrow_width=(1, 2)) < ink(
        0.9, sim_range=(0.6, 1.0), arrow_width=(8, 16)
    )
