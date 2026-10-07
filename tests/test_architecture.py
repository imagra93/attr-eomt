"""Auxiliary box head, deep-supervision switch, attention mask, dense matcher / mask loss, attribute
gate and bf16 inference (CPU, no network). The HF-parity of the default model is in test_parity.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from PIL import Image

from eomt import build_model
from eomt.box_loss import masks_to_norm_boxes
from eomt.postprocess import boxes_from_masks, postprocess_instance
from eomt.serialization import load_model, save_checkpoint, wrap_checkpoint

from test_augment import _mini_coco  # noqa: E402  (tests/ is on sys.path)

IMGSZ, NC = 140, 3


def _batch(b=2, size=IMGSZ, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 3, size, size, generator=g)
    masks = [(torch.rand(n, size, size, generator=g) > 0.7).float() for n in (2, 1)[:b]]
    classes = [torch.tensor([0, 1]), torch.tensor([2])][:b]
    return x, masks, classes


def _model(**kw):
    torch.manual_seed(0)
    return build_model("s", nc=NC, imgsz=IMGSZ, **kw)


# ------------------------------------------------------------------------------------------- attention mask


def test_compact_attention_mask_semantics():
    m = _model().eval()
    enc = m.eomt
    nq, start = enc.config.num_queries, enc.config.num_queries + enc.embeddings.num_prefix_tokens
    n = start + 100
    hidden = torch.zeros(2, n, 8)
    logits = torch.full((2, nq, 20, 20), -1.0)
    logits[:, 3, :10] = 1.0  # query 3 predicts the top half of the image
    mask = enc._build_attention_mask(hidden, logits, 1.0, grid_size=(10, 10))
    assert mask.shape == (2, 1, n, n) and mask.dtype == torch.bool  # broadcast over heads, no dense fp32 bias
    q3 = mask[0, 0, 3, start:].view(10, 10)
    assert q3[:5].all() and not q3[5:].any()        # may only attend to the patches inside its predicted mask
    assert mask[0, 0, 0, start:].all()              # an empty prediction attends to every patch (Mask2Former), not none
    assert mask[:, 0, nq:, :].all() and mask[:, 0, :nq, :start].all()  # everything else is unrestricted
    # probability 0 unmasks every query
    assert enc._build_attention_mask(hidden, logits, 0.0, grid_size=(10, 10)).all()


def test_attention_mask_keeps_a_small_predicted_mask():
    """A mask covering a single logit cell still opens its patch (bilinear downsampling dropped it: blind query)."""
    m = _model().eval()
    enc = m.eomt
    nq, start = enc.config.num_queries, enc.config.num_queries + enc.embeddings.num_prefix_tokens
    hidden = torch.zeros(1, start + 100, 8)
    logits = torch.full((1, nq, 40, 40), -1.0)      # 4 x 4 logit cells per patch, as 184 -> 46
    logits[0, 5, 13, 30] = 1.0                      # query 5: one cell, off the centre of patch (3, 7)
    q5 = enc._build_attention_mask(hidden, logits, 1.0, grid_size=(10, 10))[0, 0, 5, start:].view(10, 10)
    assert q5[3, 7] and q5.sum() == 1


def test_blocks_at_probability_zero_build_no_mask_and_no_extra_prediction():
    m = _model().train()
    enc = m.eomt
    built, predicted = [], []
    orig_build, orig_predict = enc._build_attention_mask, enc.predict
    enc._build_attention_mask = lambda *a, **k: (built.append(1), orig_build(*a, **k))[1]
    enc.predict = lambda *a, **k: (predicted.append(1), orig_predict(*a, **k))[1]
    x, masks, classes = _batch()
    enc.deep_supervision = False
    m.eomt.set_attn_mask_probs([0.0, 0.0, 0.0, 0.0])
    m(x, mask_labels=masks, class_labels=classes)
    assert built == [] and len(predicted) == 1        # only the final prediction: nothing needs the intermediate ones
    built.clear(); predicted.clear()
    m.eomt.set_attn_mask_probs([0.0, 0.0, 0.5, 1.0])
    m(x, mask_labels=masks, class_labels=classes)
    assert len(built) == 2 and len(predicted) == 3    # two masked blocks (+ the final output) predict for the mask
    built.clear(); predicted.clear()
    enc.deep_supervision = True
    m(x, mask_labels=masks, class_labels=classes)
    assert len(predicted) == 5                        # deep supervision: 4 blocks + final
    built.clear(); predicted.clear()
    m.eomt.set_attn_mask_probs([0.0, 0.0, 0.0, 0.0])
    m(x, mask_labels=masks, class_labels=classes)
    assert built == [] and len(predicted) == 5        # supervised blocks still predict, but build no (all-True) mask


def test_deep_supervision_off_still_supervises_the_blocks_that_mask():
    """Off = a block's prediction is supervised while it builds the attention mask, and dropped once it no longer does."""
    def n_losses(ds, probs):
        m = _model().train()
        m.eomt.deep_supervision = ds
        m.eomt.set_attn_mask_probs(probs)
        calls = []
        orig = m.eomt.get_loss_dict
        m.eomt.get_loss_dict = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        x, masks, classes = _batch()
        out = m(x, mask_labels=masks, class_labels=classes)
        out["loss"].backward()
        assert all(p.grad is not None for p in m.eomt.class_predictor.parameters())
        return len(calls), float(out["loss"])

    assert n_losses(True, [1, 1, 1, 1])[0] == 5 and n_losses(True, [0, 0, 0, 0])[0] == 5
    assert n_losses(False, [1, 1, 1, 1])[0] == 5          # every block masks: all supervised (the masks need it)
    assert n_losses(False, [1.0, 0.5, 0.0, 0.0])[0] == 3  # two masking blocks + the final output
    final_only = n_losses(False, [0, 0, 0, 0])
    assert final_only[0] == 1 and final_only[1] < n_losses(True, [0, 0, 0, 0])[1]


def test_intermediate_predictions_stop_at_four_times_the_patch_grid():
    """With a 3-step head only the final output is made at 8x the patch grid; the block predictions (which build 1x
    attention masks) are made and supervised at 4x."""
    m = _model(num_upscale_blocks=3).train()
    shapes = []
    orig = m.eomt.get_loss_dict
    m.eomt.get_loss_dict = lambda g, *a, **k: (shapes.append(tuple(g.shape[-2:])), orig(g, *a, **k))[1]
    x, masks, classes = _batch()
    out = m(x, mask_labels=masks, class_labels=classes)
    out["loss"].backward()
    grid = IMGSZ // 14
    assert shapes == [(4 * grid, 4 * grid)] * 4 + [(8 * grid, 8 * grid)]
    assert out["masks_queries_logits"].shape[-2:] == (8 * grid, 8 * grid)


def test_attention_mask_probabilities_follow_the_host_mirror_in_training_only():
    m = _model()
    m.eomt.set_attn_mask_probs([1.0, 0.5, 0.0, 0.0])
    assert m.eomt.attn_mask_probs.tolist() == [1.0, 0.5, 0.0, 0.0]
    m.train()
    assert [m.eomt._block_prob(i) for i in range(4)] == [1.0, 0.5, 0.0, 0.0]
    m.eomt.attn_mask_probs.zero_()                     # the trainer zeroes the buffer before evaluating
    m.eval()
    assert [m.eomt._block_prob(i) for i in range(4)] == [0.0] * 4   # eval reads the buffer, not the stale mirror


# ------------------------------------------------------------------------------------------- box head


def test_box_head_is_instance_only():
    with pytest.raises(ValueError, match="instance family"):
        build_model("s", nc=NC, imgsz=IMGSZ, family="detect", aux_box_head=True)


def test_masks_to_boxes_and_vectorised_boxes_from_masks():
    rng = np.random.default_rng(0)
    masks = torch.from_numpy(rng.random((7, 30, 40)) > 0.97)
    masks[3] = False                                                     # an empty mask stays all zeros
    masks[5, 4:9, 10:30] = True
    ref = torch.zeros(7, 4)
    for i in range(7):                                                   # the loop the vectorised version replaces
        ys, xs = torch.where(masks[i])
        if ys.numel():
            ref[i] = torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=torch.float32)
    assert torch.equal(boxes_from_masks(masks), ref)
    assert boxes_from_masks(torch.zeros(0, 5, 5, dtype=torch.bool)).shape == (0, 4)
    soft = torch.zeros(1, 20, 20)
    soft[0, 4:8, 6:12] = 0.3                                             # soft, thin masks never reach 0.5
    assert masks_to_norm_boxes(soft)[0].tolist() == pytest.approx([9 / 20, 6 / 20, 6 / 20, 4 / 20])


def _chip(size=644, y=300, x=400, side=4):
    gt = torch.zeros(1, size, size)
    gt[0, y : y + side, x : x + side] = 1.0
    return gt


def _cells(gt, size=(184, 184)):
    """The logit-grid cells an instance touches."""
    return F.adaptive_max_pool2d(gt[None].float(), size)[0] > 0


def test_dense_matcher_assigns_a_tiny_instance_to_the_query_that_predicts_it():
    """A 16 px chip at 644 px (~1.3 cells of the 184 grid) is matched to the query predicting it, every time. The point
    sampler it replaces put no point inside such a chip ~60 % of the time, and then preferred an empty query."""
    from eomt.loss import HungarianMatcher

    gt = _chip()
    logits = torch.full((1, 10, 184, 184), -8.0)
    logits[0, 6][_cells(gt)[0]] = 8.0                          # query 6 predicts the chip
    logits[0, 2, :60] = 8.0                                    # query 2 predicts a large region elsewhere
    cls = torch.zeros(1, 10, 4)                                # equal class scores: the mask terms decide
    matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)
    for _ in range(3):
        src, tgt = matcher(logits, cls, [gt], [torch.tensor([1])])[0]
        assert src.tolist() == [6] and tgt.tolist() == [0]


def test_dense_mask_loss_sees_a_tiny_instance():
    crit = _model().eomt.criterion
    gt = _chip()
    y = F.interpolate(gt[None], size=(184, 184), mode="area")[0, 0]
    logits = torch.full((1, 2, 184, 184), -8.0)
    logits[0, 0] = torch.logit(y.clamp(1e-4, 1 - 1e-4))      # query 0 predicts the chip exactly; query 1 predicts nothing
    hit = crit.loss_masks(logits, [gt], [(torch.tensor([0]), torch.tensor([0]))], 1.0)
    miss = crit.loss_masks(logits, [gt], [(torch.tensor([1]), torch.tensor([0]))], 1.0)
    assert hit["loss_dice"] < miss["loss_dice"] - 0.1               # dice is dense: it always sees the chip
    again = crit.loss_masks(logits, [gt], [(torch.tensor([0]), torch.tensor([0]))], 1.0)
    assert torch.equal(again["loss_dice"], hit["loss_dice"])        # and deterministic (the BCE samples points)


def test_an_instance_smaller_than_a_logit_cell_is_cheaper_to_predict_than_to_miss():
    """A 2 x 2 px instance at a 644 px input straddles four 3.5 px cells of the 184 grid. Scored on the grid against its
    area-averaged GT (0.08 per cell), dice preferred a query predicting nothing; at the GT resolution a query predicting
    a tight blob there wins. (The point-sampled BCE rarely lands on such an instance; dice carries it.)"""
    crit = _model().eomt.criterion
    gt = _chip(y=300, x=300, side=2)                           # pixels 300-301: cells 85-86 on both axes
    logits = torch.full((1, 2, 184, 184), -8.0)
    logits[0, 0, 85:87, 85:87] = 1.0                           # query 0 predicts the instance; query 1 predicts nothing
    hit = crit.loss_masks(logits, [gt], [(torch.tensor([0]), torch.tensor([0]))], 1.0)
    miss = crit.loss_masks(logits, [gt], [(torch.tensor([1]), torch.tensor([0]))], 1.0)
    assert hit["loss_dice"] < miss["loss_dice"]


def test_checkpoints_of_dropped_matcher_options_still_load(tmp_path):
    m = _model().eval()
    lw = {**m.loss_weights, "match_class_weight": 0.0, "match_full_res": True}   # written by runs of the dropped options
    save_checkpoint(wrap_checkpoint(m.state_dict(), size="s", nc=NC, imgsz=IMGSZ, loss_weights=lw), tmp_path / "m.pt")
    assert load_model(tmp_path / "m.pt", device="cpu").eomt.criterion.matcher.cost_class == 2.0


def test_iou_aware_class_targets_reduce_to_the_weighted_ce_and_follow_mask_quality():
    crit = _model(loss_weights={"iou_aware_cls": True}).eomt.criterion
    assert crit.iou_aware_cls and not _model().eomt.criterion.iou_aware_cls
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, NC + 1, generator=g)
    labels = [torch.tensor([1, 2]), torch.tensor([0])]
    idx = [(torch.tensor([3, 0]), torch.tensor([0, 1])), (torch.tensor([4]), torch.tensor([0]))]
    hard = crit.loss_labels(logits, labels, idx)["loss_cross_entropy"]
    soft = crit.loss_labels(logits, labels, idx, quality=torch.ones(3))["loss_cross_entropy"]
    assert torch.allclose(hard, soft, atol=1e-6)                       # IoU 1: exactly the weighted CE
    # a matched query is pushed towards (1 + IoU) / 2: the class logit that minimises its loss tracks its mask IoU
    best = []
    for iou in (0.2, 0.8):
        x = torch.zeros(1, 1, NC + 1, requires_grad=True)
        opt = torch.optim.SGD([x], lr=1.0)
        for _ in range(300):
            opt.zero_grad()
            crit.loss_labels(x, [torch.tensor([0])], [(torch.tensor([0]), torch.tensor([0]))],
                             quality=torch.tensor([iou]))["loss_cross_entropy"].backward()
            opt.step()
        best.append(float(x.softmax(-1)[0, 0, 0]))
    assert abs(best[0] - 0.6) < 0.02 and abs(best[1] - 0.9) < 0.02
    # matched IoU is measured per pair, in permutation order
    gt = torch.zeros(2, 64, 64)
    gt[0, :32] = 1.0
    gt[1, 32:] = 1.0
    mql = torch.full((1, 3, 16, 16), -10.0)
    mql[0, 2, :8] = 10.0                                               # query 2 predicts GT 0 exactly
    mql[0, 1, :, :8] = 10.0                                            # query 1 predicts the left half: IoU 1/3 with GT 1
    iou = crit.matched_iou(mql, [gt], [(torch.tensor([2, 1]), torch.tensor([0, 1]))])
    assert torch.allclose(iou, torch.tensor([1.0, 1 / 3]), atol=1e-3)


def test_mask_quality_head_learns_the_matched_iou_and_scores_the_masks(tmp_path):
    assert _model().eomt.quality_head is None
    m = _model(loss_weights={"quality_weight": 2.0})
    enc = m.eomt
    assert enc.quality_head is not None and enc.weight_dict["loss_quality"] == 2.0
    x, masks, cls = _batch()
    m.train()(x, mask_labels=masks, class_labels=cls)["loss"].backward()
    assert enc.quality_head[-1].weight.grad.abs().sum() > 0           # trained through the final prediction
    # the loss pulls each matched query's predicted IoU towards its measured one
    crit = enc.criterion
    gt = torch.zeros(1, 64, 64)
    gt[0, :32] = 1.0
    mql = torch.full((1, 2, 16, 16), -10.0)
    mql[0, 0, :, :8] = 10.0                                            # IoU 1/3 with the GT
    cql = torch.zeros(1, 2, NC + 1)
    lo = crit(mql, cql, [gt], [torch.tensor([0])], pred_quality=torch.tensor([[torch.logit(torch.tensor(1 / 3)), 0.0]]))
    hi = crit(mql, cql, [gt], [torch.tensor([0])], pred_quality=torch.tensor([[4.0, 0.0]]))
    assert lo["loss_quality"] < hi["loss_quality"]
    # postprocess scores class prob x predicted IoU when the head is there
    out = {"masks_queries_logits": mql, "class_queries_logits": cql, "quality_logits": torch.tensor([[2.0, -2.0]])}
    res = postprocess_instance(out, 0.0, (64, 64), max_det=2)
    p = cql[0].softmax(-1)[:, :-1].max(-1).values
    assert torch.allclose(res["scores"].sort().values, (p * torch.tensor([2.0, -2.0]).sigmoid()).sort().values)
    save_checkpoint(wrap_checkpoint(m.state_dict(), size="s", nc=NC, imgsz=IMGSZ, loss_weights=m.loss_weights),
                    tmp_path / "m.pt")
    back = load_model(tmp_path / "m.pt", device="cpu")
    assert back.eomt.quality_head is not None and "quality_logits" in back.eval()(x)


def test_attribute_gate_keeps_a_tiny_instance():
    """The IoU gate compares at the GT's resolution: a predicted 4 px chip passes (nearest-shrinking the GT erased it)."""
    from eomt.aux_cls import gate_indices

    gt = 0.8 * _chip(y=298, x=298, side=2)   # soft, as training masks are; between the pixels a 644 -> 184 "nearest" reads
    logits = torch.full((1, 3, 184, 184), -8.0)
    logits[0, 1][_cells(gt)[0]] = 8.0
    out = {"masks_queries_logits": logits, "class_queries_logits": torch.zeros(1, 3, 4)}
    kept = gate_indices(out, [(torch.tensor([1]), torch.tensor([0]))], [gt], [torch.tensor([1])], iou_thr=0.05)
    assert kept[0][0].tolist() == [1]


def test_box_term_in_the_matcher_decides_between_otherwise_identical_queries():
    m = _model(aux_box_head=True).eval()
    matcher = m.eomt.criterion.matcher
    assert matcher.cost_bbox == 5.0 and matcher.cost_giou == 2.0
    gt = torch.zeros(2, 40, 40)
    gt[0, 2:10, 2:10] = 1                                                # box A: top-left
    gt[1, 28:38, 28:38] = 1                                              # box B: bottom-right
    q, c = 4, 3
    logits = torch.zeros(1, q, 10, 10)                                   # uninformative masks...
    cls = torch.zeros(1, q, c + 1)                                       # ...and identical classes
    boxes = torch.tensor([[[0.8, 0.8, 0.3, 0.3], [0.15, 0.15, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1], [0.85, 0.85, 0.25, 0.25]]])
    labels = [torch.tensor([0, 0])]
    (src, tgt), = matcher(logits, cls, [gt], labels, pred_boxes=boxes)
    assignment = dict(zip(tgt.tolist(), src.tolist()))
    assert assignment == {0: 1, 1: 3}                                    # GT A <- the box near it (query 1), GT B <- query 3
    (src2, tgt2), = matcher(logits, cls, [gt.flip(0)], labels, pred_boxes=boxes)
    assert dict(zip(tgt2.tolist(), src2.tolist())) == {0: 3, 1: 1}       # follows the labels when they swap


def test_box_loss_is_zero_for_perfect_boxes_and_the_head_is_trained():
    m = _model(aux_box_head=True).train()
    x, masks, classes = _batch()
    out = m(x, mask_labels=masks, class_labels=classes)
    assert "aux_boxes" in out and out["aux_boxes"].shape == (2, m.config.num_queries, 4) and "pred_boxes" not in out
    out["loss"].backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.eomt.aux_box_head.parameters())
    crit = m.eomt.criterion
    gt = [masks_to_norm_boxes(t) for t in masks]
    perfect = torch.zeros(2, m.config.num_queries, 4) + 0.5
    for i, g in enumerate(gt):
        perfect[i, : len(g)] = g
    idx = [(torch.arange(len(g)), torch.arange(len(g))) for g in gt]
    res = crit.loss_boxes(perfect, gt, idx, num_boxes=torch.tensor(3.0))
    assert res["loss_bbox"].item() == pytest.approx(0, abs=1e-6) and res["loss_giou"].item() == pytest.approx(0, abs=1e-6)


def test_attribute_matching_uses_the_same_box_aware_assignment():
    from eomt.aux_cls import match_queries

    m = _model(aux_box_head=True).eval()
    x, masks, classes = _batch()
    out = m(x)
    got = match_queries(m, out, masks, classes)
    want = m.eomt.criterion.matcher(out["masks_queries_logits"], out["class_queries_logits"], masks, classes,
                                    pred_boxes=out["aux_boxes"])
    # (the matcher samples random points, so only the sizes are deterministic here)
    assert [len(s) for s, _ in got] == [len(s) for s, _ in want] == [2, 1]


def test_postprocess_exposes_the_head_boxes():
    q, c = 6, 3
    logits = torch.full((1, q, 8, 8), -5.0)
    logits[0, 2, 2:6, 2:6] = 5.0
    cls = torch.full((1, q, c + 1), -5.0)
    cls[0, 2, 1] = 5.0
    out = {"masks_queries_logits": logits, "class_queries_logits": cls,
           "aux_boxes": torch.tensor([[[0.5, 0.5, 0.25, 0.25]] * q])}
    res = postprocess_instance(out, 0.3, (80, 80))
    assert res["num_detections"] == 1 and res["head_boxes"].shape == (1, 4)
    assert res["head_boxes"][0].tolist() == pytest.approx([30.0, 30.0, 50.0, 50.0])
    assert "head_boxes" not in postprocess_instance({k: v for k, v in out.items() if k != "aux_boxes"}, 0.3, (80, 80))


# ------------------------------------------------------------------------------------------- checkpoints


def test_checkpoint_round_trip_rebuilds_the_new_modules(tmp_path):
    m = _model(fpn_scales=None, aux_box_head=True).eval()
    ckpt = wrap_checkpoint(m.state_dict(), size="s", nc=NC, imgsz=IMGSZ, fpn_scales=None, aux_box_head=m.aux_box_head)
    path = tmp_path / "m.pt"
    save_checkpoint(ckpt, path)
    back = load_model(path, device="cpu")
    assert back.aux_box_head and not back.fpn_scales
    x = torch.randn(1, 3, IMGSZ, IMGSZ)
    assert torch.allclose(m(x)["masks_queries_logits"], back(x)["masks_queries_logits"], atol=1e-5)
    # a checkpoint stripped of the new metadata is still rebuilt from its tensors
    bare = {"model": m.state_dict(), "size": "s", "nc": NC, "imgsz": IMGSZ, "task": "instance"}
    save_checkpoint(bare, tmp_path / "bare.pt")
    inferred = load_model(tmp_path / "bare.pt", device="cpu")
    assert inferred.aux_box_head
    # and a plain model has none of them
    plain = _model().eval()
    save_checkpoint(wrap_checkpoint(plain.state_dict(), size="s", nc=NC, imgsz=IMGSZ), tmp_path / "plain.pt")
    again = load_model(tmp_path / "plain.pt", device="cpu")
    assert not again.aux_box_head


def test_a_pixel_detail_checkpoint_fails_to_load_with_a_clear_error(tmp_path):
    state = {**_model().state_dict(), "eomt.detail.stem.0.weight": torch.zeros(32, 3, 3, 3)}
    for name, ckpt in {"meta": {"detail": ["mask"]}, "bare": {}}.items():
        save_checkpoint({"model": state, "size": "s", "nc": NC, "imgsz": IMGSZ, **ckpt}, tmp_path / f"{name}.pt")
        with pytest.raises(ValueError, match="pixel-detail pathway"):
            load_model(tmp_path / f"{name}.pt", device="cpu")


# ------------------------------------------------------------------------------------------- training / inference


def test_train_with_the_new_architecture_then_resume_keeps_it(tmp_path):
    from eomt.engine.train import train

    img_dir, jf = _mini_coco(tmp_path)
    kw = dict(train_images=str(img_dir), train_json=str(jf), size="s", imgsz=112, batch=1, accum=1, workers=0, device="cpu",
              amp=False, pretrained=False, ema=True, seed=0, project=str(tmp_path / "runs"), name="a", warmup_steps=(1, 1))
    res = train(epochs=1, fpn_scales=None, box_head=True, deep_supervision=False, **kw)
    args = yaml.safe_load((tmp_path / "runs" / "a" / "args.yaml").read_text())
    assert args["box_head"] is True and args["deep_supervision"] is False and args["fpn_scales"] is None
    ck = torch.load(res["last"], weights_only=False)
    assert ck["aux_box_head"] is True and not ck.get("fpn_scales")
    assert {"epoch", "optimizer", "ema"} <= set(ck)
    # resuming ignores the (default) arguments and keeps the checkpoint's architecture
    res2 = train(epochs=2, resume=res["last"], **kw)
    ck2 = torch.load(res2["last"], weights_only=False)
    assert ck2["epoch"] == 1 and ck2["aux_box_head"] is True
    assert load_model(res2["last"], device="cpu").aux_box_head


def test_predict_image_amp_flag(tmp_path):
    from eomt.engine.predict import _amp_dtype, predict_image

    assert _amp_dtype("cpu", True) is None and _amp_dtype("cpu", "bf16") is None and _amp_dtype("cuda", False) is None
    with pytest.raises(ValueError):
        _amp_dtype("cuda", "fp8")
    m = _model().eval()
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8))
    a = predict_image(m, img, device=torch.device("cpu"), imgsz=IMGSZ, conf_thres=0.0, max_det=5)
    b = predict_image(m, img, device=torch.device("cpu"), imgsz=IMGSZ, conf_thres=0.0, max_det=5, amp=True)  # CPU: no-op
    assert torch.equal(a["scores"], b["scores"]) and a["masks"].shape == b["masks"].shape


def test_stop_after_epochs_truncates_a_run_but_keeps_its_schedule(tmp_path):
    from eomt.engine.train import _poly_two_stage_factors, train

    img_dir, jf = _mini_coco(tmp_path)
    kw = dict(train_images=str(img_dir), train_json=str(jf), size="s", imgsz=112, batch=1, accum=1, workers=0, device="cpu",
              amp=False, pretrained=False, ema=False, seed=0, project=str(tmp_path / "runs"), name="t", warmup_steps=(1, 1))
    res = train(epochs=3, stop_after_epochs=1, **kw)
    ck = torch.load(res["last"], weights_only=False)
    assert ck["epoch"] == 0                                              # one epoch ran, not three
    head_f, _ = _poly_two_stage_factors(3, 12, (1, 1), 0.9)             # last micro-step of epoch 0 in a 3-epoch schedule
    group = next(g for g in ck["optimizer"]["param_groups"] if not g["is_backbone"])
    assert group["lr"] == pytest.approx(group["initial_lr"] * head_f, rel=1e-6) and group["lr"] > 0.5 * group["initial_lr"]
    res2 = train(epochs=3, stop_after_epochs=2, resume=res["last"], **{**kw, "seed": None})   # composes with resume
    assert torch.load(res2["last"], weights_only=False)["epoch"] == 1


def test_last_checkpoint_records_the_best_metric_of_its_own_epoch(tmp_path):
    from eomt.engine.train import train

    img_dir, jf = _mini_coco(tmp_path)
    res = train(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s", imgsz=112,
                epochs=1, batch=1, accum=1, workers=0, device="cpu", amp=False, pretrained=False, ema=False, seed=0,
                project=str(tmp_path / "runs"), name="b", warmup_steps=(1, 1))
    ck = torch.load(res["last"], weights_only=False)
    # one epoch is the discriminating case: a stale last.pt would still hold the initial -1
    assert ck["best_metric"] == pytest.approx(res["best_metric"]) and ck["best_metric"] >= 0
