"""Several photos of one subject predicted together (cross-view modules, matching head, synthetic sets)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from eomt import build_model
from eomt.aux_cls import harm_loss, match_loss
from eomt.data import CocoViewSets, collate_sets
from eomt.data.sets import DISTRACTOR_IDS, FLIPPED_IDS, UNKNOWN_IDS

IMGSZ = 140
KEYS = ("masks_queries_logits", "class_queries_logits", "query_embed")


def _pair(mode: str):
    """A single-photo model and the same weights with the cross-view modules, as the post-training starts."""
    torch.manual_seed(0)
    base = build_model("s", nc=3, imgsz=IMGSZ, fpn_scales=(2.0, 1.0, 0.5)).eval()
    xv = build_model("s", nc=3, imgsz=IMGSZ, fpn_scales=(2.0, 1.0, 0.5), xview={"mode": mode}, match_dim=16).eval()
    xv.load_state_dict(base.state_dict(), strict=False)
    xv.init_xview()
    return base, xv


@pytest.mark.parametrize("mode", ["blocks", "refiner"])
def test_a_photo_alone_is_predicted_exactly_as_before(mode):
    base, xv = _pair(mode)
    x = torch.randn(3, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        ref = base(x)
        outs = [xv(x), xv(x, set_ids=torch.tensor([0, 1, 2])), xv(x, set_ids=torch.tensor([0, 0, 1]))]
    for k in KEYS:
        assert torch.equal(outs[0][k], ref[k]) and torch.equal(outs[1][k], ref[k])
        assert torch.equal(outs[2][k][2], ref[k][2])  # alone in its set, inside a batch with a set of two


@pytest.mark.parametrize("mode", ["blocks", "refiner"])
def test_a_set_starts_within_1e4_of_the_photos_alone(mode):
    base, xv = _pair(mode)
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        ref, out = base(x), xv(x, set_ids=torch.tensor([0, 0]))
    for k in KEYS:
        rel = float((out[k] - ref[k]).norm() / ref[k].norm())
        assert 0 < rel < 1e-4 or (mode == "refiner" and k == "masks_queries_logits" and rel == 0)
    if mode == "refiner":  # the level-1 refiner only relabels: the masks are untouched
        assert torch.equal(out["masks_queries_logits"], ref["masks_queries_logits"])


def test_cross_view_modules_get_gradients_and_the_backbone_is_frozen():
    from eomt.engine.train import build_multiview_optimizer

    _, xv = _pair("blocks")
    opt = build_multiview_optimizer(xv.train(), 1e-4, 0.05, {"xview": 1.0, "match_head": 1.0})
    trainable = {n for n, p in xv.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("eomt.xview.", "match_head.")) for n in trainable)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(trainable)
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    masks = [(torch.rand(2, IMGSZ, IMGSZ) > 0.7).float() for _ in range(2)]
    out = xv(x, mask_labels=masks, class_labels=[torch.tensor([0, 1])] * 2, set_ids=torch.tensor([0, 0]))
    out["loss"].backward()
    assert all(xv.get_parameter(n).grad is not None for n in trainable if n.startswith("eomt.xview."))


def test_match_loss_pulls_the_same_instance_together_across_photos():
    q = torch.nn.functional.normalize(torch.randn(2, 6, 8), dim=-1)
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1])), (torch.tensor([2, 3]), torch.tensor([0, 1]))]
    inst_ids = [torch.tensor([5, 7]), torch.tensor([5, 7])]  # instances 5 and 7 in both photos
    aligned = q.clone()
    aligned[1, 2], aligned[1, 3] = q[0, 0], q[0, 1]
    swapped = q.clone()
    swapped[1, 2], swapped[1, 3] = q[0, 1], q[0, 0]
    sets = torch.tensor([0, 0])
    good = match_loss({"match_embed": aligned}, indices, inst_ids, sets)
    bad = match_loss({"match_embed": swapped}, indices, inst_ids, sets)
    assert good < bad
    # other sets, or one photo, have no positive: nothing to learn
    assert match_loss({"match_embed": aligned}, indices, inst_ids, torch.tensor([0, 1])) == 0


def _mini_coco(tmp_path, n_images=4, subjects=None):
    """Tiny photos with a left blob and a centre line each; ``side`` is flip-aware, ``area`` a crop-scaled value.
    ``subjects``: each photo's ``subject_id`` (None: no subject)."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    imgs, anns = [], []
    subjects = subjects or [None] * n_images
    for i, subject in enumerate(subjects, start=1):
        name = f"im{i}.png"
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(img_dir / name)
        imgs.append({"id": i, "file_name": name, "width": 160, "height": 120,
                     **({"subject_id": subject} if subject is not None else {})})
        for cat, side, (x0, y0, x1, y1) in ((1, 0, (10, 10, 70, 60)), (2, 1, (60, 80, 150, 92))):
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cat, "iscrowd": 0,
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                         "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                         "attributes": {"side": side, "area": float((x1 - x0) * (y1 - y0))}})
    coco = {"images": imgs, "annotations": anns,
            "categories": [{"id": 1, "name": "blob"}, {"id": 2, "name": "line"}],
            "attributes": [{"name": "side", "flip_aware": {"left": "right"},
                            "categories": [{"id": 0, "name": "left"}, {"id": 1, "name": "center"},
                                           {"id": 2, "name": "right"}]},
                           {"name": "area", "type": "regression", "log": True, "crop": "scale"}]}
    jf = tmp_path / "instances.json"
    jf.write_text(json.dumps(coco))
    return img_dir, jf


def test_sets_keep_instance_ids_and_flip_once_per_set(tmp_path):
    img_dir, jf = _mini_coco(tmp_path)
    ds = CocoViewSets(img_dir, jf, imgsz=112, aug={"min_mask_mass": 0.0},
                      sets={"p_single": 0.0, "p_distractor": 1.0, "p_flipped_copy": 1.0}, seed=0)
    for idx in range(len(ds)):
        item = ds[idx]
        views = item["views"]
        assert len(views) == 4 and item["case"] in ("small", "degraded", "closeup", "occluded")
        side = {}  # instance id -> its side in the photo's views
        for k, (img, masks, cls, attrs, ids) in enumerate(views):
            assert img.shape == (3, 112, 112) and masks.shape[0] == cls.numel() == ids.numel()
            if k == 2:  # the distractor: other instances
                assert (ids >= DISTRACTOR_IDS).all() and (ids < FLIPPED_IDS).all()
                continue
            for i, c, s in zip(ids.tolist(), cls.tolist(), attrs["side"].tolist()):
                base = i % FLIPPED_IDS
                assert c == base % 2  # the class of annotation `base` (blob, line, blob, ...)
                if k == 3:  # the mirrored copy: the photo's other flip
                    assert i >= FLIPPED_IDS
                    if base in side and side[base] != 1:
                        assert s == 2 - side[base]
                else:
                    assert i < DISTRACTOR_IDS and side.setdefault(base, s) == s  # one flip for the whole set
    # fixed sets: the same draws every time
    a, b = ds[1], ds[1]
    assert a["case"] == b["case"] and all(torch.equal(x[0], y[0]) for x, y in zip(a["views"], b["views"]))


def test_locate_sets_zoom_on_the_focus_instance(tmp_path):
    img_dir, jf = _mini_coco(tmp_path)
    ds = CocoViewSets(img_dir, jf, imgsz=112, seed=0,
                      sets={"p_single": 0.0, "p_focus": 1.0, "focus": {"class": "line", "side": "center"},
                            "p_distractor": 0.0, "p_flipped_copy": 0.0})
    line = 1  # the line is annotation 1 of every photo
    for idx in range(len(ds)):
        item = ds[idx]
        assert item["case"] == "locate"
        (_, a_masks, _, a_attrs, a_ids), (_, b_masks, _, b_attrs, b_ids) = item["views"]
        assert set(a_attrs) == set(b_attrs) == {"side", "area"}  # the temporary focus mark never leaves the view
        assert line in b_ids.tolist()  # the close-up is of the focus instance
        if line in a_ids.tolist():
            share = lambda masks, ids: float(masks[ids.tolist().index(line)].sum()) / masks[0].numel()  # noqa: E731
            assert share(b_masks, b_ids) > share(a_masks, a_ids)
    with pytest.raises(ValueError, match="not one of"):
        CocoViewSets(img_dir, jf, imgsz=112, sets={"p_focus": 1.0, "focus": {"class": "wheel"}})


def test_collate_flattens_sets_into_photos(tmp_path):
    img_dir, jf = _mini_coco(tmp_path)
    ds = CocoViewSets(img_dir, jf, imgsz=112, sets={"p_single": 0.5}, seed=1)
    batch = [ds[0], ds[1], ds[2]]
    pixel_values, masks, classes, attrs, set_ids, inst_ids, extras = collate_sets(batch)
    assert extras is None  # synthetic sets carry no partial labels
    n = sum(len(item["views"]) for item in batch)
    assert pixel_values.shape == (n, 3, 112, 112) and len(masks) == len(classes) == len(inst_ids) == n
    assert set_ids.tolist() == [s for s, item in enumerate(batch) for _ in item["views"]]
    assert set(attrs) == {"side", "area"}


def test_xview_checkpoint_round_trip(tmp_path):
    from eomt.serialization import load_model, save_checkpoint, wrap_checkpoint

    _, xv = _pair("blocks")
    xv.match_thres = 0.42
    path = tmp_path / "x.pt"
    save_checkpoint(wrap_checkpoint(xv.state_dict(), size="s", nc=3, imgsz=IMGSZ, fpn_scales=xv.fpn_scales,
                                    xview=xv.xview_cfg, match_dim=xv.match_dim, match_thres=xv.match_thres), path)
    back = load_model(path, device="cpu")
    assert back.xview_cfg == xv.xview_cfg and back.match_dim == 16 and back.match_thres == 0.42
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        a, b = xv(x, set_ids=torch.tensor([0, 0])), back(x, set_ids=torch.tensor([0, 0]))
    assert torch.allclose(a["match_embed"], b["match_embed"]) and torch.allclose(a["query_embed"], b["query_embed"])


def test_sets_fill_to_their_drawn_size_with_photos_of_the_same_subject(tmp_path):
    img_dir, jf = _mini_coco(tmp_path, subjects=["A", "A", "B", "B", None])
    ds = CocoViewSets(img_dir, jf, imgsz=112, aug={"min_mask_mass": 0.0}, seed=0,
                      sets={"p_single": 0.0, "sizes": {4: 1.0}, "fill": {"subject": 1.0}, "p_subject": 1.0})
    assert ds._subject_idx == [0, 1, 2, 3]  # the photo with no subject is not one
    for idx in range(len(ds)):
        views = ds[idx]["views"]
        assert len(views) == 4
        assert all((v[4] >= 0).all() and (v[4] < DISTRACTOR_IDS).all() for v in views[:2])  # A and B: the photo
        for slot, v in enumerate(views[2:], start=2):  # the subject's other photo: ids unknown, a block per slot
            assert (v[4] < 0).all() and (v[4] >= UNKNOWN_IDS * slot).all() and (v[4] < UNKNOWN_IDS * (slot - 1)).all()
    # sizes are drawn; a subject view without another photo of the subject falls back to another part of the photo
    ds = CocoViewSets(img_dir, jf, imgsz=112, aug={"min_mask_mass": 0.0}, seed=0,
                      sets={"p_single": 0.0, "sizes": {2: 1.0, 3: 1.0}, "fill": {"subject": 1.0}})
    assert {len(ds[i]["views"]) for i in range(len(ds))} <= {2, 3}
    assert all((v[4] >= 0).all() for v in ds[4]["views"][2:])  # no subject: no other photo of it
    with pytest.raises(ValueError):
        CocoViewSets(img_dir, jf, imgsz=112, sets={"sizes": {5: 1.0}})
    with pytest.raises(ValueError, match="subject"):  # the old name says where it went
        CocoViewSets(img_dir, jf, imgsz=112, sets={"sizes": {2: 1.0}, "fill": {"case": 1.0}})


def test_match_loss_leaves_out_instances_of_unknown_correspondence():
    q = torch.nn.functional.normalize(torch.randn(3, 6, 8), dim=-1)
    two = [(torch.tensor([0, 1]), torch.tensor([0, 1])), (torch.tensor([2, 3]), torch.tensor([0, 1]))]
    ids = [torch.tensor([5, 7]), torch.tensor([5, 7])]
    base = match_loss({"match_embed": q[:2]}, two, ids, torch.tensor([0, 0]))
    three = two + [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    unknown = ids + [torch.tensor([UNKNOWN_IDS * 2, UNKNOWN_IDS * 2 + 1])]
    assert torch.allclose(match_loss({"match_embed": q}, three, unknown, torch.tensor([0, 0, 0])), base)


def test_harm_loss_holds_true_instances_to_their_score_alone():
    alone = torch.zeros(3, 4, 3)
    alone[:, 0, 1] = 3.0  # query 0 of every photo: confident in class 1, its target's class
    lower, higher = alone.clone(), alone.clone()
    lower[:, 0, 1], higher[:, 0, 1] = 1.0, 5.0
    indices = [(torch.tensor([0]), torch.tensor([0]))] * 3
    labels = [torch.tensor([1])] * 3
    sets = torch.tensor([0, 0, 1])  # photos 0 and 1 together, photo 2 alone
    hurt = harm_loss({"class_queries_logits": lower}, {"class_queries_logits": alone}, indices, labels, sets)
    p_alone, p = alone[0, 0].softmax(-1)[1], lower[0, 0].softmax(-1)[1]
    assert torch.isclose(hurt, p_alone - p)  # the photo alone (photo 2) takes no part
    assert harm_loss({"class_queries_logits": higher}, {"class_queries_logits": alone}, indices, labels, sets) == 0
    assert harm_loss({"class_queries_logits": lower}, {"class_queries_logits": alone}, indices, labels,
                     torch.tensor([0, 1, 2])) == 0


def test_stage_a_post_training_runs_and_leaves_the_checkpoint_untouched(tmp_path):
    from eomt.data import CocoInstanceSeg
    from eomt.engine.train import train
    from eomt.serialization import load_model, save_checkpoint, wrap_checkpoint

    img_dir, jf = _mini_coco(tmp_path)
    specs = CocoInstanceSeg(img_dir, jf, imgsz=112).aux_specs
    torch.manual_seed(0)
    start = build_model("s", nc=2, imgsz=112, aux_heads=specs, fpn_scales=(2.0, 1.0, 0.5))
    init = tmp_path / "init.pt"
    save_checkpoint(wrap_checkpoint(start.state_dict(), size="s", nc=2, imgsz=112, aux_heads=specs,
                                    aux_head_arch=start.aux_head_arch, fpn_scales=start.fpn_scales), init)
    res = train(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s",
                imgsz=112, epochs=1, batch=1, accum=1, workers=0, device="cpu", amp=False, ema=False, seed=0,
                project=str(tmp_path / "runs"), name="xv", init_weights=str(init),
                multiview={"match_dim": 16, "sets": {"p_single": 0.5}})
    post = load_model(res["best"], device="cpu")
    assert post.xview_cfg["mode"] == "blocks" and post.match_dim == 16
    before, after = start.state_dict(), post.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # stage A: a photo alone is predicted as before
    assert any(k.startswith("eomt.xview.") for k in after) and any(k.startswith("match_head.") for k in after)


def test_stage_a_with_same_subject_sets_and_do_no_harm_leaves_the_checkpoint_untouched(tmp_path):
    from eomt.data import CocoInstanceSeg
    from eomt.engine.train import train
    from eomt.serialization import load_model, save_checkpoint, wrap_checkpoint

    img_dir, jf = _mini_coco(tmp_path, subjects=[1, 1, 2, 2])
    specs = CocoInstanceSeg(img_dir, jf, imgsz=112).aux_specs
    torch.manual_seed(0)
    start = build_model("s", nc=2, imgsz=112, aux_heads=specs, fpn_scales=(2.0, 1.0, 0.5))
    init = tmp_path / "init.pt"
    save_checkpoint(wrap_checkpoint(start.state_dict(), size="s", nc=2, imgsz=112, aux_heads=specs,
                                    aux_head_arch=start.aux_head_arch, fpn_scales=start.fpn_scales), init)
    res = train(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s",
                imgsz=112, epochs=1, batch=1, accum=1, workers=0, device="cpu", amp=False, ema=False, seed=0,
                project=str(tmp_path / "runs"), name="xv", init_weights=str(init),
                multiview={"match_dim": 16, "harm_w": 1.0,
                           "sets": {"p_single": 0.0, "sizes": {3: 1.0}, "p_subject": 1.0,
                                    "fill": {"subject": 1.0, "positive": 1.0, "elsewhere": 1.0}}})
    post = load_model(res["best"], device="cpu")
    before, after = start.state_dict(), post.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    rows = (tmp_path / "runs" / "xv" / "metrics.csv").read_text().splitlines()
    assert "train/harm_loss" in rows[0].split(",")


def test_stage_b_continues_a_cross_view_checkpoint_with_locate_sets(tmp_path):
    import csv

    from eomt.data import CocoInstanceSeg
    from eomt.engine.train import train
    from eomt.serialization import load_model, save_checkpoint, wrap_checkpoint

    img_dir, jf = _mini_coco(tmp_path)
    specs = CocoInstanceSeg(img_dir, jf, imgsz=112).aux_specs
    torch.manual_seed(0)
    start = build_model("s", nc=2, imgsz=112, aux_heads=specs, fpn_scales=(2.0, 1.0, 0.5))
    init = tmp_path / "init.pt"
    save_checkpoint(wrap_checkpoint(start.state_dict(), size="s", nc=2, imgsz=112, aux_heads=specs,
                                    aux_head_arch=start.aux_head_arch, fpn_scales=start.fpn_scales), init)
    common = dict(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s",
                  imgsz=112, epochs=1, batch=1, accum=1, workers=0, device="cpu", amp=False, ema=False, seed=0,
                  project=str(tmp_path / "runs"))
    a = train(**common, name="a", init_weights=str(init), multiview={"match_dim": 16, "sets": {"p_single": 0.5}})
    b = train(**common, name="b", init_weights=a["best"], warmup_steps=(0, 0),  # the query blocks train from step 0
              multiview={"trainable": {"xview": 1.0, "query_blocks": 0.1},
                         "sets": {"p_single": 0.0, "p_focus": 0.5, "focus": {"class": "line"}}})
    post_a, post_b = load_model(a["best"], device="cpu"), load_model(b["best"], device="cpu")
    assert post_b.xview_cfg == post_a.xview_cfg and post_b.match_dim == 16  # the cross-view modules carry over
    sa, sb = post_a.state_dict(), post_b.state_dict()
    last = post_a.config.num_hidden_layers - 1
    assert not torch.equal(sa[f"eomt.layers.{last}.mlp.fc1.weight"], sb[f"eomt.layers.{last}.mlp.fc1.weight"])  # trained
    assert torch.equal(sa["eomt.layers.0.mlp.fc1.weight"], sb["eomt.layers.0.mlp.fc1.weight"])  # frozen
    header = next(csv.reader(open(tmp_path / "runs" / "b" / "metrics.csv")))
    assert "val/sets/locate/mAP_set" in header
    # the query blocks alone, single photos only (a plain fine-tune of the last blocks) also run
    c = train(**common, name="c", init_weights=a["best"], warmup_steps=(0, 0),
              multiview={"trainable": {"query_blocks": 0.1}, "sets": {"p_single": 1.0}})
    sc = load_model(c["best"], device="cpu").state_dict()
    assert not torch.equal(sa[f"eomt.layers.{last}.mlp.fc1.weight"], sc[f"eomt.layers.{last}.mlp.fc1.weight"])


def test_post_training_with_the_main_run_config_keeps_the_checkpoint_heads_and_inference(tmp_path):
    """The run config of the main training (a high-resolution phase, ``attributes`` naming one head's crop) over
    data-only annotations: the phase stays off, the head keeps the checkpoint's ``log``, the run config's ``crop``
    applies, the head it leaves out is kept, and the checkpoint's inference settings carry over."""
    from eomt.data import CocoInstanceSeg
    from eomt.engine.train import train
    from eomt.serialization import load_model, load_raw, save_checkpoint, wrap_checkpoint

    img_dir, jf = _mini_coco(tmp_path)
    specs = CocoInstanceSeg(img_dir, jf, imgsz=112).aux_specs  # "area" trained with log: true
    torch.manual_seed(0)
    start = build_model("s", nc=2, imgsz=112, aux_heads=specs, fpn_scales=(2.0, 1.0, 0.5))
    init = tmp_path / "init.pt"
    save_checkpoint(wrap_checkpoint(start.state_dict(), size="s", nc=2, imgsz=112, aux_heads=specs,
                                    aux_head_arch=start.aux_head_arch, fpn_scales=start.fpn_scales,
                                    inference={"conf_thres": 0.25}), init)
    data = json.loads(jf.read_text())
    data["attributes"][1] = {"name": "area", "type": "regression"}  # data only: how it learns is the run config's
    jf.write_text(json.dumps(data))
    res = train(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s",
                imgsz=112, epochs=1, batch=1, accum=1, workers=0, device="cpu", amp=False, ema=False, seed=0,
                project=str(tmp_path / "runs"), name="xv", init_weights=str(init), hires_epochs=1, hires_imgsz=168,
                attributes={"area": {"crop": "scale"}}, multiview={"match_dim": 16})
    post = load_model(res["best"], device="cpu")
    assert {s.name: s.log for s in post.aux_specs} == {"side": False, "area": True}
    assert post.inference == {"conf_thres": 0.25}
    assert load_raw(res["best"])["run_config"]["attributes"]["area"]["crop"] == "scale"


def test_subject_sets_place_every_photo():
    from eomt.engine.predict import subject_sets

    feats = torch.randn(7, 8)
    assert subject_sets(feats, 4, mode="order") == [[0, 1, 2, 3], [4, 5, 6]]
    sets = subject_sets(feats, 4)
    assert sorted(i for s in sets for i in s) == list(range(7)) and all(len(s) <= 4 for s in sets)
    anchors = subject_sets(feats, 4, mode="anchor", k=2)
    assert [s[0] for s in anchors] == list(range(7)) and all(len(s) == 3 and len(set(s)) == 3 for s in anchors)


def test_predict_subject_returns_one_result_per_photo():
    from eomt.engine.predict import predict_subject, predict_set

    base, xv = _pair("blocks")
    photos = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (90 + 10 * i, 120, 3), dtype=np.uint8))
              for i in range(5)]
    for mode in ("order", "partition", "anchor"):
        res = predict_subject(xv, photos, device="cpu", imgsz=IMGSZ, mode=mode, conf_thres=0.0, max_det=5,
                            feature_size=70)
        assert len(res) == 5 and all(r["masks"].shape[1:] == (90 + 10 * i, 120) for i, r in enumerate(res))
    # a model without cross-view modules predicts each photo alone
    alone = predict_subject(base, photos, device="cpu", imgsz=IMGSZ, conf_thres=0.0, max_det=5)
    single = [predict_set(base, [p], device="cpu", imgsz=IMGSZ, conf_thres=0.0, max_det=5)[0] for p in photos]
    assert all(torch.equal(a["scores"], s["scores"]) for a, s in zip(alone, single))


def test_predict_takes_a_subfolder_per_subject(tmp_path, monkeypatch):
    import importlib

    P = importlib.import_module("eomt.engine.predict")  # the module, not the predict() that eomt.engine re-exports

    _, xv = _pair("blocks")
    subjects = {"a": 2, "b": 5}
    for name, n in subjects.items():
        (tmp_path / "in" / name).mkdir(parents=True)
        for i in range(n):
            photo = np.random.default_rng(i).integers(0, 255, (90, 120, 3), dtype=np.uint8)
            Image.fromarray(photo).save(tmp_path / "in" / name / f"{i}.png")
    sizes, predict_set = [], P.predict_set

    def spy(model, images, **kw):
        sizes.append(len(images))
        return predict_set(model, images, **kw)

    monkeypatch.setattr(P, "predict_set", spy)
    res = P.predict(xv, tmp_path / "in", subject=True, plot=True, save=tmp_path / "out", conf_thres=0.0, max_det=5)
    assert sizes == [2, 4, 1]  # each subject apart, at most max_views photos at a time
    assert [r["plot_path"] for r in res] == [str(tmp_path / "out" / name / f"{i}.png")
                                             for name, n in subjects.items() for i in range(n)]
    with pytest.raises(ValueError, match="augment"):
        P.predict(xv, tmp_path / "in", subject=True, augment=True)


def test_predict_set_returns_one_result_per_photo_with_match_embeddings():
    from eomt.engine.predict import predict_set

    _, xv = _pair("blocks")
    photos = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (100, 130, 3), dtype=np.uint8)) for i in range(3)]
    res = predict_set(xv, photos, device="cpu", imgsz=IMGSZ, conf_thres=0.0, max_det=5)
    assert len(res) == 3 and all(r["embed"].shape == (r["num_detections"], 16) for r in res)
    assert res[0]["masks"].shape[1:] == (100, 130)
    with pytest.raises(ValueError, match="1 to 4 photos"):
        predict_set(xv, photos + photos, device="cpu", imgsz=IMGSZ)


# --- 2.2.0: batched cross-view attention, keys, view types, matching head and losses ---------------------------------


def _loop_cross_view(sub, hidden_states, groups, nq):
    """The cross-view sublayer as 2.1.0 wrote it (the reference for its default settings)."""
    import torch.nn.functional as F  # noqa: N812

    a = sub.attention
    h, dh, d = a.num_heads, a.head_dim, a.embed_dim
    rows = list(hidden_states[:, :nq].unbind(0))
    for idx in groups:
        x = sub.norm(hidden_states[idx])
        q, k, v = a.q_proj(x[:, :nq]), a.k_proj(x), a.v_proj(x)
        for i in range(len(idx)):
            others = [j for j in range(len(idx)) if j != i]
            kk = torch.cat([k[others].reshape(-1, d), sub.null_key[None]])
            vv = torch.cat([v[others].reshape(-1, d), v.new_zeros(1, d)])
            n = kk.shape[0]
            qh = q[i].view(-1, h, dh).transpose(0, 1)[None]
            bias = torch.cat([qh.new_zeros(h, 1, n - 1), sub.null_logit.view(h, 1, 1)], -1)
            out = F.scaled_dot_product_attention(qh, kk.view(n, h, dh).transpose(0, 1)[None],
                                                 vv.view(n, h, dh).transpose(0, 1)[None], attn_mask=bias[None],
                                                 scale=a.scale)
            rows[int(idx[i])] = rows[int(idx[i])] + sub.layer_scale * a.out_proj(out[0].transpose(0, 1).reshape(-1, d))
    return torch.cat((torch.stack(rows), hidden_states[:, nq:]), dim=1)


def _cross_view_sublayer(**kw):
    from eomt.config import build_eomt_config
    from eomt.model import CrossViewAttention

    torch.manual_seed(0)
    sub = CrossViewAttention(build_eomt_config("s", nc=3, image_size=IMGSZ), init_scale=0.5, **kw).eval()
    with torch.no_grad():  # the same null key whatever else was drawn at init
        sub.null_key.copy_(torch.randn(384, generator=torch.Generator().manual_seed(1)))
    return sub


def test_cross_view_attention_with_its_default_settings_computes_what_it_did_before():
    sub = _cross_view_sublayer()
    x = torch.randn(6, 31, 384)
    groups = [torch.tensor([0, 2, 3]), torch.tensor([4, 5])]  # photo 1 is alone
    with torch.no_grad():
        out, ref = sub(x, groups, 10), _loop_cross_view(sub, x, groups, 10)
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.equal(out[1], x[1])  # a photo alone is untouched


def test_cross_view_keys_and_view_types():
    x = torch.randn(3, 31, 384)
    groups = [torch.tensor([0, 1, 2])]
    with torch.no_grad():
        both = _cross_view_sublayer()(x, groups, 10)
        queries = _cross_view_sublayer(keys="queries")(x, groups, 10)
        typed = _cross_view_sublayer(view_types=3)
        start = typed(x, groups, 10, torch.tensor([1, 2, 0]))
        typed.view_embed.weight.normal_()
        moved = typed(x, groups, 10, torch.tensor([1, 2, 0]))
    assert not torch.allclose(both, queries)  # the queries of the others only
    assert torch.allclose(start, both, atol=1e-6)  # the view embedding starts at 0: no change
    assert not torch.allclose(moved, both)
    with pytest.raises(ValueError, match="keys"):
        _cross_view_sublayer(keys="all")


def test_a_view_conditioned_model_round_trips_its_vocabulary(tmp_path):
    from eomt.engine.predict import _view_type_ids
    from eomt.serialization import load_model, save_checkpoint, wrap_checkpoint

    xv = build_model("s", nc=3, imgsz=IMGSZ, xview={"mode": "blocks", "view_types": ["front", "rear"]}, match_dim=16)
    path = tmp_path / "v.pt"
    save_checkpoint(wrap_checkpoint(xv.state_dict(), size="s", nc=3, imgsz=IMGSZ, xview=xv.xview_cfg, match_dim=16), path)
    back = load_model(path, device="cpu")
    assert back.xview_cfg["view_types"] == ["front", "rear"] and back.eomt.xview[0].view_embed.weight.shape[0] == 3
    assert _view_type_ids(back, ["rear", None, "side"], 3).tolist() == [2, 0, 0]


def test_a_residual_matching_head_starts_as_re_identification():
    torch.manual_seed(0)
    full = build_model("s", nc=3, imgsz=IMGSZ, xview={"mode": "blocks"}, match_dim=384, match_residual=True).eval()
    small = build_model("s", nc=3, imgsz=IMGSZ, xview={"mode": "blocks"}, match_dim=64, match_residual=True).eval()
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        out, out_small = full(x, set_ids=torch.tensor([0, 0])), small(x, set_ids=torch.tensor([0, 0]))
    reid = torch.nn.functional.normalize(out["query_embed"].float(), dim=-1)
    assert torch.allclose(out["match_embed"], reid, atol=1e-6)  # exactly re-ID's cosines at the start
    assert out_small["match_embed"].shape[-1] == 64 and small.match_skip is not None


def _match_case():
    """Two photos of one set with instances 5 and 7 each, their embeddings aligned (the same instance alike) or swapped."""
    q = torch.nn.functional.normalize(torch.randn(2, 6, 8), dim=-1)
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1])), (torch.tensor([2, 3]), torch.tensor([0, 1]))]
    aligned, swapped = q.clone(), q.clone()
    aligned[1, 2], aligned[1, 3] = q[0, 0], q[0, 1]
    swapped[1, 2], swapped[1, 3] = q[0, 1], q[0, 0]
    return aligned, swapped, indices, [torch.tensor([5, 7]), torch.tensor([5, 7])], torch.tensor([0, 0])


@pytest.mark.parametrize("kind", ["sigmoid", "dustbin"])
def test_the_calibrated_matching_losses_prefer_the_right_links(kind):
    from eomt.aux_cls import match_loss_params

    aligned, swapped, indices, ids, sets = _match_case()
    params = match_loss_params(kind)
    good = match_loss({"match_embed": aligned}, indices, ids, sets, kind=kind, params=params)
    bad = match_loss({"match_embed": swapped}, indices, ids, sets, kind=kind, params=params)
    assert good < bad
    good.backward()
    assert all(p.grad is not None for p in params.values())  # the scale / dustbin learn


def test_the_matching_loss_sees_false_detections_hard_negatives_and_reports_what_it_sees():
    torch.manual_seed(0)
    aligned, _, indices, ids, sets = _match_case()
    base = match_loss({"match_embed": aligned}, indices, ids, sets)
    stats: dict = {}
    copy = aligned.clone()
    copy[1, 5] = aligned[0, 0]  # an unmatched query of photo 1 that looks like instance 5 of photo 0
    negatives = torch.zeros(2, 6, dtype=torch.bool)
    negatives[1, 5] = True
    with_false = match_loss({"match_embed": copy}, indices, ids, sets, negatives=negatives, stats=stats)
    assert with_false > base  # the duplicate now competes with the true partner
    # each query's hardest other: the duplicate (cos 1) for instance 5 of photo 0, the other instance for the rest
    other = float(aligned[0, 0] @ aligned[0, 1])
    assert stats["pos_cos"] == pytest.approx(1.0) and stats["hardneg_cos"] == pytest.approx((1 + 3 * other) / 4)
    assert stats["unpaired"] == 0.0
    keys = [torch.tensor([3, 3]), torch.tensor([3, 3])]  # the two instances look alike (one key): hard negatives
    harder = match_loss({"match_embed": aligned}, indices, ids, sets, hard_keys=keys, hard_w=4.0)
    assert harder > base


def test_consistency_loss_pulls_the_views_of_one_instance_together():
    from eomt.aux_cls import consistency_loss
    from eomt.config import AuxHeadSpec

    torch.manual_seed(0)
    specs = [AuxHeadSpec("side", 3, {0: "l", 1: "c", 2: "r"}), AuxHeadSpec("size", 1, regression=True)]
    model = build_model("s", nc=3, imgsz=IMGSZ, aux_heads=specs)
    q = torch.randn(2, 4, 384)
    same = q.clone()
    same[1, 1] = q[0, 0]  # instance 9: query 0 of photo 0 and query 1 of photo 1, alike
    indices = [(torch.tensor([0]), torch.tensor([0])), (torch.tensor([1]), torch.tensor([0]))]
    args = (indices, [torch.tensor([9]), torch.tensor([9])], torch.tensor([0, 0]), ["side", "size"],
            [torch.tensor([1]), torch.tensor([1])])
    assert float(consistency_loss(model, {"query_embed": same}, *args)) == pytest.approx(0.0, abs=1e-6)
    assert float(consistency_loss(model, {"query_embed": q}, *args)) > 0
    apart = (indices, [torch.tensor([9]), torch.tensor([9])], torch.tensor([0, 1]), ["side"], args[4])
    assert float(consistency_loss(model, {"query_embed": q}, *apart)) == 0  # other sets: nothing to agree on


def test_linked_instances_weights_and_hard_keys():
    from eomt.engine.train import _hard_keys, _linked_weights

    ids = [torch.tensor([5, 7, -100_000]), torch.tensor([5]), torch.tensor([7])]
    w = _linked_weights(ids, torch.tensor([0, 0, 1]), 3.0)  # photo 2 is another set
    assert w[0].tolist() == [3.0, 1.0, 1.0] and w[1].tolist() == [3.0] and w[2].tolist() == [1.0]
    keys = _hard_keys([torch.tensor([0, 0, 1])], {"side": [torch.tensor([2, -100, 2])]}, ["class", "side"])[0]
    assert keys[0] >= 0 and keys[1] == -1 and keys[2] != keys[0]


def test_instance_weights_of_one_leave_the_loss_unchanged_and_weigh_it_otherwise():
    torch.manual_seed(0)
    model = build_model("s", nc=3, imgsz=IMGSZ).train()
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    masks = [(torch.rand(2, IMGSZ, IMGSZ) > 0.7).float() for _ in range(2)]
    cls = [torch.tensor([0, 1])] * 2
    torch.manual_seed(1)  # the mask BCE samples its points at random
    base = model(x, mask_labels=masks, class_labels=cls)["loss"]
    torch.manual_seed(1)
    ones = model(x, mask_labels=masks, class_labels=cls, target_weights=[torch.ones(2)] * 2)["loss"]
    torch.manual_seed(1)
    weighted = model(x, mask_labels=masks, class_labels=cls, target_weights=[torch.tensor([4.0, 1.0])] * 2)["loss"]
    assert torch.allclose(base, ones, rtol=1e-5) and not torch.allclose(base, weighted, rtol=1e-3)


def test_multiview_trainable_takes_parameter_prefixes_and_the_layer_scales():
    from eomt.config import AuxHeadSpec
    from eomt.engine.train import build_multiview_optimizer

    specs = [AuxHeadSpec("side", 3), AuxHeadSpec("size", 1, regression=True)]
    m = build_model("s", nc=3, imgsz=IMGSZ, aux_heads=specs, xview={"mode": "blocks"}, match_dim=16)
    opt = build_multiview_optimizer(m, 1e-4, 0.05, {"heads": 0.2, "aux_heads.size": 1.0, "xview_scale": 10.0})
    lr = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    named = dict(m.named_parameters())
    assert lr[id(named["aux_heads.size.0.weight"])] == pytest.approx(1e-4)
    assert lr[id(named["aux_heads.side.0.weight"])] == pytest.approx(2e-5)
    assert lr[id(named["eomt.xview.0.layer_scale"])] == pytest.approx(1e-3)
    assert not named["eomt.xview.0.attention.q_proj.weight"].requires_grad  # xview itself is not listed
    build_multiview_optimizer(m, 1e-4, 0.05, {"aux_heads.size": 1.0})
    assert {n for n, p in m.named_parameters() if p.requires_grad} == {n for n in named if n.startswith("aux_heads.size.")}
    with pytest.raises(ValueError, match="prefix"):
        build_multiview_optimizer(m, 1e-4, 0.05, {"aux_heads.sise": 1.0})


def test_predict_subject_with_groups_links_identities_over_the_whole_subject():
    from eomt.engine.predict import predict_subject

    _, xv = _pair("blocks")
    photos = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (90, 120, 3), dtype=np.uint8)) for i in range(5)]
    out = predict_subject(xv, photos, device="cpu", imgsz=IMGSZ, conf_thres=0.0, max_det=3, groups=[[0, 3], [1, 2, 4]],
                          identities=True, sim_thres=0.0, embed="reid")
    assert out["config"]["groups"] == [[0, 3], [1, 2, 4]] and out["config"]["embed"] == "reid"
    assert len(out["images"]) == 5 and all(len(r["identity_ids"]) == r["num_detections"] for r in out["images"])
    members = [(m["image_index"], m["det_index"]) for rec in out["identities"] for m in rec["members"]]
    assert len(members) == len(set(members)) == sum(r["num_detections"] for r in out["images"])
    assert any(rec["num_photos"] > 1 for rec in out["identities"])  # at threshold 0 identities span the groups
    with pytest.raises(ValueError, match="groups"):
        predict_subject(xv, photos, device="cpu", imgsz=IMGSZ, groups=[[0, 1], [1, 2]])


def test_post_training_with_the_new_options_records_its_story_and_validates_standalone(tmp_path):
    import csv
    import hashlib

    from eomt import EoMT
    from eomt.data import CocoInstanceSeg
    from eomt.engine.train import train
    from eomt.serialization import format_summary, load_model, load_raw, save_checkpoint, summarize_checkpoint, wrap_checkpoint

    img_dir, jf = _mini_coco(tmp_path, subjects=[1, 1, 2, 2])
    data = json.loads(jf.read_text())
    for im in data["images"]:
        im["split"] = "a" if im["id"] <= 2 else "b"  # an image field to group the validation by
    jf.write_text(json.dumps(data))
    specs = CocoInstanceSeg(img_dir, jf, imgsz=112).aux_specs
    torch.manual_seed(0)
    start = build_model("s", nc=2, imgsz=112, aux_heads=specs, fpn_scales=(2.0, 1.0, 0.5))
    init = tmp_path / "init.pt"
    save_checkpoint(wrap_checkpoint(start.state_dict(), size="s", nc=2, imgsz=112, aux_heads=specs,
                                    aux_head_arch=start.aux_head_arch, fpn_scales=start.fpn_scales), init)
    res = train(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s",
                imgsz=112, epochs=2, batch=1, accum=1, workers=0, device="cpu", amp=False, ema=False, seed=0,
                project=str(tmp_path / "runs"), name="xv", init_weights=str(init), val_interval=1,
                best_key={"sets/segm/mAP_set": 1.0, "sets/match_gated/f1": 0.1}, val_group_by="split",
                multiview={"match_dim": 384, "match_residual": True, "match_loss": "dustbin", "match_neg_conf": 0.0,
                           "match_hard_w": 2.0, "match_hard_by": ["class", "side"], "linked_w": 2.0,
                           "consistency_w": 1.0, "link_group_by": "class", "xview": {"mode": "blocks", "keys": "queries"},
                           "trainable": {"xview": 1.0, "xview_scale": 10.0, "match_head": 1.0},
                           "sets": {"p_single": 0.0}, "val_interval": {"photos": 2}})
    post = load_model(res["best"], device="cpu")
    before, after = start.state_dict(), post.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # stage A: a photo alone is predicted as before
    assert post.match_residual and post.xview_cfg["keys"] == "queries" and post.link_group_by == ("class",)
    assert post.link_embed in ("match", "reid") and set(post.link_f1) == {"match", "reid"}  # (a toy model: no threshold)

    raw = load_raw(res["best"])
    header = next(csv.reader(open(tmp_path / "runs" / "xv" / "metrics.csv")))
    rows = list(csv.DictReader(open(tmp_path / "runs" / "xv" / "metrics.csv")))
    assert raw["epoch"] in (0, 1) and raw["best_key"] == {"sets/segm/mAP_set": 1.0, "sets/match_gated/f1": 0.1}
    assert set(raw["metrics"]) <= set(header) and raw["metrics"]["epoch"] == raw["epoch"]
    assert raw["run_config"]["provenance"]["init_sha256"] == hashlib.sha256(init.read_bytes()).hexdigest()
    for col in ("train/match/pos_cos", "train/consistency_loss", "train/data_wait", "val/sets/segm/mAP_linked_set",
                "val/sets/match_gated/f1", "val/sets/match/f1_stored", "val/split=a/segm/mAP", "val/sets/split=b/match/f1"):
        assert col in header, col
    assert rows[0]["val/segm/mAP"] == "nan" and rows[1]["val/segm/mAP"] != "nan"  # the photos every 2 epochs
    assert rows[0]["val/sets/match/f1_stored"] == "nan"  # nothing stored before the first validation
    text = format_summary(summarize_checkpoint(res["best"]))
    assert "links    :" in text and "started  :" in text and "metrics  :" in text

    # the trainer's validation, standalone from the checkpoint: the same keys as metrics.csv
    metrics = EoMT(res["best"], device="cpu").val(multiview=True, workers=0)
    assert set(metrics) == {k[4:] for k in header if k.startswith("val/")}
    # the checkpoint the run started from, validated as the run validated its own (its args.yaml)
    base = EoMT(init, device="cpu").val(multiview=True, run=tmp_path / "runs" / "xv", workers=0, link_group_by=("class",))
    assert set(base) == set(metrics)


def test_the_linked_embedding_and_its_threshold_default_to_the_calibrated_ones():
    from eomt.engine.predict import link_defaults, predict_set

    _, xv = _pair("blocks")
    assert link_defaults(xv) == ("match", 0.7, None)  # nothing calibrated: the head, 0.7
    xv.link_embed, xv.reid_thres, xv.match_thres, xv.link_group_by = "reid", 0.55, 0.8, ("class",)
    assert link_defaults(xv) == ("reid", 0.55, ("class",)) and link_defaults(xv, "match")[1] == 0.8
    photos = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (60, 80, 3), dtype=np.uint8)) for i in range(2)]
    res = predict_set(xv, photos, device="cpu", imgsz=IMGSZ, conf_thres=0.0, max_det=3)
    assert res[0]["embed"].shape[-1] == 384  # the per-query embedding, not the 16-d head
    base, _ = _pair("blocks")
    with pytest.raises(ValueError, match="matching head"):
        link_defaults(base, "match")


def test_the_dustbin_only_teaches_no_partner_where_it_is_known():
    from eomt.aux_cls import SINKHORN_ITERS, _log_sinkhorn, match_loss_params

    torch.manual_seed(0)
    emb = torch.nn.functional.normalize(torch.randn(2, 6, 8), dim=-1)
    params = match_loss_params("dustbin")

    def only_column_term(row, col):  # the loss when only the column's dustbin term counts
        z = _log_sinkhorn((emb[row] @ emb[col].T)[None, None].reshape(1, 1) / 0.1, params["dustbin"], SINKHORN_ITERS)
        return -z[-1, 0]

    # photo 1 shows instance 5 too, but its query was gated out of the matching: no target for photo 0's query
    indices = [(torch.tensor([0]), torch.tensor([0])), (torch.tensor([2]), torch.tensor([1]))]
    gated = match_loss({"match_embed": emb}, indices, [torch.tensor([5]), torch.tensor([5, 7])], torch.tensor([0, 0]),
                       kind="dustbin", params=params)
    assert torch.allclose(gated, only_column_term((0, 0), (1, 2)))
    absent = match_loss({"match_embed": emb}, indices, [torch.tensor([5]), torch.tensor([9, 7])], torch.tensor([0, 0]),
                        kind="dustbin", params=params)
    assert not torch.allclose(absent, gated)  # photo 1 does not show 5: now its "no partner" target counts
    # a companion whose instances' ids are unknown: only its false detection (an extra negative) takes the dustbin
    negatives = torch.zeros(2, 6, dtype=torch.bool)
    negatives[1, 4] = True
    unknown = match_loss({"match_embed": emb}, [(torch.tensor([0]), torch.tensor([0])), (torch.tensor([2]), torch.tensor([0]))],
                         [torch.tensor([5]), torch.tensor([-100_000])], torch.tensor([0, 0]), kind="dustbin",
                         params=params, negatives=negatives)
    assert torch.allclose(unknown, only_column_term((0, 0), (1, 4)))
    incomplete = match_loss({"match_embed": emb}, indices, [torch.tensor([5]), torch.tensor([9, 7])],
                            torch.tensor([0, 0]), kind="dustbin", params=params, incomplete=torch.tensor([False, True]))
    assert torch.allclose(incomplete, only_column_term((0, 0), (1, 2)))  # photo 1 partly annotated: may show 5


@pytest.mark.parametrize("kind", ["sigmoid", "dustbin"])
def test_the_calibrated_matching_losses_compute_in_fp32_under_autocast(kind):
    from eomt.aux_cls import match_loss_params

    torch.manual_seed(0)
    aligned, _, indices, ids, sets = _match_case()
    params = match_loss_params(kind)
    full = match_loss({"match_embed": aligned}, indices, ids, sets, kind=kind, params=params)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = match_loss({"match_embed": aligned}, indices, ids, sets, kind=kind, params=params)
    assert mixed.dtype == torch.float32 and torch.allclose(mixed, full, rtol=1e-6)


def test_unpaired_counts_the_instances_of_sets_of_several_photos_only():
    aligned, _, indices, ids, _ = _match_case()
    stats: dict = {}
    emb = torch.cat([aligned, aligned[:1]])  # a third photo, alone in its set, showing instance 5
    match_loss({"match_embed": emb}, indices + [(torch.tensor([0]), torch.tensor([0]))], ids + [torch.tensor([5])],
               torch.tensor([0, 0, 1]), stats=stats)
    assert stats["unpaired"] == 0.0


def test_an_uncalibrated_model_links_a_subject_within_each_class():
    from eomt.engine.predict import predict_subject

    base, _ = _pair("blocks")
    photos = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (60, 80, 3), dtype=np.uint8)) for i in range(3)]
    out = predict_subject(base, photos, device="cpu", imgsz=IMGSZ, conf_thres=0.0, max_det=3, identities=True)
    assert out["config"]["group_by"] == ["class"] and out["config"]["embed"] == "reid"
