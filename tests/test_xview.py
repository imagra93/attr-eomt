"""Several photos of one subject predicted together (cross-view modules, matching head, synthetic sets)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from eomt import build_model
from eomt.aux_cls import match_loss
from eomt.data import CocoViewSets, collate_sets
from eomt.data.sets import DISTRACTOR_IDS, FLIPPED_IDS

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


def _mini_coco(tmp_path, n_images=4):
    """Tiny photos with a left blob and a centre line each; ``side`` is flip-aware, ``area`` a crop-scaled value."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    imgs, anns = [], []
    for i in range(1, n_images + 1):
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(img_dir / f"im{i}.png")
        imgs.append({"id": i, "file_name": f"im{i}.png", "width": 160, "height": 120})
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
    pixel_values, masks, classes, attrs, set_ids, inst_ids = collate_sets(batch)
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
