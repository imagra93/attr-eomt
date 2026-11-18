"""Several photos of one subject: the sets (eomt.data.subjects), the cross-view sublayers, the matching head and its
loss, the post-training, its validation, and prediction / linking of a subject's photos (CPU, tiny models)."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from eomt import EoMT, build_model
from eomt.aux_cls import match_loss, match_loss_params
from eomt.data import CocoInstanceSeg, CocoSubjectSets, collate_sets, subjects_from_file_names
from eomt.data.subjects import UNKNOWN_IDS
from eomt.loss import EoMTLoss
from eomt.preprocess import preprocess_numpy
from eomt.serialization import load_model, migrate_checkpoint, save_checkpoint, wrap_checkpoint

BLOB, LINE = (10, 10, 70, 60), (60, 80, 150, 92)
AUG = {"min_mask_mass": 0.0, "rotate_prob": 0.0, "shear_prob": 0.0, "perspective_prob": 0.0, "erasing_prob": 0.0}


def _poly(box):
    x0, y0, x1, y1 = box
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def _subject_coco(tmp_path, n_s2: int = 2):
    """Two subjects: s1 of 3 photos, s2 of ``n_s2``, and a photo with no subject. In each subject the blob is one
    instance in every photo (in "s1" instance_id 7, a number in photo 1 and a string in the others; in "s2" the text id
    "blob-b"); photo 2 is annotated with positives only, photo 3 is input only; photos 3 and 4 have an ignore region (a
    crowd annotation) away from the blob and the line; the line has no instance id."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    images = [{"id": 1, "subject_id": "s1"}, {"id": 2, "subject_id": "s1", "supervision": "positives"},
              {"id": 3, "subject_id": "s1", "supervision": "none"}]
    images += [{"id": 4 + i, "subject_id": "s2"} for i in range(n_s2)] + [{"id": 4 + n_s2}]
    anns = []
    for im in images:
        im.update(file_name=f"im{im['id']}.png", width=160, height=120)
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(img_dir / im["file_name"])
        for cat, side, box in ((1, 0, BLOB), (2, 1, LINE)):
            x0, y0, x1, y1 = box
            anns.append({"id": len(anns) + 1, "image_id": im["id"], "category_id": cat, "iscrowd": 0,
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "segmentation": _poly(box),
                         "attributes": {"side": side, "area": float((x1 - x0) * (y1 - y0))},
                         **({"instance_id": {1: 7, 2: "7", 3: "7"}.get(im["id"], "blob-b")}
                            if cat == 1 and "subject_id" in im else {})})
    for image_id in (3, 4):
        anns.append({"id": len(anns) + 1, "image_id": image_id, "category_id": 1, "iscrowd": 1,
                     "bbox": [100, 10, 50, 40], "area": 2000, "segmentation": _poly((100, 10, 150, 50))})
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "blob"}, {"id": 2, "name": "line"}],
            "attributes": [{"name": "side", "flip_aware": {"left": "right"},
                            "categories": [{"id": 0, "name": "left"}, {"id": 1, "name": "center"},
                                           {"id": 2, "name": "right"}]},
                           {"name": "area", "type": "regression", "log": True, "crop": "scale"}]}
    jf = tmp_path / "subjects.json"
    jf.write_text(json.dumps(coco))
    return img_dir, jf


# ------------------------------------------------------------------------------------------------------------ sets


def test_sets_draw_photos_of_one_subject_with_their_links_and_partial_labels(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    ds = CocoSubjectSets(img_dir, jf, imgsz=112, aug={**AUG, "min_scale": 1.0, "max_scale": 1.0}, seed=0,
                         sets={"max_views": 3, "p_shared": 1.0})
    assert len(ds) == 4 and ds.num_subjects == 2  # anchors: every photo of a subject that trains
    item = ds[0]  # anchor im1, subject s1: im2 shares the blob, im3 is input only (and shows the blob too)
    files = [ds.coco.loadImgs(i)[0]["file_name"] for i in item["photos"]]
    assert files[0] == "im1.png" and set(files) == {"im1.png", "im2.png", "im3.png"}
    s2 = ds[3]
    blobs_b = [v[4][v[2] == 0].tolist() for v in s2["views"]]  # a text id links the blob of both photos too
    assert len(blobs_b) == 2 and blobs_b[0] == blobs_b[1] and len(blobs_b[0]) == 1 and blobs_b[0][0] >= 0
    by_file = dict(zip(files, zip(item["views"], item["extras"])))
    (v1, e1), (v2, e2), (v3, e3) = by_file["im1.png"], by_file["im2.png"], by_file["im3.png"]
    assert (e1["neg"], e2["neg"], e3["neg"]) == (1.0, 0.0, 0.0)
    blob1 = v1[4][v1[2] == 0]
    assert len(blob1) == 1 and blob1[0] >= 0 and v2[4][v2[2] == 0].tolist() == blob1.tolist()  # 7 and "7": one
    assert (v1[4][v1[2] == 1] <= UNKNOWN_IDS).all()  # the line: no instance id, left out of the links
    assert len(v3[2]) == 0 and e3["ignore"].sum() > 0 and e1["ignore"].sum() == 0
    batch = collate_sets([item, ds[2]])
    assert len(batch) == 7 and batch[6]["neg_weight"].tolist()[:3] == [e["neg"] for e in item["extras"]]
    assert [m is not None for m in batch[6]["ignore"][:3]] == [f == "im3.png" for f in files]  # empty maps: None


def test_sets_flip_every_photo_of_a_set_together(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    ds = CocoSubjectSets(img_dir, jf, imgsz=112, aug={**AUG, "flip_prob": 0.5, "min_scale": 1.0, "max_scale": 1.0},
                         sets={"max_views": 2, "p_shared": 1.0})
    torch.manual_seed(0)
    for _ in range(12):
        sides = {int(v[3]["side"][v[2] == 0][0]) for v in ds[3]["views"]}  # s2: the blob on the left in both
        assert len(sides) == 1  # left in both, or right in both: never mixed within a set


def test_unaugmented_sets_give_the_inputs_of_inference_and_the_same_after_a_round_trip(tmp_path):
    """Validation sets fit every photo to the input exactly where the inference letterbox puts it (top left), so the
    same sets give the same tensors however they are drawn: from the seed, or read back from a file."""
    img_dir, jf = _subject_coco(tmp_path)
    ds = CocoSubjectSets(img_dir, jf, imgsz=112, aug={"min_mask_mass": 0.0}, seed=0, augment=False)
    path = tmp_path / "sets.json"
    ds.write_sets(path)
    back = CocoSubjectSets(img_dir, jf, imgsz=112, aug={"min_mask_mass": 0.0}, augment=False,
                           fixed=CocoSubjectSets.read_sets(path))
    for i in range(len(ds)):
        a, b = ds[i], back[i]
        assert a["photos"] == b["photos"]
        for va, vb in zip(a["views"], b["views"]):
            assert torch.equal(va[0], vb[0]) and torch.equal(va[1], vb[1])
    view = ds[0]["views"][0]
    want, _ = preprocess_numpy(np.asarray(Image.open(img_dir / "im1.png").convert("RGB")), 112, letterbox=True)
    assert view[0][:, 84:].abs().max() < 1e-6                       # the padding below the photo, as inference has it
    assert (view[0] - torch.from_numpy(want)).abs().mean() < 0.05   # the same pixels (resampling aside)
    blob = view[1][view[2] == 0][0]
    assert abs(float(blob.sum()) - 60 * 50 * (112 / 160) ** 2) < 60 and int(view[3]["side"][view[2] == 0][0]) == 0


def test_fixed_sets_keep_what_a_new_version_of_the_data_shares(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    ds = CocoSubjectSets(img_dir, jf, imgsz=112, aug=AUG, seed=0, augment=False, sets={"max_views": 3})
    path = tmp_path / "sets.json"
    assert ds.write_sets(path) == len(ds)
    data = json.loads(jf.read_text())  # a new version of the data: photo 2 gone
    data["images"] = [im for im in data["images"] if im["id"] != 2]
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] != 2]
    jf2 = tmp_path / "v2.json"
    jf2.write_text(json.dumps(data))
    v2 = CocoSubjectSets(img_dir, jf2, imgsz=112, aug=AUG, augment=False, fixed=CocoSubjectSets.read_sets(path))
    assert v2.fixed_report["dropped"] == 1 and v2.fixed_report["photos_left_out"] >= 1
    assert all(2 not in v2[i]["photos"] for i in range(len(v2)))


def test_hard_anchors_change_the_anchor_only(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    ds = CocoSubjectSets(img_dir, jf, imgsz=112, aug=AUG, sets={"max_views": 2, "p_hard": 1.0, "hard": {"small": 1.0}})
    torch.manual_seed(0)
    item = ds[3]
    assert item["case"] == "real_small"
    sizes = [float(v[1][v[2] == 0][0].sum()) for v in item["views"]]
    assert sizes[0] < 0.5 * sizes[1]  # the anchor shrunk (0.35-0.6x the photo), its companion as usual


def test_subjects_from_file_names(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps({"images": [{"id": 1, "file_name": "x/c1_a.jpg"}, {"id": 2, "file_name": "c1_b.jpg"},
                                          {"id": 3, "file_name": "c2_a.jpg"}, {"id": 4, "file_name": "solo.jpg"}],
                               "annotations": [], "categories": []}))
    assert subjects_from_file_names(src, tmp_path / "b.json") == {"subjects": 1, "images": 2}
    out = json.loads((tmp_path / "b.json").read_text())
    assert [im.get("subject_id") for im in out["images"]] == ["c1", "c1", None, None]


class _Cfg:
    num_labels, no_object_weight, class_weight, dice_weight, mask_weight = 2, 0.1, 2.0, 5.0, 5.0


def test_partial_labels_only_remove_no_object_terms():
    torch.manual_seed(0)
    crit = EoMTLoss(_Cfg())
    masks, logits = torch.randn(2, 5, 16, 16), torch.randn(2, 5, 3)
    gt = [torch.zeros(1, 32, 32), torch.zeros(1, 32, 32)]
    gt[0][0, :16, :16] = 1
    gt[1][0, 16:, 16:] = 1
    cls = [torch.tensor([0]), torch.tensor([1])]
    loss = lambda **kw: (torch.manual_seed(1), crit(masks, logits, gt, cls, **kw)[0])[1]  # noqa: E731 (BCE samples)
    base = loss()
    assert torch.allclose(base["loss_cross_entropy"], loss(neg_weight=torch.ones(2), ignore_masks=[None, None])["loss_cross_entropy"])
    part = loss(neg_weight=torch.tensor([1.0, 0.0]))  # photo 1 with positives only: its unmatched queries drop out
    assert part["loss_cross_entropy"] < base["loss_cross_entropy"]
    assert torch.allclose(part["loss_mask"], base["loss_mask"]) and torch.allclose(part["loss_dice"], base["loss_dice"])
    full = torch.ones(32, 32)  # an ignore map over the whole of photo 0 spares every unmatched query of it
    assert torch.allclose(loss(ignore_masks=[full, None])["loss_cross_entropy"],
                          loss(neg_weight=torch.tensor([0.0, 1.0]))["loss_cross_entropy"])


# ------------------------------------------------------------------------------------------- model and losses


def test_cross_view_starts_as_the_identity_and_leaves_a_photo_alone_unchanged():
    torch.manual_seed(0)
    m = build_model("s", nc=2, imgsz=112, xview={"max_views": 3}, match_dim=384, match_residual=True).eval()
    m.init_xview()
    x = torch.randn(3, 3, 112, 112)
    with torch.no_grad():
        alone = m(x)
        apart = m(x, set_ids=torch.tensor([0, 1, 2]))                 # every photo alone in its set
        together = m(x, set_ids=torch.zeros(3, dtype=torch.long))
    assert torch.equal(alone["class_queries_logits"], apart["class_queries_logits"])
    assert torch.allclose(alone["class_queries_logits"], together["class_queries_logits"], atol=1e-3)  # LayerScale 1e-5
    # the residual matching head starts as re-identification: the normalized query embedding
    ref = torch.nn.functional.normalize(alone["query_embed"].float(), dim=-1)
    assert torch.allclose(alone["match_embed"], ref, atol=1e-6)
    with pytest.raises(ValueError, match="hidden size"):
        build_model("s", nc=2, imgsz=112, match_dim=128, match_residual=True)


def test_the_matching_loss_rewards_linked_instances_and_reports_what_it_sees():
    torch.manual_seed(0)
    params = match_loss_params()
    a, b = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    emb = torch.zeros(2, 3, 8)
    emb[0, 0], emb[0, 1], emb[1, 0], emb[1, 1] = a, b, a, b        # instance 1 and 2 in two photos of one set
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1])), (torch.tensor([0, 1]), torch.tensor([0, 1]))]
    ids = [torch.tensor([1, 2]), torch.tensor([1, 2])]
    stats: dict = {}
    good = match_loss({"match_embed": emb}, indices, ids, torch.tensor([0, 0]), params, stats=stats)
    swapped = match_loss({"match_embed": emb}, indices, [torch.tensor([1, 2]), torch.tensor([2, 1])], torch.tensor([0, 0]),
                         params)
    assert good < swapped
    assert torch.is_tensor(stats["pos_cos"]) and float(stats["pos_cos"]) == pytest.approx(1.0, abs=1e-5)
    assert float(stats["unpaired"]) == 0.0
    # two photos of two different sets never pair: nothing to learn
    assert float(match_loss({"match_embed": emb}, indices, ids, torch.tensor([0, 1]), params)) == 0.0


def test_old_cross_view_checkpoints_are_upgraded_or_refused():
    m = build_model("s", nc=2, imgsz=112, xview={}, match_dim=384, match_residual=True)
    sd = m.state_dict()
    base = wrap_checkpoint(sd, size="s", nc=2, imgsz=112, loss_weights=m.loss_weights, xview=m.xview_cfg, match_dim=384,
                           match_residual=True)
    # view types (2.x): their embedding and vocabulary are dropped
    old = {**base, "schema_version": "1.0", "xview": {**m.xview_cfg, "keys": "both", "view_types": ["front", "rear"]},
           "model": {**sd, "eomt.xview.0.view_embed.weight": torch.zeros(3, 384), "eomt.attn_mask_probs": torch.zeros(4)}}
    up = migrate_checkpoint(old)
    assert "view_types" not in up["xview"] and not any("view_embed" in k or "attn_mask" in k for k in up["model"])
    # a validation that chose re-identification keeps linking that way: no matching head, its threshold
    reid = migrate_checkpoint({**base, "link_embed": "reid", "reid_thres": 0.8, "match_thres": 0.9})
    assert reid["match_dim"] == 0 and reid["match_thres"] == 0.8 and not any(k.startswith("match_head.") for k in reid["model"])
    with pytest.raises(ValueError, match="refiner"):
        migrate_checkpoint({**base, "xview": {"mode": "refiner"}})
    with pytest.raises(ValueError, match="compressed"):
        migrate_checkpoint({**base, "compression": {"recipe": "int8"}})


# ------------------------------------------------------------------------------------- post-training end to end


def _init_checkpoint(tmp_path, img_dir, jf):
    """A single-photo checkpoint to post-train: ``(model, path)``."""
    specs = CocoInstanceSeg(img_dir, jf, imgsz=112).aux_specs
    torch.manual_seed(0)
    start = build_model("s", nc=2, imgsz=112, aux_heads=specs)
    init = tmp_path / "init.pt"
    save_checkpoint(wrap_checkpoint(start.state_dict(), size="s", nc=2, imgsz=112, aux_heads=specs,
                                    aux_head_arch=start.aux_head_arch, fpn_scales=start.fpn_scales,
                                    loss_weights=start.loss_weights, inference={"conf_thres": 0.0}), init)
    return start, init


def _post_train(tmp_path, img_dir, jf, init, name, **multiview):
    from eomt.engine.train import train

    return train(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s",
                 imgsz=112, epochs=1, batch=2, nominal_batch=2, workers=0, device="cpu", amp=False, seed=0,
                 project=str(tmp_path / "runs"), name=name, init_weights=str(init), optim={"ema": False},
                 multiview={"sets": {"max_views": 3, "sizes": {2: 1.0, 3: 1.0}}, **multiview})


def test_post_training_freezes_the_checkpoint_and_its_validation_runs_standalone(tmp_path):
    img_dir, jf = _subject_coco(tmp_path, n_s2=5)
    start, init = _init_checkpoint(tmp_path, img_dir, jf)
    res = _post_train(tmp_path, img_dir, jf, init, "xv", gate=["class"])
    post = load_model(res["best"], device="cpu")
    before, after = start.state_dict(), post.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)        # "new": only the added modules trained
    assert post.eomt.xview is not None and post.match_residual and post.match_dim == 384
    assert post.link_group_by == ("class",) and res["best_key"] == "sets/segm/mAP_set"
    row = next(csv.DictReader(open(tmp_path / "runs" / "xv" / "metrics.csv")))
    thres = float(row["val/sets/match_gated/thres"])  # the links are calibrated as they are made: within the gate
    assert post.match_thres == pytest.approx(thres) if math.isfinite(thres) else post.match_thres is None
    for key in ("val/segm/mAP", "val/sets/segm/mAP_set", "val/sets/match/f1", "val/sets/match_gated/f1",
                "val/sets/reid/f1", "val/sets/match/cluster_f1", "train/match_loss"):
        assert key in row
    # the standalone validation rebuilds the run's and gives its numbers; read back, the fixed sets give them too
    model = EoMT(res["best"], device="cpu")
    for kw in ({}, {"sets_file": str(tmp_path / "runs" / "xv" / "val_sets.json")}):
        got = model.val(multiview=True, workers=0, **kw)
        for key in ("segm/mAP", "sets/segm/mAP_alone", "sets/segm/mAP_set", "sets/match/f1", "sets/reid/f1"):
            assert got[key] == pytest.approx(float(row[f"val/{key}"]), abs=1e-6), (kw, key)

    # a subject of more photos than a set holds: predicted max_views at a time, linked across the groups
    from eomt.engine.predict import predict_subject

    photos = [Image.open(img_dir / f"im{i}.png").convert("RGB") for i in range(4, 9)]
    out = predict_subject(post, photos, device="cpu", imgsz=112, conf_thres=0.0, max_det=5, identities=True)
    assert out["config"]["groups"] == [[0, 1, 2], [3, 4]] and len(out["images"]) == 5
    assert all(len(r["identity_ids"]) == r["num_detections"] for r in out["images"])
    folder = tmp_path / "subject"
    folder.mkdir()
    for i, p in enumerate(photos):
        p.save(folder / f"{i}.png")
    linked = model.match(folder, plot=False, save=None, max_det=5)     # 5 photos: more than max_views is fine
    from eomt.engine.predict import link_defaults

    assert linked["num_images"] == 5 and linked["config"]["sim_thres"] == pytest.approx(link_defaults(post)[0])


def test_matching_without_cross_view(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    _, init = _init_checkpoint(tmp_path, img_dir, jf)
    res = _post_train(tmp_path, img_dir, jf, init, "m", xview=False)
    post = load_model(res["best"], device="cpu")
    assert post.eomt.xview is None and post.match_head is not None and res["best_key"] == "sets/match/f1"
    with pytest.raises(ValueError, match="nothing to train"):
        _post_train(tmp_path, img_dir, jf, init, "none", xview=False, matching=False)


def test_trainable_presets_and_prefixes():
    from eomt.engine.train import MultiviewConfig, build_multiview_optimizer

    m = build_model("s", nc=2, imgsz=112, xview={}, match_dim=384, match_residual=True)
    opt = build_multiview_optimizer(m, 1e-4, 0.05, MultiviewConfig(trainable="query_blocks").groups())
    lrs = {round(g["lr"] / 1e-4, 3) for g in opt.param_groups}
    assert lrs == {1.0, 0.4, 0.2}
    trained = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("eomt.layers.11.") for n in trained) and not any(n.startswith("eomt.layers.0.") for n in trained)
    build_multiview_optimizer(m, 1e-4, 0.05, {"xview": 1.0, "eomt.class_predictor": 0.0})
    assert not any(p.requires_grad for p in m.eomt.class_predictor.parameters())
    with pytest.raises(ValueError, match="neither a group"):
        build_multiview_optimizer(m, 1e-4, 0.05, {"xviews": 1.0})
    assert MultiviewConfig(xview=False).groups() == {"match_head": 1.0}
