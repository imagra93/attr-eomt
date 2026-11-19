"""Several photos of one subject: the sets (eomt.data.subjects), the cross-view sublayers, the set matcher and its
loss, LoRA, the post-training, its validation, and prediction / linking of a subject's photos (CPU, tiny models)."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from eomt import EoMT, build_model
from eomt.lora import add_lora, merged_state_dict
from eomt.matcher import SetMatcher, dual_log_softmax, mask_boxes, matcher_loss
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
    m = build_model("s", nc=2, imgsz=112, xview={"max_views": 3}, matcher={}).eval()
    m.init_xview()
    x = torch.randn(3, 3, 112, 112)
    with torch.no_grad():
        alone = m(x)
        apart = m(x, set_ids=torch.tensor([0, 1, 2]))                 # every photo alone in its set
        together = m(x, set_ids=torch.zeros(3, dtype=torch.long))
    assert torch.equal(alone["class_queries_logits"], apart["class_queries_logits"])
    assert torch.equal(alone["masks_queries_logits"], together["masks_queries_logits"])  # closed gates: the identity
    for blk in m.eomt.xview:                                          # open gates: the set changes each photo
        torch.nn.init.normal_(blk.gate.bias)
        torch.nn.init.normal_(blk.patch_gate.bias)
    with torch.no_grad():
        live = m(x, set_ids=torch.zeros(3, dtype=torch.long))
        trunk = m.eomt.encode_trunk(x)
        again = m(x, set_ids=torch.zeros(3, dtype=torch.long), trunk=trunk)
    assert not torch.allclose(live["masks_queries_logits"], alone["masks_queries_logits"])
    assert torch.equal(live["masks_queries_logits"], again["masks_queries_logits"])  # the trunk once, the queries twice
    with torch.no_grad():
        assert torch.equal(m(x)["class_queries_logits"], alone["class_queries_logits"])  # a photo alone: unchanged


def test_the_set_matcher_gives_dual_softmax_probabilities_and_its_loss_rewards_the_links():
    torch.manual_seed(0)
    scores = torch.randn(5, 5)
    photo = torch.tensor([0, 0, 1, 1, 1])
    valid = photo[:, None] != photo[None]
    log_p, log_bin = dual_log_softmax(scores, photo, valid, torch.tensor(0.5))
    p = log_p.exp()
    assert torch.equal(p[~valid], torch.zeros_like(p[~valid])) and float(p.max()) <= 1.0
    # each direction is a softmax over the other photo's instances and the bin
    assert torch.allclose(log_bin[0, 1].exp() + torch.softmax(torch.cat((scores[0, 2:], torch.tensor([0.5]))), 0)[:3].sum(),
                          torch.tensor(1.0), atol=1e-5)
    # the loss: low when the scores agree with the ids, high when swapped
    ids = torch.tensor([1, 2, 1, 2, 3])
    known = torch.ones(5, dtype=torch.bool)
    group = torch.zeros(5, dtype=torch.long)
    good = torch.full((5, 5), -5.0)
    good[0, 2] = good[2, 0] = good[1, 3] = good[3, 1] = 5.0
    swapped = torch.full((5, 5), -5.0)
    swapped[0, 3] = swapped[3, 0] = swapped[1, 2] = swapped[2, 1] = 5.0
    stats: dict = {}
    lg = matcher_loss(*dual_log_softmax(good, photo, valid, torch.tensor(0.0)), ids, photo, group, known, stats)
    ls = matcher_loss(*dual_log_softmax(swapped, photo, valid, torch.tensor(0.0)), ids, photo, group, known)
    assert lg < ls and float(stats["pos_p"]) > 0.9 and float(stats["paired"]) == pytest.approx(4 / 5)
    # the network: probabilities for every two instances of two photos of one group, 0 elsewhere
    matcher = SetMatcher(64, dim=32, layers=2, heads=4)
    feats, boxes = torch.randn(6, 64), torch.rand(6, 4)
    lp, _ = matcher(feats, boxes, torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 0, 0, 0, 1, 1]))
    assert torch.isinf(lp[0, 4]) and torch.isinf(lp[0, 1]) and torch.isfinite(lp[0, 2])
    masks = torch.full((2, 10, 20), -1.0)
    masks[0, 2:5, 4:10] = 1.0
    assert torch.allclose(mask_boxes(masks), torch.tensor([[0.2, 0.2, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]))


def test_lora_adapts_the_blocks_below_the_queries_and_merges_exactly():
    torch.manual_seed(0)
    m = build_model("s", nc=2, imgsz=112).eval()
    cfg = add_lora(m, rank=4, blocks=2)
    adapted = {n.split(".")[2] for n, _ in m.named_parameters() if ".lora_" in n}
    first = m.config.num_hidden_layers - m.config.num_blocks
    assert cfg == {"rank": 4, "alpha": 4.0, "blocks": 2} and adapted == {str(first - 2), str(first - 1)}
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        start = m(x)["class_queries_logits"]
    for n, p in m.named_parameters():
        if "lora_b" in n:
            torch.nn.init.normal_(p, std=0.02)
    with torch.no_grad():
        adapted_out = m(x)["class_queries_logits"]
    plain = build_model("s", nc=2, imgsz=112).eval()
    plain.load_state_dict(merged_state_dict(m))                       # strict: plain keys only
    with torch.no_grad():
        merged = plain(x)["class_queries_logits"]
    assert not torch.allclose(start, adapted_out) and torch.allclose(adapted_out, merged, atol=1e-5)


def test_multi_view_checkpoints_before_3_1_are_refused():
    m = build_model("s", nc=2, imgsz=112, xview={})
    sd = m.state_dict()
    base = wrap_checkpoint(sd, size="s", nc=2, imgsz=112, loss_weights=m.loss_weights, xview=m.xview_cfg)
    old = {**base, "model": {**sd, "eomt.xview.0.null_logit": torch.zeros(6)}}
    with pytest.raises(ValueError, match="before attr-eomt 3.1"):
        migrate_checkpoint(old)
    with pytest.raises(ValueError, match="before attr-eomt 3.1"):
        migrate_checkpoint({**base, "match_dim": 384, "model": {**sd, "match_head.0.weight": torch.zeros(1)}})
    with pytest.raises(ValueError, match="compressed"):
        migrate_checkpoint({**base, "compression": {"recipe": "int8"}})
    assert migrate_checkpoint(base)["xview"] == m.xview_cfg


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
    assert post.eomt.xview is not None and post.matcher is not None
    assert post.link_group_by == ("class",) and res["best_key"] == "sets/segm/mAP_set"
    row = next(csv.DictReader(open(tmp_path / "runs" / "xv" / "metrics.csv")))
    thres = float(row["val/sets/match_gated/thres"])  # the links are calibrated as they are made: within the gate
    assert post.match_thres == pytest.approx(thres) if math.isfinite(thres) else post.match_thres is None
    for key in ("val/segm/mAP", "val/sets/segm/mAP_set", "val/sets/match/f1", "val/sets/match_gated/f1",
                "val/sets/reid/f1", "val/sets/match/cluster_f1", "val/sets/match_oracle/f1", "val/sets/reid_oracle/f1",
                "train/match_loss", "train/match/pos_p"):
        assert key in row
    # the standalone validation rebuilds the run's and gives its numbers; read back, the fixed sets give them too
    model = EoMT(res["best"], device="cpu")
    for kw in ({}, {"sets_file": str(tmp_path / "runs" / "xv" / "val_sets.json")}):
        got = model.val(multiview=True, workers=0, **kw)
        for key in ("segm/mAP", "sets/segm/mAP_alone", "sets/segm/mAP_set", "sets/match/f1", "sets/reid/f1",
                    "sets/match_oracle/f1"):
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


def test_post_training_with_lora_and_attention_supervision(tmp_path):
    img_dir, jf = _subject_coco(tmp_path, n_s2=5)
    start, init = _init_checkpoint(tmp_path, img_dir, jf)
    res = _post_train(tmp_path, img_dir, jf, init, "lora", lora={"rank": 4, "blocks": 2}, attn_w=0.5)
    row = next(csv.DictReader(open(tmp_path / "runs" / "lora" / "metrics.csv")))
    assert math.isfinite(float(row["train/attn_loss"]))
    best = load_model(res["best"], device="cpu")                     # exported: the factors merged, plain keys
    last = load_model(res["last"], device="cpu")                     # the trainer state: merged on loading
    raw = torch.load(res["best"], map_location="cpu", weights_only=True)["model"]
    assert not any(".parametrizations." in k for k in raw)
    assert any(".parametrizations." in k for k in torch.load(res["last"], map_location="cpu", weights_only=True)["model"])
    first = start.config.num_hidden_layers - start.config.num_blocks
    moved = [k for k, v in start.state_dict().items() if not torch.equal(v, best.state_dict()[k])]
    assert moved and all(k.startswith((f"eomt.layers.{first - 2}.attention.", f"eomt.layers.{first - 1}.attention."))
                         for k in moved)                             # only the adapted projections changed
    assert set(last.state_dict()) == set(best.state_dict())


def test_matching_without_cross_view(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    _, init = _init_checkpoint(tmp_path, img_dir, jf)
    res = _post_train(tmp_path, img_dir, jf, init, "m", xview=False)
    post = load_model(res["best"], device="cpu")
    assert post.eomt.xview is None and post.matcher is not None and res["best_key"] == "sets/match/f1"
    with pytest.raises(ValueError, match="nothing to train"):
        _post_train(tmp_path, img_dir, jf, init, "none", xview=False, matching=False)


def test_trainable_presets_and_prefixes():
    from eomt.engine.train import MultiviewConfig, build_multiview_optimizer

    m = build_model("s", nc=2, imgsz=112, xview={}, matcher={})
    opt = build_multiview_optimizer(m, 1e-4, 0.05, MultiviewConfig(trainable="query_blocks").groups())
    lrs = {round(g["lr"] / 1e-4, 3) for g in opt.param_groups}
    assert lrs == {1.0, 0.4, 0.2}
    trained = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("eomt.layers.11.") for n in trained) and not any(n.startswith("eomt.layers.0.") for n in trained)
    build_multiview_optimizer(m, 1e-4, 0.05, {"xview": 1.0, "eomt.class_predictor": 0.0})
    assert not any(p.requires_grad for p in m.eomt.class_predictor.parameters())
    with pytest.raises(ValueError, match="neither a group"):
        build_multiview_optimizer(m, 1e-4, 0.05, {"xviews": 1.0})
    assert MultiviewConfig(xview=False).groups() == {"matcher": 1.0}             # no LoRA unless asked
    assert "lora" in MultiviewConfig(lora={"rank": 4}).groups()


def test_sets_from_groups_with_a_share_of_random_ones(tmp_path):
    """sets.groups: the photos of a subject with the same group value are one set (as inference groups them); p_random
    of the sets are random combinations on top; a photo with no group is a set alone."""
    img_dir, jf = _subject_coco(tmp_path, n_s2=4)
    data = json.loads(jf.read_text())
    group = {1: "a", 2: "a", 3: "b", 4: "x", 5: "x", 6: "x", 7: "x"}  # im8: no group; im3 is input only: no set
    for im in data["images"]:
        if im["id"] in group:
            im["group"] = group[im["id"]]
    jf.write_text(json.dumps(data))
    ds = CocoSubjectSets(img_dir, jf, imgsz=112, aug=AUG, seed=0, augment=False, sets={"max_views": 3, "groups": "group"})
    files = [[ds.coco.loadImgs(i)[0]["file_name"] for i in ds[k]["photos"]] for k in range(len(ds))]
    assert sorted(map(sorted, files)) == [["im1.png", "im2.png"], ["im4.png", "im5.png", "im6.png"], ["im7.png"]]
    assert ds.first_photos() == [ds[k]["photos"][0] for k in range(len(ds))]
    mixed = CocoSubjectSets(img_dir, jf, imgsz=112, aug=AUG, seed=0, sets={"max_views": 3, "groups": "group",
                                                                         "p_random": 0.25})
    assert len(mixed) == 3 + 1                                   # 25 % of the sets random: one more on three groups
    assert len(mixed[3]["photos"]) >= 1 and all(p in {1, 2, 3, 4, 5, 6, 7} for p in mixed[3]["photos"])
    with pytest.raises(ValueError, match="p_random"):
        CocoSubjectSets(img_dir, jf, imgsz=112, aug=AUG, sets={"p_random": 0.3})


# ------------------------------------------------------------------------------------- groups and prepared inputs

def test_groups_fill_the_last_one_and_may_repeat_a_photo_as_context():
    from types import SimpleNamespace

    from eomt.engine.predict import group_owners, subject_groups

    m = SimpleNamespace(xview_cfg={"max_views": 4})
    assert subject_groups(m, 5) == [[0, 1, 2, 3], [4]]                       # the default is unchanged
    assert subject_groups(m, 5, fill=True) == [[0, 1, 2, 3], [4, 3, 2, 1]]   # 4 and 4, not 4 + 1
    assert subject_groups(m, 6, fill=True) == [[0, 1, 2, 3], [4, 5, 3, 2]]
    assert subject_groups(m, 8, fill=True) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert subject_groups(m, 3, fill=True) == [[0, 1, 2]]                    # fewer photos than a set holds
    given = [[0, 1, 2, 3], [4, 0, 1, 2]]                                      # photos 0-2 are context in the second
    assert subject_groups(m, 6, given) == given + [[5]]
    assert group_owners(given) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}
    for bad in ([[0, 0]], [[0, 1, 2, 3, 4]], [[6]], [[]]):
        with pytest.raises(ValueError):
            subject_groups(m, 6, bad)


def test_similar_groups_put_together_the_photos_that_show_the_same_things():
    from eomt.engine.predict import similar_groups

    # photos 0 and 2 show mostly class A, 1 and 3 mostly class B, 4 both; 5 nothing
    f = np.array([[1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0, 0]])
    assert similar_groups(f[:4], 2) == [[0, 2], [1, 3]]
    assert similar_groups(f[:4], 2, order=[1]) == [[1, 3], [0, 2]]           # the order chooses who picks first
    assert similar_groups(f[:5], 2, fill=False) == [[0, 2], [1, 3], [4]]
    assert similar_groups(f[:5], 2) == [[0, 2], [1, 3], [4, 3]]               # filled: 3 is context for 4
    assert similar_groups(f, 3) == [[0, 2, 4], [1, 3, 5]]                    # a row of zeros joins by order
    assert similar_groups(np.zeros((0, 2)), 4) == []
    with pytest.raises(ValueError):
        similar_groups(f, 2, order=[7])


def test_photos_alike_in_one_way_are_not_split_by_an_error_in_another():
    """Three close-ups of one door, the last read on the wrong side by the parts model: by parts alone it goes with
    the real other-side door; with the photos' appearance as well it stays with its look-alikes."""
    from eomt.engine.predict import photo_similarity, similar_groups

    # columns: left door, right door, rear bumper. Photos 0, 2, 5: the left door (5 read as the right one);
    # 1, 3: the bumper; 4: the right door.
    parts = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    looks = np.array([[1, 0], [0, 1], [1, 0.05], [0.05, 1], [0.7, 0.7], [1, 0.1]])
    assert similar_groups(parts, 3, fill=False) == [[0, 2, 1], [3, 4, 5]]
    assert similar_groups([parts, looks], 3, fill=False) == [[0, 2, 5], [1, 3, 4]]
    assert similar_groups([parts, looks], 3, weights=[1, 0], fill=False) == similar_groups(parts, 3, fill=False)
    shared = looks + 10.0                                   # a component every photo shares: centering takes it out
    assert similar_groups([parts, shared], 3, center=[False, True], fill=False) == [[0, 2, 5], [1, 3, 4]]
    sim = photo_similarity([parts, looks], weights=[3, 1])
    assert sim.shape == (6, 6) and np.allclose(np.diag(sim), 1.0)
    assert sim[0, 2] == pytest.approx((3 * 1 + looks[0] @ looks[2] / np.linalg.norm(looks[2])) / 4)
    for bad in ({"weights": [1]}, {"weights": [-1, 1]}, {"center": [True]}):
        with pytest.raises(ValueError):
            photo_similarity([parts, looks], **bad)
    with pytest.raises(ValueError):
        photo_similarity([parts, looks[:5]])


def test_a_subject_predicted_from_prepared_inputs_in_filled_groups():
    """A photo repeated as context keeps the result of its first group; prepared inputs (tensors) and an output size
    give the detections of the photos themselves, at that size."""
    from eomt.engine.predict import predict_batch, predict_set, predict_subject

    torch.manual_seed(0)
    m = build_model("s", nc=2, imgsz=112, xview={"max_views": 3}, matcher={}).eval()
    m.init_xview()
    for blk in m.eomt.xview:                                          # open gates: a photo's result depends on its set
        torch.nn.init.normal_(blk.gate.bias)
        torch.nn.init.normal_(blk.patch_gate.bias)
    rng = np.random.default_rng(0)
    photos = [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(4)]   # RGB arrays, not PIL
    kw = dict(device="cpu", imgsz=112, conf_thres=0.0, max_det=5, amp=False)
    out = predict_subject(m, photos, fill=True, identities=True, **kw)
    assert out["config"]["groups"] == [[0, 1, 2], [3, 2, 1]]
    first = predict_set(m, photos[:3], **kw)
    for got, want in zip(out["images"][:3], first):
        assert torch.equal(got["scores"], want["scores"])
    assert torch.equal(out["images"][3]["scores"], predict_set(m, [photos[3], photos[2], photos[1]], **kw)[0]["scores"])
    assert all(len(r["identity_ids"]) == r["num_detections"] for r in out["images"])

    prepared = [preprocess_numpy(p, 112, letterbox=True) for p in photos[:3]]
    tensors = [(torch.from_numpy(chw), meta) for chw, meta in prepared]
    again = predict_set(m, photos[:3], prepared=tensors, sizes=[(80, 60)] * 3, **kw)
    for got, want in zip(again, first):
        assert torch.allclose(got["scores"], want["scores"]) and got["masks"].shape[-2:] == (60, 80)
        assert torch.allclose(got["embed"], want["embed"])
    alone = predict_batch(m, photos[:2], prepared=tensors[:2], sizes=[(80, 60)] * 2, photo_embed=True, **kw)
    assert [r["masks"].shape[-2:] for r in alone] == [(60, 80)] * 2
    with torch.no_grad():                                   # the photo's mean final patch token, alone
        want = m(torch.stack([t for t, _ in tensors[:2]]))["photo_embed"]
    assert torch.allclose(torch.stack([r["photo_embed"] for r in alone]), want, atol=1e-6)
    assert again[0]["photo_embed"].shape == want[0].shape   # predict_set gives it too (in its set)
    with pytest.raises(ValueError, match="sizes"):
        predict_set(m, photos[:3], sizes=[(80, 60)], **kw)


def test_photo_embed_leaves_the_letterbox_padding_out():
    """The photo's mean patch token is taken over the photo: the patches of a letterbox's padding (all exactly 0) are
    left out, a patch the photo partly covers is kept, and a photo without padding is the plain mean."""
    torch.manual_seed(0)
    m = build_model("s", nc=2, imgsz=112).eval()                     # an 8 x 8 grid of 14 px patches
    rng = np.random.default_rng(0)
    wide = preprocess_numpy(rng.integers(0, 255, (60, 112, 3), dtype=np.uint8), 112, letterbox=True)
    full = preprocess_numpy(rng.integers(0, 255, (112, 112, 3), dtype=np.uint8), 112, letterbox=True)
    assert tuple(wide[1]["content_hw"]) == (60, 112)                 # rows 60.. are padding: patch row 4 is partly photo
    x = torch.from_numpy(np.stack([wide[0], full[0]]))
    with torch.no_grad():
        got = m(x)["photo_embed"]
        tokens = m.eomt(pixel_values=x).last_hidden_state[:, m.config.num_queries + m.eomt.embeddings.num_prefix_tokens:]
    assert torch.allclose(got[0], tokens[0, : 5 * 8].mean(0), atol=1e-5)   # patch rows 0-4 of 8
    assert not torch.allclose(got[0], tokens[0].mean(0), atol=1e-3)
    assert torch.allclose(got[1], tokens[1].mean(0), atol=1e-5)
    with torch.no_grad():
        blank = m(torch.zeros(1, 3, 112, 112))["photo_embed"]        # nothing but padding: every patch, not NaN
    assert torch.isfinite(blank).all()
