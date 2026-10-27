"""Real multi-photo sets (eomt.data.subjects): the data format, the sets, the partial-label loss and a post-training run
on them."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from eomt.data import CocoSubjectSets, CocoViewSets, MixedSets, collate_sets, subjects_from_file_names
from eomt.data.sets import UNKNOWN_IDS
from eomt.loss import EoMTLoss

BLOB, LINE = (10, 10, 70, 60), (60, 80, 150, 92)


def _poly(box):
    x0, y0, x1, y1 = box
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def _subject_coco(tmp_path):
    """Two subjects of 3 and 2 photos and a photo with no subject. In each subject the blob is one instance in every
    photo (in "s1" instance_id 7, a number in photo 1 and a string in the others; in "s2" the text id "blob-b"); photo 2
    is annotated with positives only, photo 3 is input only; photos 3 and 4 have an ignore region (a crowd annotation)
    away from the blob and the line; the line has no instance id."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    images = [
        {"id": 1, "subject_id": "s1"},
        {"id": 2, "subject_id": "s1", "supervision": "positives"},
        {"id": 3, "subject_id": "s1", "supervision": "none"},
        {"id": 4, "subject_id": "s2"},
        {"id": 5, "subject_id": "s2"},
        {"id": 6},
    ]
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


AUG = {"min_mask_mass": 0.0, "rotate_prob": 0.0, "shear_prob": 0.0, "perspective_prob": 0.0, "erasing_prob": 0.0}


def test_real_sets_draw_photos_of_one_subject_with_their_links_and_partial_labels(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    ds = CocoSubjectSets(img_dir, jf, imgsz=112, aug={**AUG, "min_scale": 1.0, "max_scale": 1.0}, seed=0,
                         sets={"max_views": 3, "p_shared": 1.0})
    # anchors: every photo of a subject that trains (not the input-only one); photo 6 has no subject
    assert len(ds) == 4 and ds.num_subjects == 2
    item = ds[0]  # anchor im1, subject s1: im2 shares the blob, im3 is input only (and shows the blob too)
    files = [ds.coco.loadImgs(i)[0]["file_name"] for i in item["photos"]]
    assert files[0] == "im1.png" and set(files) == {"im1.png", "im2.png", "im3.png"}
    s2 = ds[3]
    assert {ds.coco.loadImgs(i)[0]["subject_id"] for i in s2["photos"]} == {"s2"}
    blobs_b = [v[4][v[2] == 0].tolist() for v in s2["views"]]  # a text id links the blob of both photos too
    assert len(blobs_b) == 2 and blobs_b[0] == blobs_b[1] and len(blobs_b[0]) == 1 and blobs_b[0][0] >= 0
    views, extras = item["views"], item["extras"]
    assert len(views) == 3 and item["case"] == "real"
    by_file = dict(zip(files, zip(views, extras)))
    (v1, e1), (v2, e2), (v3, e3) = by_file["im1.png"], by_file["im2.png"], by_file["im3.png"]
    assert e1["neg"] == 1.0 and e2["neg"] == 0.0 and e3["neg"] == 0.0
    blob1 = v1[4][v1[2] == 0]
    assert len(blob1) == 1 and blob1[0] >= 0  # linked: 7 and "7" are one instance in both photos
    assert v2[4][v2[2] == 0].tolist() == blob1.tolist()
    assert (v1[4][v1[2] == 1] <= UNKNOWN_IDS).all()  # the line: no instance id, unknown to the matching head
    assert len(v3[2]) == 0  # input only: no targets
    assert e3["ignore"].sum() > 0 and e1["ignore"].sum() == 0  # the crowd annotation is the ignore map
    batch = collate_sets([item, ds[2]])
    assert len(batch) == 7
    ex = batch[6]
    assert ex["neg_weight"].tolist()[:3] == [e["neg"] for e in extras]
    assert len(ex["ignore"]) == len(batch[0])
    assert [m is not None for m in ex["ignore"][:3]] == [f == "im3.png" for f in files]  # empty maps come as None


def test_real_sets_flip_every_photo_of_a_set_together(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    ds = CocoSubjectSets(img_dir, jf, imgsz=112, aug={**AUG, "flip_prob": 0.5, "min_scale": 1.0, "max_scale": 1.0},
                         sets={"max_views": 2, "p_shared": 1.0})
    torch.manual_seed(0)
    for _ in range(12):
        views = ds[3]["views"]  # s2: two fully annotated photos, the blob on the left
        sides = {int(v[3]["side"][v[2] == 0][0]) for v in views}
        assert len(sides) == 1  # left in both, or right in both: never mixed within a set


def test_real_val_sets_are_not_augmented(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    ds = CocoSubjectSets(img_dir, jf, imgsz=112, aug={"min_mask_mass": 0.0}, seed=0, augment=False)
    fit = 112 / 160  # the photo fitted to the input
    blobs = [v for i in range(len(ds)) for v in ds[i]["views"] if (v[2] == 0).any()]
    assert blobs
    for v in blobs:
        blob = v[1][v[2] == 0][0]
        assert abs(float(blob.sum()) - 60 * 50 * fit**2) < 60  # no scale jitter, no crop
        assert int(v[3]["side"][v[2] == 0][0]) == 0  # no flip: the blob stays on the left


def test_links_of_detections_nobody_annotated_are_left_out():
    from eomt.engine.validate import _set_links

    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[0, :4, :4] = True
    det = {"scores": torch.tensor([0.9]), "masks": mask, "query_idx": torch.tensor([0])}
    embeds = [torch.ones(1, 4), torch.ones(1, 4)]
    gt_masks = [mask, torch.zeros(0, 8, 8, dtype=torch.bool)]  # photo 0: instance 5; photo 1: nothing annotated
    gt_ids = [torch.tensor([5]), torch.zeros(0, dtype=torch.long)]
    pairs = _set_links([det, det], embeds, gt_masks, gt_ids, 0.3, 0.5)
    assert len(pairs) == 1 and not pairs[0][1]  # a fully annotated photo 1: the link is wrong
    unsure = [torch.tensor([False]), torch.tensor([True])]  # photo 1 not fully annotated: unknown, left out
    assert _set_links([det, det], embeds, gt_masks, gt_ids, 0.3, 0.5, unsure) == []
    gt_masks[1], gt_ids[1] = mask, torch.tensor([5])  # an annotated instance still links
    assert _set_links([det, det], embeds, gt_masks, gt_ids, 0.3, 0.5, unsure)[0][1]


def test_mixed_sets_keep_the_real_share(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    real = CocoSubjectSets(img_dir, jf, imgsz=112, aug=AUG)
    synthetic = CocoViewSets(img_dir, jf, imgsz=112, aug=AUG)
    mix = MixedSets(real, synthetic, 0.25)
    assert len(mix) == 4 + 12
    assert mix[0]["case"] == "real" and mix[10]["case"] != "real"
    assert len(MixedSets(real, synthetic, 1.0)) == 4
    with pytest.raises(ValueError):
        MixedSets(real, synthetic, 0.0)


def test_subjects_from_file_names(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps({"images": [{"id": 1, "file_name": "x/c1_a.jpg"}, {"id": 2, "file_name": "c1_b.jpg"},
                                          {"id": 3, "file_name": "c2_a.jpg"}, {"id": 4, "file_name": "solo.jpg"}],
                               "annotations": [], "categories": []}))
    counts = subjects_from_file_names(src, tmp_path / "b.json")
    out = json.loads((tmp_path / "b.json").read_text())
    assert counts == {"subjects": 1, "images": 2}
    assert [im.get("subject_id") for im in out["images"]] == ["c1", "c1", None, None]


class _Cfg:
    num_labels, no_object_weight, class_weight, dice_weight, mask_weight = 2, 0.1, 2.0, 5.0, 5.0


def test_partial_labels_only_remove_no_object_terms():
    torch.manual_seed(0)
    crit = EoMTLoss(_Cfg(), {})
    masks = torch.randn(2, 5, 16, 16)
    logits = torch.randn(2, 5, 3)
    gt = [torch.zeros(1, 32, 32), torch.zeros(1, 32, 32)]
    gt[0][0, :16, :16] = 1
    gt[1][0, 16:, 16:] = 1
    cls = [torch.tensor([0]), torch.tensor([1])]
    torch.manual_seed(1)  # the mask BCE samples its points at random
    base = crit(masks, logits, gt, cls)
    same = crit(masks, logits, gt, cls, neg_weight=torch.ones(2), ignore_masks=[None, None])
    assert torch.allclose(base["loss_cross_entropy"], same["loss_cross_entropy"])
    # photo 1 with positives only: its unmatched queries drop out, the matched ones and the mask terms stay
    torch.manual_seed(1)
    base = crit(masks, logits, gt, cls)
    torch.manual_seed(1)
    part = crit(masks, logits, gt, cls, neg_weight=torch.tensor([1.0, 0.0]))
    assert part["loss_cross_entropy"] < base["loss_cross_entropy"]
    assert torch.allclose(part["loss_mask"], base["loss_mask"]) and torch.allclose(part["loss_dice"], base["loss_dice"])
    # an ignore map over the whole of photo 0 spares every unmatched query of it, as neg_weight 0 does
    full = torch.ones(32, 32)
    ign = crit(masks, logits, gt, cls, ignore_masks=[full, None])
    neg0 = crit(masks, logits, gt, cls, neg_weight=torch.tensor([0.0, 1.0]))
    assert torch.allclose(ign["loss_cross_entropy"], neg0["loss_cross_entropy"])
    w = crit.negative_weights(masks, [full, None], None)
    assert w[0].sum() == 0 and (w[1] == 1).all()


def _init_checkpoint(tmp_path, img_dir, jf):
    """A single-photo checkpoint to post-train: ``(model, path)``."""
    from eomt import build_model
    from eomt.data import CocoInstanceSeg
    from eomt.serialization import save_checkpoint, wrap_checkpoint

    specs = CocoInstanceSeg(img_dir, jf, imgsz=112).aux_specs
    torch.manual_seed(0)
    start = build_model("s", nc=2, imgsz=112, aux_heads=specs, fpn_scales=(2.0, 1.0, 0.5))
    init = tmp_path / "init.pt"
    save_checkpoint(wrap_checkpoint(start.state_dict(), size="s", nc=2, imgsz=112, aux_heads=specs,
                                    aux_head_arch=start.aux_head_arch, fpn_scales=start.fpn_scales), init)
    return start, init


def _post_train(tmp_path, img_dir, jf, init, real: dict):
    from eomt.engine.train import train

    return train(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s",
                 imgsz=112, epochs=1, batch=2, accum=1, workers=0, device="cpu", amp=False, ema=False, seed=0,
                 project=str(tmp_path / "runs"), name="real", init_weights=str(init),
                 multiview={"match_dim": 16, "harm_w": 1.0, "sets": {"p_single": 0.0}, "real": real})


def test_stage_a_on_real_and_synthetic_sets_scores_real_sets_and_leaves_the_checkpoint_untouched(tmp_path):
    from eomt.serialization import load_model

    img_dir, jf = _subject_coco(tmp_path)
    start, init = _init_checkpoint(tmp_path, img_dir, jf)
    res = _post_train(tmp_path, img_dir, jf, init, {"p": 0.5, "sets": {"max_views": 3, "sizes": {2: 1.0, 3: 1.0}}})
    post = load_model(res["best"], device="cpu")
    before, after = start.state_dict(), post.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    header = (tmp_path / "runs" / "real" / "metrics.csv").read_text().splitlines()[0].split(",")
    assert "val/real/segm/mAP_set" in header and "val/real/match/f1" in header and "val/sets/segm/mAP_set" in header


def test_a_real_val_json_asked_for_must_load(tmp_path):
    img_dir, jf = _subject_coco(tmp_path)
    _, init = _init_checkpoint(tmp_path, img_dir, jf)
    other = json.loads(jf.read_text())
    other["categories"][0]["name"] = "disc"
    bad = tmp_path / "other.json"
    bad.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="categories differ"):  # not a silent fallback to the synthetic sets
        _post_train(tmp_path, img_dir, jf, init, {"val_json": str(bad)})
