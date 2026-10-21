"""CPU tests of test-time augmentation and tiled inference (:mod:`eomt.tta`)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from eomt import build_model
from eomt.config import AuxHeadSpec
from eomt.tta import TTAConfig, View, _clusters, make_views, predict_views

IMGSZ = 140


class _Blobs(torch.nn.Module):
    """Stand-in instance model: query 0 segments the bright pixels of its input as class 0 (with ``left_only``, only
    while most of them are in the input's left half); the other queries predict "no object"."""

    family, patch_size, num_upscale_blocks, aux_specs = "instance", 14, 2, []

    def __init__(self, left_only: bool = False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))  # gives the model a device
        self.left_only = left_only
        self.config = SimpleNamespace(hidden_size=4)

    def forward(self, x):
        b, n = x.shape[0], x.shape[-1] // 14 * 4
        bright = F.adaptive_avg_pool2d((x.mean(1, keepdim=True) > 0).float(), n)[:, 0]  # (B, n, n)
        masks = torch.full((b, 3, n, n), -20.0)
        masks[:, 0] = (bright - 0.5) * 40
        cls = torch.zeros(b, 3, 2)
        cls[..., 1] = 20.0  # "no object"
        cls[:, 0] = torch.tensor([20.0, 0.0])
        if self.left_only:
            right = bright[..., n // 2 :].sum((1, 2)) > bright[..., : n // 2].sum((1, 2))
            cls[right, 0] = torch.tensor([0.0, 20.0])
        return {"masks_queries_logits": masks, "class_queries_logits": cls, "query_embed": torch.zeros(b, 3, 4)}


def _image(h, w, x0, y0, x1, y1):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[y0:y1, x0:x1] = 255
    return img


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal masks up to a few pixels at the threshold (bilinear resampling of a mirrored or shifted grid rounds
    differently, ~1e-6)."""
    return a.shape == b.shape and int((a ^ b).sum()) <= max(1, a.numel() // 10_000)


def _iou(mask: torch.Tensor, x0, y0, x1, y1) -> float:
    gt = torch.zeros_like(mask)
    gt[y0:y1, x0:x1] = True
    return float((mask & gt).sum() / (mask | gt).sum())


def test_views_cover_the_image_with_the_requested_overlap():
    views = make_views(960, 1280, 644, TTAConfig(tiles=True, overlap=0.25, full=False))
    xs = sorted({v.rect[0] for v in views})
    ys = sorted({v.rect[1] for v in views})
    assert xs == [0, 318, 636] and ys == [0, 316]
    assert all(v.size == 644 and v.rect[2] - v.rect[0] == 644 and v.rect[3] - v.rect[1] == 644 for v in views)
    assert max(v.rect[2] for v in views) == 1280 and max(v.rect[3] for v in views) == 960
    assert min(b - a for a, b in zip(xs, xs[1:])) <= 0.75 * 644  # neighbours overlap by >= a quarter of a tile

    tta = make_views(960, 1280, 644, TTAConfig(flip=True, scales=(1.0, 1.5)))
    assert sorted((v.size, v.flip) for v in tta) == [(644, False), (644, True), (966, False), (966, True)]
    assert all(v.rect == (0, 0, 1280, 960) for v in tta)
    # A tile as large as the image is the whole-image view: not run twice.
    assert make_views(100, 120, 140, TTAConfig(tiles=True)) == [View((0, 0, 120, 100), 140)]


def test_augment_and_tiles_resolve_to_their_defaults():
    assert TTAConfig.resolve() is None and TTAConfig.resolve(False, None) is None
    aug = TTAConfig.resolve(augment=True)
    assert (aug.flip, aug.scales, aug.tiles, aug.score) == (False, (1.0, 1.5), False, "mean")
    til = TTAConfig.resolve(tiles=True)
    assert (til.flip, til.tiles, til.tile_size, til.overlap, til.full, til.score) == (False, True, None, 0.25, True, "mean")
    assert TTAConfig.resolve(tiles=1024).tile_size == 1024
    both = TTAConfig.resolve({"flip": True, "scales": [1, 1.25], "iou": 0.6}, {"overlap": 0.2, "score": "max"})
    assert (both.flip, both.scales, both.overlap, both.iou, both.score) == (True, (1.0, 1.25), 0.2, 0.6, "max")
    with pytest.raises(ValueError, match="unknown tiles options"):
        TTAConfig.resolve(tiles={"tile": 644})
    with pytest.raises(ValueError, match="differently"):
        TTAConfig.resolve({"iou": 0.5}, {"iou": 0.7})


def test_clusters_follow_views_and_chains():
    # an instance never joins a cluster that already has one from its view
    iou = np.array([[0, 0.9, 0.8], [0.9, 0, 0], [0.8, 0, 0]], dtype=np.float32)
    assert _clusters(iou, np.array([0.9, 0.8, 0.7]), np.array([0, 1, 1]), 0.5) == [[0, 1], [2]]
    # a chain of tile pieces 0-1-2-3 is one object whatever order the scores visit it in
    chain = np.zeros((4, 4), dtype=np.float32)
    for a in range(3):
        chain[a, a + 1] = chain[a + 1, a] = 0.9
    clusters = _clusters(chain, np.array([0.8, 0.7, 0.6, 0.9]), np.array([0, 1, 2, 3]), 0.5)
    assert len(clusters) == 1 and sorted(clusters[0]) == [0, 1, 2, 3]
    # two clusters holding the same view are not merged through a third instance
    iou = np.array([[0, 0, 0.6], [0, 0, 0.6], [0.6, 0.6, 0]], dtype=np.float32)
    assert sorted(map(sorted, _clusters(iou, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]), 0.5))) == [[0, 2], [1]]


def test_a_single_view_reproduces_the_plain_prediction():
    """Views are decoded exactly as a single prediction is (logits interpolated to the image, then thresholded):
    averaging probabilities, or decoding through a coarser grid, reshapes thin masks."""
    from eomt.engine.predict import predict_image

    torch.manual_seed(0)
    model = build_model("s", nc=3, imgsz=IMGSZ).eval()
    image = Image.fromarray(np.random.default_rng(3).integers(0, 255, (150, 230, 3), dtype=np.uint8))
    plain = predict_image(model, image, device="cpu", imgsz=IMGSZ, conf_thres=0.0, max_det=20)
    viewed = predict_image(model, image, device="cpu", imgsz=IMGSZ, conf_thres=0.0, max_det=20,
                           augment={"scales": [1.0]})  # one view, through the merge
    assert torch.allclose(viewed["scores"], plain["scores"]) and torch.equal(viewed["classes"], plain["classes"])
    assert torch.equal(viewed["masks"], plain["masks"])


def test_flipped_view_is_the_mirror_of_the_plain_view(monkeypatch):
    import eomt.tta as tta

    torch.manual_seed(0)
    model = build_model("s", nc=3, imgsz=IMGSZ).eval()
    img = np.random.default_rng(0).integers(0, 255, (112, 196, 3), dtype=np.uint8)
    plain = predict_views(model, np.ascontiguousarray(img[:, ::-1]), TTAConfig(), imgsz=IMGSZ, max_det=20)
    monkeypatch.setattr(tta, "make_views", lambda h, w, imgsz, cfg, patch=14: [View((0, 0, w, h), imgsz, True)])
    flipped = predict_views(model, img, TTAConfig(), imgsz=IMGSZ, max_det=20)
    assert torch.equal(flipped["scores"], plain["scores"]) and torch.equal(flipped["classes"], plain["classes"])
    assert _same(flipped["masks"], plain["masks"].flip(-1))


def test_a_tile_view_is_the_crop_put_back_in_place(monkeypatch):
    import eomt.tta as tta

    torch.manual_seed(0)
    model = build_model("s", nc=3, imgsz=IMGSZ).eval()
    img = np.random.default_rng(1).integers(0, 255, (280, 280, 3), dtype=np.uint8)
    crop = predict_views(model, np.ascontiguousarray(img[:140, 140:]), TTAConfig(), imgsz=IMGSZ, max_det=20)
    monkeypatch.setattr(tta, "make_views", lambda h, w, imgsz, cfg, patch=14: [View((140, 0, 280, 140), imgsz)])
    tile = predict_views(model, img, TTAConfig(), imgsz=IMGSZ, max_det=20)
    assert torch.equal(tile["scores"], crop["scores"]) and torch.equal(tile["classes"], crop["classes"])
    # identical away from the tile's inner border (where the image edge is clamped but the tile edge fades out)
    assert _same(tile["masks"][:, :136, 144:], crop["masks"][:, :136, 4:])
    assert not tile["masks"][:, 144:].any() and not tile["masks"][:, :, :136].any()


def test_tiles_reassemble_an_object_cut_by_their_borders():
    img = _image(140, 420, 70, 42, 315, 98)  # spans all four 140 px tiles
    res = predict_views(_Blobs(), img, TTAConfig(tiles=True, tile_size=140, full=False), imgsz=IMGSZ, conf_thres=0.3)
    assert res["num_detections"] == 1
    assert _iou(res["masks"][0], 70, 42, 315, 98) > 0.95
    assert res["boxes"][0].tolist() == pytest.approx([70, 42, 315, 98], abs=3)


def test_flip_tta_merges_the_two_views_and_scores_their_agreement():
    img = _image(140, 140, 14, 35, 56, 91)  # in the left half; mirrored it is in the right half
    both = predict_views(_Blobs(), img, TTAConfig(flip=True), imgsz=IMGSZ, conf_thres=0.3)
    assert both["num_detections"] == 1 and _iou(both["masks"][0], 14, 35, 56, 91) > 0.95
    assert float(both["scores"][0]) > 0.9  # both views found it
    # A model that finds it in one view only: "mean" halves the score (the other view votes 0), "max" keeps it.
    mean = predict_views(_Blobs(left_only=True), img, TTAConfig(flip=True), imgsz=IMGSZ, conf_thres=0.3)
    best = predict_views(_Blobs(left_only=True), img, TTAConfig(flip=True, score="max"), imgsz=IMGSZ, conf_thres=0.3)
    assert float(mean["scores"][0]) == pytest.approx(float(both["scores"][0]) / 2, rel=1e-3)
    assert float(best["scores"][0]) == pytest.approx(float(both["scores"][0]), rel=1e-3)
    assert _iou(mean["masks"][0], 14, 35, 56, 91) > 0.95


def test_device_preprocessed_views_find_the_same_instances():
    img = _image(280, 420, 70, 42, 315, 98)
    cfg = TTAConfig(flip=True, scales=(1.0, 1.5), tiles=True, tile_size=140)
    cpu = predict_views(_Blobs(), img, cfg, imgsz=IMGSZ, conf_thres=0.3)
    dev = predict_views(_Blobs(), img, cfg, imgsz=IMGSZ, conf_thres=0.3, device_preprocess=True)
    assert dev["num_detections"] == cpu["num_detections"] == 1
    assert torch.equal(dev["scores"], cpu["scores"]) and torch.equal(dev["masks"], cpu["masks"])


def test_predict_image_with_tta_keeps_the_result_contract():
    from eomt.engine.predict import predict_image

    torch.manual_seed(0)
    model = build_model("s", nc=3, imgsz=IMGSZ, aux_heads=[AuxHeadSpec("tone", 2)]).eval()
    image = Image.fromarray(np.random.default_rng(2).integers(0, 255, (150, 230, 3), dtype=np.uint8))
    res = predict_image(model, image, device="cpu", imgsz=IMGSZ, conf_thres=0.0, max_det=30, embed=True,
                        augment={"scales": [1.0, 1.4]}, tiles={"size": 100, "overlap": 0.2})
    n = res["num_detections"]
    assert 0 < n <= 30 and "query_idx" not in res
    assert res["masks"].shape == (n, 150, 230) and res["masks"].dtype == torch.bool
    assert res["boxes"].shape == (n, 4) and res["classes"].shape == (n,)
    assert torch.all(res["scores"][:-1] >= res["scores"][1:])
    assert res["aux"]["tone"]["probs"].shape == (n, 2) and res["aux"]["tone"]["ids"].shape == (n,)
    assert torch.allclose(res["aux"]["tone"]["probs"].sum(1), torch.ones(n), atol=1e-4)
    assert res["embed"].shape == (n, model.config.hidden_size)


def test_evaluate_with_tta_and_tiles_scores_the_ground_truth(tmp_path):
    from eomt.data import CocoValImages
    from eomt.engine.validate import evaluate

    rects = [(70, 42, 315, 98), (28, 14, 112, 126)]
    images, anns = [], []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        Image.fromarray(_image(140, 420, x0, y0, x1, y1)).save(tmp_path / f"{i}.png")
        images.append({"id": i + 1, "file_name": f"{i}.png", "width": 420, "height": 140})
        anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "iscrowd": 0, "area": (x1 - x0) * (y1 - y0),
                     "bbox": [x0, y0, x1 - x0, y1 - y0], "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]})
    jf = tmp_path / "val.json"
    jf.write_text(json.dumps({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "blob"}]}))
    ds = CocoValImages(tmp_path, jf, imgsz=IMGSZ)
    for augment, tiles in ((False, False), (True, False), ({"flip": True}, False), (False, {"full": False}), (True, 140)):
        m = evaluate(_Blobs(), ds, device="cpu", batch_size=2, num_workers=0, verbose=False, augment=augment,
                     tiles=tiles)
        assert m["segm/mAP50"] > 0.99, (augment, tiles, m)


class _Sided(_Blobs):
    """_Blobs plus a "side" head that says "left" (0) when the object is in the left half of its input, else "right"."""

    def __init__(self, flip_aware):
        super().__init__()
        self.aux_specs = [AuxHeadSpec("side", 2, {0: "left", 1: "right"}, flip_aware=flip_aware)]

    def forward(self, x):
        out = super().forward(x)
        n = out["masks_queries_logits"].shape[-1]
        left = (out["masks_queries_logits"][:, 0, :, : n // 2] > 0).sum((1, 2)) > (out["masks_queries_logits"][:, 0, :, n // 2 :] > 0).sum((1, 2))
        logits = torch.zeros(x.shape[0], 3, 2)
        logits[left, :, 0], logits[~left, :, 1] = 8.0, 8.0
        return {**out, "aux_queries_logits": {"side": logits}}


def test_flip_tta_swaps_a_flip_aware_head_back_and_the_mapping_survives_the_checkpoint():
    from eomt.config import aux_specs_from_meta, aux_specs_to_meta

    img = _image(140, 140, 14, 35, 56, 91)  # on the left: the plain view says "left", the mirrored view "right"
    aware = predict_views(_Sided({0: 1, 1: 0}), img, TTAConfig(flip=True), imgsz=IMGSZ, conf_thres=0.3)
    naive = predict_views(_Sided(None), img, TTAConfig(flip=True), imgsz=IMGSZ, conf_thres=0.3)
    assert aware["aux"]["side"]["ids"].tolist() == [0] and float(aware["aux"]["side"]["probs"][0, 0]) > 0.99
    assert float(naive["aux"]["side"]["probs"][0, 0]) == pytest.approx(0.5, abs=0.01)  # the views cancel out
    spec = _Sided({0: 1, 1: 0}).aux_specs[0]
    assert aux_specs_from_meta(aux_specs_to_meta([spec]))[0].flip_aware == {0: 1, 1: 0}
    assert aux_specs_from_meta(aux_specs_to_meta([_Sided(None).aux_specs[0]]))[0].flip_aware is None  # old checkpoints



def test_evaluate_scores_the_attribute_heads_plain_and_under_tta(tmp_path):
    from eomt.data import CocoValImages
    from eomt.engine.validate import evaluate

    class Attr(_Sided):
        """_Sided plus a "size" regression head that always says 100."""

        def __init__(self, flip_aware):
            super().__init__(flip_aware)
            self.aux_specs.append(AuxHeadSpec("size", 1, regression=True))

        def forward(self, x):
            out = super().forward(x)
            return {**out, "aux_queries_values": {"size": torch.full(out["aux_queries_logits"]["side"].shape[:2], 100.0)}}

    rects = [(70, 42, 315, 98), (28, 14, 112, 126)]  # both mostly in the left half: the model says "left" for both
    images, anns = [], []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        Image.fromarray(_image(140, 420, x0, y0, x1, y1)).save(tmp_path / f"{i}.png")
        images.append({"id": i + 1, "file_name": f"{i}.png", "width": 420, "height": 140})
        anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "iscrowd": 0, "area": (x1 - x0) * (y1 - y0),
                     "bbox": [x0, y0, x1 - x0, y1 - y0], "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                     "attributes": {"side": i, "size": 100.0 + 50.0 * i}})  # GT: left / right; 100 / 150
    anns[1]["attributes"]["side"] = 1
    cats = [{"id": 0, "name": "left"}, {"id": 1, "name": "right"}]
    jf = tmp_path / "val.json"
    jf.write_text(json.dumps({
        "images": images, "annotations": anns, "categories": [{"id": 1, "name": "blob"}],
        "attributes": [{"name": "side", "categories": cats, "flip_aware": {"left": "right"}},
                       {"name": "size", "type": "regression"}],
    }))
    ds = CocoValImages(tmp_path, jf, imgsz=IMGSZ)
    for augment in (False, True, {"flip": True}):
        m = evaluate(Attr({0: 1, 1: 0}), ds, device="cpu", batch_size=2, num_workers=0, verbose=False,
                     augment=augment, attributes=True)
        assert m["aux_acc/side"] == pytest.approx(0.5) and m["aux_mae/size"] == pytest.approx(25.0), (augment, m)
        assert m["segm/mAP50"] > 0.99
    old = Attr(None)  # a checkpoint saved before the mapping was stored: mirrored views take the val JSON's
    with pytest.warns(UserWarning, match="stores no flip_aware"):
        evaluate(old, ds, device="cpu", batch_size=2, num_workers=0, verbose=False, augment={"flip": True}, attributes=True)
    assert old.aux_specs[0].flip_aware == {0: 1, 1: 0}
    assert "aux_acc/side" not in evaluate(Attr(None), ds, device="cpu", batch_size=2, num_workers=0, verbose=False)
    bad = Attr(None)
    bad.aux_specs[0].names = {0: "right", 1: "left"}  # the model's ids mean the opposite of the val JSON's
    with pytest.raises(ValueError, match="differ from the model's"):
        evaluate(bad, ds, device="cpu", batch_size=2, num_workers=0, verbose=False, attributes=True)
