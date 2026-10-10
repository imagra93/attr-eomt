"""Regression attributes (``"type": "regression"``): a head that regresses one real number per instance next to the
categorical ones. Dataset labels and statistics, the loss, training and inference in real units (CPU, no network)."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from eomt import build_model
from eomt.aux_cls import aux_loss
from eomt.config import AuxHeadSpec
from eomt.data import CocoInstanceSeg, TrainAugment
from eomt.data.transforms import _homography

from test_augment import _mini_coco, _rects  # noqa: E402  (tests/ is on sys.path)


def _with_cost(tmp_path):
    """The mini COCO plus a log-regressed 'cost' (10 on the blobs, 1000 on the lines), missing on the first
    annotation and outside its range on the third (both blobs)."""
    img_dir, jf = _mini_coco(tmp_path)
    coco = json.loads(jf.read_text())
    coco["attributes"].append({"name": "cost", "type": "regression", "log": True, "range": [0, 5000]})
    for a in coco["annotations"]:
        a["attributes"]["cost"] = 10.0 if a["attributes"]["kind"] == 0 else 1000.0
    del coco["annotations"][0]["attributes"]["cost"]
    coco["annotations"][2]["attributes"]["cost"] = 9999.0
    jf.write_text(json.dumps(coco))
    return img_dir, jf


def test_dataset_reads_regression_values_and_standardizes_on_the_split(tmp_path):
    img_dir, jf = _with_cost(tmp_path)
    kind, cost = CocoInstanceSeg(img_dir, jf, imgsz=112).aux_specs
    assert not kind.regression and cost.regression and cost.log and cost.num_classes == 1
    valid = torch.tensor([10.0, 10.0, 1000.0, 1000.0, 1000.0, 1000.0], dtype=torch.float64).log1p()
    assert cost.mean == pytest.approx(float(valid.mean())) and cost.std == pytest.approx(float(valid.std()))
    attrs = CocoInstanceSeg(img_dir, jf, imgsz=112)._load_raw(0)[3]   # image 1: the blob without a cost, a line
    assert attrs["kind"].dtype == torch.long and attrs["cost"].dtype == torch.float32
    assert attrs["cost"][0].isnan() and attrs["cost"][1] == 1000.0
    y = torch.tensor([0.0, 10.0, 1000.0])
    assert torch.allclose(cost.decode(cost.encode(y)), y, rtol=1e-5, atol=1e-3)
    bad = json.loads(jf.read_text())
    bad["attributes"][1]["type"] = "regresion"
    jf.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown type"):
        CocoInstanceSeg(img_dir, jf, imgsz=112)


def test_regression_loss_is_huber_on_the_standardized_value_and_skips_nan():
    spec = AuxHeadSpec("cost", 1, regression=True, mean=1.0, std=2.0)
    model = build_model("s", nc=2, imgsz=112, aux_heads=[spec])
    q = torch.randn(1, 3, model.config.hidden_size, requires_grad=True)
    out = {"query_embed": q}
    loss, per_head = aux_loss(model, out, None, None, {"cost": [torch.tensor([3.0, float("nan")])]},
                              indices=[(torch.tensor([0, 1]), torch.tensor([0, 1]))])
    want = F.huber_loss(model.aux_heads["cost"](q[0, :1])[:, 0], torch.tensor([1.0]))   # (3 - 1) / 2
    assert per_head["cost"].item() == pytest.approx(want.item(), rel=1e-5)
    loss.backward()
    assert q.grad[0, 2].abs().sum() == 0 and q.grad[0, 0].abs().sum() > 0   # unmatched query untouched


def test_train_with_a_regression_head_then_predict_real_values(tmp_path):
    from eomt.engine.predict import predict_image
    from eomt.engine.train import train
    from eomt.reid import validate_group_by
    from eomt.serialization import load_model

    img_dir, jf = _with_cost(tmp_path)
    res = train(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s",
                imgsz=112, epochs=1, batch=1, accum=1, workers=0, device="cpu", amp=False, pretrained=False, ema=False,
                seed=0, project=str(tmp_path / "runs"), name="r", warmup_steps=(1, 1), aux_iou_gate=0.0,
                aux_class_gate=False, aux_class_weights=True, aux_balance="uncertainty")
    row = next(csv.DictReader(open(tmp_path / "runs" / "r" / "metrics.csv")))
    assert "train/aux_acc/kind" in row and "val/aux_acc/kind" in row
    assert math.isfinite(float(row["train/aux_mae/cost"])) and math.isfinite(float(row["val/aux_mae/cost"]))
    assert float(row["train/aux_grad/cost"]) > 0 and float(row["train/aux_grad/kind"]) > 0
    assert set(torch.load(res["last"], weights_only=False)["aux_log_vars"]) == {"kind", "cost"}

    model = load_model(res["last"], device="cpu")
    cost = model.aux_specs[1]
    assert cost.regression and cost.log and cost.value_range == (0.0, 5000.0) and cost.std > 0
    pred = predict_image(model, Image.open(img_dir / "im1.png"), device=torch.device("cpu"), imgsz=112,
                         conf_thres=0.0, max_det=5)
    values = pred["aux_values"]["cost"]
    assert values.shape == (pred["num_detections"],) and ((values >= 0) & (values <= 5000)).all()
    assert set(pred["aux"]) == {"kind"}
    with pytest.raises(ValueError, match="regression"):
        validate_group_by(("class", "cost"), model.aux_specs)


def test_augmentation_measures_the_fraction_of_each_instance_kept_in_view():
    img, masks, boxes = _rects()  # green 140x80 at x 90-230, red 30x70 at x 40-70, in a 320x240 image
    tf = TrainAugment(112)
    m = masks.float()
    for tgt, is_boxes in ((m, False), (boxes, True)):  # a window over x >= 160: half the green one, none of the red
        vis = tf._crop(img, tgt, is_boxes, (240, 320), (240, 160), (0, 160), None)[2]
        assert torch.allclose(vis, torch.tensor([0.5, 0.0]))
    # a warp that halves the size and bends the perspective keeps both whole: the shrink must not count as a loss
    quad = np.array([[0, 0], [160, 0], [160, 120], [0, 120]], dtype=np.float64)
    fwd = _homography(quad, quad + [[4, 3], [-6, 2], [-3, -5], [5, -4]]) @ np.diag([0.5, 0.5, 1.0])
    for tgt, is_boxes in ((m, False), (boxes, True)):
        vis = tf._crop(img, tgt, is_boxes, (240, 320), (120, 160), (0, 0), fwd)[2]
        assert torch.allclose(vis, torch.ones(2), atol=0.03), vis


def test_crop_aware_values_follow_the_augmentation(tmp_path):
    img_dir, jf = _with_cost(tmp_path)
    coco = json.loads(jf.read_text())
    coco["attributes"] += [{"name": "area", "type": "regression", "crop": "scale"},
                           {"name": "whole", "type": "regression", "crop": "ignore", "min_visible": 0.8}]
    for a in coco["annotations"]:
        a["attributes"].update(area=100.0, whole=7.0)
    jf.write_text(json.dumps(coco))
    # zoomed in (the window sees ~half of the image) and min_mask_mass 0, so every instance is kept, cut or not; the
    # multi-image ops add instances from other samples, whose values must follow their own crops
    aug = {"min_scale": 2.0, "max_scale": 2.0, "flip_prob": 0.0, "min_mask_mass": 0.0,
           "mosaic_prob": 0.25, "copy_paste_prob": 0.25, "mixup_prob": 0.25}
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, aug=aug)
    assert [s.crop for s in ds.aux_specs] == [None, None, "scale", "ignore"]
    torch.manual_seed(0)
    partial, multi = False, set()
    for _ in range(20):
        a = ds[0][3]
        vis = ds.transform.last_info["visible"]
        multi.add(ds.transform.last_info["multi"])
        assert torch.allclose(a["area"], 100.0 * vis)
        assert torch.equal(a["whole"].isnan(), vis < 0.8) and (a["whole"][vis >= 0.8] == 7.0).all()
        assert a["cost"][1] == 1000.0  # not crop-aware: untouched
        partial |= bool(((vis > 0.05) & (vis < 0.95)).any())
    assert partial and multi == {None, "mosaic", "copy_paste", "mixup"}  # the crops did cut instances
    coco["attributes"][0]["crop"] = "scale"  # "kind", a categorical head
    jf.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="regression attributes only"):
        CocoInstanceSeg(img_dir, jf, imgsz=112)
