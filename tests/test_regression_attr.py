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
    assert torch.equal(tf._crop(img, m, False, (240, 320), (120, 160), (0, 0), fwd)[2], torch.ones(2))  # masks: exact
    # the same warp into the left half of that window (x < 80): the green one (x ~45-115) is cut, the red one stays whole
    vis = tf._crop(img, m, False, (240, 320), (120, 80), (0, 0), fwd)[2]
    assert 0.0 < vis[0] < 1.0 and vis[1] == 1.0, vis


def test_crop_aware_values_follow_the_augmentation(tmp_path):
    img_dir, jf = _with_cost(tmp_path)
    coco = json.loads(jf.read_text())
    coco["attributes"] += [{"name": "surface", "type": "regression", "crop": "scale"},
                           {"name": "whole", "type": "regression", "crop": "ignore", "min_visible": 0.8}]
    for a in coco["annotations"]:
        a["attributes"].update(surface=100.0, whole=7.0)
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
        assert torch.allclose(a["surface"], 100.0 * vis)
        assert torch.equal(a["whole"].isnan(), vis < 0.8) and (a["whole"][vis >= 0.8] == 7.0).all()
        assert a["cost"][1] == 1000.0  # not crop-aware: untouched
        partial |= bool(((vis > 0.05) & (vis < 0.95)).any())
    assert partial and multi == {None, "mosaic", "copy_paste", "mixup"}  # the crops did cut instances
    coco["attributes"][0]["crop"] = "scale"  # "kind", a categorical head
    jf.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="regression attributes only"):
        CocoInstanceSeg(img_dir, jf, imgsz=112)



def test_zoom_closes_in_on_an_instance_with_the_attribute_and_its_area_follows_the_kept_share(tmp_path):
    """zoom_prob / zoom_scale / zoom_attr: a 4x close-up placed on an instance that has a surface, never on a sample
    without one; the surface of every instance (cut or whole) is its full surface x the share of its mask mass kept in
    view, checked against the output masks themselves (mass / (source mass x scale^2), exact for a crop with no
    warp)."""
    img_dir, jf = _with_cost(tmp_path)
    coco = json.loads(jf.read_text())
    coco["attributes"].append({"name": "surface", "type": "regression", "log": True, "crop": "scale"})
    for a in coco["annotations"]:
        if a["attributes"]["kind"] == 1:  # only the lines have a surface
            a["attributes"]["surface"] = 500.0
    lines_of_image0 = [a for a in coco["annotations"] if a["image_id"] == coco["images"][0]["id"] and a["attributes"]["kind"] == 1]
    assert lines_of_image0
    jf.write_text(json.dumps(coco))
    aug = {"zoom_prob": 1.0, "zoom_scale": (4.0, 4.0), "zoom_attr": "surface", "flip_prob": 0.0, "rotate_prob": 0.0,
           "shear_prob": 0.0, "perspective_prob": 0.0, "min_mask_mass": 0.0}
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, aug=aug)
    kind = next(s for s in ds.aux_specs if s.name == "kind")
    torch.manual_seed(0)
    cut = False
    for i in range(len(ds)):
        _, src_masks, _, src_attrs, _ = ds._load_raw(i)
        _, masks, _, attrs = ds[i]
        info = ds.transform.last_info
        assert info["mode"] == "zoom" and info["scale"] == 4.0
        assert not torch.isnan(src_attrs["surface"][info["focus"]])  # the close-up is on an instance with a surface
        assert int(src_attrs["kind"][info["focus"]]) == 1 and kind.names[1] == "line"
        kept = masks.flatten(1).sum(1) / (src_masks.flatten(1).float().sum(1) * info["eff_scale"] ** 2)
        assert torch.allclose(info["visible"], kept.clamp(max=1.0), atol=0.02), (info["visible"], kept)
        has = ~torch.isnan(src_attrs["surface"])
        assert torch.allclose(attrs["surface"][has], 500.0 * info["visible"][has])
        cut |= bool(((info["visible"] > 0.05) & (info["visible"] < 0.95) & has).any())
    assert cut  # some close-ups cut the instance: the surface followed the share kept
    for a in coco["annotations"]:  # no instance with a surface: never a close-up
        a["attributes"].pop("surface", None)
    jf.write_text(json.dumps(coco))
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, aug=aug)
    ds[0]
    assert ds.transform.last_info["mode"] != "zoom"


def test_log_regressed_area_follows_the_crop_in_real_units_before_the_log(tmp_path):
    """A crop-scaled log1p head (a surface): the value is scaled by the kept fraction in real units, then log1p'd."""
    img_dir, jf = _with_cost(tmp_path)
    coco = json.loads(jf.read_text())
    coco["attributes"].append({"name": "surface", "type": "regression", "log": True, "crop": "scale"})
    for i, a in enumerate(coco["annotations"]):
        a["attributes"]["surface"] = 400.0 * (1 + i % 3)  # spread, so the standardization is not trivial
    jf.write_text(json.dumps(coco))
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, aug={"min_scale": 0.5, "max_scale": 2.0, "flip_prob": 0.0, "min_mask_mass": 0.0})
    spec = next(s for s in ds.aux_specs if s.name == "surface")
    torch.manual_seed(0)
    cut = False
    for _ in range(20):
        whole = torch.tensor([400.0 * (1 + i % 3) for i in range(2)])  # image 0: annotations 0 and 1
        a = ds[0][3]["surface"]
        vis = ds.transform.last_info["visible"]
        assert torch.allclose(a, whole * vis)  # the proportion in real units ...
        z = spec.encode(a.double())
        assert torch.allclose(z, (torch.log1p(whole.double() * vis.double()) - spec.mean) / spec.std, atol=1e-5)  # ... then the log
        cut |= bool((vis < 1).any())
    assert cut


def test_metric_plots_show_each_regression_head_as_its_mae(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from eomt.plotting import plot_aux_per_class, plot_metrics_csv

    figs = []
    monkeypatch.setattr(plt, "close", figs.append)  # keep the figures to look at their panels
    heads = ["kind", "cost", "surface"]
    cols = ["epoch", "train/loss", "val/segm/mAP"] + [f"{s}/{'aux_acc' if h == 'kind' else 'aux_mae'}/{h}"
                                                      for h in heads for s in ("train", "val")]
    (tmp_path / "metrics.csv").write_text(",".join(cols) + "\n0,2.0,0.1,0.5,0.4,90,120,30,35\n1,1.5,0.2,0.6,0.5,60,80,20,25\n")
    plot_metrics_csv(tmp_path / "metrics.csv", tmp_path / "metrics.png")
    panels = {ax.get_title(): ax for ax in figs[0].axes if ax.axison}
    assert {"Loss", "Aux head accuracy", "Aux 'cost' mean absolute error",
            "Aux 'surface' mean absolute error"} <= set(panels)
    assert [ln.get_label() for ln in panels["Aux head accuracy"].get_lines()] == ["tr kind", "va kind"]
    assert list(panels["Aux 'cost' mean absolute error"].get_lines()[1].get_ydata()) == [120.0, 80.0]
    assert (tmp_path / "metrics.png").stat().st_size > 0

    plot_aux_per_class({"kind": {0: (3, 4)}, "cost": {0: (50.0, 2), 1: (10.0, 1)}}, {0: "a", 1: "b"},
                       tmp_path / "aux_per_class.png", regression={"cost"})
    acc, mae = figs[1].axes
    assert acc.get_title() == "aux 'kind' accuracy by primary class" and acc.get_ylim()[1] == 1.08
    assert mae.get_title() == "aux 'cost' mean absolute error by primary class"
    assert [b.get_height() for b in mae.patches] == [25.0, 10.0] and mae.get_ylim()[1] > 25.0
