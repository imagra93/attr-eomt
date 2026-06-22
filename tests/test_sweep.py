"""CPU tests for the inference-knob sweep tool and min_mask_area filter."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from eomt import build_model, postprocess_instance

IMGSZ = 140
NC = 3


def _make_synthetic_coco(tmp_path, n_images=3):
    """Write a tiny COCO val set (images + JSON) and return (img_dir, json_path)."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    images, annotations, ann_id = [], [], 1
    rng = np.random.default_rng(0)
    for i in range(1, n_images + 1):
        h, w = 32, 40
        arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{i}.png")
        images.append({"id": i, "file_name": f"{i}.png", "width": w, "height": h})
        # one square-ish polygon annotation per image
        x0, y0, bw, bh = 5, 5, 12, 10
        poly = [x0, y0, x0 + bw, y0, x0 + bw, y0 + bh, x0, y0 + bh]
        annotations.append({
            "id": ann_id, "image_id": i, "category_id": (i % NC) + 1,
            "segmentation": [poly], "area": float(bw * bh),
            "bbox": [x0, y0, bw, bh], "iscrowd": 0,
        })
        ann_id += 1
    categories = [{"id": c + 1, "name": f"class_{c}"} for c in range(NC)]
    data = {"images": images, "annotations": annotations, "categories": categories}
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(data))
    return img_dir, json_path


def test_min_mask_area_filters_specks():
    """min_mask_area drops binarized masks below the threshold; 0 keeps everything."""
    torch.manual_seed(0)
    q = 8
    # Build logits where exactly one query has a tiny positive mask and the rest big.
    masks = torch.full((1, q, 10, 10), -10.0)
    masks[0, 0, :1, :1] = 10.0          # query 0: 1 low-res px -> tiny after upsample
    masks[0, 1, :, :] = 10.0            # query 1: full mask
    cls = torch.full((1, q, NC + 1), -10.0)
    cls[0, 0, 0] = 10.0
    cls[0, 1, 1] = 10.0
    out = {"masks_queries_logits": masks, "class_queries_logits": cls}

    base = postprocess_instance(out, conf_thres=0.0, original_size=(20, 20), max_det=q)
    filt = postprocess_instance(out, conf_thres=0.0, original_size=(20, 20), max_det=q,
                                min_mask_area=20.0)
    assert filt["num_detections"] < base["num_detections"]
    assert (filt["masks"].flatten(1).sum(1) >= 20).all()


def test_sweep_reproduces_evaluate_at_default_knobs(tmp_path):
    """The (0.0, 0.5, 100, 0) grid point must equal `evaluate` on the same checkpoint."""
    from eomt.data import CocoValImages
    from eomt.engine import evaluate, sweep

    torch.manual_seed(0)
    img_dir, json_path = _make_synthetic_coco(tmp_path)
    model = build_model("s", nc=NC, imgsz=IMGSZ).eval()
    val_ds = CocoValImages(img_dir, json_path, imgsz=IMGSZ, letterbox=True)

    ref = evaluate(model, val_ds, device="cpu", batch_size=2, num_workers=0,
                   conf_thres=0.0, max_det=100, mask_thresh=0.5, verbose=False)
    grid = [{"conf_thres": 0.0, "mask_thresh": 0.5, "max_det": 100, "min_mask_area": 0.0}]
    rows = sweep(model, val_ds, device="cpu", grid=grid, batch_size=2, num_workers=0,
                 verbose=False)

    assert len(rows) == 1
    for k in ("segm/mAP", "segm/mAP50", "segm/mAP_small", "segm/AR100"):
        assert rows[0][k] == pytest.approx(ref[k], abs=1e-6), k


def test_sweep_returns_one_row_per_grid_point(tmp_path):
    from eomt.data import CocoValImages
    from eomt.engine import sweep

    torch.manual_seed(0)
    img_dir, json_path = _make_synthetic_coco(tmp_path)
    model = build_model("s", nc=NC, imgsz=IMGSZ).eval()
    val_ds = CocoValImages(img_dir, json_path, imgsz=IMGSZ, letterbox=True)
    grid = [
        {"conf_thres": c, "mask_thresh": 0.5, "max_det": 100, "min_mask_area": 0.0}
        for c in (0.0, 0.3, 0.6)
    ]
    rows = sweep(model, val_ds, device="cpu", grid=grid, batch_size=2, num_workers=0,
                 verbose=False)
    assert len(rows) == 3
    assert all("segm/mAP" in r and r["conf_thres"] == g["conf_thres"]
               for r, g in zip(rows, grid))
