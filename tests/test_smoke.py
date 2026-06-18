"""CPU, no-network smoke tests for the libre-eomt package."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eomt import build_eomt_config, build_model, postprocess_instance
from eomt.config import EOMT_CONFIGS

IMGSZ = 140  # 14 * 10 -> tiny patch grid keeps the test fast
NC = 3


@pytest.mark.parametrize("size", ["s", "b", "l"])
def test_config_matches_size_preset(size):
    cfg = build_eomt_config(size, nc=NC, image_size=IMGSZ)
    preset = EOMT_CONFIGS[size]
    assert cfg.hidden_size == preset.hidden_size
    assert cfg.num_hidden_layers == preset.num_hidden_layers
    assert cfg.num_attention_heads == preset.num_attention_heads
    assert cfg.num_queries == preset.num_queries


def test_config_rejects_non_patch_aligned_imgsz():
    with pytest.raises(ValueError):
        build_eomt_config("s", nc=NC, image_size=100)  # 100 % 14 != 0


def test_build_forward_shapes():
    torch.manual_seed(0)
    model = build_model("s", nc=NC, imgsz=IMGSZ).eval()
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        out = model(x)
    assert set(out) == {"masks_queries_logits", "class_queries_logits"}
    q = EOMT_CONFIGS["s"].num_queries
    assert out["class_queries_logits"].shape == (2, q, NC + 1)
    assert out["masks_queries_logits"].shape[:2] == (2, q)


def test_train_step_backward():
    torch.manual_seed(0)
    model = build_model("s", nc=NC, imgsz=IMGSZ).train()
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    mask_labels = [
        (torch.rand(2, IMGSZ, IMGSZ) > 0.5).float(),
        (torch.rand(1, IMGSZ, IMGSZ) > 0.5).float(),
    ]
    class_labels = [torch.tensor([0, 1]), torch.tensor([2])]
    loss = model(x, mask_labels=mask_labels, class_labels=class_labels)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"


def test_postprocess_instance_contract():
    q = 100
    out = {
        "masks_queries_logits": torch.randn(1, q, 10, 10),
        "class_queries_logits": torch.randn(1, q, NC + 1),
    }
    res = postprocess_instance(out, conf_thres=0.0, original_size=(20, 15), max_det=10)
    assert {"num_detections", "boxes", "scores", "classes", "masks"} <= set(res)
    assert res["masks"].shape[1:] == (15, 20)  # (orig_h, orig_w)
    assert res["boxes"].shape[1] == 4
    assert res["num_detections"] <= 10


def test_build_model_rejects_unimplemented_family():
    with pytest.raises(NotImplementedError):
        build_model("s", nc=NC, family="detect")
