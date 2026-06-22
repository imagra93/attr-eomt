"""CPU, no-network smoke tests for the high-level :class:`eomt.EoMT` interface."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from eomt import EoMT, EoMTModel

IMGSZ = 140  # 14 * 10 -> tiny patch grid keeps the test fast
NC = 3


def test_init_from_size_builds_lazily():
    """``EoMT(size)`` defers building (and the DINOv2 load) until the model is used."""
    m = EoMT("s", device="cpu", pretrained=False, nc=NC, imgsz=IMGSZ)
    assert m.size == "s"
    assert m._model is None  # not built yet
    model = m.model  # triggers the lazy build
    assert isinstance(model, EoMTModel) and model.nc == NC


def test_init_rejects_unknown_spec():
    with pytest.raises(ValueError):
        EoMT("xl")  # neither a known size nor an existing checkpoint


def test_save_reload_and_predict_roundtrip(tmp_path):
    """save() writes a self-describing ckpt; EoMT(path) reloads it and predict() runs."""
    torch.manual_seed(0)
    m = EoMT("s", device="cpu", pretrained=False, nc=NC, imgsz=IMGSZ)
    ckpt = tmp_path / "m.pt"
    m.save(ckpt)

    reloaded = EoMT(ckpt, device="cpu")
    assert reloaded.size == "s" and reloaded.model.nc == NC

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.fromarray(np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)).save(img_dir / "a.png")

    out_dir = tmp_path / "out"
    results = reloaded.predict(img_dir, plot=True, save=str(out_dir), conf_thres=0.0)
    assert len(results) == 1
    r = results[0]
    assert {"num_detections", "boxes", "scores", "classes", "masks"} <= set(r)
    assert "plot_path" in r and (out_dir / "a.png").is_file()
