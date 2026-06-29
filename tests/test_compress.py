"""Int8 compression round-trip test (gated on torchao + CUDA).

Int8 weight-only quant uses torchao tensor subclasses whose kernels need a GPU, so
this test skips automatically when either torchao or CUDA is unavailable.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from eomt import EoMT
from eomt.serialization import load_raw

IMGSZ = 140  # 14 * 10 -> tiny patch grid keeps the test fast
NC = 3

_no_torchao = importlib.util.find_spec("torchao") is None
_no_cuda = not torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    _no_torchao or _no_cuda, reason="int8 compression needs torchao + CUDA"
)


def test_int8_compress_save_reload_roundtrip(tmp_path):
    """compress() shrinks the state dict; the recipe round-trips through save/reload."""
    m = EoMT("s", device="cuda", pretrained=False, nc=NC, imgsz=IMGSZ)
    ckpt = tmp_path / "int8.pt"

    result = m.compress("int8", validate=False, save=ckpt)
    assert result["recipe"] == "int8"
    # Quantizing the transformer blocks meaningfully shrinks the serialized weights.
    assert result["size_mb_after"] < result["size_mb_before"]
    assert result["size_ratio"] < 0.6
    assert m._compression and m._compression["recipe"] == "int8"

    # The compressed checkpoint records the recipe so reload re-creates the layout.
    assert load_raw(ckpt).get("compression", {}).get("recipe") == "int8"

    # A metrics sidecar is dropped in the run folder (parent of weights/).
    import json

    sidecar = tmp_path / "compression_metrics.json"
    assert sidecar.is_file()
    saved_metrics = json.loads(sidecar.read_text())
    assert saved_metrics["recipe"] == "int8" and "size_ratio" in saved_metrics

    reloaded = EoMT(ckpt, device="cuda")
    assert reloaded._compression and reloaded._compression["recipe"] == "int8"

    x = torch.randn(1, 3, int(reloaded.model.image_size), int(reloaded.model.image_size), device="cuda")
    with torch.no_grad():
        out = reloaded.model(x)
    assert {"class_queries_logits", "masks_queries_logits"} <= set(out)

    # Re-saving a reloaded compressed model round-trips the recipe (no metadata loss).
    again = tmp_path / "int8_again.pt"
    reloaded.save(again)
    assert load_raw(again).get("compression", {}).get("recipe") == "int8"
