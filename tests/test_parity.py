"""Numerical parity between EoMTEncoder and HuggingFace EomtForUniversalSegmentation.

The EoMTEncoder must (a) expose the exact same ``state_dict`` keys as the HF
model and (b) produce numerically identical outputs/loss on CPU/fp32, so existing
checkpoints load unchanged. HF-dependent tests skip if transformers' EoMT model
code is unavailable; the real-checkpoint test skips when no weights are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eomt.config import build_eomt_config
from eomt.model import EoMTEncoder

hf_eomt = pytest.importorskip("transformers")
try:
    from transformers import EomtForUniversalSegmentation
except Exception:  # pragma: no cover
    pytest.skip("transformers EoMT model code unavailable", allow_module_level=True)

IMGSZ = 140  # 14 * 10 keeps the forward fast
NC = 3
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size", ["s", "b", "l"])
def test_state_dict_keys_match(size):
    cfg = build_eomt_config(size, nc=NC, image_size=IMGSZ)
    hf = EomtForUniversalSegmentation(cfg)
    nat = EoMTEncoder(cfg)
    assert set(hf.state_dict()) == set(nat.state_dict())


@pytest.mark.parametrize("probs", [1.0, 0.0])
def test_forward_parity_fresh_weights(probs):
    torch.manual_seed(0)
    cfg = build_eomt_config("s", nc=NC, image_size=IMGSZ)
    hf = EomtForUniversalSegmentation(cfg).eval()
    nat = EoMTEncoder(cfg).eval()
    missing, unexpected = nat.load_state_dict(hf.state_dict())
    assert not missing and not unexpected

    hf.attn_mask_probs.fill_(probs)
    nat.attn_mask_probs.fill_(probs)
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        o_hf = hf(pixel_values=x)
        o_nat = nat(pixel_values=x)
    assert torch.allclose(o_hf.masks_queries_logits, o_nat.masks_queries_logits, atol=1e-4, rtol=1e-3)
    assert torch.allclose(o_hf.class_queries_logits, o_nat.class_queries_logits, atol=1e-5, rtol=1e-4)


def test_loss_parity_fresh_weights():
    cfg = build_eomt_config("s", nc=NC, image_size=IMGSZ)
    hf = EomtForUniversalSegmentation(cfg).train()
    nat = EoMTEncoder(cfg).train()
    nat.load_state_dict(hf.state_dict())

    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    mask_labels = [(torch.rand(2, IMGSZ, IMGSZ) > 0.5).float(), (torch.rand(1, IMGSZ, IMGSZ) > 0.5).float()]
    class_labels = [torch.tensor([0, 1]), torch.tensor([2])]

    # Re-seed identically before each loss so the matcher/point-sampler rand draws
    # happen in the same order.
    torch.manual_seed(1234)
    l_hf = hf(pixel_values=x, mask_labels=mask_labels, class_labels=class_labels).loss
    torch.manual_seed(1234)
    l_nat = nat(pixel_values=x, mask_labels=mask_labels, class_labels=class_labels).loss
    assert torch.allclose(l_hf, l_nat, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("ckpt_rel", ["runs/train/eomt-b/weights/best.pt"])
def test_real_checkpoint_parity(ckpt_rel):
    """A real trained checkpoint loads into EoMTEncoder cleanly and matches HF output."""
    ckpt_path = REPO / ckpt_rel
    if not ckpt_path.is_file():
        pytest.skip(f"no checkpoint at {ckpt_rel}")
    from eomt.serialization import load_raw

    ckpt = load_raw(ckpt_path)
    size, nc, imgsz = ckpt["size"], int(ckpt["nc"]), int(ckpt["imgsz"])
    nub = ckpt.get("num_upscale_blocks")
    cfg = build_eomt_config(size, nc=nc, image_size=imgsz,
                            num_upscale_blocks=nub, **(ckpt.get("loss_weights") or {}))
    # eomt.* subset of the (EoMTModel) state dict, with the prefix stripped.
    state = {k[len("eomt."):]: v for k, v in ckpt["model"].items() if k.startswith("eomt.")}

    nat = EoMTEncoder(cfg).eval()
    missing, unexpected = nat.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])

    hf = EomtForUniversalSegmentation(cfg).eval()
    hf.load_state_dict(state, strict=False)

    hf.attn_mask_probs.zero_()
    nat.attn_mask_probs.zero_()
    torch.manual_seed(0)
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        o_hf = hf(pixel_values=x)
        o_nat = nat(pixel_values=x)
    assert torch.allclose(o_hf.masks_queries_logits, o_nat.masks_queries_logits, atol=1e-4, rtol=1e-3)
    assert torch.allclose(o_hf.class_queries_logits, o_nat.class_queries_logits, atol=1e-5, rtol=1e-4)
