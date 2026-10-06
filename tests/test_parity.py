"""Numerical parity between EoMTEncoder and HuggingFace EomtForUniversalSegmentation.

The EoMTEncoder must (a) expose the exact same ``state_dict`` keys as the HF
model and (b) produce numerically identical outputs on CPU/fp32, so existing
checkpoints load unchanged. Two deliberate deviations are swapped out where they
would differ: the attention mask (max-pool + attend-all fallback, see
``EoMTEncoder._build_attention_mask``) and the dense mask terms of the loss (see
:mod:`eomt.loss`). HF-dependent tests skip if transformers' EoMT model code is
unavailable; the real-checkpoint test skips when no weights are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

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


def _use_reference_attention_mask(enc):
    """Give ``enc`` the HF / official-EoMT attention mask (bilinear downsampling; an empty mask sees no patch)."""
    def build(hidden_states, masks_queries_logits, prob, grid_size=None):
        nq = enc.config.num_queries
        start = nq + enc.embeddings.num_prefix_tokens
        b, n = hidden_states.shape[:2]
        mask = torch.ones(b, n, n, dtype=torch.bool)
        logits = F.interpolate(masks_queries_logits, size=grid_size or enc.grid_size, mode="bilinear")
        mask[:, :nq, start:] = logits.flatten(2) > 0
        if prob < 1:
            mask[:, :nq, start:][torch.rand(b, nq) > prob] = True
        return mask[:, None]
    enc._build_attention_mask = build


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
    _use_reference_attention_mask(nat)  # everything but the mask construction must match HF
    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        o_hf = hf(pixel_values=x)
        o_nat = nat(pixel_values=x)
    assert torch.allclose(o_hf.masks_queries_logits, o_nat.masks_queries_logits, atol=1e-4, rtol=1e-3)
    assert torch.allclose(o_hf.class_queries_logits, o_nat.class_queries_logits, atol=1e-5, rtol=1e-4)


def test_loss_parity_fresh_weights():
    """Class CE, matching by class, per-layer summation and weighting match HF. The mask terms are zero-weighted:
    HF point-samples them, ours are dense on the logit grid (tested in test_architecture.py)."""
    cfg = build_eomt_config("s", nc=NC, image_size=IMGSZ, mask_weight=0.0, dice_weight=0.0)
    hf = EomtForUniversalSegmentation(cfg).train()
    nat = EoMTEncoder(cfg).train()
    nat.load_state_dict(hf.state_dict())
    _use_reference_attention_mask(nat)

    x = torch.randn(2, 3, IMGSZ, IMGSZ)
    mask_labels = [(torch.rand(2, IMGSZ, IMGSZ) > 0.5).float(), (torch.rand(1, IMGSZ, IMGSZ) > 0.5).float()]
    class_labels = [torch.tensor([0, 1]), torch.tensor([2])]

    # Re-seed identically before each forward (HF's point sampler draws random numbers).
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
