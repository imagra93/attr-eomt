"""Low-rank adaptation (LoRA) of the frozen ViT blocks of a multi-view post-training.

The attention projections of the ``blocks`` frozen blocks right below the query blocks become ``W + (alpha / rank)
B A`` (``A`` ``rank x in``, ``B`` ``out x rank``, ``B`` starting at 0: the model starts exactly as loaded), so the patch
features can adapt to the sets of photos while ``W`` stays frozen. It is a parametrization of each weight
(:mod:`torch.nn.utils.parametrize`): the state dict keeps ``W`` under ``...weight.original`` and the two factors beside
it, which a training resumes from; the checkpoints the trainer exports for inference (``best.pt``) hold the merged
weights (:func:`merged_state_dict`), and :func:`merge_state` merges the factors of any other one on loading.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

#: The projections of an attention block that are adapted.
TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


class LoRA(nn.Module):
    """The parametrization ``W -> W + scale * B A`` of one weight."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float):
        super().__init__()
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = float(alpha) / float(rank)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + (self.lora_b @ self.lora_a).to(weight.dtype) * self.scale


def lora_blocks(model, blocks: int) -> list[int]:
    """The indices of the ``blocks`` ViT blocks right below the query blocks."""
    first = model.config.num_hidden_layers - model.config.num_blocks
    return list(range(max(0, first - int(blocks)), first))


def add_lora(model, *, rank: int = 16, alpha: float | None = None, blocks: int = 8) -> dict:
    """Adapt the attention projections (:data:`TARGETS`) of :func:`lora_blocks` with rank-``rank`` factors (``alpha``
    defaults to ``rank``: scale 1). Returns the settings, as the checkpoints record them."""
    alpha = float(rank if alpha is None else alpha)
    for i in lora_blocks(model, blocks):
        att = model.eomt.layers[i].attention
        for name in TARGETS:
            lin = getattr(att, name)
            dev = lin.weight.device
            parametrize.register_parametrization(lin, "weight", LoRA(lin.out_features, lin.in_features, rank, alpha).to(dev))
    return {"rank": int(rank), "alpha": alpha, "blocks": int(blocks)}


def merged_state_dict(model) -> dict[str, torch.Tensor]:
    """``model``'s state dict with every LoRA weight merged (``W + scale * B A``) under its plain name."""
    return merge_state(model.state_dict(), scale=None, model=model)


def merge_state(state: dict[str, torch.Tensor], scale: float | None, model=None) -> dict[str, torch.Tensor]:
    """A state dict with the LoRA factors of each adapted weight merged into it (``...weight.original`` + ``scale``
    ``lora_b @ lora_a`` -> ``...weight``). ``scale`` is ``alpha / rank`` of the checkpoint's ``lora`` record (or read
    off ``model``'s parametrizations)."""
    out = {}
    suffix = ".parametrizations.weight.original"
    for key, value in state.items():
        if ".parametrizations.weight." in key and not key.endswith(suffix):
            continue  # a factor: merged below
        if not key.endswith(suffix):
            out[key] = value
            continue
        base = key[: -len(suffix)]
        a = state[f"{base}.parametrizations.weight.0.lora_a"]
        b = state[f"{base}.parametrizations.weight.0.lora_b"]
        s = scale
        if s is None:
            s = model.get_submodule(base).parametrizations.weight[0].scale
        out[f"{base}.weight"] = (value.float() + (b.float() @ a.float()) * s).to(value.dtype)
    return out


def has_lora(state: dict) -> bool:
    return any(".parametrizations.weight.original" in k for k in state)
