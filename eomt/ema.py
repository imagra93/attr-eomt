"""Exponential moving average of model weights (Polyak averaging).

A frozen shadow copy of the model whose **parameters** track the live training
weights with an exponentially decaying average, while **buffers** are copied
straight through. Validating and exporting the EMA copy instead of the live
weights gives transformer detectors/segmenters a small but consistent mAP gain
and a smoother best-checkpoint signal.

The decay is ramped in YOLO-style — ``d = decay * (1 - exp(-updates / tau))`` —
so the average is responsive early (when weights move fast) and slow later.
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch


def _unwrap(model):
    """Return the underlying module, transparently unwrapping ``torch.compile``."""
    return getattr(model, "_orig_mod", model)


class ModelEMA:
    """Maintain a moving-average copy of ``model``'s parameters.

    Buffers (e.g. EoMT's annealed ``attn_mask_probs``) are copied verbatim rather
    than averaged, so the EMA model is configured exactly like the live model and
    can be switched to the mask-free regime for validation the same way.
    """

    def __init__(self, model, *, decay: float = 0.9999, tau: float = 2000.0, device=None):
        self.module = deepcopy(_unwrap(model)).eval()
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)

    @torch.no_grad()
    def update(self, model) -> None:
        """Pull one EMA step from the live ``model`` (call once per optimizer step)."""
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau)) if self.tau > 0 else self.decay
        src = _unwrap(model)
        for ema_p, p in zip(self.module.parameters(), src.parameters()):
            # ``.to(device=..)`` makes the step valid when the EMA copy lives on a
            # different device than the live model (e.g. EMA held on CPU to save VRAM).
            ema_p.mul_(d).add_(p.detach().to(device=ema_p.device, dtype=ema_p.dtype), alpha=1.0 - d)
        for ema_b, b in zip(self.module.buffers(), src.buffers()):
            ema_b.copy_(b)

    def state_dict(self) -> dict:
        return {
            "module": self.module.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
            "tau": self.tau,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd["module"])
        self.updates = int(sd.get("updates", 0))
        self.decay = float(sd.get("decay", self.decay))
        self.tau = float(sd.get("tau", self.tau))
