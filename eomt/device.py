"""Device resolution shared across train / val / predict.

``resolve_device("auto")`` returns ``cuda:0`` when CUDA is available (even with
several GPUs present — pick a specific card with an explicit ``"cuda:N"``), else
CPU. Defaulting to a single fixed device keeps a run on one GPU: ``auto`` is
resolved more than once per run (model load, then training), and a "least busy"
heuristic could land those calls on different cards and split the run across both.
An explicit ``"cpu"`` / ``"cuda:N"`` (or a ``torch.device``) is honored as-is.
The first time each device is resolved, one line is logged so it is always clear
which device a run landed on.
"""

from __future__ import annotations

import torch

#: Devices already announced, so the "using ..." line is printed once each.
_logged: set[str] = set()


def resolve_device(device: str | torch.device = "auto", *, verbose: bool = True) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device` (see module docstring)."""
    if isinstance(device, torch.device):
        dev = device
    elif device in ("", "auto"):
        dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        dev = torch.device(device)
    if verbose:
        _log_once(dev)
    return dev


def _log_once(dev: torch.device) -> None:
    key = str(dev)
    if key in _logged:
        return
    _logged.add(key)
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        try:
            name = torch.cuda.get_device_name(idx)
            free, _total = torch.cuda.mem_get_info(idx)
            print(f"[eomt] using cuda:{idx} ({name}, {free / 1e9:.1f} GB free)")
        except Exception:  # pragma: no cover
            print(f"[eomt] using cuda:{idx}")
    else:
        print(f"[eomt] using {dev}")
