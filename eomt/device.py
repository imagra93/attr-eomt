"""Device resolution shared across train / val / predict.

``resolve_device("auto")`` returns CUDA when available — picking the GPU with the
most free memory when several are present (handy when ``cuda:0`` is busy) — else
CPU. An explicit ``"cpu"`` / ``"cuda:N"`` (or a ``torch.device``) is honored as-is.
The first time each device is resolved, one line is logged so it is always clear
which device a run landed on.
"""

from __future__ import annotations

import torch

#: Devices already announced, so the "using ..." line is printed once each.
_logged: set[str] = set()


def _least_busy_cuda() -> int:
    """Return the index of the CUDA device with the most free memory."""
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:  # pragma: no cover - driver quirks; fall back to index 0
            free = 0
        if free > best_free:
            best_idx, best_free = i, free
    return best_idx


def resolve_device(device: str | torch.device = "auto", *, verbose: bool = True) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device` (see module docstring)."""
    if isinstance(device, torch.device):
        dev = device
    elif device in ("", "auto"):
        if torch.cuda.is_available():
            idx = _least_busy_cuda() if torch.cuda.device_count() > 1 else 0
            dev = torch.device(f"cuda:{idx}")
        else:
            dev = torch.device("cpu")
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
