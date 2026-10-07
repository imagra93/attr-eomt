"""Multi-GPU training: one process per GPU (DistributedDataParallel), started by ``train(device="0,1")``.

:func:`launch` spawns a process per GPU; each joins a NCCL process group and runs the same function on its own
``cuda:N``. Inside, :func:`rank` / :func:`world_size` / :func:`is_main` describe the process, :func:`shard` splits a
dataset between the processes without padding (an evaluation must see every image exactly once), and :func:`gather` /
:func:`sum_over_processes` collect per-process results on the main process. Without a process group they all reduce
to the single-process case, so the same code serves both.
"""

from __future__ import annotations

import functools
import multiprocessing
import os
import pickle
import socket
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset

#: How long a process waits at a collective: after each validation the main process alone runs COCOeval and writes
#: the checkpoints and plots while the others wait at their next collective.
TIMEOUT = timedelta(minutes=60)


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def shard(ds):
    """This process's share of ``ds``: every ``world_size``-th item, no padding (all of ``ds`` on one process)."""
    return ds if world_size() == 1 else Subset(ds, range(rank(), len(ds), world_size()))


def gather(obj) -> list | None:
    """``[obj of process 0, obj of process 1, ...]`` on the main process, ``None`` on the others (``[obj]`` on one)."""
    if world_size() == 1:
        return [obj]
    out = [None] * world_size() if is_main() else None
    dist.gather_object(obj, out, dst=0)
    return out


def _add(a, b):
    if isinstance(a, dict):
        return {k: _add(a[k], b[k]) if k in a and k in b else (a[k] if k in a else b[k]) for k in {**a, **b}}
    if isinstance(a, (list, tuple)):
        return type(a)(_add(x, y) for x, y in zip(a, b))
    return a + b


def sum_over_processes(obj):
    """``obj`` (numbers, or dicts / lists / tuples of them, e.g. counts) summed over the processes, on the main
    process; ``None`` on the others."""
    parts = gather(obj)
    return None if parts is None else functools.reduce(_add, parts)


def launch(fn, gpus: list[int], kwargs: dict):
    """``fn(**kwargs)`` in one process per GPU, each with ``device="cuda:<gpu>"``; returns the main process's result.

    The processes are spawned, so ``fn`` and every value in ``kwargs`` must be picklable (a custom callable such as a
    ``train_transform`` must be importable, not a lambda or a function defined in a notebook).
    """
    with socket.socket() as s:  # a free port for the process group's rendezvous
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmp:  # the main process's result, read once every process has exited
        result = Path(tmp) / "result.pkl"
        mp.spawn(_worker, args=(fn, gpus, port, kwargs, result), nprocs=len(gpus), join=True)
        return pickle.loads(result.read_bytes())


def _worker(index: int, fn, gpus: list[int], port: int, kwargs: dict, result: Path) -> None:
    # A spawned process starts its own children (DataLoader workers) by spawning too, which pickles the datasets;
    # restore the platform's default (fork on Linux), as in a single-process run.
    multiprocessing.set_start_method(None, force=True)
    device = torch.device(f"cuda:{gpus[index]}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=index, world_size=len(gpus),
                            timeout=TIMEOUT, device_id=device)
    if index > 0:
        sys.stdout = open(os.devnull, "w")  # one log: the main process speaks for all (errors still reach stderr)
    try:
        out = fn(**{**kwargs, "device": str(device)})
        if index == 0:
            result.write_bytes(pickle.dumps(out))
    finally:
        dist.destroy_process_group()
