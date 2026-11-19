"""Checkpoint metadata wrapping and model (re)loading.

Checkpoints are plain ``torch.save`` dicts carrying the state dict plus enough
metadata (size / task / nc / names / imgsz) to rebuild the model with
:func:`load_model` — no external registry needed.
"""

from __future__ import annotations

import functools
import re
import subprocess
import warnings
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

import torch

from .config import HIDDEN_TO_SIZE, aux_specs_from_meta, aux_specs_to_meta
from .device import resolve_device
from .lora import has_lora, merge_state
from .model import EoMTModel, build_model
from .preprocess import IMAGENET_MEAN, IMAGENET_STD

#: 2.0: written by eomt 3 (no masked-attention buffer, no cross-view view types / refiner, one link calibration).
#: Older checkpoints are upgraded on load by :func:`migrate_checkpoint`.
SCHEMA_VERSION = "2.0"

#: Checkpoint metadata about where the weights come from, kept on a loaded model (``model.meta``) so that a re-save
#: (:meth:`eomt.EoMT.save`) carries it over: the run that made them, the epoch and the metrics row it was saved at, and
#: the key ``best.pt`` was chosen on.
CARRIED_KEYS = ("run_config", "epoch", "metrics", "best_key", "best_metric")


def module_counts(keys) -> str:
    """``"eomt.xview (52), matcher (40)"``: state-dict keys counted per module (two levels under ``eomt.``)."""
    counts: dict[str, int] = {}
    for k in keys:
        parts = k.split(".")
        mod = ".".join(parts[:2]) if parts[0] == "eomt" and len(parts) > 2 else parts[0]
        counts[mod] = counts.get(mod, 0) + 1
    return ", ".join(f"{m} ({c})" for m, c in sorted(counts.items(), key=lambda kv: -kv[1]))


@functools.lru_cache(maxsize=1)
def get_version() -> str:
    """The version of the eomt code that is running, recorded in every checkpoint: the installed distribution's when
    this module is it; else (a source checkout or a copy of one) the ``pyproject.toml`` version next to the package,
    plus the git commit and ``.dirty`` when the package has uncommitted changes, e.g. ``3.0.0+g1a2b3c4.dirty``."""
    here = Path(__file__).resolve().parent
    try:
        dist = distribution("attr-eomt")
        if Path(str(dist.locate_file("eomt/__init__.py"))).resolve().parent == here:
            return dist.version
    except PackageNotFoundError:
        pass
    ver = "0.0.0"
    pyproject = here.parent / "pyproject.toml"
    if pyproject.is_file():
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
        ver = m.group(1) if m else ver
    try:
        git = functools.partial(subprocess.run, cwd=here, capture_output=True, text=True, timeout=5, check=False)
        sha = git(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
        dirty = bool(git(["git", "status", "--porcelain", "--", "."]).stdout.strip()) if sha else False
    except (OSError, subprocess.SubprocessError):
        sha, dirty = "", False
    return f"{ver}+g{sha}{'.dirty' if dirty else ''}" if sha else f"{ver}+unknown"


def normalize_names(names: Any, nc: int) -> dict[int, str]:
    """Normalize class names to a canonical ``{int: str}`` covering 0..nc-1."""
    if names is None:
        return {i: f"class_{i}" for i in range(nc)}
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise ValueError("names must be a dict[int, str] or list[str].")
    normalized = {int(k): str(v) for k, v in names.items()}
    missing = [i for i in range(nc) if i not in normalized]
    if missing:
        warnings.warn(
            f"names missing class indices {missing}; padding with class_i labels.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {i: normalized.get(i, f"class_{i}") for i in range(nc)}


def wrap_checkpoint(
    state_dict: dict[str, torch.Tensor],
    *,
    size: str,
    nc: int,
    imgsz: int,
    names: dict[int, str] | list[str] | None = None,
    task: str = "instance",
    family: str = "eomt",
    aux_heads: list | None = None,
    aux_head_arch: dict | None = None,
    letterbox: bool = False,
    loss_weights: dict | None = None,
    num_upscale_blocks: int | None = None,
    fpn_scales: tuple[float, ...] | list[float] | None = None,
    norm_mean: Any = None,
    norm_std: Any = None,
    patch_size: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a metadata-wrapped checkpoint that :func:`load_model` can restore.

    ``aux_head_arch`` records the secondary-head network shape (``layers`` /
    ``hidden`` / ``dropout``) so an MLP head is rebuilt identically on reload —
    without it a non-linear head would be silently dropped by ``strict=False``.
    ``loss_weights`` records the segmentation-loss/criterion weights so a tuned
    objective is rebuilt on reload (the criterion is built once from the config, so
    a bare reload would otherwise reset them to defaults). ``num_upscale_blocks``
    records the mask-head depth, which is load-bearing for the ``upscale_block``
    weight shapes.
    """
    checkpoint: dict[str, Any] = {
        "model": state_dict,
        "schema_version": SCHEMA_VERSION,
        "eomt_version": get_version(),
        "model_family": family,
        "size": size,
        "task": task,
        "nc": int(nc),
        "names": normalize_names(names, nc),
        "imgsz": int(imgsz),
        "letterbox": bool(letterbox),
        "patch_size": int(patch_size) if patch_size is not None else 14,
        # Pixel normalization travels with the checkpoint so preprocessing is fully
        # reproducible from the file alone (defaults to ImageNet).
        "norm_mean": [float(x) for x in (norm_mean if norm_mean is not None else IMAGENET_MEAN)],
        "norm_std": [float(x) for x in (norm_std if norm_std is not None else IMAGENET_STD)],
        "aux_heads": aux_specs_to_meta(aux_heads),
    }
    if aux_heads and aux_head_arch is not None:
        checkpoint["aux_head_arch"] = dict(aux_head_arch)
    if loss_weights is not None:
        checkpoint["loss_weights"] = dict(loss_weights)
    if num_upscale_blocks is not None:
        checkpoint["num_upscale_blocks"] = int(num_upscale_blocks)
    # Multi-scale (B1): load-bearing metadata — the conv weights don't encode which
    # level is upsample vs downsample, so the scales must be recorded to rebuild the
    # SimpleFPN + cross-attn modules with matching shapes on reload.
    if fpn_scales is not None:
        checkpoint["fpn_scales"] = [float(s) for s in fpn_scales]
    checkpoint.update({k: v for k, v in extra.items() if v is not None})
    return checkpoint


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    """Atomically write a checkpoint to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.rename(path)


def load_raw(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict from disk, as plain data: tensors and Python containers only (``weights_only``), so a
    checkpoint can never run code when it is loaded."""
    return torch.load(path, map_location=map_location, weights_only=True)


#: Default checkpoint filename used for Hugging Face Hub repos when none is given.
HF_DEFAULT_FILENAME = "model.pt"


def is_hf_ref(spec: str | Path) -> bool:
    """True if ``spec`` is a Hugging Face Hub reference (``hf://owner/repo[/file]``)."""
    return str(spec).startswith("hf://")


def download_from_hub(ref: str | Path, *, revision: str | None = None) -> Path:
    """Download a checkpoint from the Hugging Face Hub and return its local cached path.

    ``ref`` is ``hf://<owner>/<repo>`` (resolves to :data:`HF_DEFAULT_FILENAME`) or
    ``hf://<owner>/<repo>/<path/to/file.pt>`` for a specific file. The download is
    cached by ``huggingface_hub``, so repeated loads don't re-fetch.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "loading weights from the Hugging Face Hub needs 'huggingface_hub' "
            "(pip install huggingface_hub)."
        ) from e

    body = str(ref)[len("hf://"):].strip("/")
    parts = body.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"invalid hf reference {ref!r}; expected 'hf://<owner>/<repo>[/<file>]'."
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:]) or HF_DEFAULT_FILENAME
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))


def resolve_checkpoint(path: str | Path, *, prefer: str = "best") -> Path:
    """Resolve a checkpoint path that may be a file, a run/weights folder, or a Hub ref.

    A ``hf://owner/repo[/file]`` reference is downloaded from the Hugging Face Hub
    (cached) and the local path returned. A file is returned as-is. A directory is
    searched — itself and its ``weights/`` subdir — for ``best.pt`` / ``last.pt``.
    ``prefer`` sets the order: ``"best"`` for inference, ``"last"`` for resuming
    training. So ``runs/train/eomt`` (or ``runs/train/eomt/weights``) resolves to
    the right checkpoint without naming the file.
    """
    if is_hf_ref(path):
        return download_from_hub(path)
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        order = ("best.pt", "last.pt") if prefer == "best" else ("last.pt", "best.pt")
        for d in (p, p / "weights"):
            for fname in order:
                cand = d / fname
                if cand.is_file():
                    return cand
        raise FileNotFoundError(
            f"no {' or '.join(order)} found under {p} or {p / 'weights'}"
        )
    raise FileNotFoundError(f"checkpoint path does not exist: {p}")




def migrate_checkpoint(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Bring a checkpoint written by an older eomt to the current format (a new dict; the state dict is copied, its
    tensors are not). Every old-format rule is here, so the rest of the package reads one format:

    * metadata predating its field: the size, classes, input size and family read off the weights; single Linear
      attribute heads (``aux_head_arch``); stretch preprocessing (``letterbox``); ImageNet normalisation; the mask-head
      depth and the multi-scale levels counted from the weights;
    * the masked-attention probability buffer (``eomt.attn_mask_probs``) is dropped: checkpoints are saved mask-free;
    * the auxiliary box head is dropped with a warning (it only added boxes to the outputs);
    * link records of older versions (``reid_thres``, ``link_embed``, ``link_f1``) are dropped;
    * multi-view post-trainings before 3.1 (cross-view sublayers with a null token, an embedding matching head) and
      int8-compressed and pixel-detail checkpoints are refused.
    """
    if "model" not in ckpt:
        raise ValueError("not a wrapped EoMT checkpoint (no 'model' key)")
    ck = dict(ckpt)
    state = dict(ck["model"])
    if ck.get("compression"):
        raise ValueError("int8-compressed checkpoints are no longer supported: load the fp32 / bf16 one")
    if ck.get("detail") or any(k.startswith("eomt.detail.") for k in state):
        raise ValueError("this checkpoint uses the pixel-detail pathway, which eomt no longer has")
    for key, infer in (("size", _infer_size), ("nc", _infer_nc), ("imgsz", _infer_imgsz)):
        if not ck.get(key):
            ck[key] = infer(state)
    if not ck.get("task"):
        ck["task"] = "detect" if any(k.startswith("eomt.box_head.") for k in state) else "instance"
    if ck.get("aux_heads") and not ck.get("aux_head_arch"):
        ck["aux_head_arch"] = {"layers": 1}  # attribute heads predating the MLP were single Linear layers
    ck.setdefault("letterbox", False)  # stretch resize, before letterboxing existed
    ck.setdefault("norm_mean", list(IMAGENET_MEAN))
    ck.setdefault("norm_std", list(IMAGENET_STD))
    if not ck.get("num_upscale_blocks"):
        ck["num_upscale_blocks"] = _infer_num_upscale_blocks(state)
    if not ck.get("fpn_scales"):
        ck["fpn_scales"] = _infer_fpn_scales(state)
    if ck.get("schema_version") != SCHEMA_VERSION and ck["task"] != "detect":
        # the IoU-aware class targets and the quality head became defaults in schema 2.0: off where not recorded
        ck["loss_weights"] = {"iou_aware_cls": False, "quality_weight": 0.0, **(ck.get("loss_weights") or {})}
    state.pop("eomt.attn_mask_probs", None)
    if ck.get("xview") and any(k.startswith("eomt.xview.") and k.endswith(".null_logit") for k in state) or \
            ck.get("match_dim") or any(k.startswith(("match_head.", "match_skip.")) for k in state):
        raise ValueError("a multi-view post-training before attr-eomt 3.1 (cross-view null token / embedding matching "
                         "head): no longer supported; its base single-photo checkpoint still loads")
    if ck.pop("aux_box_head", None) or any(k.startswith("eomt.aux_box_head.") for k in state):
        warnings.warn("the auxiliary box head is no longer supported: its weights are ignored (it only added boxes "
                      "to the outputs; masks, classes and attributes are unchanged)", RuntimeWarning, stacklevel=3)
        state = {k: v for k, v in state.items() if not k.startswith("eomt.aux_box_head.")}
    for key in ("reid_thres", "link_embed", "link_f1", "match_dim", "match_residual"):
        ck.pop(key, None)
    ck["model"] = state
    ck["schema_version"] = SCHEMA_VERSION
    return ck


def load_model(path: str | Path, *, device: str = "auto") -> EoMTModel:
    """Rebuild an :class:`EoMTModel` from a checkpoint and load its weights (strictly: every weight must fit).

    ``path`` may be a checkpoint file, a run/weights folder (``best.pt`` is preferred) or a ``hf://`` reference.
    Checkpoints of older versions are upgraded first (:func:`migrate_checkpoint`).
    """
    path = resolve_checkpoint(path, prefer="best")
    ckpt = migrate_checkpoint(load_raw(path))
    model = build_model(
        ckpt["size"],
        nc=int(ckpt["nc"]),
        imgsz=int(ckpt["imgsz"]),
        names=ckpt.get("names"),
        family=ckpt["task"],
        aux_heads=aux_specs_from_meta(ckpt.get("aux_heads")),
        aux_head_arch=ckpt.get("aux_head_arch"),
        loss_weights=ckpt.get("loss_weights"),
        num_upscale_blocks=ckpt.get("num_upscale_blocks"),
        fpn_scales=ckpt.get("fpn_scales"),
        xview=ckpt.get("xview"),
        matcher=ckpt.get("matcher"),
    )
    state = ckpt["model"]
    if has_lora(state):  # a trainer state (last.pt) of a post-training with LoRA: its factors merged
        lora = ckpt.get("lora") or {}
        state = merge_state(state, scale=float(lora.get("alpha", 1)) / float(lora.get("rank", 1)))
    model.load_state_dict(state)
    model.eomt.set_attn_mask_probs([0.0] * model.config.num_blocks)  # checkpoints are saved and run mask-free
    model.match_thres = ckpt.get("match_thres")  # the links' calibration (eomt.engine.predict.link_defaults)
    model.link_group_by = tuple(ckpt["link_group_by"]) if ckpt.get("link_group_by") else None
    model.meta = {k: ckpt[k] for k in CARRIED_KEYS if ckpt.get(k) is not None}
    model.preprocess_letterbox = bool(ckpt["letterbox"])
    model.pixel_mean = tuple(float(x) for x in ckpt["norm_mean"])
    model.pixel_std = tuple(float(x) for x in ckpt["norm_std"])
    # How the model is meant to be run (the run config's `inference`: conf_thres, ...): the defaults of predict().
    model.inference = dict(ckpt.get("inference") or {})
    return model.to(resolve_device(device)).eval()


# --- metadata read off the weights, for checkpoints predating a field --------


def _infer_size(state: dict) -> str:
    key = "eomt.layernorm.weight"
    size = HIDDEN_TO_SIZE.get(int(state[key].shape[0])) if key in state else None
    if size is None:
        raise ValueError("could not infer the model size from the state dict")
    return size


def _infer_nc(state: dict) -> int:
    if "eomt.class_predictor.weight" not in state:
        raise ValueError("could not infer nc from the state dict")
    return int(state["eomt.class_predictor.weight"].shape[0]) - 1  # drop the +1 null class


def _infer_imgsz(state: dict, patch_size: int = 14) -> int:
    import math

    key = "eomt.embeddings.position_embeddings.weight"
    if key not in state:
        raise ValueError("could not infer imgsz from the state dict")
    return int(math.isqrt(int(state[key].shape[0]))) * patch_size


def _layer_count(state: dict, prefix: str) -> int:
    """``max(i) + 1`` over the keys ``<prefix><i>.``, 0 when there is none."""
    idxs = {int(k[len(prefix):].split(".", 1)[0]) for k in state
            if k.startswith(prefix) and k[len(prefix):].split(".", 1)[0].isdigit()}
    return max(idxs) + 1 if idxs else 0


def _infer_num_upscale_blocks(state: dict) -> int | None:
    """The mask-head upscale depth (``None`` without a mask head: the size preset's)."""
    return _layer_count(state, "eomt.upscale_block.block.") or None


def _infer_fpn_scales(state: dict) -> list[float] | None:
    """The multi-scale levels of an FPN checkpoint predating ``fpn_scales``: the weights only give their count, so the
    default scales when it matches."""
    from .model import DEFAULT_FPN_SCALES

    n = _layer_count(state, "eomt.fpn.blocks.")
    if not n:
        return None
    if n != len(DEFAULT_FPN_SCALES):
        raise ValueError(f"an FPN of {n} levels without its fpn_scales: cannot rebuild it")
    return list(DEFAULT_FPN_SCALES)


# --- checkpoint inspection --------------------------------------------------


def summarize_checkpoint(path: str | Path) -> dict[str, Any]:
    """Return a human-oriented summary of a checkpoint's metadata.

    Accepts a checkpoint file, a run/weights folder, or a ``hf://`` ref (resolved
    via :func:`resolve_checkpoint`). Metadata absent from older checkpoints is
    inferred from the state dict where possible.
    """
    resolved = resolve_checkpoint(path, prefer="best")
    raw = load_raw(resolved)
    ckpt = migrate_checkpoint(raw)
    state = ckpt["model"]
    summary: dict[str, Any] = {
        "path": str(resolved),
        "file_mb": round(resolved.stat().st_size / 1e6, 1),
        "num_tensors": len(state),
        "schema_version": raw.get("schema_version"),
        "eomt_version": ckpt.get("eomt_version"),
        "size": ckpt["size"],
        "task": ckpt["task"],
        "nc": ckpt["nc"],
        "imgsz": ckpt["imgsz"],
        "patch_size": ckpt.get("patch_size", 14),
        "letterbox": ckpt.get("letterbox"),
        "norm_mean": ckpt.get("norm_mean", list(IMAGENET_MEAN)),
        "norm_std": ckpt.get("norm_std", list(IMAGENET_STD)),
        "names": ckpt.get("names"),
        "aux_heads": ckpt.get("aux_heads"),
        "loss_weights": ckpt.get("loss_weights"),
        "num_upscale_blocks": ckpt.get("num_upscale_blocks"),
        "fpn_scales": ckpt.get("fpn_scales"),
        "xview": ckpt.get("xview"),
        "matcher": ckpt.get("matcher"),
        "match_thres": ckpt.get("match_thres"),
        "link_group_by": ckpt.get("link_group_by"),
        "dtype": ckpt.get("dtype"),
        # Where the weights come from: the epoch and metrics row they were saved at, best.pt's key, the run.
        "epoch": ckpt.get("epoch"),
        "best_metric": ckpt.get("best_metric"),
        "best_key": ckpt.get("best_key"),
        "metrics": ckpt.get("metrics"),
        "provenance": (ckpt.get("run_config") or {}).get("provenance"),
        "has_optimizer": "optimizer" in ckpt,
        "has_ema": "ema" in ckpt,
    }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """Pretty-print a :func:`summarize_checkpoint` dict as an aligned block."""
    names = summary.get("names") or {}
    if isinstance(names, dict):
        names_items = sorted(names.items(), key=lambda kv: int(kv[0]))
        names_str = ", ".join(f"{k}:{v}" for k, v in names_items)
    else:
        names_str = ", ".join(map(str, names))
    if len(names_str) > 200:
        names_str = names_str[:197] + "..."

    mean = ", ".join(f"{x:.3f}" for x in summary.get("norm_mean") or [])
    std = ", ".join(f"{x:.3f}" for x in summary.get("norm_std") or [])
    aux = summary.get("aux_heads") or []
    aux_str = ", ".join(
        f"{a.get('name')}({'regression' + (', log' if a.get('log') else '') if a.get('regression') else a.get('num_classes')})"
        for a in aux
    ) if aux else "none"

    lines = [
        f"checkpoint : {summary['path']}",
        f"  file     : {summary['file_mb']} MB, {summary['num_tensors']} tensors"
        f" (schema {summary.get('schema_version')}, eomt {summary.get('eomt_version')})",
        f"  model    : eomt-{summary.get('size')}  task={summary.get('task')}"
        f"  nc={summary.get('nc')}",
        f"  input    : imgsz={summary.get('imgsz')}  patch={summary.get('patch_size')}"
        f"  letterbox={summary.get('letterbox')}",
        f"  norm     : mean=[{mean}]  std=[{std}]",
        f"  aux heads: {aux_str}",
        f"  classes  : {names_str or '(none)'}",
    ]
    arch = []
    if summary.get("fpn_scales"):
        arch.append(f"FPN{tuple(summary['fpn_scales'])}")
    if summary.get("xview"):
        arch.append(f"cross-view (<= {summary['xview'].get('max_views')} photos)")
    if summary.get("matcher"):
        arch.append(f"set matcher {summary['matcher'].get('layers')} x {summary['matcher'].get('dim')}-d")
    if summary.get("match_thres") is not None:
        gate = summary.get("link_group_by")
        what = "probability" if summary.get("matcher") else "cosine"
        arch.append(f"links at {what} {summary['match_thres']:.2f}" + (f" within {', '.join(gate)}" if gate else ""))
    if summary.get("dtype"):
        arch.append(f"{summary['dtype']} Linear / Conv weights")
    if arch:
        lines.insert(5, "  extras   : " + ", ".join(arch))
    if summary.get("epoch") is not None or summary.get("has_optimizer"):
        bm = summary.get("best_metric")
        bm_str = f"{bm:.4f}" if isinstance(bm, (int, float)) else str(bm)
        key = summary.get("best_key")
        key = key if isinstance(key, str) or key is None else " + ".join(f"{w:g} x {k}" for k, w in key.items())
        lines.append(
            f"  training : epoch={summary.get('epoch')}  best_metric={bm_str}" + (f" ({key})" if key else "")
            + f"  optimizer={summary.get('has_optimizer')}  ema={summary.get('has_ema')}"
        )
    metrics = summary.get("metrics") or {}
    shown = [k for k in _HEADLINE if isinstance(metrics.get(k), (int, float)) and metrics[k] == metrics[k]]
    if shown:
        lines.append("  metrics  : " + ", ".join(f"{k[4:] if k.startswith('val/') else k}={metrics[k]:.4f}" for k in shown))
    prov = summary.get("provenance") or {}
    if prov.get("init_weights"):
        lines.append(f"  started  : {prov['init_weights']} (sha256 {str(prov.get('init_sha256'))[:12]})")
    for name, rec in (prov.get("data") or {}).items():
        lines.append(f"  data     : {name} {rec.get('path')} (sha256 {str(rec.get('sha256'))[:12]})")
    return "\n".join(lines)


#: The metrics :func:`format_summary` shows of a checkpoint's metrics row, when it has them.
_HEADLINE = ("val/segm/mAP", "val/bbox/mAP", "val/sets/segm/mAP_alone", "val/sets/segm/mAP_set",
             "val/sets/segm/mAP_linked_set", "val/sets/match/f1", "val/sets/match/cluster_f1")
