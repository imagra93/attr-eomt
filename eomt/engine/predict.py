"""Inference + rendering for EoMT instance segmentation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..postprocess import postprocess_detection, postprocess_instance
from ..preprocess import preprocess_numpy
from ..serialization import load_model
from ..tta import TTAConfig, predict_views
from ..visualize import draw_instances

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _iter_sources(source: str | Path):
    source = Path(source)
    if source.is_dir():
        yield from sorted(p for p in source.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    else:
        yield source


def _amp_dtype(device, amp):
    """The autocast dtype for ``amp`` on ``device`` (``None`` = run in the model's own precision).

    ``False`` / ``"off"`` -> None; ``True`` / ``"auto"`` -> bf16 where the GPU supports it, else fp16;
    ``"bf16"`` / ``"fp16"`` -> that dtype. Only CUDA autocasts.
    """
    if amp is False or amp is None or amp == "off":
        return None
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda":
        return None
    if amp is True or amp == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("amp='bf16' but this GPU does not support bf16.")
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    raise ValueError(f"amp must be False, True, 'auto', 'bf16' or 'fp16', got {amp!r}")


@torch.no_grad()
def predict_image(
    model,
    image: Image.Image,
    *,
    device,
    imgsz: int,
    conf_thres: float = 0.3,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    letterbox: bool = True,
    embed: bool = False,
    amp: bool | str = False,
    augment: bool | dict = False,
    tiles: bool | int | dict = False,
) -> dict:
    """Run the model on one PIL image and return a postprocess dict.

    Returns a box-only :func:`~eomt.postprocess.postprocess_detection` dict for
    ``family="detect"`` models, else a :func:`~eomt.postprocess.postprocess_instance`
    dict (with masks).

    ``amp`` runs the network under autocast (``True`` / ``"auto"`` = bf16 where supported, or
    ``"bf16"`` / ``"fp16"``): ~2.5x faster at 644 px on an RTX 5090 with the same accuracy (AP
    within 0.002 in every size bucket on a trained ViT-L model). Off by default.

    With ``embed``, the result also carries ``"embed"`` ``(N, hidden)``: the per-query
    embedding of each kept detection, sliced out of the same forward pass. It is the
    appearance fingerprint cross-photo re-identification matches on
    (:mod:`eomt.reid`). The vectors are **raw**, not normalized — normalizing (and
    optionally mean-centering) is :func:`eomt.reid.similarity_matrix`'s job, and it
    needs the raw vectors to do it.

    ``augment=True`` adds the image zoomed to 1.5x ``imgsz`` (test-time augmentation) and ``tiles=True`` runs overlapping
    native-resolution tiles plus the whole image (tiled inference); the views' instances are merged. Each also takes a
    dict of options (``tiles`` an int tile size); see :mod:`eomt.tta`. Instance family only. The result then has no
    ``query_idx`` (a merged instance comes from several queries); ``embed`` is the score-weighted mean of the merged
    queries' embeddings.
    """
    # Class-scoped aux heads: pass their primary-class scope so postprocess emits
    # ``ids = -1`` for detections the head does not apply to (the inference side of
    # the hard class-routing). Unscoped heads (``applies_to=None``) are omitted here.
    aux_scopes = {
        s.name: s.applies_to
        for s in getattr(model, "aux_specs", [])
        if s.applies_to is not None
    }
    cfg = TTAConfig.resolve(augment, tiles)
    if cfg is not None:
        return predict_views(
            model, np.array(image.convert("RGB")), cfg, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det,
            mask_thresh=mask_thresh, letterbox=letterbox, mean=getattr(model, "pixel_mean", None),
            std=getattr(model, "pixel_std", None), amp_dtype=_amp_dtype(device, amp), aux_scopes=aux_scopes,
            embed=embed,
        )
    orig_w, orig_h = image.size
    chw, meta = preprocess_numpy(
        np.array(image.convert("RGB")), imgsz, letterbox=letterbox,
        mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None),
    )
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    dtype = _amp_dtype(device, amp)
    if dtype is None:
        out = model(tensor)
    else:  # only the network runs in reduced precision; postprocess casts the logits back to fp32
        with torch.autocast("cuda", dtype=dtype):
            out = model(tensor)
    if getattr(model, "family", "instance") == "detect":
        result = postprocess_detection(
            out, conf_thres, (orig_w, orig_h), max_det=max_det, preprocess_meta=meta,
            aux_scopes=aux_scopes,
        )
    else:
        result = postprocess_instance(
            out, conf_thres, (orig_w, orig_h), max_det=max_det,
            mask_thresh=mask_thresh, preprocess_meta=meta, aux_scopes=aux_scopes,
        )
    if embed:
        # Index with the returned ``query_idx``, never re-derive from the scores: the
        # ``max_det`` topk reorders rows, so only ``query_idx`` is guaranteed aligned
        # with the detections. ``.float()`` because the forward may run under autocast;
        # ``.cpu()`` so a folder's worth of embeddings does not pin GPU memory.
        qe = out.get("query_embed")
        result["embed"] = (
            qe[0, result["query_idx"]].detach().float().cpu()
            if qe is not None
            else torch.zeros((result["num_detections"], 0))
        )
    return result


def predict(
    model,
    source: str,
    *,
    plot: bool = False,
    save: str | None = "runs/predict",
    conf_thres: float = 0.3,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    device: str = "auto",
    alpha: float = 0.35,
    draw_boxes: bool = True,
    aux_multiline: bool = True,
    show_scores: bool = True,
    color_by: str = "class",
    imgsz: int | None = None,
    embed: bool = False,
    amp: bool | str = False,
    augment: bool | dict = False,
    tiles: bool | int | dict = False,
) -> list[dict]:
    """Run inference on an image or a directory of images.

    ``model`` may be a loaded :class:`~eomt.model.EoMTModel` or a checkpoint path /
    run folder (loaded with :func:`~eomt.serialization.load_model`). Returns one
    result dict per image (``boxes`` / ``scores`` / ``classes`` / ``masks`` and,
    for models with secondary heads, ``aux``), each annotated with its source
    ``path``. When ``plot`` is set, every image is rendered with masks/boxes/labels
    and written under ``save`` (default ``runs/predict``); the output path is added
    to the result dict as ``plot_path``.

    With ``embed``, each result also carries ``"embed"`` ``(N, hidden)`` — the
    per-instance appearance fingerprint (see :func:`predict_image`). ``amp`` runs the
    network under autocast; ``augment`` / ``tiles`` run test-time augmentation / tiled
    inference (see :func:`predict_image`).
    """
    if isinstance(model, (str, Path)):
        model = load_model(model, device=device)

    dev = next(model.parameters()).device
    imgsz = int(imgsz if imgsz is not None else model.image_size)
    letterbox = bool(getattr(model, "preprocess_letterbox", False))
    names = getattr(model, "names", None)
    aux_names = {s.name: s.names for s in getattr(model, "aux_specs", [])}

    family = getattr(model, "family", "instance")
    print(
        f"[predict] eomt-{getattr(model, 'size', '?')} ({family}, "
        f"nc={getattr(model, 'nc', '?')}, imgsz={imgsz}) on {dev}"
    )

    out_root = Path(save) if (plot and save) else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total_t0 = time.perf_counter()
    for path in _iter_sources(source):
        image = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        result = predict_image(
            model, image, device=dev, imgsz=imgsz,
            conf_thres=conf_thres, max_det=max_det,
            mask_thresh=mask_thresh, letterbox=letterbox, embed=embed, amp=amp,
            augment=augment, tiles=tiles,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        result["elapsed_ms"] = elapsed_ms
        result["path"] = str(path)
        dst = None
        if plot:
            rendered = draw_instances(
                image, result, names=names, aux_names=aux_names or None,
                alpha=alpha, draw_boxes=draw_boxes, aux_multiline=aux_multiline,
                show_scores=show_scores, color_by=color_by,
            )
            if out_root is not None:
                dst = out_root / path.name
                rendered.save(dst)
                result["plot_path"] = str(dst)
            result["plot"] = rendered
        saved = f" -> {dst}" if dst is not None else ""
        print(
            f"[predict] {path.name}: {result['num_detections']} instances "
            f"({elapsed_ms:.1f} ms){saved}"
        )
        results.append(result)

    n = len(results)
    if n:
        total_s = time.perf_counter() - total_t0
        avg_ms = total_s * 1e3 / n
        fps = n / total_s if total_s > 0 else float("inf")
        dest = f" -> {out_root}" if out_root is not None else ""
        print(
            f"[predict] done: {n} image(s) in {total_s:.2f} s "
            f"({avg_ms:.1f} ms/img, {fps:.1f} FPS){dest}"
        )
    return results
