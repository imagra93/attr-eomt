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
from ..visualize import draw_instances

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _iter_sources(source: str | Path):
    source = Path(source)
    if source.is_dir():
        yield from sorted(p for p in source.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    else:
        yield source


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
) -> dict:
    """Run the model on one PIL image and return a postprocess dict.

    Returns a box-only :func:`~eomt.postprocess.postprocess_detection` dict for
    ``family="detect"`` models, else a :func:`~eomt.postprocess.postprocess_instance`
    dict (with masks).

    With ``embed``, the result also carries ``"embed"`` ``(N, hidden)``: the per-query
    embedding of each kept detection, sliced out of the same forward pass. It is the
    appearance fingerprint cross-photo re-identification matches on
    (:mod:`eomt.reid`). The vectors are **raw**, not normalized — normalizing (and
    optionally mean-centering) is :func:`eomt.reid.similarity_matrix`'s job, and it
    needs the raw vectors to do it.
    """
    orig_w, orig_h = image.size
    chw, meta = preprocess_numpy(
        np.array(image.convert("RGB")), imgsz, letterbox=letterbox,
        mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None),
    )
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    out = model(tensor)
    # Class-scoped aux heads: pass their primary-class scope so postprocess emits
    # ``ids = -1`` for detections the head does not apply to (the inference side of
    # the hard class-routing). Unscoped heads (``applies_to=None``) are omitted here.
    aux_scopes = {
        s.name: s.applies_to
        for s in getattr(model, "aux_specs", [])
        if s.applies_to is not None
    }
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
    per-instance appearance fingerprint (see :func:`predict_image`).
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
            mask_thresh=mask_thresh, letterbox=letterbox, embed=embed,
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
