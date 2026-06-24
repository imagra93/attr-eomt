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
) -> dict:
    """Run the model on one PIL image and return a postprocess dict.

    Returns a box-only :func:`~eomt.postprocess.postprocess_detection` dict for
    ``family="detect"`` models, else a :func:`~eomt.postprocess.postprocess_instance`
    dict (with masks).
    """
    orig_w, orig_h = image.size
    chw, meta = preprocess_numpy(
        np.array(image.convert("RGB")), imgsz, letterbox=letterbox,
        mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None),
    )
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    out = model(tensor)
    if getattr(model, "family", "instance") == "detect":
        return postprocess_detection(
            out, conf_thres, (orig_w, orig_h), max_det=max_det, preprocess_meta=meta,
        )
    return postprocess_instance(
        out, conf_thres, (orig_w, orig_h), max_det=max_det,
        mask_thresh=mask_thresh, preprocess_meta=meta,
    )


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
    alpha: float = 0.5,
    draw_boxes: bool = True,
) -> list[dict]:
    """Run inference on an image or a directory of images.

    ``model`` may be a loaded :class:`~eomt.model.EoMTModel` or a checkpoint path /
    run folder (loaded with :func:`~eomt.serialization.load_model`). Returns one
    result dict per image (``boxes`` / ``scores`` / ``classes`` / ``masks`` and,
    for models with secondary heads, ``aux``), each annotated with its source
    ``path``. When ``plot`` is set, every image is rendered with masks/boxes/labels
    and written under ``save`` (default ``runs/predict``); the output path is added
    to the result dict as ``plot_path``.
    """
    if isinstance(model, (str, Path)):
        model = load_model(model, device=device)

    dev = next(model.parameters()).device
    imgsz = int(model.image_size)
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
            mask_thresh=mask_thresh, letterbox=letterbox,
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
                alpha=alpha, draw_boxes=draw_boxes,
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
