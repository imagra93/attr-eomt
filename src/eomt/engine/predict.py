"""Inference + rendering for EoMT instance segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..postprocess import postprocess_instance
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
) -> dict:
    """Run the model on one PIL image and return a ``postprocess_instance`` dict."""
    orig_w, orig_h = image.size
    chw, _ = preprocess_numpy(np.array(image.convert("RGB")), imgsz)
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    out = model(tensor)
    return postprocess_instance(
        out, conf_thres, (orig_w, orig_h), max_det=max_det, mask_thresh=mask_thresh
    )


def predict(
    weights: str,
    source: str,
    *,
    out_dir: str = "runs/predict",
    conf_thres: float = 0.3,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    device: str = "auto",
    alpha: float = 0.5,
    draw_boxes: bool = True,
) -> list[str]:
    """Render predictions for an image or a directory of images.

    Returns the list of written annotated-image paths.
    """
    model = load_model(weights, device=device)
    dev = next(model.parameters()).device
    imgsz = int(model.image_size)
    names = getattr(model, "names", None)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for path in _iter_sources(source):
        image = Image.open(path).convert("RGB")
        result = predict_image(
            model,
            image,
            device=dev,
            imgsz=imgsz,
            conf_thres=conf_thres,
            max_det=max_det,
            mask_thresh=mask_thresh,
        )
        rendered = draw_instances(
            image, result, names=names, alpha=alpha, draw_boxes=draw_boxes
        )
        dst = out_root / path.name
        rendered.save(dst)
        written.append(str(dst))
        print(f"[predict] {path.name}: {result['num_detections']} instances -> {dst}")

    return written
