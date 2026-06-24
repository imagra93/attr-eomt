"""Lightweight PIL rendering of instance-segmentation results."""

from __future__ import annotations

import colorsys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def class_color(idx: int) -> tuple[int, int, int]:
    """Stable, well-spread RGB color for a class index (golden-angle hue)."""
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _aux_label(result: dict, aux_names: dict[str, dict[int, str]] | None, i: int) -> str:
    """Build the secondary-class text for instance ``i`` (rendered on its own row)."""
    aux = result.get("aux")
    if not aux:
        return ""
    parts = []
    for head, pred in aux.items():
        idx = int(pred["ids"][i])
        prob = float(pred["probs"][i][idx])
        label = str(idx)
        if aux_names and head in aux_names:
            label = aux_names[head].get(idx, label)
        parts.append(f"{label} {prob:.2f}")
    return "  ".join(parts)


def draw_instances(
    image: Image.Image,
    result: dict,
    names: dict[int, str] | None = None,
    *,
    aux_names: dict[str, dict[int, str]] | None = None,
    alpha: float = 0.5,
    draw_boxes: bool = True,
) -> Image.Image:
    """Overlay masks, boxes and labels from a ``postprocess_instance`` result.

    Args:
        image: source PIL image (RGB).
        result: dict with ``masks`` ``(N,H,W)``, ``boxes`` ``(N,4)``, ``scores``,
            ``classes`` (cpu tensors), at the image's original resolution. May
            carry ``aux`` ``{head: {"ids", "probs"}}`` from secondary heads.
        names: optional ``{class_index: name}`` mapping for labels.
        aux_names: optional ``{head: {id: name}}`` mapping for attribute labels.
    """
    img = np.array(image.convert("RGB")).astype(np.float32)
    masks = result.get("masks")  # absent for detection (box) models
    boxes = result["boxes"]
    scores = result["scores"]
    classes = result["classes"]

    have_masks = isinstance(masks, torch.Tensor)
    n = int(masks.shape[0]) if have_masks else int(result.get("num_detections", len(boxes)))
    # Composite colored masks (skipped when the model emits boxes only).
    for i in range(n) if have_masks else ():
        m = masks[i].cpu().numpy().astype(bool)
        if not m.any():
            continue
        color = np.array(class_color(int(classes[i])), dtype=np.float32)
        img[m] = img[m] * (1.0 - alpha) + color * alpha

    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    line_w = max(2, int(round(max(out.size) / 320)))
    font = _font(max(11, int(round(max(out.size) / 80))))

    for i in range(n):
        cls = int(classes[i])
        color = class_color(cls)
        label = names.get(cls, str(cls)) if names else str(cls)
        # primary class + score on the first row; secondary classes on the row below
        text = f"{label} {float(scores[i]):.2f}"
        aux = _aux_label(result, aux_names, i)
        if aux:
            text += "\n" + aux
        if draw_boxes:
            x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
            tb = draw.multiline_textbbox((x1, y1), text, font=font)
            draw.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
            draw.multiline_text((x1, y1), text, fill=(0, 0, 0), font=font)

    return out
