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


def _aux_label(
    result: dict,
    aux_names: dict[str, dict[int, str]] | None,
    i: int,
    *,
    multiline: bool = True,
    show_scores: bool = True,
) -> str:
    """Build the secondary-class text for instance ``i``.

    With ``multiline`` (default) each subclass gets its own indented, bulleted row,
    prefixed by the head name and with the probability in parentheses, e.g.::

          • scale: large (1.00)
          • occlusion: slight (0.90)

    With ``show_scores=False`` the parenthesised probabilities are omitted.
    Otherwise the legacy compact form is used (all heads on one space-joined row).
    """
    aux = result.get("aux")
    if not aux:
        return ""
    rows = []
    for head, pred in aux.items():
        idx = int(pred["ids"][i])
        prob = float(pred["probs"][i][idx])
        label = str(idx)
        if aux_names and head in aux_names:
            label = aux_names[head].get(idx, label)
        prob_txt = f" ({prob:.2f})" if show_scores else ""
        rows.append(f"  • {head}: {label}{prob_txt}" if multiline else f"{label}{prob_txt}")
    return "\n".join(rows) if multiline else "  ".join(rows)


def draw_instances(
    image: Image.Image,
    result: dict,
    names: dict[int, str] | None = None,
    *,
    aux_names: dict[str, dict[int, str]] | None = None,
    alpha: float = 0.35,
    draw_boxes: bool = True,
    aux_multiline: bool = True,
    show_scores: bool = True,
    color_by: str = "class",
) -> Image.Image:
    """Overlay masks, boxes and labels from a ``postprocess_instance`` result.

    Args:
        image: source PIL image (RGB).
        result: dict with ``masks`` ``(N,H,W)``, ``boxes`` ``(N,4)``, ``scores``,
            ``classes`` (cpu tensors), at the image's original resolution. May
            carry ``aux`` ``{head: {"ids", "probs"}}`` from secondary heads.
        names: optional ``{class_index: name}`` mapping for labels.
        aux_names: optional ``{head: {id: name}}`` mapping for attribute labels.
        alpha: mask fill opacity in ``[0, 1]`` (lower = more transparent).
        aux_multiline: render each secondary class on its own row, prefixed by the
            head name and with its probability in parentheses; ``False`` falls back
            to the compact single-row form.
        show_scores: include the primary score and attribute probabilities in the
            labels; ``False`` shows only the class/attribute names.
        color_by: ``"class"`` colors masks/boxes by class index (instances of the
            same class share a color); ``"instance"`` colors each instance distinctly
            by its position, so same-class instances stand apart.
    """
    def _color(i: int, cls: int) -> tuple[int, int, int]:
        return class_color(i if color_by == "instance" else cls)

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
        color = np.array(_color(i, int(classes[i])), dtype=np.float32)
        img[m] = img[m] * (1.0 - alpha) + color * alpha

    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    line_w = max(2, int(round(max(out.size) / 320)))
    font = _font(max(11, int(round(max(out.size) / 80))))

    W, H = out.size
    placed: list[tuple[float, float, float, float]] = []  # (x, y, w, h) of drawn labels

    def _place(x: float, y: float, w: float, h: float) -> tuple[float, float]:
        """Clamp a label box into the image and push it below any it overlaps."""
        x = max(0.0, min(x, W - w))
        y = max(0.0, min(y, H - h))
        for _ in range(2 * max(1, n)):  # bounded; resolves the few real collisions
            hit = next(
                (p for p in placed if x < p[0] + p[2] and x + w > p[0]
                 and y < p[1] + p[3] and y + h > p[1]),
                None,
            )
            if hit is None:
                break
            y = hit[1] + hit[3]
            if y + h > H:
                y = max(0.0, H - h)
                break
        placed.append((x, y, w, h))
        return x, y

    for i in range(n):
        cls = int(classes[i])
        color = _color(i, cls)
        label = names.get(cls, str(cls)) if names else str(cls)
        # primary class + score on the first row; secondary classes on the rows below
        if show_scores:
            score = float(scores[i])
            text = f"{label} ({score:.2f})" if aux_multiline else f"{label} {score:.2f}"
        else:
            text = label
        aux = _aux_label(result, aux_names, i, multiline=aux_multiline, show_scores=show_scores)
        if aux:
            text += "\n" + aux
        if draw_boxes:
            x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
            tb = draw.multiline_textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            lx, ly = _place(x1, y1, tw, th)
            draw.rectangle([lx, ly, lx + tw, ly + th], fill=color)
            draw.multiline_text((lx - tb[0], ly - tb[1]), text, fill=(0, 0, 0), font=font)

    return out
