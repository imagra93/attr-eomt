"""Lightweight PIL rendering of instance-segmentation results."""

from __future__ import annotations

import colorsys
import math
from collections.abc import Sequence

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
        if idx < 0:  # -1 sentinel: class-scoped head does not apply to this instance
            continue
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
    color_ids: Sequence[int] | None = None,
    label_prefix: Sequence[str] | None = None,
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
        color_ids: optional per-instance color index, overriding ``color_by``. Pass
            cross-photo identity ids to color each instance by *which object it is*
            rather than what class it is (see :func:`draw_identity_grid`).
        label_prefix: optional per-instance string prepended to the first label row,
            e.g. ``"#3"`` for an identity id.
    """
    def _color(i: int, cls: int) -> tuple[int, int, int]:
        if color_ids is not None:
            return class_color(int(color_ids[i]))
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
        if label_prefix is not None:
            text = f"{label_prefix[i]} {text}"
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


# ---------------------------------------------------------------------------
# Cross-photo identity grid
# ---------------------------------------------------------------------------
def _centroid(result: dict, i: int) -> tuple[float, float] | None:
    """Mask centroid for instance ``i``, falling back to the box center."""
    masks = result.get("masks")
    if isinstance(masks, torch.Tensor) and i < masks.shape[0]:
        m = masks[i].cpu().numpy().astype(bool)
        if m.any():
            ys, xs = np.nonzero(m)
            return float(xs.mean()), float(ys.mean())
    boxes = result.get("boxes")
    if boxes is not None and i < len(boxes):
        x1, y1, x2, y2 = (float(v) for v in boxes[i].tolist())
        return (x1 + x2) / 2, (y1 + y2) / 2
    return None


def _arrow(draw, p0, p1, color, width: int, head: int | None = None) -> None:
    """A double-headed connector: line plus a filled triangle at each end."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:  # both instances land on the same canvas point
        return
    ux, uy = dx / length, dy / length
    h = head if head is not None else max(6, 4 * width)
    h = min(h, length / 2)
    px, py = -uy, ux  # perpendicular
    # Shorten the stroke so it does not poke through either head.
    draw.line(
        [(x0 + 0.8 * h * ux, y0 + 0.8 * h * uy), (x1 - 0.8 * h * ux, y1 - 0.8 * h * uy)],
        fill=color, width=width,
    )
    for (tx, ty), (sx, sy) in (((x1, y1), (ux, uy)), ((x0, y0), (-ux, -uy))):
        draw.polygon(
            [
                (tx, ty),
                (tx - h * sx + 0.4 * h * px, ty - h * sy + 0.4 * h * py),
                (tx - h * sx - 0.4 * h * px, ty - h * sy - 0.4 * h * py),
            ],
            fill=color,
        )


def draw_identity_grid(
    panels: list[dict],
    *,
    names: dict[int, str] | None = None,
    aux_names: dict[str, dict[int, str]] | None = None,
    arrows: list[dict] | None = None,
    identities: list[dict] | None = None,
    cols: int | None = None,
    panel_size: int = 480,
    pad: int = 12,
    caption_h: int | None = None,
    bg: tuple[int, int, int] = (22, 22, 26),
    fg: tuple[int, int, int] = (235, 235, 235),
    alpha: float = 0.35,
    draw_boxes: bool = True,
    show_scores: bool = False,
    aux_multiline: bool = False,
    arrow_alpha: float = 0.55,
    arrow_width: tuple[int, int] | None = None,
    sim_range: tuple[float, float] | None = None,
    legend: bool = True,
    legend_cols: int = 3,
    legend_attr: str | None = None,
    max_legend: int = 24,
    title: str | None = None,
) -> Image.Image:
    """Render one grid image for a set of photos linked by cross-photo identity.

    Each panel is one photo with its instances outlined and colored by identity id
    (so the same physical object wears the same color in every photo it appears in),
    labelled ``#id class``. Connectors are drawn between the matched instances of an
    identity, so the actual pairwise links are visible and not just the final
    clustering. A legend strip summarizes each identity.

    Args:
        panels: one dict per photo, ``{"image": PIL.Image, "result": dict,
            "identity_ids": sequence[int], "caption": str}``. Images and results must
            already be at panel scale — see the note below.
        draw_boxes: :func:`draw_instances` renders labels only alongside boxes, so
            turning this off also hides the ``#id class`` labels the grid exists to
            show. Leave it on unless you want outlines alone.
        aux_names: pass to label each instance with its attributes too. ``None``
            (the default from :func:`eomt.engine.match.match`) keeps panel labels to
            ``#id class``; the legend already carries the dominant attribute.
        legend_attr: which attribute head to show beside the class in the legend.
            ``None`` (default) uses each identity's first attribute, which is the
            first head the checkpoint declares.
        arrows: connectors to draw, each ``{"a_panel", "a_det", "b_panel", "b_det",
            "similarity", "identity_id"}``. Stroke thickness scales with similarity:
            a confident match is drawn thick, a marginal one hairline.
        arrow_width: ``(thinnest, thickest)`` stroke width in pixels. ``None``
            derives a pair from ``panel_size`` so arrows stay proportionate.
        sim_range: the ``(low, high)`` similarity band mapped onto ``arrow_width``.
            Pass ``(sim_thres, 1.0)`` — as :func:`eomt.engine.match.match` does — to
            make widths **absolute**, so the same similarity is the same thickness in
            every grid and a link drawn at threshold is visibly hairline. ``None``
            falls back to min-maxing over the arrows actually drawn, which maximizes
            contrast within one grid but means nothing across grids (and makes a lone
            arrow thinnest regardless of how good the match is).
        identities: records from :func:`eomt.reid.summarize_identities`, used for the
            legend only.

    Note:
        Panels must be pre-scaled by the caller. :func:`draw_instances` derives its
        line width and font size from the image it is handed, so drawing on a 4000 px
        photo and downscaling afterwards turns every label to mush.

    Note:
        Identity colors come from :func:`class_color`, a golden-angle hue ramp. Past
        roughly 20 identities adjacent hues stop being distinguishable in a small
        legend swatch, so the ``#id`` text is the real key, not the color.
    """
    n = len(panels)
    cols = cols or max(1, math.ceil(math.sqrt(max(n, 1))))
    rows = max(1, math.ceil(n / cols)) if n else 1
    caption_h = caption_h or max(16, panel_size // 22)
    cell_h = panel_size + caption_h
    cap_font = _font(max(10, caption_h - 5))

    legend_rows = 0
    swatch = 14
    legend_row_h = swatch + 10
    shown = (identities or [])[:max_legend] if legend else []
    if shown:
        legend_rows = math.ceil(len(shown) / legend_cols) + (
            1 if identities and len(identities) > max_legend else 0
        )
    title_h = (max(18, panel_size // 18) + pad) if title else 0
    legend_h = (legend_rows * legend_row_h + 2 * pad) if legend_rows else 0

    W = cols * (panel_size + pad) + pad
    H = title_h + rows * (cell_h + pad) + pad + legend_h
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((pad, pad // 2), title, fill=fg, font=_font(max(14, panel_size // 22)))

    # --- panels -----------------------------------------------------------
    origins: list[tuple[int, int]] = []
    for k, panel in enumerate(panels):
        r, c = divmod(k, cols)
        ox = pad + c * (panel_size + pad)
        oy = title_h + pad + r * (cell_h + pad)
        image, result = panel["image"], panel["result"]
        if aux_names is None:
            # _aux_label falls back to raw numeric ids when it has no name map, which
            # would tack an unreadable "0 0 1 4 3" row onto every label. No name map
            # here means "no attribute rows", so drop the key outright.
            result = {k: v for k, v in result.items() if k != "aux"}
        ids = [int(v) for v in panel.get("identity_ids", [])]
        rendered = draw_instances(
            image, result, names=names, aux_names=aux_names,
            alpha=alpha, draw_boxes=draw_boxes, aux_multiline=aux_multiline,
            show_scores=show_scores,
            color_ids=ids or None,
            label_prefix=[f"#{i}" for i in ids] or None,
        )
        # Center the (already panel-scaled) photo in its cell.
        px = ox + (panel_size - rendered.width) // 2
        py = oy + (panel_size - rendered.height) // 2
        canvas.paste(rendered, (px, py))
        origins.append((px, py))
        caption = str(panel.get("caption", ""))
        if caption:
            draw.text((ox + 2, oy + panel_size + 2), caption, fill=fg, font=cap_font)

    # --- connectors -------------------------------------------------------
    # Drawn on an overlay and composited, so they never fully obliterate the
    # instances they point at.
    if arrows:
        # Cosine similarities bunch up well below 1.0 in practice (a strong match is
        # ~0.85, not ~0.99), so the top of the band is rarely reached — a generous
        # span is what makes "thick = confident" legible at a glance.
        w_min, w_max = arrow_width or (
            max(1, panel_size // 240), max(4, panel_size // 40)
        )
        if sim_range is not None:
            lo, hi = (float(v) for v in sim_range)
        else:
            sims = [float(a.get("similarity", 0.0)) for a in arrows]
            lo, hi = min(sims), max(sims)
        span = (hi - lo) or 1.0
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        for a in arrows:
            ia, ib = int(a["a_panel"]), int(a["b_panel"])
            if not (0 <= ia < n and 0 <= ib < n):
                continue
            ca = _centroid(panels[ia]["result"], int(a["a_det"]))
            cb = _centroid(panels[ib]["result"], int(a["b_det"]))
            if ca is None or cb is None:
                continue
            p0 = (origins[ia][0] + ca[0], origins[ia][1] + ca[1])
            p1 = (origins[ib][0] + cb[0], origins[ib][1] + cb[1])
            norm = min(1.0, max(0.0, (float(a.get("similarity", 0.0)) - lo) / span))
            color = (*class_color(int(a.get("identity_id", 0))), int(255 * arrow_alpha))
            _arrow(odraw, p0, p1, color, max(1, round(w_min + (w_max - w_min) * norm)))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)

    # --- legend -----------------------------------------------------------
    if shown:
        lfont = _font(max(11, swatch))
        base_y = title_h + rows * (cell_h + pad) + pad
        col_w = (W - 2 * pad) // max(1, legend_cols)
        for i, rec in enumerate(shown):
            r, c = divmod(i, legend_cols)
            x = pad + c * col_w
            y = base_y + r * legend_row_h
            cid = int(rec["identity_id"])
            draw.rectangle([x, y, x + swatch, y + swatch], fill=class_color(cid))
            attrs = rec.get("attributes") or {}
            # Which attribute to put beside the class. Defaulting to the first one
            # keeps this working on any checkpoint; naming a head that does not exist
            # (or that this identity has no value for) just omits it.
            key = legend_attr if legend_attr is not None else next(iter(attrs), None)
            value = attrs.get(key) if key is not None else None
            label = f"#{cid}  {rec.get('class_name', '?')}"
            if value:
                label += f"/{value}"
            label += f"  ·  {rec.get('num_photos', 0)}/{n} photos"
            draw.text((x + swatch + 6, y), label, fill=fg, font=lfont)
        if identities and len(identities) > max_legend:
            y = base_y + math.ceil(len(shown) / legend_cols) * legend_row_h
            draw.text(
                (pad, y), f"+{len(identities) - max_legend} more identities",
                fill=fg, font=lfont,
            )
    return canvas
