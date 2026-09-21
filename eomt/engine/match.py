"""Cross-photo instance re-identification over a set of photos of one subject.

Runs the still-image inference (:func:`~eomt.engine.predict.predict_image`) once per
photo, asks :mod:`eomt.reid` which detections across photos are the same physical
instance, and (optionally) renders a grid image plus a JSON summary.

This is the photo-set counterpart to :func:`~eomt.engine.track.track`: the tracker
associates across *video frames* using motion plus appearance, this associates across
*unordered photos* using appearance alone, since there is no motion prior between two
shots taken from opposite sides of a subject.

One call handles one set of photos of one subject. Multiple subjects are the caller's
loop — see ``scripts/match.py --each-subdir``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..reid import (
    cluster,
    flatten_instances,
    gate_mask,
    pairwise_matches,
    similarity_diagnostics,
    similarity_matrix,
    summarize_identities,
    validate_group_by,
)
from ..serialization import load_model
from ..visualize import draw_identity_grid
from .predict import _IMAGE_EXTS, _iter_sources, predict_image


def _panelize(image: Image.Image, result: dict, panel_size: int) -> tuple[Image.Image, dict]:
    """Downscale a photo and its masks/boxes to panel scale for rendering.

    Rendering has to happen at panel scale, not full resolution: ``draw_instances``
    sizes its strokes and font from the image it is handed. Doing the downscale here
    also keeps full-resolution masks from being retained for the whole run.
    """
    w, h = image.size
    scale = min(panel_size / max(w, h), 1.0)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = image.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (w, h) else image.copy()

    out = {k: v for k, v in result.items() if k not in ("masks", "boxes", "plot", "embed")}
    boxes = result["boxes"]
    out["boxes"] = boxes * scale if len(boxes) else boxes
    masks = result.get("masks")
    if isinstance(masks, torch.Tensor) and masks.numel():
        # NEAREST, not bilinear: interpolating a bool mask and re-binarizing shifts
        # the boundary and leaves a gray fringe.
        arr = masks.cpu().numpy()
        resized = np.stack([
            np.asarray(
                Image.fromarray(m.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)
            ) > 127
            for m in arr
        ]) if arr.shape[0] else np.zeros((0, nh, nw), dtype=bool)
        out["masks"] = torch.from_numpy(resized)
    return small, out


def _print_summary(identities: list[dict], num_images: int, top: int = 15) -> None:
    if not identities:
        return
    print(f"[match] top identities by photo count (of {len(identities)}):")
    for rec in identities[:top]:
        attrs = " ".join(f"{k}={v}" for k, v in (rec.get("attributes") or {}).items())
        sim = rec.get("mean_similarity")
        sim_txt = f" sim~{sim:.2f}" if sim is not None else ""
        print(
            f"  #{rec['identity_id']:<3} {rec['class_name']:<18} "
            f"{rec['num_photos']}/{num_images} photos{sim_txt}  {attrs}"
        )


def match(
    model,
    source: str,
    *,
    plot: bool = True,
    save: str | None = "runs/match",
    conf_thres: float = 0.3,
    max_det: int = 30,
    mask_thresh: float = 0.5,
    device: str = "auto",
    imgsz: int | None = None,
    group_by: tuple[str, ...] | None = ("class",),
    sim_thres: float = 0.6,
    center: bool = False,
    guard: str = "mean",
    guard_factor: float = 1.0,
    keep_masks: bool = False,
    panel_size: int = 480,
    cols: int | None = None,
    arrow_mode: str = "chain",
    legend_attr: str | None = None,
    alpha: float = 0.35,
    show_scores: bool = False,
) -> dict:
    """Re-identify instances across a set of photos of the same subject.

    Args:
        source: a folder of photos (non-recursive) or a single image.
        device: only consulted when ``model`` is a path to load; ignored when an
            already-loaded model is passed (as :meth:`eomt.EoMT.infer_match` does).
        group_by: gate keys — two instances may only match if they agree on all of
            them. ``("class",)`` by default; add aux head names to tighten, which
            matters when the primary class is coarser than the distinction you care
            about (e.g. ``("class", "position")`` so instances at different places on
            the subject never match, however alike they look). ``None`` disables
            gating, which needs a stricter ``sim_thres`` since far more pairs then
            compete.
        sim_thres: cosine-similarity floor for "same instance".
        center: mean-center embeddings before normalizing (transductive; see
            :func:`eomt.reid.similarity_matrix`).
        guard / guard_factor: merge guard against transitive chaining
            (:func:`eomt.reid.cluster`).
        keep_masks: retain full-resolution masks in the returned per-photo dicts.
            Off by default — 30 photos of 20 instances at 4000x3000 is over 7 GB.
            The masks are released as each photo is processed, not at the end, so
            leaving this off caps peak memory rather than only trimming the result.
        arrow_mode: ``"chain"`` draws one connector between panel-consecutive members
            of each identity (same connectivity, far fewer lines), ``"all"`` draws
            every accepted pair, ``"none"`` draws none.
        legend_attr: which attribute head the legend shows beside the class; ``None``
            uses each identity's first attribute.

    Returns:
        ``{"images", "identities", "matches", "num_identities", "num_images",
        "config", "diagnostics", "plot", "plot_path", "summary_path"}``. The last
        three are ``None`` when the corresponding output was not produced. Each
        per-photo dict gains ``identity_ids`` ``(n,)`` and ``embed`` ``(n, hidden)``
        (raw, unnormalized). Per-photo dicts hold torch tensors (as ``predict``
        does); the summaries are plain JSON-able Python (as ``track`` does).
    """
    if isinstance(model, (str, Path)):
        model = load_model(model, device=device)

    # Validate the gate before the first forward: a typo should cost milliseconds,
    # not N ViT passes.
    group_by = validate_group_by(group_by, getattr(model, "aux_specs", []))

    dev = next(model.parameters()).device
    imgsz = int(imgsz if imgsz is not None else model.image_size)
    letterbox = bool(getattr(model, "preprocess_letterbox", False))
    names = getattr(model, "names", None)

    paths = list(_iter_sources(source))
    if not paths:
        raise FileNotFoundError(
            f"No images found in {source!r} (looked for {sorted(_IMAGE_EXTS)}). "
            "Note the source is one folder of photos of one subject and is not "
            "searched recursively."
        )
    if len(paths) == 1:
        warnings.warn(
            "match() received a single photo: there is nothing to match across, so "
            "every detection becomes its own identity.",
            stacklevel=2,
        )

    print(
        f"[match] eomt-{getattr(model, 'size', '?')} "
        f"({getattr(model, 'family', 'instance')}, nc={getattr(model, 'nc', '?')}, "
        f"imgsz={imgsz}) on {dev} — {len(paths)} photo(s), "
        f"group_by={group_by or '(off)'}, sim_thres={sim_thres}"
    )

    # --- 1. one forward per photo ----------------------------------------
    # Each photo is reduced to panel scale and released within its own iteration:
    # nothing full-resolution outlives the loop body. Holding all N photos and all N
    # mask stacks to the end instead would peak at several GB on a 30-photo set of
    # 4000x3000 shots — the panel is all the renderer ever needs, and ``keep_masks``
    # is what asks for the full-resolution masks to survive.
    results, panels, t0 = [], [], time.perf_counter()
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        it0 = time.perf_counter()
        res = predict_image(
            model, image, device=dev, imgsz=imgsz, conf_thres=conf_thres,
            max_det=max_det, mask_thresh=mask_thresh, letterbox=letterbox, embed=True,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        res["elapsed_ms"] = (time.perf_counter() - it0) * 1e3
        res["path"] = str(path)
        res["image_index"] = i
        results.append(res)
        if plot:
            # Identity ids are not known until clustering; the panel gets them below.
            small, panel_res = _panelize(image, res, panel_size)
            panels.append({"image": small, "result": panel_res})
        del image
        if not keep_masks:
            res.pop("masks", None)
        print(f"[match] {path.name}: {res['num_detections']} instances "
              f"({res['elapsed_ms']:.1f} ms)")

    # --- 2. fingerprints, gate, Hungarian, link --------------------------
    dim = int(getattr(model.config, "hidden_size", 0))
    embed, photo, keys, origin = flatten_instances(results, group_by=group_by, dim=dim)
    sim = similarity_matrix(embed, center=center)
    gate = gate_mask(photo, keys)
    pairs = pairwise_matches(sim, photo, keys, sim_thres=sim_thres)
    identity, pairs = cluster(
        pairs, sim, photo, num_instances=len(origin), sim_thres=sim_thres,
        guard=guard, guard_factor=guard_factor, gate=gate,
    )
    identities = summarize_identities(
        identity, origin, results, sim, names=names,
        aux_specs=getattr(model, "aux_specs", []), paths=[str(p) for p in paths],
    )
    diagnostics = similarity_diagnostics(sim, gate, identity)

    # Stamp each detection with its identity, and drop heavy masks unless asked.
    flat_of = {o: m for m, o in enumerate(origin)}
    for i, res in enumerate(results):
        n = int(res["num_detections"])
        res["identity_ids"] = torch.tensor(
            [int(identity[flat_of[(i, j)]]) for j in range(n)], dtype=torch.long
        )

    accepted = [p for p in pairs if p["accepted"]]
    if not accepted and len(paths) > 1:
        print("[match] 0 accepted matches — try lowering sim_thres or relaxing group_by")

    matches = [
        {
            "a_image": origin[p["a"]][0], "a_det": origin[p["a"]][1],
            "b_image": origin[p["b"]][0], "b_det": origin[p["b"]][1],
            "similarity": round(p["similarity"], 4),
            "accepted": bool(p["accepted"]),
            "reason": p["reason"],
            "identity_id": p.get("identity_id"),
        }
        for p in pairs
    ]

    config = {
        "size": getattr(model, "size", None),
        "family": getattr(model, "family", "instance"),
        "nc": getattr(model, "nc", None),
        "hidden": dim,
        "imgsz": imgsz,
        "letterbox": letterbox,
        "conf_thres": conf_thres, "max_det": max_det, "mask_thresh": mask_thresh,
        "group_by": list(group_by), "sim_thres": sim_thres, "center": center,
        "guard": guard, "guard_factor": guard_factor,
    }

    result = {
        "source": str(source),
        "images": results,
        "identities": identities,
        "matches": matches,
        "num_identities": len(identities),
        "num_images": len(paths),
        "config": config,
        "diagnostics": diagnostics,
        "plot": None,
        "plot_path": None,
        "summary_path": None,
    }

    # --- 3. render + write ------------------------------------------------
    save_dir = Path(save) if save else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    if plot:
        for i, (panel, res) in enumerate(zip(panels, results)):
            panel["identity_ids"] = res["identity_ids"].tolist()
            panel["caption"] = (f"{i}. {Path(res['path']).stem[:22]} "
                                f"({res['num_detections']} inst)")
        mode = arrow_mode
        if mode != "none" and len(panels) > 36:
            print(f"[match] {len(panels)} panels: connectors disabled for legibility")
            mode = "none"
        arrows = _arrows_for(mode, accepted, origin, identity, sim)
        grid = draw_identity_grid(
            panels, names=names, aux_names=None, arrows=arrows,
            identities=identities, cols=cols, panel_size=panel_size,
            legend_attr=legend_attr, alpha=alpha, show_scores=show_scores,
            # Absolute width scale: an accepted link is >= sim_thres by definition,
            # so a threshold-grazing match draws hairline and a near-1.0 match draws
            # thick — and the same similarity means the same thickness in every grid.
            sim_range=(sim_thres, 1.0),
            title=f"{Path(str(source)).name} — {len(identities)} identities "
                  f"across {len(paths)} photos",
        )
        result["plot"] = grid
        if save_dir is not None:
            dst = save_dir / f"{Path(str(source)).name or 'match'}.png"
            grid.save(dst)
            result["plot_path"] = str(dst)

    if save_dir is not None:
        summary = {
            k: result[k] for k in (
                "source", "num_images", "num_identities", "config",
                "diagnostics", "identities", "matches", "plot_path",
            )
        }
        dst = save_dir / f"{Path(str(source)).name or 'match'}_identities.json"
        dst.write_text(json.dumps(summary, indent=2) + "\n")
        result["summary_path"] = str(dst)

    total = time.perf_counter() - t0
    _print_summary(identities, len(paths))
    print(
        f"[match] done: {len(identities)} identities from "
        f"{sum(int(r['num_detections']) for r in results)} detections across "
        f"{len(paths)} photos in {total:.2f} s"
        + (f" -> {save_dir}" if save_dir is not None else "")
    )
    return result


def _arrows_for(mode: str, accepted, origin, identity, sim) -> list[dict]:
    """Connectors to draw: every accepted pair, or one chain per identity."""
    if mode == "none":
        return []
    if mode == "all":
        return [
            {
                "a_panel": origin[p["a"]][0], "a_det": origin[p["a"]][1],
                "b_panel": origin[p["b"]][0], "b_det": origin[p["b"]][1],
                "similarity": p["similarity"],
                "identity_id": p.get("identity_id", 0),
            }
            for p in accepted
        ]
    # "chain": one link between panel-consecutive members of each identity. Same
    # connectivity, O(M) lines instead of O(M^2) — every pair of 30 photos is an
    # unreadable hairball.
    # A chained link is not always an accepted pair: an identity built from a-c and
    # b-c puts a next to b in panel order, and that cut was never matched directly.
    # Read its true cosine off the matrix rather than defaulting to 0.0, which under
    # ``sim_range=(sim_thres, 1.0)`` would clamp to hairline and paint a confidently
    # linked pair as a barely-made match.
    s = sim.detach().cpu().numpy()
    sims = {(p["a"], p["b"]): p["similarity"] for p in accepted}
    sims.update({(p["b"], p["a"]): p["similarity"] for p in accepted})
    by_identity: dict[int, list[int]] = {}
    for m, cid in enumerate(identity.tolist()):
        by_identity.setdefault(int(cid), []).append(m)
    arrows = []
    for cid, members in sorted(by_identity.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda m: origin[m])
        for a, b in zip(ordered, ordered[1:]):
            arrows.append({
                "a_panel": origin[a][0], "a_det": origin[a][1],
                "b_panel": origin[b][0], "b_det": origin[b][1],
                "similarity": sims.get((a, b), float(s[a, b])),
                "identity_id": cid,
            })
    return arrows
