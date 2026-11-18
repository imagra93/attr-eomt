"""Instance matching over the photos of one subject: which detections are the same physical instance.

Predicts every photo (:func:`~eomt.engine.predict.predict_subject`'s groups: a cross-view model predicts them
``max_views`` at a time), links the detections of every two photos (:mod:`eomt.reid`), and renders a grid image plus a
JSON summary. The embedding is the matching head's when the model has one (trained on ``instance_id`` links), else the
per-query embedding the detector computes anyway (re-identification, nothing trained for it).

One call handles the photos of one subject. Several subjects are the caller's loop — see ``scripts/match.py
--each-subdir``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..serialization import load_model
from ..visualize import draw_identity_grid
from .predict import _IMAGE_EXTS, _iter_sources, link_defaults, link_results, predict_set, subject_groups


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
    amp: bool | str = "auto",
    groups: list[list[int]] | None = None,
    group_by: tuple[str, ...] | str | None = "auto",
    sim_thres: float | None = None,
    keep_masks: bool = False,
    panel_size: int = 480,
    cols: int | None = None,
    link_mode: str = "chain",
    legend_attr: str | None = None,
    alpha: float = 0.35,
    show_scores: bool = False,
) -> dict:
    """Link the instances of the photos of one subject: one identity per physical instance.

    Args:
        source: a folder of photos (non-recursive) or a single image; any number of photos.
        device: only consulted when ``model`` is a path to load.
        groups: photo indices predicted together by a cross-view model (each group up to its ``max_views``; by default
            consecutive photos); see :func:`eomt.engine.predict.subject_groups`.
        group_by: gate keys — two instances may only match if they agree on all of them (``"class"`` and categorical
            attribute heads; ``None`` disables gating). ``"auto"``: the gate the model's links were calibrated with,
            else ``("class",)`` for a model never calibrated.
        sim_thres: cosine floor for "same instance" (Hungarian always proposes a partner; the threshold turns it down).
            ``None``: the model's calibrated ``match_thres``, else 0.7.
        keep_masks: retain the full-resolution masks in the returned per-photo dicts (off: they are released as each
            group is processed, so peak memory stays that of one group).
        link_mode: ``"chain"`` draws one link between panel-consecutive members of each identity, ``"all"`` every
            accepted pair, ``"none"`` none. ``legend_attr``: the attribute head the legend shows beside the class.

    Returns:
        ``{"images", "identities", "matches", "num_identities", "num_images", "config", "diagnostics", "plot",
        "plot_path", "summary_path"}`` (the last three ``None`` when not produced). Each per-photo dict gains
        ``identity_ids`` ``(n,)`` and ``embed`` ``(n, d)``.
    """
    if isinstance(model, (str, Path)):
        model = load_model(model, device=device)
    thres, gate = link_defaults(model)
    paths = list(_iter_sources(source))
    if not paths:
        raise FileNotFoundError(f"No images found in {source!r} (looked for {sorted(_IMAGE_EXTS)}); the source is one "
                                "folder of photos of one subject and is not searched recursively.")
    if len(paths) == 1:
        warnings.warn("match() received a single photo: every detection becomes its own identity.", stacklevel=2)
    dev = next(model.parameters()).device
    imgsz = int(imgsz if imgsz is not None else model.image_size)
    names = getattr(model, "names", None)
    sets = subject_groups(model, len(paths), groups)
    print(f"[match] eomt-{model.size} on {dev}: {len(paths)} photo(s) in {len(sets)} group(s), "
          f"group_by={gate if group_by == 'auto' else group_by}, sim_thres={thres if sim_thres is None else sim_thres}")

    # --- 1. predict each group, reduce every photo to panel scale at once (nothing full-resolution outlives its group)
    results: list[dict | None] = [None] * len(paths)
    panels: list[dict | None] = [None] * len(paths)
    t0 = time.perf_counter()
    for group in sets:
        images = [Image.open(paths[i]).convert("RGB") for i in group]
        it0 = time.perf_counter()
        res = predict_set(model, images, device=dev, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det,
                          mask_thresh=mask_thresh, letterbox=bool(getattr(model, "preprocess_letterbox", True)), amp=amp)
        ms = (time.perf_counter() - it0) * 1e3 / len(group)
        for i, image, r in zip(group, images, res):
            r.update(elapsed_ms=ms, path=str(paths[i]), image_index=i)
            if plot:
                small, panel_res = _panelize(image, r, panel_size)
                panels[i] = {"image": small, "result": panel_res}
            if not keep_masks:
                r.pop("masks", None)
            results[i] = r
            print(f"[match] {paths[i].name}: {r['num_detections']} instances ({ms:.1f} ms)")

    # --- 2. fingerprints, gate, Hungarian, link (each detection stamped with its identity)
    linked = link_results(model, results, group_by=group_by, sim_thres=sim_thres, paths=[str(p) for p in paths])
    identity, identities, pairs = linked["identity"], linked["identities"], linked["pairs"]
    origin, sim, diagnostics = linked["origin"], linked["sim"], linked["diagnostics"]
    accepted = [p for p in pairs if p["accepted"]]
    if not accepted and len(paths) > 1:
        print("[match] 0 accepted matches — try lowering sim_thres or relaxing group_by")
    config = {"size": model.size, "nc": model.nc, "imgsz": imgsz, "conf_thres": conf_thres, "max_det": max_det,
              "mask_thresh": mask_thresh, "groups": sets, **linked["config"]}
    result = {"source": str(source), "images": results, "identities": identities, "matches": linked["matches"],
              "num_identities": len(identities), "num_images": len(paths), "config": config,
              "diagnostics": diagnostics, "plot": None, "plot_path": None, "summary_path": None}

    # --- 3. render + write
    save_dir = Path(save) if save else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(str(source)).name or "match"
    if plot:
        for i, (panel, res) in enumerate(zip(panels, results)):
            panel["identity_ids"] = res["identity_ids"].tolist()
            panel["caption"] = f"{i}. {Path(res['path']).stem[:22]} ({res['num_detections']} inst)"
        mode = link_mode
        if mode != "none" and len(panels) > 36:
            print(f"[match] {len(panels)} panels: connectors disabled for legibility")
            mode = "none"
        grid = draw_identity_grid(
            panels, names=names, aux_names=None, links=_links_for(mode, accepted, origin, identity, sim),
            identities=identities, cols=cols, panel_size=panel_size, legend_attr=legend_attr, alpha=alpha,
            show_scores=show_scores,
            # an absolute width scale: an accepted link is >= the threshold, so a link grazing it draws hairline
            sim_range=(config["sim_thres"], 1.0),
            title=f"{stem} — {len(identities)} identities across {len(paths)} photos",
        )
        result["plot"] = grid
        if save_dir is not None:
            dst = save_dir / f"{stem}.png"
            grid.save(dst)
            result["plot_path"] = str(dst)
    if save_dir is not None:
        summary = {k: result[k] for k in ("source", "num_images", "num_identities", "config", "diagnostics",
                                          "identities", "matches", "plot_path")}
        dst = save_dir / f"{stem}_identities.json"
        dst.write_text(json.dumps(summary, indent=2) + "\n")
        result["summary_path"] = str(dst)
    _print_summary(identities, len(paths))
    print(f"[match] done: {len(identities)} identities from {sum(int(r['num_detections']) for r in results)} detections "
          f"across {len(paths)} photos in {time.perf_counter() - t0:.2f} s" + (f" -> {save_dir}" if save_dir else ""))
    return result


def _links_for(mode: str, accepted, origin, identity, sim) -> list[dict]:
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
    links = []
    for cid, members in sorted(by_identity.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda m: origin[m])
        for a, b in zip(ordered, ordered[1:]):
            links.append({
                "a_panel": origin[a][0], "a_det": origin[a][1],
                "b_panel": origin[b][0], "b_det": origin[b][1],
                "similarity": sims.get((a, b), float(s[a, b])),
                "identity_id": cid,
            })
    return links
