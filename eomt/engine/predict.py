"""Inference: photos one by one (batched), or the photos of one subject together (cross-view), and their links.

* :func:`predict_image` / :func:`predict_batch` — one photo (optionally with test-time augmentation / tiles), or
  several independent ones in one forward pass;
* :func:`predict` — a photo, a folder of photos, or a folder of subject folders: batched, streamed and rendered;
* :func:`predict_set` — up to ``max_views`` photos of one subject in one forward pass;
* :func:`predict_subject` — every photo of one subject, grouped into sets, optionally linked into identities.

Every function runs the network in bf16 where the GPU has it (``amp="auto"``); ``amp=False`` runs it in fp32.
"""

from __future__ import annotations

import collections
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .._log import LOG
from ..matcher import link_probabilities, mask_boxes
from ..postprocess import postprocess_detection, postprocess_instance, slice_batch
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


def _subjects(source: str | Path) -> list[list[Path]]:
    """The photos of each subject: those of ``source`` (a photo or a folder), then those of each of its subfolders."""
    source = Path(source)
    folders = [source, *sorted(p for p in source.iterdir() if p.is_dir())] if source.is_dir() else [source]
    return [photos for photos in (list(_iter_sources(f)) for f in folders) if photos]


def _chunks(items, n: int):
    it = iter(items)
    while chunk := list(itertools.islice(it, n)):
        yield chunk


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


def _forward(model, x, device, amp, **kw):
    dtype = _amp_dtype(device, amp)
    if dtype is None:
        return model(x, **kw)
    with torch.autocast("cuda", dtype=dtype):  # only the network; postprocess casts the logits back to fp32
        return model(x, **kw)


def _aux_scopes(model) -> dict:
    """The primary-class scope of each class-scoped attribute head: postprocess reports ``ids = -1`` outside it."""
    return {s.name: s.applies_to for s in getattr(model, "aux_specs", []) if s.applies_to is not None}


def _preprocess(model, image, imgsz: int, letterbox: bool):
    return preprocess_numpy(np.asarray(image.convert("RGB")), imgsz, letterbox=letterbox,
                            mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None))


def _postprocess(model, out, b, size, meta, conf_thres, max_det, mask_thresh) -> dict:
    """Photo ``b`` of a batched forward as a result dict (boxes for the detect family, else masks too)."""
    one = slice_batch(out, b)
    if getattr(model, "family", "instance") == "detect":
        return postprocess_detection(one, conf_thres, size, max_det=max_det, preprocess_meta=meta,
                                     aux_scopes=_aux_scopes(model))
    return postprocess_instance(one, conf_thres, size, max_det=max_det, mask_thresh=mask_thresh, preprocess_meta=meta,
                                aux_scopes=_aux_scopes(model))


@torch.no_grad()
def predict_batch(model, images: list[Image.Image], *, device, imgsz: int, conf_thres: float = 0.3, max_det: int = 100,
                  mask_thresh: float = 0.5, letterbox: bool = True, amp: bool | str = "auto", embed: bool = False,
                  prepared: list | None = None) -> list[dict]:
    """Several independent photos in one forward pass, one result dict per photo (see :func:`predict_image`).
    ``prepared`` (one ``(chw, meta)`` per photo, :func:`eomt.preprocess.preprocess_numpy`) skips the preprocessing."""
    prepared = prepared or [_preprocess(model, im, imgsz, letterbox) for im in images]
    x = torch.from_numpy(np.stack([chw for chw, _ in prepared])).to(device, non_blocking=True)
    out = _forward(model, x, device, amp)
    results = []
    for b, (im, (_, meta)) in enumerate(zip(images, prepared)):
        res = _postprocess(model, out, b, im.size, meta, conf_thres, max_det, mask_thresh)
        if embed:  # the per-query embedding of each kept detection (raw: normalising is the linker's job)
            res["embed"] = out["query_embed"][b, res["query_idx"]].float().cpu()
        results.append(res)
    return results


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
    amp: bool | str = "auto",
    augment: bool | dict = False,
    tiles: bool | int | dict = False,
) -> dict:
    """Run the model on one PIL image and return a result dict: ``boxes`` (xyxy, original pixels), ``scores``,
    ``classes``, ``masks`` (bool, original resolution; not for the detect family), ``query_idx`` and, with attribute
    heads, ``aux`` (``{head: {"ids", "probs"}}``) and ``aux_values`` (regression heads, real units).

    With ``embed``, the result also carries ``"embed"`` ``(N, hidden)``: the per-query embedding of each kept detection
    (re-identification's fingerprint, :mod:`eomt.reid`), raw. ``augment=True`` adds the image zoomed to 1.5x ``imgsz``
    (test-time augmentation) and ``tiles=True`` runs overlapping native-resolution tiles plus the whole image; the views'
    instances are merged. Each also takes a dict of options (``tiles`` an int tile size); see :mod:`eomt.tta`. Instance
    family only; the result then has no ``query_idx`` and ``embed`` is the score-weighted mean of the merged queries'.
    """
    cfg = TTAConfig.resolve(augment, tiles)
    if cfg is not None:
        return predict_views(
            model, np.asarray(image.convert("RGB")), cfg, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det,
            mask_thresh=mask_thresh, letterbox=letterbox, mean=getattr(model, "pixel_mean", None),
            std=getattr(model, "pixel_std", None), amp_dtype=_amp_dtype(device, amp), aux_scopes=_aux_scopes(model),
            embed=embed,
        )
    return predict_batch(model, [image], device=device, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det,
                         mask_thresh=mask_thresh, letterbox=letterbox, amp=amp, embed=embed)[0]


def link_defaults(model) -> tuple[float, tuple[str, ...] | None]:
    """``(threshold, gate keys)`` the detections of a model are linked with by default: the threshold its validation
    calibrated (``match_thres``: a probability of the set matcher, else a cosine of the query embeddings) and the gate
    it was calibrated with (``None`` = no gate); for a model never calibrated, 0.5 (a set matcher) or 0.7
    (re-identification by the query embedding of a single-photo model), within the same class."""
    if getattr(model, "match_thres", None) is None:
        return (0.5 if getattr(model, "matcher", None) is not None else 0.7), ("class",)
    gate = getattr(model, "link_group_by", None)
    return float(model.match_thres), tuple(gate) if gate else None


def max_views(model) -> int:
    """The most photos ``model`` predicts together (1 without cross-view sublayers)."""
    return int((getattr(model, "xview_cfg", None) or {}).get("max_views", 1))


@torch.no_grad()
def predict_set(
    model,
    images: list[Image.Image],
    *,
    device,
    imgsz: int,
    conf_thres: float = 0.3,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    letterbox: bool = True,
    amp: bool | str = "auto",
) -> list[dict]:
    """Predict photos of ONE subject in one forward pass: with cross-view sublayers the queries of each photo take what
    the others show (at most :func:`max_views` photos); a model without them predicts each photo alone.

    Returns one result dict per photo (:func:`predict_image`), each with ``"embed"`` ``(N, hidden)``, the per-query
    embedding of its detections, and ``"link_box"`` ``(N, 4)``, the box of each mask on the input canvas: what the set
    matcher links them by (:func:`link_results`; a model without one links by the embedding's cosine).
    """
    if getattr(model, "xview_cfg", None) and not 1 <= len(images) <= max_views(model):
        raise ValueError(f"predict_set takes 1 to {max_views(model)} photos of one subject, got {len(images)}.")
    if getattr(model, "family", "instance") == "detect":
        raise ValueError("predict_set: the detect family predicts photos one by one (predict_batch)")
    prepared = [_preprocess(model, im, imgsz, letterbox) for im in images]
    x = torch.from_numpy(np.stack([chw for chw, _ in prepared])).to(device, non_blocking=True)
    out = _forward(model, x, device, amp, set_ids=torch.zeros(len(images), dtype=torch.long))
    results = []
    for b, (im, (_, meta)) in enumerate(zip(images, prepared)):
        res = _postprocess(model, out, b, im.size, meta, conf_thres, max_det, mask_thresh)
        q = res["query_idx"].to(out["query_embed"].device)
        res["embed"] = out["query_embed"][b, q].float().cpu()
        res["link_box"] = mask_boxes(out["masks_queries_logits"][b, q]).cpu()
        results.append(res)
    return results


def subject_groups(model, n: int, groups: list[list[int]] | None = None, *, batch: int = 8) -> list[list[int]]:
    """The groups of photo indices a subject's ``n`` photos are predicted in: ``groups`` checked (1 to
    :func:`max_views` distinct photos each, every photo in one group at most; those left out are predicted alone), else
    consecutive photos, :func:`max_views` at a time. Without cross-view sublayers the photos are independent: ``batch``
    at a time."""
    k = max_views(model)
    if groups is None:
        return [list(c) for c in _chunks(range(n), k if k > 1 else batch)]
    out, seen = [], set()
    for g in groups:
        g = [int(i) for i in g]
        if not g or len(g) > k or len(set(g)) < len(g) or seen & set(g) or any(not 0 <= i < n for i in g):
            raise ValueError(f"groups: {g} must hold 1 to {k} distinct photos of 0..{n - 1}, each in one group")
        seen.update(g)
        out.append(g)
    return out + [[i] for i in range(n) if i not in seen]


def link_results(model, results: list[dict], *, group_by="auto", sim_thres: float | None = None,
                 paths: list[str] | None = None, groups: list[list[int]] | None = None) -> dict:
    """Link the detections of the photos of one subject (each result from :func:`predict_set`) into identities
    (:func:`eomt.reid.link_identities`: Hungarian per photo pair, union-find with a merge guard), by the set matcher's
    probabilities (the photos of each of ``groups``, those predicted together, matched together; any other two photos
    as a pair) or, without one, the cosine of the embeddings; with the model's calibrated threshold and gate
    (:func:`link_defaults`; ``sim_thres`` and ``group_by`` override them, ``group_by="auto"`` being the calibrated
    gate). Stamps every result with ``identity_ids`` and returns the linking (:func:`eomt.reid.link_identities`) plus
    ``"matches"`` (every candidate pair, by photo and detection) and ``"config"``."""
    from ..reid import link_identities, validate_group_by

    thres, gate = link_defaults(model)
    keys = validate_group_by(gate if group_by == "auto" else group_by, getattr(model, "aux_specs", []))
    thres = float(sim_thres) if sim_thres is not None else thres
    sim = None
    if getattr(model, "matcher", None) is not None and all("link_box" in r for r in results):
        sim = link_probabilities(model.matcher, [r["embed"] for r in results], [r["link_box"] for r in results],
                                 groups=groups)
    linked = link_identities(results, sim_thres=thres, group_by=keys, dim=model.config.hidden_size, sim=sim,
                             names=getattr(model, "names", None), aux_specs=getattr(model, "aux_specs", []), paths=paths)
    origin = linked["origin"]
    matches = [{"a_image": origin[p["a"]][0], "a_det": origin[p["a"]][1], "b_image": origin[p["b"]][0],
                "b_det": origin[p["b"]][1], "similarity": round(p["similarity"], 4), "accepted": bool(p["accepted"]),
                "reason": p["reason"], "identity_id": p.get("identity_id")} for p in linked["pairs"]]
    return {**linked, "matches": matches, "config": {"sim_thres": thres, "group_by": list(keys)}}


@torch.no_grad()
def predict_subject(
    model,
    images: list[Image.Image],
    *,
    device,
    imgsz: int,
    groups: list[list[int]] | None = None,
    conf_thres: float = 0.3,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    letterbox: bool = True,
    amp: bool | str = "auto",
    identities: bool = False,
    group_by: tuple[str, ...] | str | None = "auto",
    sim_thres: float | None = None,
) -> list[dict] | dict:
    """Predict every photo of one subject, any number of them, together with other photos of it: grouped by
    :func:`subject_groups` (``groups`` gives them, e.g. each overview with the close-ups it shows; by default
    consecutive photos), one :func:`predict_set` per group. Returns one result per photo, in input order.

    ``identities=True`` also links the detections of every photo with those of every other, across the groups too
    (:func:`link_results`), and returns ``{"images": [one result per photo, each with "identity_ids"], "identities":
    [one record per identity: its members, fused attributes and regression values], "matches", "config"}``.
    """
    results: list[dict | None] = [None] * len(images)
    sets = subject_groups(model, len(images), groups)
    kw = dict(device=device, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det, mask_thresh=mask_thresh,
              letterbox=letterbox, amp=amp)
    for group in sets:
        for i, r in zip(group, predict_set(model, [images[i] for i in group], **kw)):
            results[i] = r
    if not identities:
        return results
    linked = link_results(model, results, group_by=group_by, sim_thres=sim_thres, groups=sets)
    return {"images": results, "identities": linked["identities"], "matches": linked["matches"],
            "config": {**linked["config"], "groups": sets}}


def _prefetched(fn, items, depth: int, workers: int = 4):
    """``fn(item)`` of every item, in order, computed by ``workers`` threads at most ``depth`` items ahead (decoding the
    next photos while the GPU runs the current batch, without holding a whole folder in memory)."""
    items = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue = collections.deque(pool.submit(fn, x) for x in itertools.islice(items, depth))
        while queue:
            done = queue.popleft().result()
            queue.extend(pool.submit(fn, x) for x in itertools.islice(items, 1))
            yield done


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
    imgsz: int | None = None,
    batch: int = 8,
    stream: bool = False,
    amp: bool | str = "auto",
    augment: bool | dict = False,
    tiles: bool | int | dict = False,
    subject: bool = False,
    alpha: float = 0.35,
    draw_boxes: bool = True,
    show_scores: bool = True,
    color_by: str = "class",
    verbose: bool = True,
):
    """Run inference on an image or a folder of images, ``batch`` photos per forward pass (the next ones decoded
    meanwhile). Returns one result dict per image (:func:`predict_image`), each with its source ``path``; with
    ``stream=True`` a generator of them instead, so a large folder is never held in memory.

    ``model`` may be a loaded :class:`~eomt.model.EoMTModel` or a checkpoint path / run folder. With ``plot``, every
    image is rendered with its masks / boxes / labels and written under ``save`` (``result["plot_path"]``; with
    ``save=None`` the rendered image is kept as ``result["plot"]``). ``augment`` / ``tiles``: test-time augmentation /
    tiled inference (one photo at a time). ``subject=True`` predicts the photos of each subject together
    (:func:`predict_subject`, for a model post-trained with ``multiview``): the photos of ``source`` are one subject and
    each of its subfolders is another, whose annotated images keep the subfolder under ``save``.
    """
    if subject and (augment or tiles):
        raise ValueError("predict(subject=True) predicts the photos together, without augment / tiles.")
    if isinstance(model, (str, Path)):
        model = load_model(model, device=device)
    gen = _predict_iter(model, source, plot=plot, save=save, conf_thres=conf_thres, max_det=max_det,
                        mask_thresh=mask_thresh, imgsz=imgsz, batch=batch, amp=amp, augment=augment, tiles=tiles,
                        subject=subject, alpha=alpha, draw_boxes=draw_boxes, show_scores=show_scores, color_by=color_by,
                        verbose=verbose)
    return gen if stream else list(gen)


def _predict_iter(model, source, *, plot, save, conf_thres, max_det, mask_thresh, imgsz, batch, amp, augment, tiles,
                  subject, alpha, draw_boxes, show_scores, color_by, verbose):
    dev = next(model.parameters()).device
    imgsz = int(imgsz if imgsz is not None else model.image_size)
    letterbox = bool(getattr(model, "preprocess_letterbox", True))
    names = getattr(model, "names", None)
    aux_names = {s.name: s.names for s in getattr(model, "aux_specs", [])}
    tta = TTAConfig.resolve(augment, tiles) is not None
    if verbose:
        LOG.info(f"[predict] eomt-{model.size} ({model.family}, nc={model.nc}, imgsz={imgsz}) on {dev}")
        if subject and getattr(model, "xview_cfg", None) is None:
            LOG.info("[predict] the model has no cross-view sublayers: each photo is predicted alone")
    out_root = Path(save) if (plot and save) else None
    root = Path(source) if Path(source).is_dir() else Path(source).parent
    kw = dict(device=dev, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det, mask_thresh=mask_thresh,
              letterbox=letterbox, amp=amp)

    def load(paths):  # decoded (and, for plain batches, preprocessed) in a worker thread
        images = [Image.open(p).convert("RGB") for p in paths]
        prepared = None if tta or subject else [_preprocess(model, im, imgsz, letterbox) for im in images]
        return paths, images, prepared

    groups = _subjects(source) if subject else _chunks(_iter_sources(source), batch)
    n, t0 = 0, time.perf_counter()
    for paths, images, prepared in _prefetched(load, groups, depth=2):
        if subject:
            results = predict_subject(model, images, **kw)
        elif tta:
            results = [predict_image(model, im, augment=augment, tiles=tiles, **kw) for im in images]
        else:
            results = predict_batch(model, images, prepared=prepared, **kw)
        for path, image, result in zip(paths, images, results):
            result["path"] = str(path)
            if plot:
                rendered = draw_instances(image, result, names=names, aux_names=aux_names or None, alpha=alpha,
                                          draw_boxes=draw_boxes, show_scores=show_scores, color_by=color_by)
                if out_root is not None:
                    dst = out_root / path.relative_to(root)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    rendered.save(dst)
                    result["plot_path"] = str(dst)
                else:
                    result["plot"] = rendered
            n += 1
            yield result
    if verbose and n:
        s = time.perf_counter() - t0
        LOG.info(f"[predict] {n} image(s) in {s:.2f} s ({s * 1e3 / n:.1f} ms/img)" + (f" -> {out_root}" if out_root else ""))
