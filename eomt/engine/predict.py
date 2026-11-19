"""Inference: photos one by one (batched), or the photos of one subject together (cross-view), and their links.

* :func:`predict_image` / :func:`predict_batch` — one photo (optionally with test-time augmentation / tiles), or
  several independent ones in one forward pass;
* :func:`predict` — a photo, a folder of photos, or a folder of subject folders: batched, streamed and rendered;
* :func:`predict_set` — up to ``max_views`` photos of one subject in one forward pass;
* :func:`predict_subject` — every photo of one subject, grouped into sets, optionally linked into identities.
* :func:`subject_groups` / :func:`similar_groups` — the groups a subject's photos are predicted in: consecutive
  (``fill``: no short last group), given, or by what the photos show; a photo may be context in several.

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


def _rgb(image) -> np.ndarray:
    """A PIL image or an RGB ``(H, W, 3)`` array, as an array."""
    return np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)


def _size(image) -> tuple[int, int]:
    """``(w, h)`` of a PIL image or an ``(H, W, ...)`` array."""
    if isinstance(image, Image.Image):
        return image.size
    h, w = np.shape(image)[:2]
    return int(w), int(h)


def _preprocess(model, image, imgsz: int, letterbox: bool):
    return preprocess_numpy(_rgb(image), imgsz, letterbox=letterbox,
                            mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None))


def _stack(prepared: list, device) -> torch.Tensor:
    """The batch of ``prepared`` photos (one ``(chw, meta)`` each, from :func:`eomt.preprocess.preprocess_numpy` or
    :func:`eomt.preprocess.preprocess_torch`: numpy arrays or tensors, on any device) on ``device``."""
    chws = [chw for chw, _ in prepared]
    if any(isinstance(c, torch.Tensor) for c in chws):
        return torch.stack([torch.as_tensor(c).to(device, non_blocking=True) for c in chws])
    return torch.from_numpy(np.stack(chws)).to(device, non_blocking=True)


def _inputs(model, images: list, prepared: list | None, sizes: list | None, imgsz: int, letterbox: bool):
    """``(prepared, sizes)`` of photos: ``prepared`` as given (else preprocessed here) and the ``(w, h)`` each result is
    returned at (``sizes`` as given, else the photo's own)."""
    if prepared is not None and len(prepared) != len(images):
        raise ValueError(f"prepared: {len(prepared)} inputs for {len(images)} photos")
    if sizes is not None and len(sizes) != len(images):
        raise ValueError(f"sizes: {len(sizes)} sizes for {len(images)} photos")
    prepared = prepared or [_preprocess(model, im, imgsz, letterbox) for im in images]
    sizes = [tuple(int(v) for v in s) for s in sizes] if sizes is not None else [_size(im) for im in images]
    return prepared, sizes


def _postprocess(model, out, b, size, meta, conf_thres, max_det, mask_thresh) -> dict:
    """Photo ``b`` of a batched forward as a result dict (boxes for the detect family, else masks too)."""
    one = slice_batch(out, b)
    if getattr(model, "family", "instance") == "detect":
        return postprocess_detection(one, conf_thres, size, max_det=max_det, preprocess_meta=meta,
                                     aux_scopes=_aux_scopes(model))
    return postprocess_instance(one, conf_thres, size, max_det=max_det, mask_thresh=mask_thresh, preprocess_meta=meta,
                                aux_scopes=_aux_scopes(model))


@torch.no_grad()
def predict_batch(model, images: list, *, device, imgsz: int, conf_thres: float = 0.3, max_det: int = 100,
                  mask_thresh: float = 0.5, letterbox: bool = True, amp: bool | str = "auto", embed: bool = False,
                  prepared: list | None = None, sizes: list | None = None, photo_embed: bool = False) -> list[dict]:
    """Several independent photos (PIL images or RGB arrays) in one forward pass, one result dict per photo (see
    :func:`predict_image`). ``prepared`` (one ``(chw, meta)`` per photo, from
    :func:`eomt.preprocess.preprocess_numpy` or :func:`eomt.preprocess.preprocess_torch`, e.g. already on the GPU)
    skips the preprocessing; ``sizes`` (one ``(w, h)`` per photo) returns the boxes and masks at that size instead of
    the photo's own (e.g. a fixed mask grid). ``photo_embed`` adds ``"photo_embed"`` ``(hidden,)``, the photo's mean
    final patch token (to compare photos by appearance, :func:`similar_groups`)."""
    prepared, sizes = _inputs(model, images, prepared, sizes, imgsz, letterbox)
    out = _forward(model, _stack(prepared, device), device, amp)
    results = []
    for b, (size, (_, meta)) in enumerate(zip(sizes, prepared)):
        res = _postprocess(model, out, b, size, meta, conf_thres, max_det, mask_thresh)
        if embed:  # the per-query embedding of each kept detection (raw: normalising is the linker's job)
            res["embed"] = out["query_embed"][b, res["query_idx"]].float().cpu()
        if photo_embed and "photo_embed" in out:
            res["photo_embed"] = out["photo_embed"][b].float().cpu()
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
    images: list,
    *,
    device,
    imgsz: int,
    conf_thres: float = 0.3,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    letterbox: bool = True,
    amp: bool | str = "auto",
    prepared: list | None = None,
    sizes: list | None = None,
) -> list[dict]:
    """Predict photos (PIL images or RGB arrays) of ONE subject in one forward pass: with cross-view sublayers the
    queries of each photo take what the others show (at most :func:`max_views` photos); a model without them predicts
    each photo alone. ``prepared`` and ``sizes`` as in :func:`predict_batch`.

    ``model`` may be a wrapper of the network (e.g. one rescoring its outputs) whose ``forward`` passes ``set_ids`` on
    and keeps ``query_embed`` / ``masks_queries_logits`` in its output.

    Returns one result dict per photo (:func:`predict_image`), each with ``"embed"`` ``(N, hidden)``, the per-query
    embedding of its detections, and ``"link_box"`` ``(N, 4)``, the box of each mask on the input canvas: what the set
    matcher links them by (:func:`link_results`; a model without one links by the embedding's cosine); and
    ``"photo_embed"`` ``(hidden,)``, the photo's mean final patch token in this set.
    """
    if getattr(model, "xview_cfg", None) and not 1 <= len(images) <= max_views(model):
        raise ValueError(f"predict_set takes 1 to {max_views(model)} photos of one subject, got {len(images)}.")
    if getattr(model, "family", "instance") == "detect":
        raise ValueError("predict_set: the detect family predicts photos one by one (predict_batch)")
    prepared, sizes = _inputs(model, images, prepared, sizes, imgsz, letterbox)
    out = _forward(model, _stack(prepared, device), device, amp, set_ids=torch.zeros(len(images), dtype=torch.long))
    results = []
    for b, (size, (_, meta)) in enumerate(zip(sizes, prepared)):
        res = _postprocess(model, out, b, size, meta, conf_thres, max_det, mask_thresh)
        q = res["query_idx"].to(out["query_embed"].device)
        res["embed"] = out["query_embed"][b, q].float().cpu()
        res["link_box"] = mask_boxes(out["masks_queries_logits"][b, q]).cpu()
        if "photo_embed" in out:
            res["photo_embed"] = out["photo_embed"][b].float().cpu()
        results.append(res)
    return results


def subject_groups(model, n: int, groups: list[list[int]] | None = None, *, batch: int = 8,
                   fill: bool = False) -> list[list[int]]:
    """The groups of photo indices a subject's ``n`` photos are predicted in: ``groups`` checked (1 to
    :func:`max_views` distinct photos each; those in no group are predicted alone), else consecutive photos,
    :func:`max_views` at a time. Without cross-view sublayers the photos are independent: ``batch`` at a time.

    A photo may be in several groups: its result comes from the first one (:func:`group_owners`), and the later ones
    only show it to the others. ``fill`` (consecutive groups only) tops a short last group up to :func:`max_views` with
    the photos just before it, so that no photo is predicted with fewer companions than the subject has: 5 photos at 4
    a time are ``[0, 1, 2, 3]`` and ``[4, 3, 2, 1]``, not ``[0, 1, 2, 3]`` and ``[4]``."""
    k = max_views(model)
    if groups is None:
        out = [list(c) for c in _chunks(range(n), k if k > 1 else batch)]
        if fill and k > 1 and len(out) > 1 and len(out[-1]) < k:
            last = out[-1]
            out[-1] = last + list(range(last[0] - 1, -1, -1))[: k - len(last)]
        return out
    out = []
    for g in groups:
        g = [int(i) for i in g]
        if not g or len(g) > k or len(set(g)) < len(g) or any(not 0 <= i < n for i in g):
            raise ValueError(f"groups: {g} must hold 1 to {k} distinct photos of 0..{n - 1}")
        out.append(g)
    seen = {i for g in out for i in g}
    return out + [[i] for i in range(n) if i not in seen]


def group_owners(groups: list[list[int]]) -> dict[int, int]:
    """Photo index -> the index of the group its result comes from: the first group listing it (in a later one it is
    context for the others only)."""
    owners: dict[int, int] = {}
    for g, members in enumerate(groups):
        for i in members:
            owners.setdefault(int(i), g)
    return owners


def photo_similarity(features, *, weights=None, center=False) -> np.ndarray:
    """``(n, n)`` similarity of a subject's photos: the cosine of their rows in each of ``features`` (one ``(n, d)``
    array, or a list of them), averaged with ``weights`` (default equal). Several arrays describe the photos in several
    ways, e.g. how much of each class a model finds in them, the same with the sides merged, and their appearance
    (``photo_embed``), so that an error in one (a part read on the wrong side) is outvoted by the others (the photos
    still look alike). ``center`` (one bool, or one per array) subtracts the photos' mean row first: for embeddings
    that every photo of a subject shares much of. A row of zeros is like no other photo in that array."""
    blocks = list(features) if isinstance(features, (list, tuple)) and len(features) and \
        all(np.ndim(b) == 2 for b in features) else [features]
    blocks = [np.asarray(b, dtype=np.float64) for b in blocks]
    n = len(blocks[0])
    if any(b.ndim != 2 or len(b) != n for b in blocks):
        raise ValueError(f"features: one (n, d) array per description, with the same n; got {[b.shape for b in blocks]}")
    w = np.ones(len(blocks)) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if len(w) != len(blocks) or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"weights: one non-negative weight per array of features, not all 0; got {w.tolist()}")
    centers = [bool(center)] * len(blocks) if isinstance(center, (bool, int)) else [bool(c) for c in center]
    if len(centers) != len(blocks):
        raise ValueError(f"center: one bool, or one per array of features ({len(blocks)})")
    sim = np.zeros((n, n))
    for b, wk, c in zip(blocks, w, centers):
        if c and n > 1:
            b = b - b.mean(0, keepdims=True)
        norm = np.linalg.norm(b, axis=1, keepdims=True)
        b = np.divide(b, norm, out=np.zeros_like(b), where=norm > 0)
        sim += wk * (b @ b.T)
    return sim / w.sum()


def similar_groups(features, max_views: int, *, weights=None, center=False, fill: bool = True,
                   order=None) -> list[list[int]]:
    """Groups of up to ``max_views`` photos of one subject by what they show, for :func:`predict_subject`'s ``groups``.

    ``features``: one ``(n, d)`` array (one row per photo) or a list of them, compared by :func:`photo_similarity`
    (``weights``, ``center``); e.g. how much of each class a model finds in each photo (a close-up then goes with the
    photos showing the same things, whatever its metadata says) and the photo's appearance (``photo_embed`` of
    :func:`predict_batch`; very close close-ups of one part stay together even when the model reads one of them on the
    other side). Each photo still free, in ``order`` (default: input order; photos it leaves out follow in input
    order), starts a group with the free photos most like it (ties: the closest in the input). With ``fill``, a group
    left short tops up with the photos most like its first one among those already grouped, as context
    (:func:`subject_groups`): a subject of 5 photos at 4 a time gives two groups of 4.
    """
    sim = photo_similarity(features, weights=weights, center=center)
    n = len(sim)
    if n == 0:
        return []
    k = max(1, int(max_views))
    seeds = list(dict.fromkeys(int(i) for i in (order if order is not None else range(n))))
    if any(not 0 <= i < n for i in seeds):
        raise ValueError(f"order: photo indices must be in 0..{n - 1}")
    seeds += [i for i in range(n) if i not in set(seeds)]
    free, groups = set(range(n)), []
    for i in seeds:
        if i not in free:
            continue
        ranked = sorted((j for j in range(n) if j != i), key=lambda j: (-sim[i, j], abs(j - i), j))
        group = [i] + [j for j in ranked if j in free][: k - 1]
        free -= set(group)
        if fill and len(group) < k:
            group += [j for j in ranked if j not in group][: k - len(group)]
        groups.append(group)
    return groups


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
    images: list,
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
    fill: bool = False,
    prepared: list | None = None,
    sizes: list | None = None,
) -> list[dict] | dict:
    """Predict every photo (PIL images or RGB arrays) of one subject, any number of them, together with other photos of
    it: grouped by :func:`subject_groups` (``groups`` gives them, e.g. each overview with the close-ups it shows, or
    :func:`similar_groups`; by default consecutive photos, ``fill`` topping the last group up), one :func:`predict_set`
    per group. A photo in several groups gets its result from the first (:func:`group_owners`). Returns one result per
    photo, in input order. ``prepared`` / ``sizes``: one per photo, as in :func:`predict_batch`.

    ``identities=True`` also links the detections of every photo with those of every other, across the groups too
    (:func:`link_results`), and returns ``{"images": [one result per photo, each with "identity_ids"], "identities":
    [one record per identity: its members, fused attributes and regression values], "matches", "config"}``.
    """
    for name, given in (("prepared", prepared), ("sizes", sizes)):
        if given is not None and len(given) != len(images):
            raise ValueError(f"{name}: {len(given)} entries for {len(images)} photos")
    results: list[dict | None] = [None] * len(images)
    sets = subject_groups(model, len(images), groups, fill=fill)
    kw = dict(device=device, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det, mask_thresh=mask_thresh,
              letterbox=letterbox, amp=amp)
    pick = lambda seq, group: None if seq is None else [seq[i] for i in group]  # noqa: E731
    for group in sets:
        res = predict_set(model, [images[i] for i in group], prepared=pick(prepared, group), sizes=pick(sizes, group),
                          **kw)
        for i, r in zip(group, res):
            if results[i] is None:  # the first group listing a photo gives its result; a later one shows it only
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
    fill: bool = False,
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
    each of its subfolders is another, whose annotated images keep the subfolder under ``save``; ``fill`` tops each
    subject's last group up with the photos before it (:func:`subject_groups`).
    """
    if subject and (augment or tiles):
        raise ValueError("predict(subject=True) predicts the photos together, without augment / tiles.")
    if isinstance(model, (str, Path)):
        model = load_model(model, device=device)
    gen = _predict_iter(model, source, plot=plot, save=save, conf_thres=conf_thres, max_det=max_det,
                        mask_thresh=mask_thresh, imgsz=imgsz, batch=batch, amp=amp, augment=augment, tiles=tiles,
                        subject=subject, fill=fill, alpha=alpha, draw_boxes=draw_boxes, show_scores=show_scores,
                        color_by=color_by, verbose=verbose)
    return gen if stream else list(gen)


def _predict_iter(model, source, *, plot, save, conf_thres, max_det, mask_thresh, imgsz, batch, amp, augment, tiles,
                  subject, fill, alpha, draw_boxes, show_scores, color_by, verbose):
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
            results = predict_subject(model, images, fill=fill, **kw)
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
