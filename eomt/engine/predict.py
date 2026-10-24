"""Inference + rendering for EoMT instance segmentation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
    amp: bool | str = False,
    augment: bool | dict = False,
    tiles: bool | int | dict = False,
) -> dict:
    """Run the model on one PIL image and return a postprocess dict.

    Returns a box-only :func:`~eomt.postprocess.postprocess_detection` dict for
    ``family="detect"`` models, else a :func:`~eomt.postprocess.postprocess_instance`
    dict (with masks).

    ``amp`` runs the network under autocast (``True`` / ``"auto"`` = bf16 where supported, or
    ``"bf16"`` / ``"fp16"``): ~2.5x faster at 644 px on an RTX 5090 with the same accuracy (AP
    within 0.002 in every size bucket on a trained ViT-L model). Off by default.

    With ``embed``, the result also carries ``"embed"`` ``(N, hidden)``: the per-query
    embedding of each kept detection, sliced out of the same forward pass. It is the
    appearance fingerprint cross-photo re-identification matches on
    (:mod:`eomt.reid`). The vectors are **raw**, not normalized — normalizing (and
    optionally mean-centering) is :func:`eomt.reid.similarity_matrix`'s job, and it
    needs the raw vectors to do it.

    ``augment=True`` adds the image zoomed to 1.5x ``imgsz`` (test-time augmentation) and ``tiles=True`` runs overlapping
    native-resolution tiles plus the whole image (tiled inference); the views' instances are merged. Each also takes a
    dict of options (``tiles`` an int tile size); see :mod:`eomt.tta`. Instance family only. The result then has no
    ``query_idx`` (a merged instance comes from several queries); ``embed`` is the score-weighted mean of the merged
    queries' embeddings.
    """
    # Class-scoped aux heads: pass their primary-class scope so postprocess emits
    # ``ids = -1`` for detections the head does not apply to (the inference side of
    # the hard class-routing). Unscoped heads (``applies_to=None``) are omitted here.
    aux_scopes = {
        s.name: s.applies_to
        for s in getattr(model, "aux_specs", [])
        if s.applies_to is not None
    }
    cfg = TTAConfig.resolve(augment, tiles)
    if cfg is not None:
        return predict_views(
            model, np.array(image.convert("RGB")), cfg, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det,
            mask_thresh=mask_thresh, letterbox=letterbox, mean=getattr(model, "pixel_mean", None),
            std=getattr(model, "pixel_std", None), amp_dtype=_amp_dtype(device, amp), aux_scopes=aux_scopes,
            embed=embed,
        )
    orig_w, orig_h = image.size
    chw, meta = preprocess_numpy(
        np.array(image.convert("RGB")), imgsz, letterbox=letterbox,
        mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None),
    )
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    dtype = _amp_dtype(device, amp)
    if dtype is None:
        out = model(tensor)
    else:  # only the network runs in reduced precision; postprocess casts the logits back to fp32
        with torch.autocast("cuda", dtype=dtype):
            out = model(tensor)
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
    amp: bool | str = False,
) -> list[dict]:
    """Predict the photos of ONE subject together, in one forward pass: with a cross-view model (``xview``) the
    queries of each photo take what the others show; any other model predicts each photo as :func:`predict_image`.

    Returns one :func:`~eomt.postprocess.postprocess_instance` dict per photo, each with ``"embed"`` ``(N, d)``: the
    matching head's embedding of each kept detection when the model has one (``match_embed``, L2-normalized), else the
    per-query embedding (re-identification's fingerprint, :mod:`eomt.reid`). At most the model's ``max_views`` photos.
    """
    max_views = int((getattr(model, "xview_cfg", None) or {}).get("max_views", len(images)))
    if not 1 <= len(images) <= max_views:
        raise ValueError(f"predict_set takes 1 to {max_views} photos of one subject, got {len(images)}.")
    aux_scopes = {s.name: s.applies_to for s in getattr(model, "aux_specs", []) if s.applies_to is not None}
    chws, metas = zip(*(
        preprocess_numpy(np.array(im.convert("RGB")), imgsz, letterbox=letterbox,
                         mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None))
        for im in images
    ))
    x = torch.from_numpy(np.stack(chws)).to(device)
    set_ids = torch.zeros(len(images), dtype=torch.long)
    dtype = _amp_dtype(device, amp)
    if dtype is None:
        out = model(x, set_ids=set_ids)
    else:
        with torch.autocast("cuda", dtype=dtype):
            out = model(x, set_ids=set_ids)
    embed = out.get("match_embed", out["query_embed"])
    results = []
    for b, (im, meta) in enumerate(zip(images, metas)):
        res = postprocess_instance(slice_batch(out, b), conf_thres, im.size, max_det=max_det, mask_thresh=mask_thresh,
                                   preprocess_meta=meta, aux_scopes=aux_scopes)
        res["embed"] = embed[b, res["query_idx"]].float().cpu()
        results.append(res)
    return results


def subject_sets(feats: torch.Tensor, max_views: int, *, mode: str = "partition", k: int = 3) -> list[list[int]]:
    """The sets in which the photos of one subject are predicted together, from one feature row per photo.

    ``"order"``: consecutive photos, ``max_views`` at a time (``feats`` is only counted). ``"partition"``: each photo in
    exactly one set, of up to ``max_views`` photos — the first photo not yet placed and the most similar photos (cosine
    of ``feats``) not yet placed. Both predict every photo once. ``"anchor"``: one set per photo, the photo first with
    its ``k`` most similar others (``k`` capped at ``max_views - 1``); only the first photo's result of each set is its
    own. On real multi-photo sets (close-ups plus overviews of each subject) the three gave about the same gain.
    """
    if mode not in ("order", "partition", "anchor"):
        raise ValueError(f"subject_sets: mode {mode!r} is 'order', 'partition' or 'anchor'")
    if mode == "order":
        return [list(range(s, min(s + max_views, len(feats)))) for s in range(0, len(feats), max_views)]
    f = torch.nn.functional.normalize(feats.float(), dim=-1)
    order = (f @ f.T).argsort(dim=1, descending=True).tolist()
    if mode == "anchor":
        return [[t] + [j for j in order[t] if j != t][: min(k, max_views - 1)] for t in range(len(f))]
    placed, sets = set(), []
    for t in range(len(f)):
        if t in placed:
            continue
        group = [t] + [j for j in order[t] if j != t and j not in placed][: max_views - 1]
        placed.update(group)
        sets.append(group)
    return sets


@torch.no_grad()
def predict_subject(
    model,
    images: list[Image.Image],
    *,
    device,
    imgsz: int,
    mode: str = "order",
    k: int = 3,
    feature_size: int = 322,
    conf_thres: float = 0.3,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    letterbox: bool = True,
    amp: bool | str = False,
) -> list[dict]:
    """Predict every photo of one subject together with other photos of it: one result per photo, in
    input order, as :func:`predict_set` gives them.

    The photos are grouped by :func:`subject_sets`: ``mode="order"`` takes them ``max_views`` at a time in the given
    order; ``"partition"`` and ``"anchor"`` group them on the mean patch token of a pass at ``feature_size`` (a small
    input, about a quarter of the cost of a prediction at 644 px). ``"order"`` and ``"partition"`` predict every photo
    once, ``"anchor"`` once per photo with its ``k`` most similar others. A model without cross-view modules predicts
    each photo alone either way.
    """
    if not images:
        return []
    kw = dict(device=device, imgsz=imgsz, conf_thres=conf_thres, max_det=max_det, mask_thresh=mask_thresh,
              letterbox=letterbox, amp=amp)
    max_views = int((getattr(model, "xview_cfg", None) or {}).get("max_views", 1))
    if max_views < 2 or len(images) == 1:
        return [predict_set(model, [im], **kw)[0] for im in images]
    feats = torch.zeros(len(images), 1) if mode == "order" else _mean_patch_tokens(model, images, feature_size, **kw)
    return _predict_groups(model, images, subject_sets(feats, max_views, mode=mode, k=k), mode, **kw)


@torch.no_grad()
def _mean_patch_tokens(model, images, size, *, device, letterbox, amp, **_) -> torch.Tensor:
    """``(N, hidden)``: the mean patch token of each photo predicted alone at ``size``."""
    grab = {}
    hook = model.eomt.layernorm.register_forward_hook(lambda m, i, o: grab.__setitem__("x", o))
    pre = model.config.num_queries + model.eomt.embeddings.num_prefix_tokens
    dtype = _amp_dtype(device, amp)
    feats = []
    try:
        for im in images:
            chw, _ = preprocess_numpy(np.array(im.convert("RGB")), size, letterbox=letterbox,
                                      mean=getattr(model, "pixel_mean", None), std=getattr(model, "pixel_std", None))
            x = torch.from_numpy(chw)[None].to(device)
            if dtype is None:
                model(x)
            else:
                with torch.autocast("cuda", dtype=dtype):
                    model(x)
            feats.append(grab["x"][0, pre:].float().mean(0))
    finally:
        hook.remove()
    return torch.stack(feats)


def _predict_groups(model, images, groups, mode, **kw) -> list[dict]:
    """One :func:`predict_set` per group; every photo keeps its own result (the first photo only, for ``"anchor"``)."""
    results: list[dict | None] = [None] * len(images)
    for group in groups:
        res = predict_set(model, [images[i] for i in group], **kw)
        for i, r in zip(group if mode != "anchor" else group[:1], res):
            results[i] = r
    return results


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
    amp: bool | str = False,
    augment: bool | dict = False,
    tiles: bool | int | dict = False,
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
    per-instance appearance fingerprint (see :func:`predict_image`). ``amp`` runs the
    network under autocast; ``augment`` / ``tiles`` run test-time augmentation / tiled
    inference (see :func:`predict_image`).
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
            mask_thresh=mask_thresh, letterbox=letterbox, embed=embed, amp=amp,
            augment=augment, tiles=tiles,
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
