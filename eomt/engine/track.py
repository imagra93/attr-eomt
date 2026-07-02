"""Video object tracking for EoMT (per-frame inference + ByteTrack association).

Runs the still-image inference (:func:`~eomt.engine.predict.predict_image`) on every
frame of a video, associates detections across frames into persistent ``track_id``s
with ByteTrack, and (optionally) writes an annotated video. For models with auxiliary
attribute heads (e.g. ``laterality`` / ``typology``) the tracker associates on the
**main class + box only** — attributes flicker frame-to-frame and must not drive
identity — while each track keeps a running mean of the attribute probabilities and
reports a temporally-smoothed label (``aux_track``).

``cv2`` and ``supervision`` are imported lazily so that importing ``eomt`` or running
``predict`` never requires them; they are only needed when :func:`track` is called.
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
from .predict import predict_image

_AUX_DATA_PREFIX = "aux::"  # namespaces aux probs stashed in sv.Detections.data


def track(
    model,
    source: str,
    *,
    plot: bool = True,
    save: str | None = "runs/track",
    conf_thres: float = 0.3,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    device: str = "auto",
    imgsz: int | None = None,
    tracker_kwargs: dict | None = None,
    min_hits: int = 3,
    trace: bool = False,
    hud: bool = True,
    alpha: float = 0.35,
    show_scores: bool = True,
) -> list[dict]:
    """Track instances across a video, returning one result dict per frame.

    ``model`` may be a loaded :class:`~eomt.model.EoMTModel` or a checkpoint path / run
    folder (loaded with :func:`~eomt.serialization.load_model`). Each returned dict
    carries the usual :func:`~eomt.engine.predict.predict_image` keys
    (``boxes`` / ``scores`` / ``classes`` / optional ``masks`` / optional ``aux``) plus
    a persistent ``track_ids`` tensor aligned to the detections, a ``frame`` index, and
    — for models with secondary heads — ``aux_track`` with the temporally-smoothed
    attribute ids/probs per track. When ``plot`` is set, an annotated ``.mp4`` (colored
    by track id, optional ``Frame i/N | Tracks: n`` ``hud`` banner) is written under
    ``save`` and, alongside it, a ``<stem>_tracks.json`` per-track summary.

    Args:
        tracker_kwargs: forwarded to ``supervision.ByteTrack`` (e.g.
            ``{"lost_track_buffer": 90}`` to keep an id longer when a part leaves view).
        min_hits: a track must be seen this many *consecutive* frames before it is
            emitted or drawn — suppresses one-frame false-positive specks. Once a track
            clears the bar it stays confirmed for the rest of the video. Set ``1`` to
            emit every track immediately.
    """
    import cv2  # lazy: only needed for tracking
    import supervision as sv

    if isinstance(model, (str, Path)):
        model = load_model(model, device=device)

    dev = next(model.parameters()).device
    imgsz = int(imgsz if imgsz is not None else model.image_size)
    letterbox = bool(getattr(model, "preprocess_letterbox", False))
    names = getattr(model, "names", None)
    aux_specs = list(getattr(model, "aux_specs", []))
    aux_head_names = [s.name for s in aux_specs]
    aux_label_names = {s.name: s.names for s in aux_specs}
    aux_ns = {s.name: int(s.num_classes) for s in aux_specs}
    family = getattr(model, "family", "instance")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(
        f"[track] eomt-{getattr(model, 'size', '?')} ({family}, "
        f"nc={getattr(model, 'nc', '?')}, imgsz={imgsz}) on {dev} | "
        f"{source}: {W}x{H} @ {fps:.1f} fps, {n_frames or '?'} frames"
    )

    # ByteTrack is deprecated in supervision>=0.28 (removed in 0.30); we pin <0.30 and
    # keep using it per the chosen design. Silence the noisy per-call FutureWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        tracker = sv.ByteTrack(**(tracker_kwargs or {}))

    text_scale = max(0.4, W / 1600.0)
    thickness = max(1, round(W / 640.0))
    lookup = sv.ColorLookup.TRACK
    mask_ann = sv.MaskAnnotator(color_lookup=lookup, opacity=alpha) if family != "detect" else None
    box_ann = sv.BoxAnnotator(color_lookup=lookup, thickness=thickness)
    label_ann = sv.LabelAnnotator(color_lookup=lookup, text_scale=text_scale, smart_position=True)
    trace_ann = sv.TraceAnnotator(color_lookup=lookup, thickness=thickness) if trace else None

    save_dir = Path(save) if save else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    video_path = None
    if plot and save_dir is not None:
        video_path = str(save_dir / f"{Path(source).stem}_track.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Running per-track aux history: tid -> head -> [sum_probs (ns,), count].
    hist: dict[int, dict[str, list]] = {}
    # min_hits confirmation state and per-track summary accumulators.
    streak: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    confirmed: set[int] = set()
    summary: dict[int, dict] = {}

    results: list[dict] = []
    idx = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()  # BGR uint8
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            raw = predict_image(
                model, image, device=dev, imgsz=imgsz,
                conf_thres=conf_thres, max_det=max_det,
                mask_thresh=mask_thresh, letterbox=letterbox,
            )

            # ---- to sv.Detections (aux probs ride along in .data so they survive) ----
            n = int(raw["num_detections"])
            data = {}
            for name in aux_head_names:
                probs = raw.get("aux", {}).get(name, {}).get("probs")
                data[_AUX_DATA_PREFIX + name] = (
                    probs.cpu().numpy() if probs is not None else np.zeros((n, 0), np.float32)
                )
            det = sv.Detections(
                xyxy=raw["boxes"].cpu().numpy().reshape(-1, 4).astype(np.float32),
                confidence=raw["scores"].cpu().numpy().astype(np.float32),
                class_id=raw["classes"].cpu().numpy().astype(int),
                mask=(raw["masks"].cpu().numpy().astype(bool) if "masks" in raw else None),
                data=data,
            )

            tracked = tracker.update_with_detections(det)
            all_tids = (
                tracked.tracker_id.astype(int)
                if tracked.tracker_id is not None
                else np.zeros((len(tracked),), int)
            )

            # ---- min_hits: keep only tracks seen >= min_hits consecutive frames ----
            keep = np.zeros(len(tracked), bool)
            for k, tid in enumerate(all_tids):
                tid = int(tid)
                streak[tid] = (streak.get(tid, 0) + 1) if last_seen.get(tid) == idx - 1 else 1
                last_seen[tid] = idx
                if streak[tid] >= min_hits:
                    confirmed.add(tid)
                keep[k] = tid in confirmed
            tracked = tracked[keep]
            m = len(tracked)
            tids = all_tids[keep]

            # ---- rebuild an aligned per-frame result from the tracked detections ----
            res: dict = {
                "frame": idx,
                "num_detections": m,
                "boxes": torch.from_numpy(tracked.xyxy).float(),
                "scores": torch.from_numpy(np.asarray(tracked.confidence, np.float32)),
                "classes": torch.from_numpy(np.asarray(tracked.class_id, int)).long(),
                "track_ids": torch.from_numpy(tids).long(),
            }
            if tracked.mask is not None:
                res["masks"] = torch.from_numpy(tracked.mask).bool()

            # ---- aux: raw per-frame + temporally-smoothed per-track ----
            if aux_head_names:
                res["aux"], res["aux_track"] = {}, {}
                for name in aux_head_names:
                    # sv drops .data on empty frames; fall back to a correctly-shaped
                    # empty so downstream indexing stays aligned.
                    probs = np.asarray(
                        tracked.data.get(_AUX_DATA_PREFIX + name, np.zeros((m, aux_ns[name]), np.float32)),
                        np.float32,
                    ).reshape(m, aux_ns[name])
                    res["aux"][name] = {
                        "ids": torch.from_numpy(probs.argmax(1)).long() if m else torch.zeros(0, dtype=torch.long),
                        "probs": torch.from_numpy(probs),
                    }
                    smoothed = np.zeros_like(probs)
                    for j in range(m):
                        acc = hist.setdefault(int(tids[j]), {}).setdefault(
                            name, [np.zeros(probs.shape[1], np.float32), 0]
                        )
                        acc[0] += probs[j]
                        acc[1] += 1
                        smoothed[j] = acc[0] / acc[1]
                    res["aux_track"][name] = {
                        "ids": torch.from_numpy(smoothed.argmax(1)).long() if m else torch.zeros(0, dtype=torch.long),
                        "probs": torch.from_numpy(smoothed),
                    }

            # ---- per-track summary accumulators (confirmed detections only) ----
            cls_list = res["classes"].tolist()
            for j in range(m):
                tid = int(tids[j])
                s = summary.setdefault(tid, {"cls": {}, "first": idx, "last": idx, "n": 0})
                s["cls"][int(cls_list[j])] = s["cls"].get(int(cls_list[j]), 0) + 1
                s["last"] = idx
                s["n"] += 1

            # ---- render ----
            if writer is not None:
                scene = frame
                if m:
                    if mask_ann is not None and tracked.mask is not None:
                        scene = mask_ann.annotate(scene, tracked)
                    if trace_ann is not None:
                        scene = trace_ann.annotate(scene, tracked)
                    scene = box_ann.annotate(scene, tracked)
                    scene = label_ann.annotate(scene, tracked, labels=_labels(
                        m, tids, res, names, aux_head_names, aux_label_names, show_scores,
                    ))
                if hud:
                    _draw_hud(cv2, scene, idx + 1, n_frames, m, text_scale)
                writer.write(scene)

            if video_path is not None:
                res["video_path"] = video_path
            results.append(res)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    total = time.perf_counter() - t0
    fps_out = idx / total if total > 0 else float("inf")
    dest = f" -> {video_path}" if video_path else ""
    print(f"[track] done: {idx} frame(s) in {total:.2f} s ({fps_out:.1f} FPS){dest}")

    # ---- per-track summary: id -> class, smoothed attrs, frame span, frames seen ----
    tracks = _build_summary(summary, hist, names, aux_head_names, aux_label_names)
    if save_dir is not None:
        out = save_dir / f"{Path(source).stem}_tracks.json"
        out.write_text(json.dumps({"source": str(source), "frames": idx, "tracks": tracks}, indent=2) + "\n")
        print(f"[track] {len(tracks)} track(s) -> {out}")
    _print_summary(tracks)
    return results


def _build_summary(summary, hist, names, aux_head_names, aux_label_names) -> list[dict]:
    """One record per track: majority class, smoothed attributes, span, frames seen."""
    tracks = []
    for tid, s in sorted(summary.items()):
        cls = max(s["cls"], key=s["cls"].get)
        attrs = {}
        for name in aux_head_names:
            acc = hist.get(tid, {}).get(name)
            if acc and acc[1]:
                aid = int((acc[0] / acc[1]).argmax())
                attrs[name] = aux_label_names.get(name, {}).get(aid, str(aid))
        tracks.append({
            "track_id": tid,
            "class": cls,
            "class_name": (names.get(cls, str(cls)) if names else str(cls)),
            "attributes": attrs,
            "first_frame": s["first"],
            "last_frame": s["last"],
            "frames_seen": s["n"],
        })
    return tracks


def _print_summary(tracks: list[dict], top: int = 15) -> None:
    """Print a compact per-track table (most-seen first)."""
    if not tracks:
        print("[track] no confirmed tracks")
        return
    ranked = sorted(tracks, key=lambda t: t["frames_seen"], reverse=True)
    print(f"[track] top tracks by frames seen (of {len(tracks)}):")
    for t in ranked[:top]:
        attrs = " ".join(t["attributes"].values())
        span = f"{t['first_frame']}-{t['last_frame']}"
        print(f"  #{t['track_id']:>3} {t['class_name']:<24} {attrs:<14} frames {span} ({t['frames_seen']})")


def _draw_hud(cv2, scene, i: int, n: int, ntracks: int, scale: float) -> None:
    """Draw a top-left ``Frame i/N | Tracks: n`` banner (n = tracks in this frame)."""
    text = f"Frame {i}/{n or '?'} | Tracks: {ntracks}"
    fs = max(0.6, scale * 1.4)
    th = max(1, round(fs * 2))
    (tw, hh), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = int(hh * 0.5)
    cv2.rectangle(scene, (0, 0), (tw + 2 * pad, hh + base + 2 * pad), (0, 0, 0), -1)
    cv2.putText(scene, text, (pad, hh + pad), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), th, cv2.LINE_AA)


def _labels(m, tids, res, names, aux_head_names, aux_label_names, show_scores) -> list[str]:
    """Build per-detection label strings: ``#id class [score] [smoothed attrs]``."""
    classes = res["classes"].tolist()
    scores = res["scores"].tolist()
    out: list[str] = []
    for j in range(m):
        cls = int(classes[j])
        cname = names.get(cls, str(cls)) if names else str(cls)
        parts = [f"#{int(tids[j])} {cname}"]
        if show_scores:
            parts.append(f"{float(scores[j]):.2f}")
        for name in aux_head_names:
            aid = int(res["aux_track"][name]["ids"][j])
            parts.append(aux_label_names.get(name, {}).get(aid, str(aid)))
        out.append(" ".join(parts))
    return out
