#!/usr/bin/env python
"""Track segmentation instances (+ smoothed attributes) across a video.

    python scripts/track.py runs/train/eomt-l video.mp4 --conf 0.3
    python scripts/track.py PATH/TO/WEIGHTS video.mp4 --trace       # draw motion trails
    python scripts/track.py PATH/TO/WEIGHTS video.mp4 --with-attr   # compound cls-attr tracks
    python scripts/track.py PATH/TO/WEIGHTS video.mp4 --details     # per-frame attribute evolution

Writes an annotated .mp4 to runs/track/, colored by persistent track id. Each track is
labelled with its id, class and — for models with auxiliary heads — the temporally
smoothed attribute (steadier than per-frame predict output).
"""

from __future__ import annotations

import argparse

from eomt import EoMT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="Checkpoint .pt or a run/weights folder.")
    p.add_argument("source", help="Video file (mp4/avi/...).")
    p.add_argument("--out", default="runs/track", help="Output directory for the annotated video.")
    p.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    p.add_argument("--imgsz", type=int, default=None, help="Inference image size (defaults to trained size).")
    p.add_argument("--min-hits", type=int, default=3, help="Frames a track must persist before it is drawn.")
    p.add_argument("--track-buffer", type=int, default=None, help="ByteTrack lost_track_buffer (frames to keep an id after it leaves view).")
    p.add_argument("--with-attr", action="store_true", help="Treat attributes as part of the class (compound cls-attr tracking).")
    p.add_argument("--details", action="store_true", help="Record per-frame attribute evolution per track (returns {frames, tracks}).")
    p.add_argument("--trace", action="store_true", help="Draw per-track motion trails.")
    p.add_argument("--no-hud", action="store_true", help="Hide the Frame/Tracks banner.")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    tracker_kwargs = {"lost_track_buffer": args.track_buffer} if args.track_buffer else None
    model = EoMT(args.weights, device=args.device)
    out = model.track(
        args.source, plot=True, save=args.out,
        conf_thres=args.conf, imgsz=args.imgsz, min_hits=args.min_hits,
        with_attr=args.with_attr, details=args.details,
        trace=args.trace, hud=not args.no_hud, tracker_kwargs=tracker_kwargs,
    )
    n_frames = len(out["frames"]) if args.details else len(out)
    print(f"[done] tracked {n_frames} frame(s); annotated video under {args.out}")


if __name__ == "__main__":
    main()
