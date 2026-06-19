#!/usr/bin/env python3
"""Convert the YOLO ``parts`` dataset to (main class + per-instance laterality).

The source is a YOLO-segmentation dataset whose 110 class names encode the side
in the name (e.g. ``front_left_door``, ``front_right_door``, ``left_mirror``).
This rewrites it into:

  * a **main class** = the part name with the ``left`` / ``right`` tokens removed
    (so ``front_left_door`` and ``front_right_door`` both become ``front_door``),
    keeping any explicit ``center`` token in the name (``upper_center_front_bumper``
    stays as-is unless ``--strip-center`` is given); and
  * a **per-instance laterality subclass** in {center, left, right}: ``left`` /
    ``right`` if that token is present in the original name, otherwise ``center``.

Two output formats (``--format``):

``coco`` (default) — trainable directly by this repo. Writes COCO JSON with the
per-instance ``attributes`` extension the secondary-cls head reads::

    <dst>/
      data.yaml                         # this repo's schema (train_json/val_json/...)
      annotations/instances_<split>.json  # categories=main classes; top-level
                                          #   `attributes` head + per-ann
                                          #   `attributes: {laterality: id}`
      images/<split>                    # symlink (default) / copy of source images
      class_map.csv

``yolo`` — keeps the YOLO layout and adds a line-aligned laterality sidecar
(mirrors the aux-cls-examples ``typ_labels/`` convention)::

    <dst>/
      data.yaml                         # merged `names` + a `laterality` head
      labels/<split>/<stem>.txt         # YOLO seg lines, first field = main class id
      laterality/<split>/<stem>.txt     # one side id per line, LINE-ALIGNED
      images/<split>
      class_map.csv

Side ids: ``0=center, 1=left, 2=right``. YOLO polygon coords are normalized; the
COCO format denormalizes them to absolute pixels (the only step that reads image
sizes — header-only, no full decode).

Usage::

    python scripts/convert_parts_laterality.py --src data/parts/main --format coco
    python scripts/convert_parts_laterality.py --src data/parts/main --format yolo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

LATERALITY = ["center", "left", "right"]  # contiguous ids: center=0, left=1, right=2
SIDE_ID = {name: i for i, name in enumerate(LATERALITY)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def split_name(name: str, *, strip_center: bool) -> tuple[str, str]:
    """Return ``(main_class_name, side)`` for an original class name.

    Drops the ``left`` / ``right`` tokens (and ``center`` too when ``strip_center``)
    from the underscore-separated name; ``side`` is ``left`` / ``right`` if such a
    token was present, else ``center``.
    """
    side = "center"
    kept: list[str] = []
    for tok in name.split("_"):
        if tok == "left":
            side = "left"
        elif tok == "right":
            side = "right"
        elif tok == "center" and strip_center:
            continue  # fold center variants into the same main class
        else:
            kept.append(tok)
    return "_".join(kept), side


def build_maps(names: list[str], *, strip_center: bool):
    """Build old->new class maps from the ordered list of original names.

    Returns ``(main_names, old2new, old2side)`` where ``main_names`` is the sorted
    list of merged main-class names (its index is the new class id).
    """
    per_old = [split_name(n, strip_center=strip_center) for n in names]
    main_names = sorted({m for m, _ in per_old})
    name2id = {m: i for i, m in enumerate(main_names)}
    old2new = [name2id[m] for m, _ in per_old]
    old2side = [SIDE_ID[s] for _, s in per_old]
    return main_names, old2new, old2side


def _iter_label_files(d: Path):
    """Yield ``*.txt`` paths under ``d`` without materializing all entries at once."""
    with os.scandir(d) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".txt"):
                yield Path(entry.path)


def _image_index(images_dir: Path) -> dict[str, str]:
    """Map ``stem -> filename`` for images present in ``images_dir``."""
    idx: dict[str, str] = {}
    if not images_dir.exists():
        return idx
    with os.scandir(images_dir) as it:
        for e in it:
            if e.is_file() and Path(e.name).suffix.lower() in IMG_EXTS:
                idx.setdefault(Path(e.name).stem, e.name)
    return idx


def _poly_area(xs: list[float], ys: list[float]) -> float:
    """Shoelace polygon area (absolute), for COCO ``area`` (drives size-based AP)."""
    n = len(xs)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0


def _parse_line(raw: str, n_old: int):
    """Parse a YOLO line -> ``(old_id, coords)`` or ``None`` if malformed."""
    parts = raw.split()
    if len(parts) < 7:  # class id + at least 3 (x, y) points
        return None
    try:
        old = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if not (0 <= old < n_old) or len(coords) % 2 != 0:
        return None
    return old, coords


# --------------------------------------------------------------------------- #
# YOLO output (labels + laterality sidecar)
# --------------------------------------------------------------------------- #
def convert_split_yolo(src_labels, dst_labels, dst_lat, old2new, old2side, *, n_old):
    dst_labels.mkdir(parents=True, exist_ok=True)
    dst_lat.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "instances": 0, "bad": 0, "sides": [0] * len(LATERALITY)}

    for src_path in _iter_label_files(src_labels):
        out_lines, side_lines = [], []
        for raw in src_path.read_text().splitlines():
            if not raw.strip():
                continue
            parsed = _parse_line(raw, n_old)
            if parsed is None:
                stats["bad"] += 1
                continue
            old, coords = parsed
            rest = " ".join(f"{c:g}" for c in coords)
            out_lines.append(f"{old2new[old]} {rest}")
            side = old2side[old]
            side_lines.append(str(side))
            stats["sides"][side] += 1
            stats["instances"] += 1

        (dst_labels / src_path.name).write_text(
            "\n".join(out_lines) + ("\n" if out_lines else "")
        )
        (dst_lat / src_path.name).write_text(
            "\n".join(side_lines) + ("\n" if side_lines else "")
        )
        stats["files"] += 1
        if stats["files"] % 5000 == 0:
            print(f"    {stats['files']} files...", flush=True)
    return stats


# --------------------------------------------------------------------------- #
# COCO output (instances_<split>.json with per-annotation attributes)
# --------------------------------------------------------------------------- #
def convert_split_coco(src_labels, images_dir, out_json, old2new, old2side, main_names, *, n_old):
    from PIL import Image

    img_index = _image_index(images_dir)
    images: list[dict] = []
    annotations: list[dict] = []
    stats = {"files": 0, "instances": 0, "bad": 0, "missing_img": 0,
             "sides": [0] * len(LATERALITY)}
    ann_id = 1
    img_id = 0

    for src_path in _iter_label_files(src_labels):
        fname = img_index.get(src_path.stem)
        if fname is None:
            stats["missing_img"] += 1
            continue
        try:
            with Image.open(images_dir / fname) as im:
                w, h = im.size  # header read only, no full decode
        except Exception:
            stats["missing_img"] += 1
            continue

        img_id += 1
        images.append({"id": img_id, "file_name": fname, "width": int(w), "height": int(h)})

        for raw in src_path.read_text().splitlines():
            if not raw.strip():
                continue
            parsed = _parse_line(raw, n_old)
            if parsed is None:
                stats["bad"] += 1
                continue
            old, coords = parsed
            xs = [coords[i] * w for i in range(0, len(coords), 2)]
            ys = [coords[i] * h for i in range(1, len(coords), 2)]
            poly = [round(v, 2) for pair in zip(xs, ys) for v in pair]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            side = old2side[old]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": old2new[old] + 1,  # 1-based COCO category ids
                "segmentation": [poly],
                "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                "area": round(_poly_area(xs, ys), 2),
                "iscrowd": 0,
                "attributes": {"laterality": side},
            })
            ann_id += 1
            stats["instances"] += 1
            stats["sides"][side] += 1

        stats["files"] += 1
        if stats["files"] % 5000 == 0:
            print(f"    {stats['files']} images...", flush=True)

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(main_names)]
    attributes = [{
        "name": "laterality",
        "categories": [{"id": SIDE_ID[s], "name": s} for s in LATERALITY],
    }]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(
            {"images": images, "annotations": annotations,
             "categories": categories, "attributes": attributes},
            f, separators=(",", ":"),
        )
    return stats


def link_images(src_images: Path, dst_images: Path, mode: str) -> None:
    """Make the converted root self-contained: symlink / copy / skip images."""
    if mode == "skip" or not src_images.exists():
        if mode != "skip" and not src_images.exists():
            print(f"  [warn] no images at {src_images}; skipping image linking")
        return
    dst_images.parent.mkdir(parents=True, exist_ok=True)
    if dst_images.is_symlink() or dst_images.exists():
        if dst_images.is_symlink() or dst_images.is_file():
            dst_images.unlink()
        else:
            shutil.rmtree(dst_images)
    if mode == "link":
        dst_images.symlink_to(src_images.resolve(), target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(src_images, dst_images)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, default=Path("data/parts/main"),
                    help="Source dataset root (contains data.yaml, images/, labels/).")
    ap.add_argument("--dst", type=Path, default=None,
                    help="Output root (default: data/parts/main_coco | main_laterality).")
    ap.add_argument("--format", choices=["coco", "yolo"], default="coco",
                    help="coco (trainable here) | yolo (labels + laterality sidecar).")
    ap.add_argument("--splits", default="train,val", help="Comma-separated splits.")
    ap.add_argument("--images", choices=["link", "copy", "skip"], default="link",
                    help="How to provide images in the output root (default: symlink).")
    ap.add_argument("--strip-center", action="store_true",
                    help="Also drop the 'center' token so left/right/center variants "
                         "merge into one main class (default: keep 'center' in name).")
    args = ap.parse_args()

    if args.dst is None:
        args.dst = Path(f"data/parts/main_{'coco' if args.format == 'coco' else 'laterality'}")

    src_yaml = args.src / "data.yaml"
    if not src_yaml.exists():
        print(f"[error] {src_yaml} not found", file=sys.stderr)
        return 1
    names = list(yaml.safe_load(src_yaml.read_text())["names"])
    n_old = len(names)

    main_names, old2new, old2side = build_maps(names, strip_center=args.strip_center)
    print(f"[map] {n_old} original classes -> {len(main_names)} main classes "
          f"(format={args.format}, strip_center={args.strip_center})")

    args.dst.mkdir(parents=True, exist_ok=True)
    with (args.dst / "class_map.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_id", "old_name", "new_id", "new_name", "side_id", "side"])
        for old, name in enumerate(names):
            w.writerow([old, name, old2new[old], main_names[old2new[old]],
                        old2side[old], LATERALITY[old2side[old]]])

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    done_splits: list[str] = []
    totals = {"files": 0, "instances": 0, "bad": 0, "missing_img": 0,
              "sides": [0] * len(LATERALITY)}

    for split in splits:
        src_labels = args.src / "labels" / split
        if not src_labels.exists():
            print(f"[skip] no labels for split '{split}' ({src_labels})")
            continue
        print(f"[split] {split}: converting ({args.format})...")
        if args.format == "coco":
            st = convert_split_coco(
                src_labels, args.src / "images" / split,
                args.dst / "annotations" / f"instances_{split}.json",
                old2new, old2side, main_names, n_old=n_old,
            )
        else:
            st = convert_split_yolo(
                src_labels, args.dst / "labels" / split, args.dst / "laterality" / split,
                old2new, old2side, n_old=n_old,
            )
        link_images(args.src / "images" / split, args.dst / "images" / split, args.images)
        extra = f", missing_img={st['missing_img']}" if "missing_img" in st else ""
        print(f"  {split}: {st['files']} files, {st['instances']} instances, "
              f"sides(center/left/right)={st['sides']}, bad_lines={st['bad']}{extra}")
        for k in ("files", "instances", "bad", "missing_img"):
            totals[k] += st.get(k, 0)
        for i in range(len(LATERALITY)):
            totals["sides"][i] += st["sides"][i]
        done_splits.append(split)

    # output data.yaml
    if args.format == "coco":
        cfg = {"path": str(args.dst.resolve())}
        for split in done_splits:
            cfg[f"{split}_images"] = f"images/{split}"
            cfg[f"{split}_json"] = f"annotations/instances_{split}.json"
        cfg["download"] = False
        header = (
            "# Generated by scripts/convert_parts_laterality.py (--format coco)\n"
            "# Main classes (categories) = original part names with left/right removed.\n"
            "# Per-instance laterality is in each annotation's `attributes: {laterality: id}`\n"
            "# (0=center, 1=left, 2=right); the secondary-cls head is auto-discovered.\n"
        )
    else:
        cfg = {
            "path": str(args.dst.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(main_names),
            "names": main_names,
            "laterality": LATERALITY,
        }
        header = (
            "# Generated by scripts/convert_parts_laterality.py (--format yolo)\n"
            "# Main classes = original part names with left/right tokens removed.\n"
            "# Per-instance laterality sidecar: laterality/<split>/<stem>.txt\n"
            "#   (0=center, 1=left, 2=right), line-aligned with labels/<split>/.\n"
        )
    (args.dst / "data.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False))

    miss = f", missing_img={totals['missing_img']}" if args.format == "coco" else ""
    print(f"[done] {totals['files']} files, {totals['instances']} instances "
          f"(center/left/right={totals['sides']}), bad_lines={totals['bad']}{miss}")
    print(f"[done] wrote {args.format} dataset to {args.dst} (images {args.images})")
    if args.format == "coco":
        print(f"[next] eomt train --data {args.dst / 'data.yaml'} --size s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
