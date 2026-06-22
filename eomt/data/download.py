"""Auto-download for the COCO 2017 dataset.

Fetches the official COCO 2017 images and annotations directly from
``cocodataset.org`` when the expected files are not present. The dataset is read
in its native COCO-JSON form (no label conversion), so the only artifacts needed
are the image folders and the ``instances_*2017.json`` annotation files.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

COCO_URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}


def _download_file(url: str, dest: Path) -> None:
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"download {dest.name}"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    print(f"[coco] extracting {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def ensure_coco(
    root: str | Path,
    *,
    train: bool = True,
    val: bool = True,
    keep_zips: bool = False,
) -> dict[str, Path]:
    """Ensure COCO 2017 exists under ``root``; download/extract missing parts.

    Layout produced::

        root/
          annotations/instances_{train,val}2017.json
          train2017/*.jpg
          val2017/*.jpg

    Returns a dict of resolved paths (``train_images``, ``train_json``,
    ``val_images``, ``val_json``).
    """
    root = Path(root)
    ann_dir = root / "annotations"

    need_ann = (train and not (ann_dir / "instances_train2017.json").exists()) or (
        val and not (ann_dir / "instances_val2017.json").exists()
    )
    if need_ann:
        zp = root / "annotations_trainval2017.zip"
        if not zp.exists():
            _download_file(COCO_URLS["annotations"], zp)
        _extract_zip(zp, root)
        if not keep_zips:
            zp.unlink(missing_ok=True)

    for split, enabled in (("train2017", train), ("val2017", val)):
        if not enabled:
            continue
        img_dir = root / split
        if img_dir.exists() and any(img_dir.iterdir()):
            continue
        zp = root / f"{split}.zip"
        if not zp.exists():
            _download_file(COCO_URLS[split], zp)
        _extract_zip(zp, root)
        if not keep_zips:
            zp.unlink(missing_ok=True)

    return {
        "train_images": root / "train2017",
        "train_json": ann_dir / "instances_train2017.json",
        "val_images": root / "val2017",
        "val_json": ann_dir / "instances_val2017.json",
    }
