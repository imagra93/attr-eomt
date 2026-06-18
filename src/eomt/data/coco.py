"""COCO-format datasets for EoMT instance segmentation.

``CocoInstanceSeg`` yields augmented ``(pixel_values, instance_masks,
class_labels)`` for training. ``CocoValImages`` yields preprocessed images plus
their ``image_id`` / original size for COCO-mAP evaluation (ground truth comes
straight from the annotation JSON via :mod:`pycocotools`).

COCO category ids are non-contiguous (up to 90); both datasets remap them to a
contiguous ``0..nc-1`` range (sorted by category id), and expose ``contig2cat``
to convert predictions back to original ids for ``COCOeval``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..preprocess import preprocess_numpy
from .transforms import build_train_transform, build_val_transform


def _build_category_maps(coco):
    """Return ``(cat2contig, contig2cat, names, num_classes)`` from a COCO handle."""
    cat_ids = sorted(coco.getCatIds())
    cat2contig = {c: i for i, c in enumerate(cat_ids)}
    contig2cat = {i: c for c, i in cat2contig.items()}
    cats = coco.loadCats(cat_ids)
    names = {cat2contig[c["id"]]: c["name"] for c in cats}
    return cat2contig, contig2cat, names, len(cat_ids)


class CocoInstanceSeg(Dataset):
    """COCO instance segmentation -> ``(pixel_values, masks, class_labels)``.

    Each item is an augmented, ImageNet-normalized square tensor plus a stack of
    binary instance masks ``(num_inst, imgsz, imgsz)`` and their contiguous class
    ids. Images with no instances are filtered out at construction.
    """

    def __init__(
        self,
        img_dir: str | Path,
        json_file: str | Path,
        imgsz: int = 644,
        *,
        transform=None,
        flip_prob: float = 0.5,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
    ):
        from pycocotools.coco import COCO

        self.img_dir = Path(img_dir)
        self.imgsz = imgsz
        self.coco = COCO(str(json_file))
        self.cat2contig, self.contig2cat, self.names, self.num_classes = (
            _build_category_maps(self.coco)
        )
        self.ids = [
            i
            for i in self.coco.getImgIds()
            if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0
        ]
        self.transform = transform or build_train_transform(
            imgsz, flip_prob=flip_prob, min_scale=min_scale, max_scale=max_scale
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        from torchvision import tv_tensors

        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        masks, classes = [], []
        for ann in anns:
            m = self.coco.annToMask(ann)  # (H, W) uint8 at original resolution
            if m.sum() == 0:
                continue
            masks.append(torch.from_numpy(m))
            classes.append(self.cat2contig[ann["category_id"]])

        image_tv = tv_tensors.Image(
            torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (3, H, W) uint8
        )
        if masks:
            masks_tv = tv_tensors.Mask(torch.stack(masks))  # (N, H, W) uint8
            class_t = torch.tensor(classes, dtype=torch.long)
        else:  # should not happen (filtered), but stay safe
            masks_tv = tv_tensors.Mask(torch.zeros((1, *image_tv.shape[1:]), dtype=torch.uint8))
            class_t = torch.zeros((1,), dtype=torch.long)

        image_t, masks_t = self.transform(image_tv, masks_tv)

        # Drop instances cropped away by augmentation.
        areas = masks_t.flatten(1).sum(1)
        keep = areas > 0
        masks_t, class_t = masks_t[keep], class_t[keep]
        if masks_t.shape[0] == 0:  # degenerate after crop -> one empty instance
            masks_t = torch.zeros((1, self.imgsz, self.imgsz))
            class_t = torch.zeros((1,), dtype=torch.long)

        return image_t, masks_t.float(), class_t


class CocoValImages(Dataset):
    """COCO images for evaluation -> ``(pixel_values, image_id, orig_w, orig_h)``.

    Ground-truth annotations are read from the JSON by ``COCOeval`` directly, so
    this dataset only needs to deliver preprocessed pixels and identity/size.
    """

    def __init__(self, img_dir: str | Path, json_file: str | Path, imgsz: int = 644):
        from pycocotools.coco import COCO

        self.img_dir = Path(img_dir)
        self.imgsz = imgsz
        self.coco = COCO(str(json_file))
        self.cat2contig, self.contig2cat, self.names, self.num_classes = (
            _build_category_maps(self.coco)
        )
        self.ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        orig_w, orig_h = img.size
        chw, _ = preprocess_numpy(np.array(img), self.imgsz)
        return torch.from_numpy(chw), int(img_id), orig_w, orig_h


def collate_train(batch):
    """Stack pixel values; keep masks/classes as per-image lists (variable N)."""
    pixel_values = torch.stack([b[0] for b in batch])
    mask_labels = [b[1] for b in batch]
    class_labels = [b[2] for b in batch]
    return pixel_values, mask_labels, class_labels


def collate_val(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    image_ids = [b[1] for b in batch]
    sizes = [(b[2], b[3]) for b in batch]
    return pixel_values, image_ids, sizes
