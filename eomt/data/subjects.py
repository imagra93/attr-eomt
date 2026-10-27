"""Real multi-photo sets: several annotated photos of one subject, predicted together by the cross-view post-training
(``train(multiview={"real": {...}})``, :data:`eomt.engine.train.DEFAULT_MULTIVIEW`) and scored the same way.

**Data.** A COCO instance-segmentation file, as for single-photo training (images, annotations, categories and the
attribute schema), where:

* every image of a subject has ``"subject_id"`` (a JSON number or string; the photos with the same id show the same
  subject, e.g. the photos of one object taken from several sides). Images without one take no part in the real sets;
* an image may say how far its annotations go, ``"supervision"``:

  - ``"full"`` (the default): every instance is annotated, so the rest of the photo is background;
  - ``"positives"``: only some are. The annotated instances train as usual, and nothing unannotated is taught as
    background: the photo's unmatched queries take no "no object" loss;
  - ``"none"``: the photo is only input for the other photos of its set; it takes no loss at all;

* an annotation may carry ``"instance_id"`` (a JSON number or string): the same physical instance in every photo of the
  subject that shows it (unique within the subject; ``7`` and ``"7"`` are the same id). The matching head learns to link
  those across the photos of a set; an instance without
  one is linked to nothing (its id is unknown to the matching head, :data:`eomt.data.sets.UNKNOWN_IDS`);
* an annotation with ``"iscrowd": 1`` is a region to leave out of the loss (unsure pixels: something that may or may
  not be an instance). A query whose mask lies mostly inside it takes no "no object" loss, and validation drops the
  detections there. Single-photo training already skips these annotations.

A top-level ``"subjects"`` list (``{"id": ..., "name": ...}``) may describe the subjects; nothing needs it.
:func:`subjects_from_file_names` adds ``subject_id`` to a file whose names start with the subject (``<subject>_...``).

**Sets.** Every photo of a subject (that trains: supervision other than ``"none"``) is the first photo, the **anchor**,
of one set per epoch; its companions are other photos of the same subject, up to :attr:`SubjectSetConfig.max_views`.
A share ``p_shared`` of the sets take first the companions that show one of the anchor's linked instances (the evidence
cross-view attention is for), the rest at random. Each photo is augmented on its own, as single-photo training would
(they are different photos), but the horizontal flip is drawn once per set: left and right of one subject never mix.

:class:`MixedSets` mixes these sets with the synthetic ones of :class:`eomt.data.sets.CocoViewSets`, a share ``p`` real.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .coco import CocoInstanceSeg, _attr_ignored, _attr_label, _attr_tensor, _follow_aug
from .sets import UNKNOWN_IDS, _draw
from .transforms import _PROB_FIELDS, AugConfig, TrainAugment, _coin, _randint

#: ``supervision`` of an image -> the weight of its unmatched queries' "no object" loss
SUPERVISION = {"full": 1.0, "positives": 0.0, "none": 0.0}


@dataclass
class SubjectSetConfig:
    """How the real sets are drawn: at most ``max_views`` photos; ``sizes`` (``{photos: weight}``, 2 to ``max_views``)
    the size of a set of several (default: as many photos as the subject has, up to ``max_views``); ``p_single`` sets of
    the anchor alone; ``p_shared`` sets whose companions are first the photos sharing a linked instance with the
    anchor."""

    max_views: int = 4
    sizes: dict | None = None
    p_single: float = 0.0
    p_shared: float = 0.5

    def __post_init__(self) -> None:
        if self.max_views < 2:
            raise ValueError("SubjectSetConfig: max_views < 2")
        if self.sizes is not None:
            self.sizes = {int(n): float(w) for n, w in self.sizes.items()}
            if not self.sizes or any(not 2 <= n <= self.max_views for n in self.sizes) or sum(self.sizes.values()) <= 0:
                raise ValueError(f"SubjectSetConfig: sizes {self.sizes} must weigh set sizes from 2 to max_views")


class CocoSubjectSets(CocoInstanceSeg):
    """Real sets of 1..``max_views`` photos of one subject (see the module docstring).

    Item: ``{"views": [(image, masks, classes, attrs, inst_ids), ...], "case": "real", "extras": [{"ignore": (S, S)
    float, "neg": float}, ...], "photos": [image id, ...]}``, the anchor first; ``extras`` holds per view its ignore
    map (1 = left out) and the weight of its unmatched queries' "no object" loss. ``seed`` fixes every set
    (validation); ``None`` draws new ones. ``augment=False`` (validation) only fits each photo to the input, with no
    random augmentation (and no flip).
    """

    def __init__(self, img_dir, json_file, imgsz: int = 644, *, aug: AugConfig | dict | None = None,
                 sets: SubjectSetConfig | dict | None = None, seed: int | None = None, augment: bool = True, **kw):
        super().__init__(img_dir, json_file, imgsz, aug=aug, keep_empty=True, **kw)
        if not augment:
            self.aug = AugConfig.resolve(self.aug, min_scale=1.0, max_scale=1.0, **dict.fromkeys(_PROB_FIELDS, 0.0))
        self.sets = sets if isinstance(sets, SubjectSetConfig) else SubjectSetConfig(**(sets or {}))
        self.seed = seed
        base = AugConfig.resolve(self.aug, mosaic_prob=0.0, mixup_prob=0.0, copy_paste_prob=0.0)
        self._tf = {flip: TrainAugment(imgsz, AugConfig.resolve(base, flip_prob=float(flip))) for flip in (0, 1)}
        self._members: dict = {}
        self._linked: dict[int, set] = {}
        self._supervision: dict[int, str] = {}
        self._inst_code: dict = {}  # (subject, instance_id as text) -> int id
        for i, img_id in enumerate(self.ids):
            info = self.coco.loadImgs(img_id)[0]
            sup = info.get("supervision", "full")
            if sup not in SUPERVISION:
                raise ValueError(f"image {img_id}: supervision {sup!r} is not one of {sorted(SUPERVISION)}")
            self._supervision[i] = sup
            subject = info.get("subject_id")
            if subject is None:
                continue
            self._members.setdefault(subject, []).append(i)
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
            self._linked[i] = {self._instance_code(subject, a["instance_id"]) for a in anns
                               if a.get("instance_id") is not None}
        self._subject_of = {i: s for s, m in self._members.items() for i in m}
        self.anchors = [i for i in sorted(self._subject_of) if self._supervision[i] != "none"]
        if not self.anchors:
            raise ValueError(f"{json_file}: no image with a subject_id (and supervision other than 'none')")

    def __len__(self) -> int:
        return len(self.anchors)

    @property
    def num_subjects(self) -> int:
        return len(self._members)

    def _instance_code(self, subject, instance_id) -> int:
        """The ``instance_id`` of an annotation of ``subject`` as a non-negative int (a JSON number or string)."""
        return self._inst_code.setdefault((subject, str(instance_id)), len(self._inst_code))

    def __getitem__(self, i: int) -> dict:
        if self.seed is None:
            return self._make_set(i)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed * 100_003 + i)
            return self._make_set(i)

    def companions(self, anchor: int, size: int) -> list[int]:
        """Up to ``size - 1`` other photos of the anchor's subject: with probability ``p_shared`` those sharing a linked
        instance with it first, the rest at random."""
        others = [j for j in self._members[self._subject_of[anchor]] if j != anchor]
        order = list(torch.randperm(len(others)).tolist())
        others = [others[k] for k in order]
        if _coin(self.sets.p_shared):
            mine = self._linked.get(anchor, set())
            others.sort(key=lambda j: not (self._linked.get(j, set()) & mine))  # stable: shared first, then random
        return others[: size - 1]

    def _make_set(self, i: int) -> dict:
        c = self.sets
        anchor = self.anchors[i]
        n_others = len(self._members[self._subject_of[anchor]]) - 1
        if not n_others or _coin(c.p_single):
            size = 1
        else:
            size = _draw(c.sizes) if c.sizes else c.max_views
            size = min(size, n_others + 1)
        flip = _coin(self.aug.flip_prob)
        photos = [anchor, *self.companions(anchor, size)]
        views, extras = [], []
        for slot, idx in enumerate(photos):
            *view, ignore, neg = self._real_view(idx, flip, slot)
            views.append(tuple(view))
            extras.append({"ignore": ignore, "neg": neg})
        return {"views": views, "case": "real", "extras": extras, "photos": [self.ids[i] for i in photos]}

    def _load_subject_raw(self, idx: int):
        """One photo before augmentation: ``(image_tv, masks_tv, class_t, attr_t, inst_ids, ignore)``, with the
        instances of a ``"none"`` photo dropped and the crowd annotations merged into ``ignore`` (H, W) uint8."""
        from torchvision import tv_tensors

        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        w, h = img.size
        masks, classes, ids = [], [], []
        attrs: dict[str, list] = {s.name: [] for s in self.aux_specs}
        ignore = np.zeros((h, w), np.uint8)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            m = self.coco.annToMask(ann)
            if m.sum() == 0:
                continue
            if ann.get("iscrowd"):
                ignore |= m
                continue
            if self._supervision[idx] == "none":
                continue
            masks.append(torch.from_numpy(m))
            cls = self.cat2contig[ann["category_id"]]
            classes.append(cls)
            ids.append(-1 if ann.get("instance_id") is None
                       else self._instance_code(info["subject_id"], ann["instance_id"]))
            for spec in self.aux_specs:
                attrs[spec.name].append(_attr_label(spec, self._attr_id_maps, ann.get("attributes") or {}, cls))
        image_tv = tv_tensors.Image(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        n = len(masks)
        masks_t = torch.stack(masks) if n else torch.zeros((0, h, w), dtype=torch.uint8)
        class_t = torch.tensor(classes, dtype=torch.long)
        attr_t = {s.name: _attr_tensor(s, attrs[s.name]) if n else _attr_ignored(s, 0) for s in self.aux_specs}
        return image_tv, masks_t, class_t, attr_t, torch.tensor(ids, dtype=torch.long), torch.from_numpy(ignore)

    def _real_view(self, idx: int, flip: bool, slot: int):
        """One augmented photo of a set: ``(image, masks, classes, attrs, inst_ids, ignore, neg)``. The ignore map rides
        the augmentation as one more mask (with no weight in the instance-aware crops) and is split off after it."""
        from torchvision import tv_tensors

        image_tv, masks_t, class_t, attr_t, ids, ignore = self._load_subject_raw(idx)
        n = class_t.numel()
        masks_tv = tv_tensors.Mask(torch.cat([masks_t, ignore[None].to(masks_t.dtype)]))
        class_all = torch.cat([class_t, torch.zeros(1, dtype=torch.long)])
        attr_all = {k: torch.cat([v, _attr_ignored(s, 1)]) for (k, v), s in zip(attr_t.items(), self.aux_specs)}
        tf = self._tf[int(flip)]
        weights = np.r_[np.ones(n), 0.0] if n else None
        img, masks, cls, attrs = tf(image_tv, masks_tv, class_all, attr_all, weights=weights)
        ign, masks, cls = masks[-1], masks[:-1], cls[:-1]
        attrs = {k: v[:-1] for k, v in attrs.items()}
        info = {k: v[:-1] if torch.is_tensor(v) and v.ndim and len(v) == n + 1 else v for k, v in tf.last_info.items()}
        attrs = _follow_aug(self.aux_specs, attrs, info)
        keep = masks.flatten(1).sum(1) >= tf.cfg.min_mask_mass
        ids = torch.where(ids >= 0, ids, UNKNOWN_IDS - 1000 * slot - torch.arange(n))
        neg = SUPERVISION[self._supervision[idx]]
        return (img, masks[keep], cls[keep], {k: v[keep] for k, v in attrs.items()}, ids[keep],
                (ign > 0.5).float(), neg)


class MixedSets(torch.utils.data.Dataset):
    """Real and synthetic sets in one epoch: every real set once (``len(real)`` items) and synthetic sets for the rest,
    so that a share ``p`` of the epoch is real (``p = 1``: real sets only). Each synthetic item is drawn at random from
    ``synthetic``, so the epochs see different ones."""

    def __init__(self, real, synthetic, p: float):
        if not 0.0 < p <= 1.0:
            raise ValueError(f"MixedSets: the real share p={p} must be in (0, 1]")
        self.real, self.synthetic, self.p = real, synthetic, float(p)
        self.n_real = len(real)
        self.n_synthetic = 0 if p >= 1.0 or synthetic is None else round(self.n_real * (1.0 - p) / p)

    def __len__(self) -> int:
        return self.n_real + self.n_synthetic

    def __getitem__(self, i: int) -> dict:
        if i < self.n_real:
            return self.real[i]
        return self.synthetic[_randint(0, len(self.synthetic) - 1)]


def subjects_from_file_names(src: str | Path, dst: str | Path, sep: str = "_", *, min_photos: int = 2) -> dict:
    """Copy the COCO file ``src`` to ``dst`` with ``subject_id`` set on every image whose file name starts with
    ``<subject><sep>`` and whose subject has ``min_photos`` or more images. Returns ``{subjects, images}`` counted."""
    data = json.loads(Path(src).read_text())
    names = {}
    for im in data["images"]:
        base = im["file_name"].rsplit("/", 1)[-1]
        if sep in base:
            names.setdefault(base.split(sep, 1)[0], []).append(im)
    n_img = 0
    for subject, ims in names.items():
        if len(ims) >= min_photos:
            for im in ims:
                im["subject_id"] = subject
                n_img += 1
    Path(dst).write_text(json.dumps(data))
    return {"subjects": sum(1 for ims in names.values() if len(ims) >= min_photos), "images": n_img}


def collate_extras(batch: list[dict]):
    """The extras of a batch of sets, flattened as :func:`eomt.data.sets.collate_sets` flattens their views:
    ``{"ignore": [(S, S) float or None], "neg_weight": (B,) float}``, or None when no set has any."""
    if not any("extras" in item for item in batch):
        return None
    ignore, neg = [], []
    for item in batch:
        for k in range(len(item["views"])):
            e = item["extras"][k] if "extras" in item else {}
            ign = e.get("ignore")
            ignore.append(ign if ign is not None and bool(ign.any()) else None)
            neg.append(float(e.get("neg", 1.0)))
    return {"ignore": ignore, "neg_weight": torch.tensor(neg)}


__all__ = ["CocoSubjectSets", "MixedSets", "SubjectSetConfig", "SUPERVISION", "collate_extras",
           "subjects_from_file_names"]
