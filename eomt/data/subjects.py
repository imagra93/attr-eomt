"""Sets of photos of one subject, predicted together by a multi-view post-training (``train(multiview={...})``,
:class:`eomt.engine.train.MultiviewConfig`) and scored the same way.

**Data.** A COCO instance-segmentation file, as for single-photo training (images, annotations, categories and the
attribute schema), where:

* every image of a subject has ``"subject_id"`` (a JSON number or string; the photos with the same id show the same
  subject, e.g. the photos of one object taken from several sides). Images without one take no part in the sets;
* an image may say how far its annotations go, ``"supervision"``:

  - ``"full"`` (the default): every instance is annotated, so the rest of the photo is background;
  - ``"positives"``: only some are. The annotated instances train as usual, and nothing unannotated is taught as
    background: the photo's unmatched queries take no "no object" loss;
  - ``"none"``: the photo is only input for the other photos of its set; it takes no loss at all;

* an annotation may carry ``"instance_id"`` (a JSON number or string): the same physical instance in every photo of the
  subject that shows it (unique within the subject; ``7`` and ``"7"`` are the same id). The matching head learns to link
  those across the photos of a set; an instance without one is linked to nothing (its id is unknown, negative);
* an annotation with ``"iscrowd": 1`` is a region to leave out of the loss (unsure pixels: something that may or may
  not be an instance). A query whose mask lies mostly inside it takes no "no object" loss, and validation drops the
  detections there. Single-photo training already skips these annotations.

:func:`subjects_from_file_names` adds ``subject_id`` to a file whose names start with the subject (``<subject>_...``).

**Sets.** Every photo of a subject (that trains: supervision other than ``"none"``) is the first photo, the **anchor**,
of one set per epoch; its companions are other photos of the same subject, up to :attr:`SubjectSetConfig.max_views`.
A share ``p_shared`` of the sets take first the companions that show one of the anchor's linked instances (the evidence
cross-view attention is for), the rest at random. Each photo is augmented on its own, as single-photo training would
(they are different photos), but the horizontal flip is drawn once per set: left and right of one subject never mix.
With ``p_hard`` the anchor is made hard (:data:`HARD_CASES`: shrunk, degraded, a close-up or occluded) while its
companions stay as they are, so the evidence has to come from the other views.

Fixed sets (validation) can be written to a JSON file (:meth:`CocoSubjectSets.write_sets`: the file names of each set's
photos, anchor first) and read back (``fixed=``), so that two models, or two versions of a dataset, are scored on the
same sets of the photos they share. Unaugmented (validation), every photo is fitted to the input exactly as inference
letterboxes it, so the same sets give the same inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .coco import CocoInstanceSeg, _attr_ignored, _attr_label, _attr_tensor, _follow_aug
from .transforms import _MEAN, _PROB_FIELDS, _STD, AugConfig, TrainAugment, _coin, _fraction, _mass, _rand, _randint, _uniform

#: ``supervision`` of an image -> the weight of its unmatched queries' "no object" loss
SUPERVISION = {"full": 1.0, "positives": 0.0, "none": 0.0}
#: Instance ids of the instances whose links are unknown (no ``instance_id``): negative, each photo of a set its own
#: block, so the matching loss and the link scores leave them out.
UNKNOWN_IDS = -100_000
#: The hard anchors (``SubjectSetConfig.p_hard``): :class:`~eomt.data.transforms.AugConfig` overrides of the run's
#: augmentation that make the anchor hard to read alone. The scales are relative to the photo fitted to the input (1.0 is
#: what inference sees). ``occluded`` also lays a band cut from another photo across part of one instance.
HARD_CASES: dict[str, dict] = {
    "small": {"min_scale": 0.35, "max_scale": 0.6},
    "degraded": {"min_scale": 0.75, "max_scale": 1.5, "blur_prob": 1.0, "blur_sigma": (1.5, 3.5), "noise_prob": 0.8,
                 "noise_std": (0.03, 0.08), "jpeg_prob": 0.8, "jpeg_quality": (10, 35), "gamma_prob": 0.7,
                 "gamma_range": (1.6, 2.6)},
    "closeup": {"zoom_prob": 1.0, "zoom_scale": (2.5, 4.0)},
    "occluded": {"min_scale": 0.75, "max_scale": 1.5},
}


def _draw(weights: dict):
    """A key of ``weights`` drawn in proportion to its weight."""
    keys = list(weights)
    cdf = np.cumsum([weights[k] for k in keys]) / sum(weights.values())
    return keys[min(int(np.searchsorted(cdf, _rand())), len(keys) - 1)]



@dataclass
class SubjectSetConfig:
    """How the sets are drawn: at most ``max_views`` photos.

    ``groups`` (an image field): the photos of a subject with the same value of that field are one set — the groups the
    photos are predicted in at inference (e.g. written by the dataset's build with the deployment's grouping rule). Every
    group is then a set once per epoch (a group larger than ``max_views`` is split), and ``p_random`` of the sets are
    drawn at random as below instead, an augmentation against learning only the fixed groups. Without ``groups``, every
    photo of a subject is the anchor of one random set per epoch.

    A random set: ``sizes`` (``{photos: weight}``, 2 to ``max_views``) the size of a set of several (default: as many
    photos as the subject has, up to ``max_views``); ``p_single`` sets of the anchor alone; ``p_shared`` sets whose
    companions are first the photos sharing a linked instance with the anchor. ``p_hard`` (training only, every set)
    the sets of several whose first photo is made hard, by one of the cases weighted by ``hard`` (``{case: weight}``,
    :data:`HARD_CASES`; default all alike); ``occluder_cover`` the share of an instance the ``occluded`` case covers."""

    max_views: int = 4
    groups: str | None = None
    p_random: float = 0.0
    sizes: dict | None = None
    p_single: float = 0.0
    p_shared: float = 0.5
    p_hard: float = 0.0
    hard: dict | None = None
    occluder_cover: tuple[float, float] = (0.3, 0.7)

    def __post_init__(self) -> None:
        if self.max_views < 2:
            raise ValueError("SubjectSetConfig: max_views < 2")
        if not 0.0 <= self.p_random < 1.0 or (self.p_random and not self.groups):
            raise ValueError(f"SubjectSetConfig: p_random {self.p_random} must be in [0, 1), with groups")
        if self.sizes is not None:
            self.sizes = {int(n): float(w) for n, w in self.sizes.items()}
            if not self.sizes or any(not 2 <= n <= self.max_views for n in self.sizes) or sum(self.sizes.values()) <= 0:
                raise ValueError(f"SubjectSetConfig: sizes {self.sizes} must weigh set sizes from 2 to max_views")
        self.hard = {c: float(w) for c, w in (self.hard or dict.fromkeys(HARD_CASES, 1.0)).items()}
        if set(self.hard) - set(HARD_CASES) or sum(self.hard.values()) <= 0:
            raise ValueError(f"SubjectSetConfig: hard {self.hard} must weigh cases of {sorted(HARD_CASES)}")
        self.occluder_cover = tuple(self.occluder_cover)


class _Occluder:
    """Lays a band cut from another photo of the dataset across part of an instance (the ``occluded`` case). Needs
    ``coco``, ``ids``, ``img_dir`` and the normalisation ``_mean`` / ``_std`` of the dataset it is mixed into."""

    def _occlude(self, img: torch.Tensor, masks: torch.Tensor, tf: TrainAugment, cover=(0.3, 0.7), n_inst=None):
        """Lay a band cut from another photo across part of one instance (a pole, an arm): its pixels and
        its share of every mask are gone, and the crop-aware attributes follow the visible share. Only the first
        ``n_inst`` masks (default: all) are instances the band may be laid on."""
        mass = _mass(masks)
        cand = (mass[:n_inst] >= 16).nonzero().flatten().tolist()
        if not cand:
            return img, masks
        i = cand[_randint(0, len(cand) - 1)]
        ys, xs = torch.nonzero(masks[i] > 0.05, as_tuple=True)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        s, share = img.shape[-1], _uniform(cover)
        if _coin(0.5):  # vertical band, longer than the instance
            w = max(2, round((x1 - x0) * share))
            bx, ext = _randint(x0, max(x0, x1 - w)), round((y1 - y0) * _uniform((0.2, 1.0)))
            box = (bx, max(0, y0 - ext), min(s, bx + w), min(s, y1 + ext))
        else:  # horizontal band
            h = max(2, round((y1 - y0) * share))
            by, ext = _randint(y0, max(y0, y1 - h)), round((x1 - x0) * _uniform((0.2, 1.0)))
            box = (max(0, x0 - ext), by, min(s, x1 + ext), min(s, by + h))
        bx0, by0, bx1, by1 = box
        img = img.clone()
        img[:, by0:by1, bx0:bx1] = self._patch(by1 - by0, bx1 - bx0)
        covered = masks.clone()
        covered[:, by0:by1, bx0:bx1] = 0.0
        tf.last_info["visible"] = tf.last_info["visible"] * _fraction(_mass(covered), mass)
        return img, covered

    def _patch(self, h: int, w: int) -> torch.Tensor:
        """A normalized ``(3, h, w)`` patch cut from another photo of the dataset, at a random scale."""
        info = self.coco.loadImgs(self.ids[_randint(0, len(self.ids) - 1)])[0]
        photo = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        pw, ph = photo.size
        k = min(pw / w, ph / h) * _uniform((0.3, 1.0))
        cw, ch = max(1, int(w * k)), max(1, int(h * k))
        x0, y0 = _randint(0, pw - cw), _randint(0, ph - ch)
        crop = photo.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1).float() / 255.0
        return (t - self._mean) / self._std


class CocoSubjectSets(_Occluder, CocoInstanceSeg):
    """Real sets of 1..``max_views`` photos of one subject (see the module docstring).

    Item: ``{"views": [(image, masks, classes, attrs, inst_ids), ...], "case": "real" (or "real_<hard case>"),
    "extras": [{"ignore": (S, S) float, "neg": float}, ...], "photos": [image id, ...]}``, the anchor first; ``extras``
    holds per view its ignore map (1 = left out) and the weight of its unmatched queries' "no object" loss. ``seed``
    fixes every set (validation); ``None`` draws new ones. ``augment=False`` (validation) only fits each photo to the
    input as inference letterboxes it, with no random augmentation (no flip, no hard anchor). ``fixed`` (file names,
    each set's anchor first; :meth:`read_sets`) replaces the drawn sets.
    """

    def __init__(self, img_dir, json_file, imgsz: int = 644, *, aug: AugConfig | dict | None = None,
                 sets: SubjectSetConfig | dict | None = None, seed: int | None = None, augment: bool = True,
                 fixed: list[list[str]] | None = None, **kw):
        super().__init__(img_dir, json_file, imgsz, aug=aug, keep_empty=True, **kw)
        if not augment:
            self.aug = AugConfig.resolve(self.aug, min_scale=1.0, max_scale=1.0, **dict.fromkeys(_PROB_FIELDS, 0.0))
        self.augment = augment
        self.sets = sets if isinstance(sets, SubjectSetConfig) else SubjectSetConfig(**(sets or {}))
        self.seed = seed
        base = AugConfig.resolve(self.aug, mosaic_prob=0.0, mixup_prob=0.0, copy_paste_prob=0.0)
        place = "random" if augment else "corner"  # unaugmented: where the inference letterbox puts the photo
        self._tf = {flip: TrainAugment(imgsz, AugConfig.resolve(base, flip_prob=float(flip)), place=place)
                    for flip in (0, 1)}
        # the hard anchors' transforms, per case and flip
        self._tf_hard = {}
        if augment and self.sets.p_hard > 0:
            hard_base = AugConfig.resolve(base, zoom_prob=0.0, instance_crop_prob=0.0, instance_crop_empty_prob=0.0)
            self._tf_hard = {(c, flip): TrainAugment(imgsz, AugConfig.resolve(hard_base, **HARD_CASES[c],
                                                                              flip_prob=float(flip)))
                             for c in self.sets.hard for flip in (0, 1)}
            self._mean = torch.tensor(_MEAN).view(3, 1, 1)
            self._std = torch.tensor(_STD).view(3, 1, 1)
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
        self._groups: list[list[int]] = []  # the photos of each group (sets.groups), in the data's order
        if self.sets.groups:
            by_key: dict = {}
            for i in sorted(self._subject_of):
                v = self.coco.loadImgs(self.ids[i])[0].get(self.sets.groups)
                by_key.setdefault((self._subject_of[i], ("photo", i) if v is None else ("group", str(v))), []).append(i)
            for photos in by_key.values():  # a group with no photo that trains is no set; a large one is split
                for k in range(0, len(photos), self.sets.max_views):
                    chunk = photos[k:k + self.sets.max_views]
                    if any(self._supervision[j] != "none" for j in chunk):
                        self._groups.append(sorted(chunk, key=lambda j: self._supervision[j] == "none"))
        self._n_random = (round(len(self._groups) * self.sets.p_random / (1.0 - self.sets.p_random))
                          if self._groups else 0)
        self.fixed = None
        if fixed is not None:
            self.fixed, self.fixed_report = self._resolve_fixed(fixed)
            if not self.fixed:
                raise ValueError(f"{json_file}: none of the fixed sets' anchors is a photo of a subject in this data")

    def _resolve_fixed(self, fixed: list[list[str]]) -> tuple[list[list[int]], dict]:
        """The fixed sets (file names) as indices of this dataset's photos: a set whose anchor is missing (or takes no
        loss) is dropped, a missing companion or one of another subject is left out, and at most ``max_views`` stay."""
        by_name = {self.coco.loadImgs(img_id)[0]["file_name"]: i for i, img_id in enumerate(self.ids)}
        out, dropped, left_out = [], 0, 0
        for names in fixed:
            anchor = by_name.get(names[0]) if names else None
            if anchor not in self._subject_of or self._supervision[anchor] == "none":
                dropped += 1
                continue
            mates = [by_name.get(n) for n in names[1:]]
            keep = [j for j in mates if j is not None and self._subject_of.get(j) == self._subject_of[anchor]]
            left_out += len(mates) - len(keep)
            out.append([anchor, *keep][: self.sets.max_views])
        return out, {"sets": len(fixed), "kept": len(out), "dropped": dropped, "photos_left_out": left_out}

    @staticmethod
    def read_sets(path) -> list[list[str]]:
        """The sets of a file written by :meth:`write_sets`."""
        return [list(s) for s in json.loads(Path(path).read_text())["sets"]]

    def write_sets(self, path) -> int:
        """Write the sets of this dataset (each set's photos as file names, anchor first) to a JSON file that ``fixed=``
        reads back. The photos are chosen without decoding them. Returns the number of sets."""
        names = [[self.coco.loadImgs(self.ids[j])[0]["file_name"] for j in self._photos(i)[0]] for i in range(len(self))]
        Path(path).write_text(json.dumps({"seed": self.seed, "sets": names}) + "\n")
        return len(names)

    def __len__(self) -> int:
        if self.fixed is not None:
            return len(self.fixed)
        return len(self._groups) + self._n_random if self._groups else len(self.anchors)

    @property
    def num_subjects(self) -> int:
        return len(self._members)

    def subject_of_set(self, i: int):
        """The subject of set ``i`` (a fixed set, a group, or the anchor of a random set without groups)."""
        if self.fixed is not None:
            return self._subject_of[self.fixed[i][0]]
        if self._groups:
            if i >= len(self._groups):
                raise IndexError("a random set's subject is drawn when the set is")
            return self._subject_of[self._groups[i][0]]
        return self._subject_of[self.anchors[i]]

    def first_photos(self) -> list:
        """The image id of each set's first photo (the random sets' anchors are drawn: not listed)."""
        if self.fixed is not None:
            return [self.ids[s[0]] for s in self.fixed]
        if self._groups:
            return [self.ids[g[0]] for g in self._groups]
        return [self.ids[a] for a in self.anchors]

    def _instance_code(self, subject, instance_id) -> int:
        """The ``instance_id`` of an annotation of ``subject`` as a non-negative int (a JSON number or string)."""
        return self._inst_code.setdefault((subject, str(instance_id)), len(self._inst_code))

    def __getitem__(self, i: int) -> dict:
        if self.seed is None:
            return self._make_set(i)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed * 100_003 + i)
            return self._make_set(i)

    def _photos(self, i: int) -> tuple[list[int], bool]:
        """The photos of set ``i`` (anchor first) and its flip: the draws that come before any photo is decoded."""
        if self.seed is None:
            return self._choose(i)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed * 100_003 + i)
            return self._choose(i)

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

    def _choose(self, i: int) -> tuple[list[int], bool]:
        if self.fixed is not None:
            return list(self.fixed[i]), _coin(self.aug.flip_prob)
        if self._groups and i < len(self._groups):
            return list(self._groups[i]), _coin(self.aug.flip_prob)
        c = self.sets
        anchor = self.anchors[_randint(0, len(self.anchors) - 1)] if self._groups else self.anchors[i]
        n_others = len(self._members[self._subject_of[anchor]]) - 1
        if not n_others or _coin(c.p_single):
            size = 1
        else:
            size = _draw(c.sizes) if c.sizes else c.max_views
            size = min(size, n_others + 1)
        flip = _coin(self.aug.flip_prob)
        return [anchor, *self.companions(anchor, size)], flip

    def _make_set(self, i: int) -> dict:
        photos, flip = self._choose(i)
        hard = None
        if self._tf_hard and len(photos) > 1 and _coin(self.sets.p_hard):  # the anchor made hard
            hard = _draw(self.sets.hard)
        views, extras = [], []
        for slot, idx in enumerate(photos):
            *view, ignore, neg = self._real_view(idx, flip, slot, hard if slot == 0 else None)
            views.append(tuple(view))
            extras.append({"ignore": ignore, "neg": neg})
        return {"views": views, "case": "real" if hard is None else f"real_{hard}", "extras": extras,
                "photos": [self.ids[i] for i in photos]}

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

    def _real_view(self, idx: int, flip: bool, slot: int, hard: str | None = None):
        """One augmented photo of a set: ``(image, masks, classes, attrs, inst_ids, ignore, neg)``. The ignore map rides
        the augmentation as one more mask (with no weight in the instance-aware crops) and is split off after it.
        ``hard`` (a case of :data:`HARD_CASES`) makes the photo that case's hard view."""
        from torchvision import tv_tensors

        image_tv, masks_t, class_t, attr_t, ids, ignore = self._load_subject_raw(idx)
        n = class_t.numel()
        masks_tv = tv_tensors.Mask(torch.cat([masks_t, ignore[None].to(masks_t.dtype)]))
        class_all = torch.cat([class_t, torch.zeros(1, dtype=torch.long)])
        attr_all = {k: torch.cat([v, _attr_ignored(s, 1)]) for (k, v), s in zip(attr_t.items(), self.aux_specs)}
        tf = self._tf[int(flip)] if hard is None else self._tf_hard[(hard, int(flip))]
        weights = np.r_[np.ones(n), 0.0] if n else None
        img, masks, cls, attrs = tf(image_tv, masks_tv, class_all, attr_all, weights=weights)
        if hard == "occluded" and n:
            img, masks = self._occlude(img, masks, tf, self.sets.occluder_cover, n_inst=len(masks) - 1)
        ign, masks, cls = masks[-1], masks[:-1], cls[:-1]
        attrs = {k: v[:-1] for k, v in attrs.items()}
        info = {k: v[:-1] if torch.is_tensor(v) and v.ndim and len(v) == n + 1 else v for k, v in tf.last_info.items()}
        attrs = _follow_aug(self.aux_specs, attrs, info)
        keep = masks.flatten(1).sum(1) >= tf.cfg.min_mask_mass
        ids = torch.where(ids >= 0, ids, UNKNOWN_IDS - 1000 * slot - torch.arange(n))
        neg = SUPERVISION[self._supervision[idx]]
        return (img, masks[keep], cls[keep], {k: v[keep] for k, v in attrs.items()}, ids[keep],
                (ign > 0.5).float(), neg)


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


def collate_sets(batch: list[dict]):
    """Flatten a batch of sets into one batch of photos: ``(pixel_values, mask_labels, class_labels, aux_labels,
    set_ids, inst_ids, extras)``, where ``set_ids[b]`` is the set of photo ``b``, ``inst_ids[b]`` the instance ids of
    its targets and ``extras`` ``{"ignore": [(S, S) float or None], "neg_weight": (B,) float}`` (ignore maps and "no
    object" weights)."""
    views = [(s, v) for s, item in enumerate(batch) for v in item["views"]]
    names = list(views[0][1][3])
    ignore, neg = [], []
    for item in batch:
        for e in item["extras"]:
            ign = e.get("ignore")
            ignore.append(ign if ign is not None and bool(ign.any()) else None)
            neg.append(float(e.get("neg", 1.0)))
    return (
        torch.stack([v[0] for _, v in views]),
        [v[1].float() for _, v in views],
        [v[2] for _, v in views],
        {n: [v[3][n] for _, v in views] for n in names},
        torch.tensor([s for s, _ in views]),
        [v[4] for _, v in views],
        {"ignore": ignore, "neg_weight": torch.tensor(neg)},
    )


__all__ = ["CocoSubjectSets", "SubjectSetConfig", "SUPERVISION", "HARD_CASES", "collate_sets",
           "subjects_from_file_names"]
