"""Synthetic multi-photo sets for the cross-view post-training, so that it needs no multi-photo annotations.

Every set is made from ONE annotated photo, so which instance is which in every view comes for free: the augmentation
keeps the instance list in order (an instance cropped away has an empty mask), and each view keeps the position of its
instances in the annotation list as their **instance id**. A set is a hard view A plus a view B that clears it up:

=========  ==========================================  =================================================
case       A (hard)                                    B (clears it up)
=========  ==========================================  =================================================
small      the photo shrunk 2-3x below the inference   a close-up of one instance
           scale: its instances cover a few pixels
degraded   strong blur, noise, JPEG and dark gamma     the photo as usual
closeup    a close-up of one instance: its class and   the photo as usual
           attributes are ambiguous
occluded   a patch of another photo across part of an  the photo as usual
           instance: its mask keeps what is visible
=========  ==========================================  =================================================

With ``SetConfig.focus`` a **locate** case joins them: the set is drawn from a photo holding a focus instance (e.g. a
small object that only a close-up shows clearly), A is the photo shrunk as in ``small`` and B a 3-6x close-up of that
instance, so that A has to find and place what only B shows clearly.

Each view is drawn independently, so A and B never share exact pixels. On top of that, with some probability, B shows
another part of the photo without A's instances (they have no partner and A must keep its own prediction), a
**distractor** photo of another subject joins the set and a **mirrored copy** of the photo joins it: their instances
are other instances (ids offset by :data:`DISTRACTOR_IDS` / :data:`FLIPPED_IDS`) — an instance and its mirror image
on the other side are not the same instance. The horizontal flip is drawn once per set and applied to every view of
the photo (the mirrored copy takes the other one), with the flip-aware attributes (e.g. left/right) swapped as in
single-photo training. A share of the sets are single photos, so the single-photo behaviour is kept in training.

With ``SetConfig.sizes`` a set's size is drawn instead, and the views after A and B **fill** it (``SetConfig.fill``):
another view of the photo's instances (**positive**), another part of the photo without them (**elsewhere**), another
annotated photo of the same **subject** (the images' ``subject_id``, :mod:`eomt.data.subjects`: in a product every
photo of a set shows the same subject, often other parts of it), the mirrored copy or a distractor. The instances of
another photo of the subject are each annotated in that photo, but which of them are A's is not known here: their ids
are negative (:data:`UNKNOWN_IDS`), so they train every head but stay out of the matching head's loss and links.
``p_subject`` draws the set's photo among those whose subject has other annotated photos. Real sets, several photos of
a subject each predicted with the others and with the instances linked across them, are
:class:`eomt.data.subjects.CocoSubjectSets`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .coco import CocoInstanceSeg, _attr_label, _follow_aug
from .transforms import _MEAN, _STD, AugConfig, TrainAugment, _coin, _fraction, _mass, _rand, _randint, _uniform

#: Instance-id offsets of the views that do not show the set's photo: their instances are other instances.
DISTRACTOR_IDS, FLIPPED_IDS = 10_000, 20_000
#: Instance-id offset of a view whose instances are not linked to the set's others (another photo of the subject):
#: its ids are negative and the matching head leaves them out (each such view takes its own block of ids below it).
UNKNOWN_IDS = -100_000
#: The views that can fill a set beyond A and B (``SetConfig.fill``).
FILLS = ("positive", "elsewhere", "subject", "flipped", "distractor")

#: View role -> :class:`~eomt.data.transforms.AugConfig` overrides of the base recipe. The scales are relative to the
#: photo fitted to the input, i.e. 1.0 is what inference sees.
ROLES: dict[str, dict] = {
    "context": {"min_scale": 0.75, "max_scale": 1.5},
    "small": {"min_scale": 0.35, "max_scale": 0.6},
    "zoom": {"zoom_prob": 1.0, "zoom_scale": (2.0, 3.5)},
    "closeup": {"zoom_prob": 1.0, "zoom_scale": (2.5, 4.0)},
    "degraded": {"min_scale": 0.75, "max_scale": 1.5, "blur_prob": 1.0, "blur_sigma": (1.5, 3.5), "noise_prob": 0.8,
                 "noise_std": (0.03, 0.08), "jpeg_prob": 0.8, "jpeg_quality": (10, 35), "gamma_prob": 0.7,
                 "gamma_range": (1.6, 2.6)},
    "elsewhere": {"instance_crop_empty_prob": 1.0, "min_scale": 1.2, "max_scale": 2.0},
    # a close-up of a focus instance (SetConfig.focus), whatever else the photo shows
    "zoom_focus": {"zoom_prob": 1.0, "zoom_scale": (3.0, 6.0), "zoom_attr": "_focus"},
    # last resort of a hard view whose instances were all cropped away: the whole photo, unwarped
    "whole": {"min_scale": 1.0, "max_scale": 1.0, "rotate_prob": 0.0, "shear_prob": 0.0, "perspective_prob": 0.0},
}
#: case -> (role of the hard view A, role of the view B that clears it up); "occluded" adds the occluder to A.
CASES: dict[str, tuple[str, str]] = {
    "small": ("small", "zoom"),
    "degraded": ("degraded", "context"),
    "closeup": ("closeup", "context"),
    "occluded": ("context", "context"),
}


def _draw(weights: dict):
    """A key of ``weights`` drawn in proportion to its weight."""
    keys = list(weights)
    cdf = np.cumsum([weights[k] for k in keys]) / sum(weights.values())
    return keys[min(int(np.searchsorted(cdf, _rand())), len(keys) - 1)]


@dataclass
class SetConfig:
    """How the sets are drawn: at most ``max_views`` photos; ``p_single`` single photos; the cases' relative weights;
    the probabilities of B showing another part of the photo (``p_elsewhere``), of a distractor and of a mirrored
    copy; the share of an instance's box the occluder covers.

    ``focus`` names the instances of a **locate** case (a class and attribute categories, e.g. ``{"class": "bird",
    "pose": "flying"}``): with probability ``p_focus`` a set of
    several photos is drawn from a photo with such an instance, A being the photo shrunk (the instance small) and B a
    close-up of that instance, so that A learns to find and place what only B shows clearly.

    ``sizes`` (``{photos: weight}``, 2 to ``max_views``) draws the size of each set of several photos, filled beyond A
    and B by the views of ``fill`` (``{kind: weight}``, kinds :data:`FILLS`; a ``subject`` view falls back to
    ``elsewhere`` when the subject has no other photo); ``p_distractor`` and ``p_flipped_copy`` then play no part.
    Without ``sizes`` the sets grow by those two coins as before. ``p_subject`` draws the photo of a set of several
    among those whose subject (the images' ``subject_id``) has other annotated photos."""

    max_views: int = 4
    p_single: float = 0.3
    cases: dict[str, float] = field(default_factory=lambda: {c: 1.0 for c in CASES})
    p_elsewhere: float = 0.2
    p_distractor: float = 0.5
    p_flipped_copy: float = 0.25
    occluder_cover: tuple[float, float] = (0.3, 0.7)
    focus: dict | None = None
    p_focus: float = 0.0
    sizes: dict | None = None
    fill: dict[str, float] = field(default_factory=lambda: {"positive": 1.0, "elsewhere": 1.0, "subject": 1.0})
    p_subject: float = 0.0

    def __post_init__(self) -> None:
        unknown = set(self.cases) - set(CASES)
        if unknown or not 2 <= self.max_views:
            raise ValueError(f"SetConfig: cases {sorted(unknown)} not in {sorted(CASES)}, or max_views < 2")
        if self.p_focus > 0 and not (self.focus and "class" in self.focus):
            raise ValueError("SetConfig: p_focus > 0 needs focus with a 'class'")
        self.occluder_cover = tuple(self.occluder_cover)
        if self.sizes is not None:
            self.sizes = {int(n): float(w) for n, w in self.sizes.items()}
            if not self.sizes or any(not 2 <= n <= self.max_views for n in self.sizes) or sum(self.sizes.values()) <= 0:
                raise ValueError(f"SetConfig: sizes {self.sizes} must weigh set sizes from 2 to max_views")
        bad = set(self.fill) - set(FILLS)
        if bad or sum(self.fill.values()) <= 0:
            raise ValueError(f"SetConfig: fill kinds {sorted(bad)} not in {FILLS}, or no positive weight"
                             + (" (the 'case' fill is now 'subject', read from the images' subject_id)"
                                if "case" in bad else ""))


class CocoViewSets(CocoInstanceSeg):
    """Synthetic sets of 1..``max_views`` photos from a COCO instance-segmentation dataset (see the module docstring).

    Item: ``{"views": [(image, masks, classes, attrs, inst_ids), ...], "case": str}``, the hard view first. ``seed``
    fixes every set (validation: the same sets at every evaluation); ``None`` draws new ones each time (training).
    """

    def __init__(self, img_dir, json_file, imgsz: int = 644, *, aug: AugConfig | dict | None = None,
                 sets: SetConfig | dict | None = None, seed: int | None = None, **kw):
        super().__init__(img_dir, json_file, imgsz, aug=aug, **kw)
        self.sets = sets if isinstance(sets, SetConfig) else SetConfig(**(sets or {}))
        self.seed = seed
        base = AugConfig.resolve(self.aug, mosaic_prob=0.0, mixup_prob=0.0, copy_paste_prob=0.0, zoom_prob=0.0,
                                 instance_crop_prob=0.0, instance_crop_empty_prob=0.0)
        # One transform per role and flip: the flip is drawn once per set, not per view.
        self._tf = {(role, flip): TrainAugment(imgsz, AugConfig.resolve(base, **over, flip_prob=float(flip)))
                    for role, over in ROLES.items() for flip in (0, 1)}
        self._mean = torch.tensor(_MEAN).view(3, 1, 1)
        self._std = torch.tensor(_STD).view(3, 1, 1)
        self._focus, self._focus_idx = None, []
        if self.sets.p_focus > 0:
            self._focus = self._focus_rule(self.sets.focus)
            self._focus_idx = [i for i, img_id in enumerate(self.ids) if self._has_focus(img_id)]
            if not self._focus_idx:
                raise ValueError(f"SetConfig.focus {self.sets.focus}: no photo has such an instance")
        # The photos of each subject (the images' subject_id), for the `subject` views and p_subject
        self._subject_of: dict[int, list[int]] = {}
        if self.sets.p_subject > 0 or (self.sets.sizes and self.sets.fill.get("subject")):
            members: dict = {}
            for i, img_id in enumerate(self.ids):
                subject = self.coco.loadImgs(img_id)[0].get("subject_id")
                if subject is not None:
                    members.setdefault(subject, []).append(i)
            self._subject_of = {i: m for m in members.values() if len(m) > 1 for i in m}
        self._subject_idx = sorted(self._subject_of)

    def _focus_rule(self, focus: dict) -> tuple[int, dict[str, int]]:
        """``focus`` as (contiguous class, {attribute: contiguous category})."""
        names = self.names if isinstance(self.names, dict) else dict(enumerate(self.names))
        inv = {n: i for i, n in names.items()}
        if focus["class"] not in inv:
            raise ValueError(f"SetConfig.focus class {focus['class']!r} is not one of {sorted(inv)}")
        attrs = {}
        for a, cat in focus.items():
            if a == "class":
                continue
            spec = next((s for s in self.aux_specs if s.name == a and not s.regression), None)
            cats = dict(enumerate(spec.names)) if spec is not None and not isinstance(spec.names, dict) else \
                (spec.names if spec is not None else {})
            ids = [i for i, n in cats.items() if n == cat]
            if not ids:
                raise ValueError(f"SetConfig.focus: {a}={cat!r} is not an attribute category of this dataset")
            attrs[a] = ids[0]
        return inv[focus["class"]], attrs

    def _has_focus(self, img_id) -> bool:
        cls_id, attrs = self._focus
        specs = {s.name: s for s in self.aux_specs}
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)):
            cls = self.cat2contig[ann["category_id"]]
            if cls == cls_id and all(_attr_label(specs[a], self._attr_id_maps, ann.get("attributes") or {}, cls) == v
                                     for a, v in attrs.items()):
                return True
        return False

    def __getitem__(self, idx: int) -> dict:
        if self.seed is None:
            return self._make_set(idx)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed * 100_003 + idx)
            return self._make_set(idx)

    def _make_set(self, idx: int) -> dict:
        c = self.sets
        flip = _coin(self.aug.flip_prob)
        if _coin(c.p_single) or not (c.cases or self._focus_idx):
            return {"views": [self._view(self._load_raw(idx), "context", flip, need=True)], "case": "single"}
        if self._focus_idx and (not c.cases or _coin(c.p_focus)):  # locate: B a close-up of a focus instance
            idx = self._focus_idx[_randint(0, len(self._focus_idx) - 1)]
            raw = self._load_raw(idx)
            case = "locate"
            views = [self._view(raw, "small", flip, need=True), self._view(raw, "zoom_focus", flip, focus=True)]
        else:
            if self._subject_idx and _coin(c.p_subject):
                idx = self._subject_idx[_randint(0, len(self._subject_idx) - 1)]
            raw = self._load_raw(idx)
            case = _draw(c.cases)
            role_a, role_b = CASES[case]
            views = [
                self._view(raw, role_a, flip, need=True, occlude=case == "occluded"),
                self._view(raw, "elsewhere" if _coin(c.p_elsewhere) else role_b, flip),
            ]
        if c.sizes:
            size = _draw(c.sizes)
            while len(views) < size:
                views.append(self._fill(raw, idx, flip, len(views)))
            return {"views": views, "case": case}
        if len(views) < c.max_views and len(self.ids) > 1 and _coin(c.p_distractor):
            views.append(self._view(self._load_raw(self._other(idx)), "context", _coin(self.aug.flip_prob), DISTRACTOR_IDS))
        if len(views) < c.max_views and _coin(c.p_flipped_copy):
            views.append(self._view(raw, "context", not flip, FLIPPED_IDS))
        return {"views": views, "case": case}

    def _other(self, idx: int) -> int:
        j = _randint(0, len(self.ids) - 2)
        return j + (j >= idx)

    def _fill(self, raw, idx: int, flip: bool, slot: int):
        """The view in position ``slot`` of a set filled to its drawn size (``SetConfig.fill``): another view of the
        photo's instances, another part of the photo without them, another photo of its subject (flipped with the set:
        left and right of one subject never mix), the mirrored copy or a distractor (each distractor its own block of
        ids: two distractors are two other subjects)."""
        kind = _draw(self.sets.fill)
        if kind == "subject":
            others = [j for j in self._subject_of.get(idx, ()) if j != idx]
            if others:
                j = others[_randint(0, len(others) - 1)]
                return self._view(self._load_raw(j), "context", flip, UNKNOWN_IDS * slot)
            kind = "elsewhere"
        if kind == "positive":
            return self._view(raw, "zoom" if _coin(0.5) else "context", flip)
        if kind == "elsewhere":
            return self._view(raw, "elsewhere", flip)
        if kind == "flipped":
            return self._view(raw, "context", not flip, FLIPPED_IDS)
        return self._view(self._load_raw(self._other(idx)), "context", _coin(self.aug.flip_prob),
                          DISTRACTOR_IDS + 1000 * slot)

    def _view(self, raw, role: str, flip: bool, offset: int = 0, *, need: bool = False, occlude: bool = False,
              focus: bool = False):
        """One view of a raw sample: ``(image, masks, classes, attrs, inst_ids)`` of the instances it shows. ``need``
        retries a view that lost every instance, the whole photo last; ``focus`` lets the close-up of the role pick
        only a focus instance (marked for the augmentation by a temporary ``_focus`` attribute)."""
        image_tv, masks_tv, class_t, attr_t = raw[:4]
        if focus:
            cls_id, attrs = self._focus
            hit = class_t == cls_id
            for a, v in attrs.items():
                hit &= attr_t[a] == v
            attr_t = {**attr_t, "_focus": torch.where(hit, 1.0, math.nan)}
        ids = torch.arange(class_t.numel()) + offset
        for r in (role, role, role, "whole") if need else (role,):
            tf = self._tf[(r, int(flip))]
            img, masks, cls, attrs = tf(image_tv, masks_tv, class_t, attr_t)
            if occlude:
                img, masks = self._occlude(img, masks, tf)
            keep = masks.flatten(1).sum(1) >= tf.cfg.min_mask_mass
            if keep.any():
                break
        attrs = _follow_aug(self.aux_specs, attrs, tf.last_info)
        attrs.pop("_focus", None)
        return img, masks[keep], cls[keep], {k: v[keep] for k, v in attrs.items()}, ids[keep]

    def _occlude(self, img: torch.Tensor, masks: torch.Tensor, tf: TrainAugment):
        """Lay a band cut from another photo across part of one instance (a pole, an arm): its pixels and
        its share of every mask are gone, and the crop-aware attributes follow the visible share."""
        mass = _mass(masks)
        cand = (mass >= 16).nonzero().flatten().tolist()
        if not cand:
            return img, masks
        i = cand[_randint(0, len(cand) - 1)]
        ys, xs = torch.nonzero(masks[i] > 0.05, as_tuple=True)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        s, cover = img.shape[-1], _uniform(self.sets.occluder_cover)
        if _coin(0.5):  # vertical band, longer than the instance
            w = max(2, round((x1 - x0) * cover))
            bx, ext = _randint(x0, max(x0, x1 - w)), round((y1 - y0) * _uniform((0.2, 1.0)))
            box = (bx, max(0, y0 - ext), min(s, bx + w), min(s, y1 + ext))
        else:  # horizontal band
            h = max(2, round((y1 - y0) * cover))
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


def collate_sets(batch: list[dict]):
    """Flatten a batch of sets into one batch of photos: ``(pixel_values, mask_labels, class_labels, aux_labels,
    set_ids, inst_ids, extras)``, where ``set_ids[b]`` is the set of photo ``b``, ``inst_ids[b]`` the instance ids of
    its targets and ``extras`` the partial-label inputs of the real sets (:func:`eomt.data.subjects.collate_extras`:
    ignore maps and "no object" weights), None when no set of the batch has any."""
    from .subjects import collate_extras

    views = [(s, v) for s, item in enumerate(batch) for v in item["views"]]
    names = list(views[0][1][3])
    return (
        torch.stack([v[0] for _, v in views]),
        [v[1].float() for _, v in views],
        [v[2] for _, v in views],
        {n: [v[3][n] for _, v in views] for n in names},
        torch.tensor([s for s, _ in views]),
        [v[4] for _, v in views],
        collate_extras(batch),
    )
