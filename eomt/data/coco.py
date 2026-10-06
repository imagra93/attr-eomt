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

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import AuxHeadSpec
from ..preprocess import preprocess_numpy
from .transforms import AugConfig, TrainAugment, build_val_transform


def _build_category_maps(coco):
    """Return ``(cat2contig, contig2cat, names, num_classes)`` from a COCO handle."""
    cat_ids = sorted(coco.getCatIds())
    cat2contig = {c: i for i, c in enumerate(cat_ids)}
    contig2cat = {i: c for c, i in cat2contig.items()}
    cats = coco.loadCats(cat_ids)
    names = {cat2contig[c["id"]]: c["name"] for c in cats}
    return cat2contig, contig2cat, names, len(cat_ids)


def _resolve_applies_to(raw, cat2contig, contig_names, attr_name):
    """Resolve a definition's ``applies_to`` to a frozenset of contiguous primary ids.

    Entries may be category **names** (matched against ``contig_names``) or raw COCO
    category **ids** (mapped through ``cat2contig``). ``None`` ⇒ head applies to every
    primary class (returns ``None``). Unresolvable entries are warned about and dropped.
    """
    if raw is None:
        return None
    name2contig = {v: k for k, v in contig_names.items()}
    out: set[int] = set()
    for entry in raw:
        if isinstance(entry, str) and entry in name2contig:
            out.add(name2contig[entry])
        elif isinstance(entry, int) and entry in cat2contig:
            out.add(cat2contig[entry])
        else:
            warnings.warn(
                f"attribute {attr_name!r} 'applies_to' entry {entry!r} matches no "
                "category (by name or raw id); ignoring it.",
                stacklevel=2,
            )
    return frozenset(out)


def _find_dataset_root(json_file: Path) -> Path:
    """Dataset root that holds optional sidecar attribute files.

    The JSON's directory, or its parent when the JSON sits in an ``annotations/``
    subfolder (the COCO convention: ``<root>/annotations/instances_train.json``).
    """
    p = json_file.parent
    return p.parent if p.name == "annotations" else p


def _split_name(json_file: Path) -> str:
    """Infer the split name from a COCO json filename (``train`` / ``val`` / stem)."""
    stem = json_file.stem.lower()
    if "train" in stem:
        return "train"
    if "val" in stem:
        return "val"
    return stem


def _load_attr_schema(root: Path):
    """Read ``<root>/attributes.yaml`` into a normalized top-level ``attributes`` list.

    Each head is ``{name, categories:[{id,name}], applies_to?}``. ``categories`` may be
    written as bare labels (auto-id 0..n-1) or explicit ``{id,name}`` dicts. Returns
    ``None`` when the schema file is absent.
    """
    import yaml

    schema_path = root / "attributes.yaml"
    if not schema_path.exists():
        return None
    doc = yaml.safe_load(schema_path.read_text()) or {}
    norm = []
    for d in doc.get("attributes") or []:
        cats = []
        for i, c in enumerate(d.get("categories") or []):
            if isinstance(c, dict):
                cats.append({"id": int(c["id"]), "name": str(c.get("name", c["id"]))})
            else:  # bare label -> auto id by position
                cats.append({"id": i, "name": str(c)})
        entry = {"name": d["name"], "categories": cats}
        if d.get("applies_to") is not None:
            entry["applies_to"] = list(d["applies_to"])
        norm.append(entry)
    return norm


def _load_attr_values(root: Path, json_file: Path):
    """Per-annotation attribute values keyed by **annotation id** (order-independent).

    Reads ``<root>/attributes/<split>.json`` (``{ann_id: {head: value}}``); values may
    be category labels or raw ids. Returns ``{int ann_id: {head: value}}`` or ``None``.
    """
    import json

    cand = root / "attributes" / f"{_split_name(json_file)}.json"
    if not cand.exists():
        return None
    raw = json.loads(cand.read_text())
    return {int(k): v for k, v in raw.items()}


def _merge_attribute_sidecar(coco, json_file) -> None:
    """Merge an optional on-disk attribute sidecar into the in-memory COCO handle.

    No-op unless ``<root>/attributes.yaml`` exists **and** the JSON has no embedded
    ``attributes`` (embedded definitions win — one source of truth). When it applies,
    the schema becomes the top-level ``attributes`` list and per-annotation values
    (keyed by annotation id) are written into each ``ann["attributes"]`` as raw
    category ids, after which discovery/loading is identical to the embedded case.
    So a plain COCO dataset always works; drop in the sidecar and it is picked up.
    """
    dataset = getattr(coco, "dataset", {}) or {}
    if dataset.get("attributes"):  # embedded schema present -> ignore sidecar
        return
    schema = _load_attr_schema(_find_dataset_root(Path(json_file)))
    if not schema:
        return
    dataset["attributes"] = schema
    name2id = {d["name"]: {c["name"]: c["id"] for c in d["categories"]} for d in schema}
    valid_ids = {d["name"]: {c["id"] for c in d["categories"]} for d in schema}
    values = _load_attr_values(_find_dataset_root(Path(json_file)), Path(json_file)) or {}
    for ann in dataset.get("annotations", []) or []:
        vals = values.get(ann["id"])
        if not vals:
            continue
        merged = {}
        for head, v in vals.items():
            if head not in name2id:
                continue
            rid = name2id[head].get(v, v)  # value may be a label or a raw id
            if isinstance(rid, int) and rid in valid_ids[head]:
                merged[head] = rid
        if merged:
            ann["attributes"] = {**ann.get("attributes", {}), **merged}


def _attr_label(spec: AuxHeadSpec, id_maps: dict, ann_attrs: dict, primary_cls: int) -> int:
    """Contiguous attribute id for one instance, or ``-100`` (ignored by the aux CE).

    ``-100`` when the attribute is missing / out-of-vocab (never silently trained as
    class 0), **or** when the head is class-scoped (``applies_to``) and this instance's
    primary class is out of scope — the hard-routing choke point, so the loss /
    accuracy paths need no change (they already skip ``-100``).
    """
    if spec.applies_to is not None and primary_cls not in spec.applies_to:
        return -100
    return id_maps[spec.name].get(ann_attrs.get(spec.name), -100)


def _build_attribute_maps(coco, only: list[str] | None = None, *, cat2contig=None, contig_names=None):
    """Discover secondary attributes from a COCO handle.

    Reads the (non-standard but valid) top-level ``attributes`` list::

        "attributes": [
          {"name": "color", "categories": [{"id": 0, "name": "red"}, ...]},
          {"name": "posture", "categories": [...], "applies_to": ["cat", 7]},
        ]

    and the per-annotation ``"attributes": {"color": 0, "posture": 2}`` field.
    Falls back to inferring each attribute's id set from the annotations when a
    definition omits ``categories``. The optional ``applies_to`` scopes a head to a
    subset of primary classes (category names or raw ids); it is resolved against
    ``cat2contig`` / ``contig_names`` (pass them from ``_build_category_maps``).
    Returns ``(specs, id_maps)`` where ``id_maps`` is ``{attr_name: {raw_id:
    contiguous_id}}``.
    """
    dataset = getattr(coco, "dataset", {}) or {}
    defs = dataset.get("attributes") or []
    if only is not None:
        defs = [d for d in defs if d.get("name") in only]
    if not defs:
        return [], {}

    cat2contig = cat2contig or {}
    contig_names = contig_names or {}
    anns = dataset.get("annotations") or []
    specs: list[AuxHeadSpec] = []
    id_maps: dict[str, dict] = {}
    for d in defs:
        name = d["name"]
        cats = d.get("categories")
        if cats:
            raw_ids = sorted({c["id"] for c in cats})
            raw2contig = {r: i for i, r in enumerate(raw_ids)}
            disp = {c["id"]: c.get("name", str(c["id"])) for c in cats}
        else:  # infer the id set straight from the annotations
            warnings.warn(
                f"attribute {name!r} has no explicit 'categories'; inferring its id "
                "set from this split's annotations. This map is NOT portable across "
                "splits — define 'categories' (or share train's id map) so train and "
                "val use the same contiguous ids.",
                stacklevel=2,
            )
            raw_ids = sorted(
                {a["attributes"][name] for a in anns if name in a.get("attributes", {})}
            )
            raw2contig = {r: i for i, r in enumerate(raw_ids)}
            disp = {r: str(r) for r in raw_ids}
        names = {raw2contig[r]: disp.get(r, str(r)) for r in raw_ids}
        applies_to = _resolve_applies_to(d.get("applies_to"), cat2contig, contig_names, name)
        specs.append(
            AuxHeadSpec(name=name, num_classes=len(raw2contig), names=names, applies_to=applies_to)
        )
        id_maps[name] = raw2contig
    return specs, id_maps


class _InstanceWeights:
    """Per-instance sampling weights for the instance-aware crop: ``freq(key) ** -power``.

    ``key`` is the instance's primary class, or its value of the attribute named by
    ``aug.instance_crop_rarity_attr`` (e.g. ``"material"``), so rare kinds are chosen as the crop's focus more often.
    Only built when the instance-aware crop is enabled (``instance_crop_prob`` / ``instance_crop_empty_prob`` > 0).
    """

    def _init_weights(self) -> None:
        c = self.aug
        self._rarity = None
        if c.instance_crop_prob <= 0 and c.instance_crop_empty_prob <= 0:
            return
        spec = None
        if c.instance_crop_rarity_attr:
            spec = next((s for s in self.aux_specs if s.name == c.instance_crop_rarity_attr), None)
            if spec is None:
                warnings.warn(
                    f"instance_crop_rarity_attr={c.instance_crop_rarity_attr!r} is not an attribute of this dataset "
                    f"({[s.name for s in self.aux_specs]}); weighting by primary class instead.",
                    stacklevel=3,
                )
        counts: Counter = Counter()
        for img_id in self.ids:
            for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)):
                counts[self._rarity_key(ann, spec)] += 1
        total = sum(counts.values()) or 1
        self._rarity = (spec, {k: v / total for k, v in counts.items()})

    def _rarity_key(self, ann: dict, spec) -> int:
        cls = self.cat2contig[ann["category_id"]]
        if spec is None:
            return cls
        return _attr_label(spec, self._attr_id_maps, ann.get("attributes", {}), cls)

    def _instance_weights(self, keys: list[int]) -> np.ndarray | None:
        if self._rarity is None or not keys:
            return None
        _, freq = self._rarity
        common = max(freq.values())
        power = self.aug.instance_crop_rarity_power
        return np.array([freq.get(k, common) ** (-power) for k in keys], dtype=np.float64)


class CocoInstanceSeg(_InstanceWeights, Dataset):
    """COCO instance segmentation -> ``(pixel_values, masks, class_labels)``.

    Each item is an augmented, ImageNet-normalized square tensor plus a stack of
    instance masks ``(num_inst, imgsz, imgsz)`` (float; soft, area-averaged values in
    [0, 1] by default, see :class:`~eomt.data.transforms.AugConfig` ``mask_resize``) and
    their contiguous class ids. Images with no instances are filtered out at construction.

    Augmentation: ``aug`` (an :class:`~eomt.data.transforms.AugConfig` or a mapping) holds every knob;
    ``flip_prob`` / ``min_scale`` / ``max_scale`` override it when given; ``transform`` replaces the built-in
    pipeline altogether.

    ``keep_empty=True`` keeps the images with no annotations as **negatives**: they yield zero targets (``masks``
    ``(0, imgsz, imgsz)``, empty ``class_t`` / attribute tensors), so every query is trained as "no object". The default
    (``False``) drops them, as before.
    """

    def __init__(
        self,
        img_dir: str | Path,
        json_file: str | Path,
        imgsz: int = 644,
        *,
        transform=None,
        flip_prob: float | None = None,
        min_scale: float | None = None,
        max_scale: float | None = None,
        aug: AugConfig | dict | None = None,
        keep_empty: bool = False,
        attributes: list[str] | bool = True,
        shared_aux: tuple[list[AuxHeadSpec], dict] | None = None,
    ):
        from pycocotools.coco import COCO

        self.img_dir = Path(img_dir)
        self.imgsz = imgsz
        self.keep_empty = keep_empty
        # One source of truth for the augmentation defaults (:class:`AugConfig`); the explicit keywords
        # (``flip_prob`` / ``min_scale`` / ``max_scale``) override ``aug`` when given.
        self.aug = AugConfig.resolve(aug, flip_prob=flip_prob, min_scale=min_scale, max_scale=max_scale)
        self.coco = COCO(str(json_file))
        _merge_attribute_sidecar(self.coco, json_file)
        self.cat2contig, self.contig2cat, self.names, self.num_classes = (
            _build_category_maps(self.coco)
        )
        # Secondary attribute heads. ``shared_aux`` ⇒ reuse another dataset's
        # ``(specs, id_maps)`` (e.g. train's, so val stays in the same contiguous
        # id space). Otherwise discover from this JSON: ``attributes=True`` ⇒ all
        # defined; a list ⇒ only those; ``False`` ⇒ ignore (detection-only).
        if shared_aux is not None:
            self.aux_specs, self._attr_id_maps = shared_aux
        else:
            only = None if attributes is True else ([] if attributes is False else attributes)
            self.aux_specs, self._attr_id_maps = (
                ([], {}) if only == [] else _build_attribute_maps(
                    self.coco, only=only, cat2contig=self.cat2contig, contig_names=self.names
                )
            )
        self.ids = [
            i
            for i in self.coco.getImgIds()
            if keep_empty or len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0
        ]
        self._init_weights()
        # ``transform`` replaces the built-in pipeline (any ``(image, masks) -> (image, masks)`` callable, e.g. a
        # torchvision v2 ``Compose``); by default :class:`TrainAugment` built from ``self.aug``.
        self.transform = transform or TrainAugment(imgsz, self.aug)
        # No-crop resize used as a fallback when augmentation crops every
        # instance away, so a sample is never forced to a fabricated empty mask.
        self._fallback_transform = build_val_transform(imgsz)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_raw(self, idx: int):
        """Decode one sample *before* augmentation: ``(image_tv, masks_tv, class_t, attr_t, rarity_keys)``."""
        from torchvision import tv_tensors

        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        masks, classes, keys = [], [], []
        attrs: dict[str, list[int]] = {s.name: [] for s in self.aux_specs}
        rarity_spec = self._rarity[0] if self._rarity is not None else None
        for ann in anns:
            m = self.coco.annToMask(ann)  # (H, W) uint8 at original resolution
            if m.sum() == 0:
                continue
            masks.append(torch.from_numpy(m))
            cls = self.cat2contig[ann["category_id"]]
            classes.append(cls)
            if self._rarity is not None:
                keys.append(self._rarity_key(ann, rarity_spec))
            ann_attrs = ann.get("attributes", {})
            for spec in self.aux_specs:
                attrs[spec.name].append(_attr_label(spec, self._attr_id_maps, ann_attrs, cls))

        image_tv = tv_tensors.Image(
            torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (3, H, W) uint8
        )
        if masks:
            masks_tv = tv_tensors.Mask(torch.stack(masks))  # (N, H, W) uint8
            class_t = torch.tensor(classes, dtype=torch.long)
            attr_t = {k: torch.tensor(v, dtype=torch.long) for k, v in attrs.items()}
        elif self.keep_empty:  # a negative image: genuinely zero targets
            masks_tv = tv_tensors.Mask(torch.zeros((0, *image_tv.shape[1:]), dtype=torch.uint8))
            class_t = torch.zeros((0,), dtype=torch.long)
            attr_t = {s.name: torch.zeros((0,), dtype=torch.long) for s in self.aux_specs}
        else:  # should not happen (filtered), but stay safe
            masks_tv = tv_tensors.Mask(torch.zeros((1, *image_tv.shape[1:]), dtype=torch.uint8))
            class_t = torch.zeros((1,), dtype=torch.long)
            attr_t = {s.name: torch.full((1,), -100, dtype=torch.long) for s in self.aux_specs}
        return image_tv, masks_tv, class_t, attr_t, keys

    def _sample_other(self) -> dict:
        """A random raw sample, for the multi-image augmentations (mosaic / mixup / copy-paste)."""
        j = int(torch.randint(0, len(self.ids), (1,)))
        image_tv, masks_tv, class_t, attr_t, keys = self._load_raw(j)
        return {"image": image_tv, "target": masks_tv, "labels": class_t, "attrs": attr_t, "weights": self._instance_weights(keys)}

    def __getitem__(self, idx: int):
        image_tv, masks_tv, class_t, attr_t, keys = self._load_raw(idx)

        if self.keep_empty and class_t.numel() == 0:  # negative: augment once, keep the empty target
            if getattr(self.transform, "is_train_augment", False):
                c = self.transform.cfg
                multi = c.mosaic_prob > 0 or c.mixup_prob > 0 or c.copy_paste_prob > 0
                image_t, masks_t, class_t, attr_t = self.transform(
                    image_tv, masks_tv, class_t, attr_t, sampler=self._sample_other if multi else None
                )
            else:
                image_t, masks_t = self.transform(image_tv, masks_tv)
            return image_t, masks_t.float(), class_t, attr_t

        # Apply augmentation; if a random crop removes every instance, retry a few
        # times, then fall back to a plain resize (no crop) that always preserves
        # them — only as a last resort emit a single empty placeholder instance.
        image_t = masks_t = keep = None
        if getattr(self.transform, "is_train_augment", False):
            tf = self.transform
            c = tf.cfg
            weights = self._instance_weights(keys)
            multi = c.mosaic_prob > 0 or c.mixup_prob > 0 or c.copy_paste_prob > 0
            class_o, attr_o = class_t, attr_t
            for _ in range(3):
                image_t, masks_t, class_o, attr_o = tf(
                    image_tv, masks_tv, class_t, attr_t, weights=weights, sampler=self._sample_other if multi else None
                )
                keep = masks_t.flatten(1).sum(1) >= c.min_mask_mass
                if keep.any():
                    break
            else:
                image_t, masks_t = self._fallback_transform(image_tv, masks_tv)
                class_o, attr_o = class_t, attr_t
                keep = masks_t.flatten(1).sum(1) > 0
            class_t, attr_t = class_o, attr_o
        else:  # a user-supplied ``(image, masks) -> (image, masks)`` transform
            for _ in range(3):
                image_t, masks_t = self.transform(image_tv, masks_tv)
                keep = masks_t.flatten(1).sum(1) > 0
                if keep.any():
                    break
            else:
                image_t, masks_t = self._fallback_transform(image_tv, masks_tv)
                keep = masks_t.flatten(1).sum(1) > 0

        masks_t, class_t = masks_t[keep], class_t[keep]
        attr_t = {k: v[keep] for k, v in attr_t.items()}
        if masks_t.shape[0] == 0:  # pathological (e.g. sub-pixel masks) -> empty
            masks_t = torch.zeros((1, self.imgsz, self.imgsz))
            class_t = torch.zeros((1,), dtype=torch.long)
            attr_t = {s.name: torch.full((1,), -100, dtype=torch.long) for s in self.aux_specs}

        return image_t, masks_t.float(), class_t, attr_t


class CocoDetection(_InstanceWeights, Dataset):
    """COCO object detection -> ``(pixel_values, boxes, class_labels, attrs)``.

    The detection counterpart to :class:`CocoInstanceSeg` for ``family="detect"``.
    Each item is an augmented, ImageNet-normalized square tensor plus per-instance
    **normalized ``cxcywh`` boxes** ``(num_inst, 4)`` in ``[0, 1]`` (relative to the
    square input) and their contiguous class ids. Boxes ride the **same** transforms
    as masks via ``tv_tensors.BoundingBoxes``, so LSJ/flip/crop apply identically.
    """

    def __init__(
        self,
        img_dir: str | Path,
        json_file: str | Path,
        imgsz: int = 644,
        *,
        transform=None,
        flip_prob: float | None = None,
        min_scale: float | None = None,
        max_scale: float | None = None,
        aug: AugConfig | dict | None = None,
        attributes: list[str] | bool = True,
        shared_aux: tuple[list[AuxHeadSpec], dict] | None = None,
    ):
        from pycocotools.coco import COCO

        self.img_dir = Path(img_dir)
        self.imgsz = imgsz
        self.aug = AugConfig.resolve(aug, flip_prob=flip_prob, min_scale=min_scale, max_scale=max_scale)
        self.coco = COCO(str(json_file))
        _merge_attribute_sidecar(self.coco, json_file)
        self.cat2contig, self.contig2cat, self.names, self.num_classes = (
            _build_category_maps(self.coco)
        )
        if shared_aux is not None:
            self.aux_specs, self._attr_id_maps = shared_aux
        else:
            only = None if attributes is True else ([] if attributes is False else attributes)
            self.aux_specs, self._attr_id_maps = (
                ([], {}) if only == [] else _build_attribute_maps(
                    self.coco, only=only, cat2contig=self.cat2contig, contig_names=self.names
                )
            )
        self.ids = [
            i
            for i in self.coco.getImgIds()
            if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0
        ]
        self._init_weights()
        self.transform = transform or TrainAugment(imgsz, self.aug)
        self._fallback_transform = build_val_transform(imgsz)

    def __len__(self) -> int:
        return len(self.ids)

    def _norm_cxcywh(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """``xyxy`` pixel boxes -> normalized ``cxcywh`` in ``[0, 1]``; clamp to canvas."""
        b = boxes_xyxy.clamp(min=0, max=self.imgsz)
        w = (b[:, 2] - b[:, 0]).clamp(min=0)
        h = (b[:, 3] - b[:, 1]).clamp(min=0)
        cx = b[:, 0] + 0.5 * w
        cy = b[:, 1] + 0.5 * h
        return torch.stack([cx, cy, w, h], dim=1) / self.imgsz

    def __getitem__(self, idx: int):
        from torchvision import tv_tensors

        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        boxes, classes, keys = [], [], []
        attrs: dict[str, list[int]] = {s.name: [] for s in self.aux_specs}
        rarity_spec = self._rarity[0] if self._rarity is not None else None
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO xywh, absolute pixels
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])  # xyxy
            cls = self.cat2contig[ann["category_id"]]
            classes.append(cls)
            if self._rarity is not None:
                keys.append(self._rarity_key(ann, rarity_spec))
            ann_attrs = ann.get("attributes", {})
            for spec in self.aux_specs:
                attrs[spec.name].append(_attr_label(spec, self._attr_id_maps, ann_attrs, cls))

        H, W = img.height, img.width
        image_tv = tv_tensors.Image(
            torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (3, H, W) uint8
        )
        if boxes:
            boxes_tv = tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(H, W),
            )
            class_t = torch.tensor(classes, dtype=torch.long)
            attr_t = {k: torch.tensor(v, dtype=torch.long) for k, v in attrs.items()}
        else:  # should not happen (filtered), but stay safe
            boxes_tv = tv_tensors.BoundingBoxes(
                torch.zeros((1, 4), dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(H, W),
            )
            class_t = torch.zeros((1,), dtype=torch.long)
            attr_t = {s.name: torch.full((1,), -100, dtype=torch.long) for s in self.aux_specs}

        # Apply augmentation; if a crop removes every box, retry, then fall back to a
        # plain resize that preserves them (mirrors CocoInstanceSeg).
        image_t = boxes_t = keep = None
        built_in = getattr(self.transform, "is_train_augment", False)
        weights = self._instance_weights(keys) if built_in else None
        for _ in range(3):
            if built_in:
                image_t, boxes_t = self.transform(image_tv, boxes_tv, weights=weights)
            else:
                image_t, boxes_t = self.transform(image_tv, boxes_tv)
            boxes_t = torch.as_tensor(boxes_t)
            keep = ((boxes_t[:, 2] - boxes_t[:, 0]) > 1) & ((boxes_t[:, 3] - boxes_t[:, 1]) > 1)
            if keep.any():
                break
        else:
            image_t, boxes_t = self._fallback_transform(image_tv, boxes_tv)
            boxes_t = torch.as_tensor(boxes_t)
            keep = ((boxes_t[:, 2] - boxes_t[:, 0]) > 1) & ((boxes_t[:, 3] - boxes_t[:, 1]) > 1)

        boxes_t, class_t = boxes_t[keep], class_t[keep]
        attr_t = {k: v[keep] for k, v in attr_t.items()}
        if boxes_t.shape[0] == 0:  # pathological -> single empty placeholder
            boxes_t = torch.tensor([[0.0, 0.0, float(self.imgsz), float(self.imgsz)]])
            class_t = torch.zeros((1,), dtype=torch.long)
            attr_t = {s.name: torch.full((1,), -100, dtype=torch.long) for s in self.aux_specs}

        return image_t, self._norm_cxcywh(boxes_t), class_t, attr_t


class CocoValImages(Dataset):
    """COCO images for evaluation -> ``(pixel_values, image_id, orig_w, orig_h)``.

    Ground-truth annotations are read from the JSON by ``COCOeval`` directly, so
    this dataset only needs to deliver preprocessed pixels and identity/size.
    """

    def __init__(
        self,
        img_dir: str | Path,
        json_file: str | Path,
        imgsz: int = 644,
        *,
        letterbox: bool = True,
        mean=None,
        std=None,
        attributes: list[str] | bool = True,
        shared_aux: tuple[list[AuxHeadSpec], dict] | None = None,
    ):
        from pycocotools.coco import COCO

        self.img_dir = Path(img_dir)
        self.imgsz = imgsz
        self.letterbox = letterbox
        self.mean = mean
        self.std = std
        self.coco = COCO(str(json_file))
        _merge_attribute_sidecar(self.coco, json_file)
        self.cat2contig, self.contig2cat, self.names, self.num_classes = (
            _build_category_maps(self.coco)
        )
        if shared_aux is not None:
            self.aux_specs, self._attr_id_maps = shared_aux
        else:
            only = None if attributes is True else ([] if attributes is False else attributes)
            self.aux_specs, self._attr_id_maps = (
                ([], {}) if only == [] else _build_attribute_maps(
                    self.coco, only=only, cat2contig=self.cat2contig, contig_names=self.names
                )
            )
        self.ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        orig_w, orig_h = img.size
        chw, meta = preprocess_numpy(
            np.array(img), self.imgsz, letterbox=self.letterbox, mean=self.mean, std=self.std,
        )
        return torch.from_numpy(chw), int(img_id), orig_w, orig_h, meta


def collate_train(batch):
    """Stack pixel values; keep masks/classes/attrs as per-image lists (variable N).

    Returns ``(pixel_values, mask_labels, class_labels, aux_labels)`` where
    ``aux_labels`` is ``{attr_name: [per-image LongTensor]}`` (empty dict when the
    dataset has no attributes).
    """
    pixel_values = torch.stack([b[0] for b in batch])
    mask_labels = [b[1] for b in batch]
    class_labels = [b[2] for b in batch]
    names = list(batch[0][3].keys())
    aux_labels = {name: [b[3][name] for b in batch] for name in names}
    return pixel_values, mask_labels, class_labels, aux_labels


def collate_val(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    image_ids = [b[1] for b in batch]
    sizes = [(b[2], b[3]) for b in batch]
    metas = [b[4] for b in batch]
    return pixel_values, image_ids, sizes, metas
