"""Training augmentation: AugConfig, the TrainAugment pipeline, dataset / train() wiring (CPU, no network)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torchvision import tv_tensors

from eomt.data import AugConfig, CocoDetection, CocoInstanceSeg, TrainAugment, build_train_transform
from eomt.data.transforms import _MEAN, _STD, build_val_transform

# every photometric / sensor op off: geometry tests then compare pure colours
PHOTO_OFF = dict(color_jitter_prob=0, gamma_prob=0, grayscale_prob=0, blur_prob=0, noise_prob=0, jpeg_prob=0, glare_prob=0, erasing_prob=0)
GEOM_OFF = dict(flip_prob=0, vflip_prob=0, rot90_prob=0, rotate_prob=0, shear_prob=0, perspective_prob=0)


def _cfg(**kw) -> AugConfig:
    return AugConfig(**{**PHOTO_OFF, **GEOM_OFF, **kw})


def _unnormalise(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tensor(_STD).view(3, 1, 1) + torch.tensor(_MEAN).view(3, 1, 1)


def _rects(h=240, w=320):
    """Green wide + red tall rectangle on a grey background, with their masks and boxes."""
    img = torch.full((3, h, w), 60, dtype=torch.uint8)
    img[:, 50:130, 90:230] = torch.tensor([0, 255, 0], dtype=torch.uint8).view(3, 1, 1)
    img[:, 150:220, 40:70] = torch.tensor([255, 0, 0], dtype=torch.uint8).view(3, 1, 1)
    masks = torch.zeros(2, h, w, dtype=torch.uint8)
    masks[0, 50:130, 90:230] = 1
    masks[1, 150:220, 40:70] = 1
    boxes = torch.tensor([[90, 50, 230, 130], [40, 150, 70, 220]], dtype=torch.float32)
    return img, masks, boxes


def _colour_masks(out_img: torch.Tensor) -> torch.Tensor:
    x = _unnormalise(out_img)
    green = (x[1] > 0.7) & (x[0] < 0.35) & (x[2] < 0.35)
    red = (x[0] > 0.7) & (x[1] < 0.35) & (x[2] < 0.35)
    return torch.stack([green, red])


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    u = (a | b).sum().item()
    return 1.0 if u == 0 else (a & b).sum().item() / u


# ---------------------------------------------------------------------------------------------------------------------
# AugConfig
# ---------------------------------------------------------------------------------------------------------------------


def test_augconfig_defaults():
    c = AugConfig()
    assert (c.flip_prob, c.min_scale, c.max_scale) == (0.5, 0.1, 2.0)  # the original recipe is kept
    # structured / orientation-changing ops are opt-in
    assert c.vflip_prob == c.rot90_prob == 0.0
    assert c.mosaic_prob == c.mixup_prob == c.copy_paste_prob == 0.0
    assert c.instance_crop_prob == 0.0 and c.instance_crop_empty_prob == 0.0  # instance-aware crop: off
    assert c.mask_resize == "area"


def test_default_pipeline_is_stronger_than_legacy():
    new, old = set(AugConfig().active()), set(AugConfig.preset("legacy").active())
    assert old == {"flip", "color_jitter"}  # exactly the original recipe
    assert old < new
    assert {"rotate", "shear", "perspective", "gamma", "grayscale", "blur", "noise", "jpeg", "glare", "erasing"} <= new
    assert AugConfig.preset("legacy").mask_resize == "nearest"


def test_augconfig_validation_and_unknown_keys():
    with pytest.raises(ValueError, match="unknown augmentation keys"):
        AugConfig.from_dict({"rotat_prob": 0.5})
    with pytest.raises(ValueError, match="probability"):
        AugConfig(flip_prob=1.5)
    with pytest.raises(ValueError, match="low"):
        AugConfig(blur_sigma=(2.0, 1.0))
    with pytest.raises(ValueError, match="min_scale"):
        AugConfig(min_scale=2.0, max_scale=1.0)
    with pytest.raises(ValueError, match="mask_resize"):
        AugConfig(mask_resize="bicubic")
    with pytest.raises(ValueError, match="preset"):
        AugConfig.from_dict({"preset": "nope"})


def test_augconfig_resolve_precedence_and_yaml_roundtrip():
    # explicit keyword > aug mapping > defaults; None means "not given"
    c = AugConfig.resolve({"flip_prob": 0.2, "rotate_prob": 0.9}, flip_prob=0.0, min_scale=None)
    assert c.flip_prob == 0.0 and c.rotate_prob == 0.9 and c.min_scale == 0.1
    # preset then overrides
    c = AugConfig.from_dict({"preset": "legacy", "blur_prob": 0.4})
    assert c.blur_prob == 0.4 and c.rotate_prob == 0.0 and c.mask_resize == "nearest"
    # ranges survive a YAML round trip (lists <-> tuples)
    d = AugConfig(blur_sigma=(0.2, 0.9), instance_crop_rarity_attr="typology").to_dict()
    again = AugConfig.from_dict(yaml.safe_load(yaml.safe_dump(d)))
    assert again == AugConfig(blur_sigma=(0.2, 0.9), instance_crop_rarity_attr="typology")


# ---------------------------------------------------------------------------------------------------------------------
# Geometry: image, masks and boxes must stay aligned
# ---------------------------------------------------------------------------------------------------------------------

GEOMETRY = {
    "identity": dict(min_scale=1.0, max_scale=1.0),
    "hflip": dict(flip_prob=1, min_scale=1.0, max_scale=1.0),
    "vflip": dict(vflip_prob=1, min_scale=1.0, max_scale=1.0),
    "rot90": dict(rot90_prob=1, min_scale=1.0, max_scale=1.0),
    "scale": dict(min_scale=0.3, max_scale=1.5),
    "rotate": dict(min_scale=0.8, max_scale=1.2, rotate_prob=1, rotate_deg=25),
    "shear": dict(min_scale=0.8, max_scale=1.2, shear_prob=1, shear_deg=10),
    "perspective": dict(min_scale=0.8, max_scale=1.2, perspective_prob=1, perspective_scale=0.2),
    "everything": dict(flip_prob=0.5, vflip_prob=0.5, rot90_prob=0.5, min_scale=0.3, max_scale=1.5, rotate_prob=1,
                       rotate_deg=20, shear_prob=1, shear_deg=8, perspective_prob=1, perspective_scale=0.15),
}


@pytest.mark.parametrize("name", list(GEOMETRY))
def test_masks_stay_aligned_with_the_image(name):
    torch.manual_seed(0)
    tf = TrainAugment(256, _cfg(**GEOMETRY[name]))
    img, masks, _ = _rects()
    ious = []
    for _ in range(25):
        out_img, out_m = tf(tv_tensors.Image(img), tv_tensors.Mask(masks))
        col = _colour_masks(out_img)
        for k in range(2):
            if out_m[k].sum() > 200:  # skip draws where the object is mostly cropped away
                ious.append(_iou(out_m[k] > 0.5, col[k]))
    assert ious and min(ious) > 0.85 and float(np.mean(ious)) > 0.95, (name, min(ious), float(np.mean(ious)))


@pytest.mark.parametrize("name", list(GEOMETRY))
def test_boxes_follow_the_same_geometry(name):
    """Replaying the same random draws, the box must match the extent of the transformed (soft) mask.

    Also guards that the mask path and the box path consume randomness identically, and that boxes of objects
    the crop window cuts are clipped as polygons (not as the clipped box of the whole rotated shape).
    """
    torch.manual_seed(1)
    img, masks, boxes = _rects()
    errs = []
    for _ in range(40):
        tf = TrainAugment(256, _cfg(**GEOMETRY[name]))
        state = torch.get_rng_state()
        _, out_m = tf(tv_tensors.Image(img), tv_tensors.Mask(masks))
        torch.set_rng_state(state)
        _, out_b = tf(tv_tensors.Image(img), tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.shape[-2:]))
        for k in range(2):
            if out_m[k].sum() < 150:  # skip draws where the object is (almost) cropped away
                continue
            ys, xs = torch.nonzero(out_m[k] > 0.02, as_tuple=True)
            support = torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=torch.float32)
            errs.append(float((support - out_b[k]).abs().max()))
    assert errs and max(errs) < 3.0, (name, max(errs))


def test_box_of_a_rotated_object_cut_by_the_window_is_clipped_as_a_polygon():
    """Clipping the shape first gives a tighter box than clamping the box of the whole shape."""
    from eomt.data.transforms import _clipped_aabb

    # a diamond centred just outside the left edge: only its right tip (x in [0, 15]) is inside the window
    diamond = np.array([[-25, 50], [-5, 30], [15, 50], [-5, 70]], dtype=np.float64)
    assert _clipped_aabb(diamond, 100, 100) == (0.0, 35.0, 15.0, 65.0)
    clamped = (max(diamond[:, 0].min(), 0), max(diamond[:, 1].min(), 0), min(diamond[:, 0].max(), 100), min(diamond[:, 1].max(), 100))
    assert clamped == (0.0, 30.0, 15.0, 70.0)  # what clamping the whole-shape box would have given: 10 px too tall
    # fully inside: unchanged; fully outside: empty
    inside = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float64)
    assert _clipped_aabb(inside, 100, 100) == (10.0, 10.0, 30.0, 30.0)
    assert _clipped_aabb(diamond + 500, 100, 100) == (0.0, 0.0, 0.0, 0.0)


def test_boxes_must_be_xyxy():
    tf = TrainAugment(64, _cfg())
    img = tv_tensors.Image(torch.zeros(3, 40, 40, dtype=torch.uint8))
    with pytest.raises(ValueError, match="XYXY"):
        tf(img, tv_tensors.BoundingBoxes(torch.tensor([[5.0, 5.0, 10.0, 10.0]]), format="XYWH", canvas_size=(40, 40)))


def test_legacy_two_arg_contract():
    """(image, masks) -> (image, masks): the call form the original pipeline had."""
    torch.manual_seed(0)
    tf = build_train_transform(112, min_scale=0.1, max_scale=2.0)
    img = tv_tensors.Image(torch.randint(0, 255, (3, 80, 120), dtype=torch.uint8))
    masks = tv_tensors.Mask(torch.ones((2, 80, 120), dtype=torch.uint8))
    out_img, out_masks = tf(img, masks)
    assert out_img.shape == (3, 112, 112) and out_img.dtype == torch.float32
    assert out_masks.shape == (2, 112, 112) and out_masks.dtype == torch.float32
    assert 0.0 <= float(out_masks.min()) and float(out_masks.max()) <= 1.0
    assert (out_masks.flatten(1).sum(1) > 0).all()


def test_build_train_transform_keywords_override_aug_and_color_jitter_flag():
    tf = build_train_transform(112, flip_prob=0.0, aug={"flip_prob": 1.0, "rotate_prob": 0.7}, color_jitter=False)
    assert tf.cfg.flip_prob == 0.0 and tf.cfg.rotate_prob == 0.7 and tf.cfg.color_jitter_prob == 0.0


# ---------------------------------------------------------------------------------------------------------------------
# Soft masks
# ---------------------------------------------------------------------------------------------------------------------


def test_area_masks_preserve_mass_and_nearest_masks_are_binary():
    torch.manual_seed(0)
    img = torch.full((3, 240, 320), 90, dtype=torch.uint8)
    masks = torch.zeros(1, 240, 320, dtype=torch.uint8)
    masks[0, 60:150, 80:250] = 1
    r = 160 / 320 * 0.5  # imgsz=160, s=0.5 -> the whole resized image fits inside the canvas (no crop)
    for mode in ("area", "nearest"):
        tf = TrainAugment(160, _cfg(min_scale=0.5, max_scale=0.5, mask_resize=mode))
        _, m = tf(tv_tensors.Image(img), tv_tensors.Mask(masks))
        expected = masks.sum().item() * r * r
        assert abs(m.sum().item() - expected) / expected < 0.06, (mode, m.sum().item(), expected)
        if mode == "nearest":
            assert set(m.unique().tolist()) <= {0.0, 1.0}
        else:
            assert ((m > 0) & (m < 1)).any()  # soft edges


def test_thin_line_stays_one_piece_with_area_masks_but_dotted_with_nearest():
    """The motivation for soft masks: nearest-neighbour shreds thin lines when downscaling (at most orientations)."""
    import math

    import cv2

    img = torch.full((3, 240, 320), 90, dtype=torch.uint8)
    area_pieces, nearest_pieces = [], []
    for ang in range(0, 90, 10):
        canvas = np.zeros((240, 320), np.uint8)
        t = math.radians(ang)
        p0 = (int(160 - 140 * math.cos(t)), int(120 - 100 * math.sin(t)))
        p1 = (int(160 + 140 * math.cos(t)), int(120 + 100 * math.sin(t)))
        cv2.line(canvas, p0, p1, 1, thickness=2)
        masks = torch.from_numpy(canvas)[None]
        for mode, store in (("area", area_pieces), ("nearest", nearest_pieces)):
            torch.manual_seed(0)
            tf = TrainAugment(96, _cfg(min_scale=1.0, max_scale=1.0, mask_resize=mode))  # r = 0.3
            _, m = tf(tv_tensors.Image(img), tv_tensors.Mask(masks))
            n, _ = cv2.connectedComponents((m[0].numpy() > 0.1).astype(np.uint8), connectivity=8)
            store.append(n - 1)
    assert area_pieces == [1] * 9, area_pieces
    assert sum(n > 1 for n in nearest_pieces) >= 4, nearest_pieces  # measured: broken into 4-13 pieces at 10-30, 70, 80 degrees


# ---------------------------------------------------------------------------------------------------------------------
# Instance-aware crop
# ---------------------------------------------------------------------------------------------------------------------


def _scattered(h=480, w=640):
    img = torch.full((3, h, w), 80, dtype=torch.uint8)
    masks = torch.zeros(3, h, w, dtype=torch.uint8)
    for k, (y, x) in enumerate([(30, 40), (400, 560), (230, 300)]):  # three 40x40 instances, far apart
        masks[k, y:y + 40, x:x + 40] = 1
    return img, masks


def test_instance_crop_is_off_by_default():
    torch.manual_seed(0)
    img, masks = _scattered()
    tf = TrainAugment(160, _cfg(min_scale=2.0, max_scale=2.0))
    for _ in range(10):
        tf(tv_tensors.Image(img), tv_tensors.Mask(masks), weights=np.array([1.0, 0.0, 0.0]))
        assert tf.last_info["mode"] == "random" and tf.last_info["focus"] is None


def test_instance_crop_keeps_the_focus_instance_fully_visible_even_with_warps():
    torch.manual_seed(0)
    img, masks = _scattered()
    cfg = _cfg(min_scale=2.0, max_scale=2.0, instance_crop_prob=1.0, rotate_prob=1, rotate_deg=10, shear_prob=1, shear_deg=4)
    tf = TrainAugment(160, cfg)  # r = 160/640*2 = 0.5: resized 240x320 > the 160 window, so cropping really happens
    full = 40 * 40 * 0.25
    for target in range(3):
        w = np.zeros(3)
        w[target] = 1.0
        for _ in range(20):
            _, m = tf(tv_tensors.Image(img), tv_tensors.Mask(masks), weights=w)
            assert tf.last_info["mode"] == "focus" and tf.last_info["focus"] == target
            assert m[target].sum().item() > 0.93 * full, (target, m[target].sum().item(), full)


def test_instance_crop_visibility_floor_raises_the_scale():
    import cv2

    canvas = np.zeros((480, 640), np.uint8)
    cv2.line(canvas, (50, 60), (600, 400), 1, thickness=4)
    masks = torch.from_numpy(canvas)[None]
    img = torch.full((3, 480, 640), 80, dtype=torch.uint8)
    torch.manual_seed(0)
    base = dict(min_scale=0.5, max_scale=1.0, instance_crop_prob=1.0)
    off = TrainAugment(160, _cfg(**base))
    on = TrainAugment(160, _cfg(**base, instance_crop_visibility_px=1.2))
    r_off, r_on = [], []
    for _ in range(20):
        off(tv_tensors.Image(img), tv_tensors.Mask(masks))
        on(tv_tensors.Image(img), tv_tensors.Mask(masks))
        r_off.append(off.last_info["eff_scale"]), r_on.append(on.last_info["eff_scale"])
    assert max(r_off) <= 0.25 + 1e-6  # 160/640 * (<= 1.0)
    assert min(r_on) >= 0.27  # raised so the ~4 px band is >= ~1.2 px wide at the model input


def test_empty_window_mode_finds_instance_free_crops_when_the_crop_is_small():
    torch.manual_seed(0)
    img, masks = _scattered()
    masks = masks[:1]  # one instance in a corner; a 128 px window at r = 0.5 has plenty of empty space
    tf = TrainAugment(128, _cfg(min_scale=4.0, max_scale=4.0, instance_crop_empty_prob=1.0))
    hits = 0
    for _ in range(15):
        _, m = tf(tv_tensors.Image(img), tv_tensors.Mask(masks))
        hits += int(tf.last_info["mode"] == "empty" and m.sum().item() == 0)
    assert hits >= 12


# ---------------------------------------------------------------------------------------------------------------------
# Optics
# ---------------------------------------------------------------------------------------------------------------------


def test_safe_erasing_never_touches_an_instance():
    torch.manual_seed(0)
    img = torch.full((3, 160, 160), 128, dtype=torch.uint8)  # flat grey
    masks = torch.zeros(1, 160, 160, dtype=torch.uint8)
    masks[0, 60:100, 20:140] = 1
    tf = TrainAugment(160, _cfg(min_scale=1.0, max_scale=1.0, erasing_prob=1.0, erasing_scale=(0.05, 0.25)))
    erased = 0
    for _ in range(40):
        out, m = tf(tv_tensors.Image(img), tv_tensors.Mask(masks))
        x = _unnormalise(out)
        inside = (m[0] > 0.05).expand(3, -1, -1)
        assert torch.allclose(x[inside], torch.full_like(x[inside], 128 / 255), atol=1e-2)  # instance pixels untouched
        erased += int((x - 128 / 255).abs().amax() > 0.05)
    assert erased > 20  # and erasing really happens elsewhere


@pytest.mark.parametrize("single", [dict(), dict(vflip_prob=1, rot90_prob=1)])
def test_every_single_image_op_forced_on_never_breaks(single):
    """Stress: all probabilities 1 (incl. grayscale followed by in-place erasing, a real regression)."""
    torch.manual_seed(0)
    cfg = AugConfig(flip_prob=1, rotate_prob=1, shear_prob=1, perspective_prob=1, color_jitter_prob=1, gamma_prob=1,
                    grayscale_prob=1, blur_prob=1, noise_prob=1, jpeg_prob=1, glare_prob=1, erasing_prob=1,
                    instance_crop_prob=0.5, instance_crop_empty_prob=0.5, **single)
    tf = TrainAugment(112, cfg)
    img, masks, boxes = _rects(120, 160)
    for _ in range(60):
        out, m = tf(tv_tensors.Image(img), tv_tensors.Mask(masks), weights=np.array([1.0, 2.0]))
        assert out.shape == (3, 112, 112) and torch.isfinite(out).all()
        assert m.shape[1:] == (112, 112) and 0.0 <= float(m.min()) and float(m.max()) <= 1.0 + 1e-5
        out_b = tf(tv_tensors.Image(img), tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(120, 160)))[1]
        assert out_b.shape == (2, 4) and (out_b >= 0).all() and (out_b <= 112).all()


# ---------------------------------------------------------------------------------------------------------------------
# Multi-image augmentations: labels must stay aligned with masks
# ---------------------------------------------------------------------------------------------------------------------


def _fake_sample(n, h=100, w=140, base=0):
    masks = torch.zeros(n, h, w, dtype=torch.uint8)
    for k in range(n):
        masks[k, 10 + 25 * k:30 + 25 * k, 20:120] = 1
    return {
        "image": torch.randint(0, 255, (3, h, w), dtype=torch.uint8),
        "target": masks,
        "labels": torch.arange(n) + base,
        "attrs": {"typ": torch.arange(n) + 10 * base},
        "weights": None,
    }


@pytest.mark.parametrize("op", ["mosaic", "mixup", "copy_paste"])
def test_multi_image_ops_keep_labels_aligned(op):
    torch.manual_seed(0)
    tf = TrainAugment(96, _cfg(min_scale=0.5, max_scale=1.2, **{f"{op}_prob": 1.0}))
    own = _fake_sample(2)
    for _ in range(30):
        calls = []

        def sampler():
            calls.append(1)
            return _fake_sample(2, base=5)

        out, m, lab, att = tf(own["image"], own["target"], own["labels"], own["attrs"], sampler=sampler)
        assert out.shape == (3, 96, 96) and m.shape[1:] == (96, 96)
        assert m.shape[0] == lab.shape[0] == att["typ"].shape[0]
        assert 0.0 <= float(m.min()) and float(m.max()) <= 1.0 + 1e-5
        assert tf.last_info["multi"] == op
        if op == "mosaic":
            assert len(calls) == 3 and m.shape[0] == 8  # own + 3 others, 2 instances each
        elif op == "mixup":
            assert len(calls) == 1 and m.shape[0] == 4
        else:
            assert len(calls) == 1 and 3 <= m.shape[0] <= 2 + 3  # pasted 1..copy_paste_max of the 2 available
            assert (lab[2:] >= 5).all()  # the pasted instances carry the donor's labels


def test_multi_image_ops_are_skipped_without_labels_or_sampler():
    torch.manual_seed(0)
    tf = TrainAugment(96, _cfg(mosaic_prob=1.0, mixup_prob=1.0, copy_paste_prob=1.0))
    s = _fake_sample(2)
    out, m = tf(s["image"], s["target"])  # legacy form: would change the instance list, so skipped
    assert m.shape[0] == 2
    out, m, lab, att = tf(s["image"], s["target"], s["labels"], s["attrs"])  # no sampler: also skipped
    assert m.shape[0] == lab.shape[0] == 2 and tf.last_info["multi"] is None


def test_copy_paste_occludes_existing_masks():
    torch.manual_seed(0)
    tf = TrainAugment(96, _cfg(min_scale=1.0, max_scale=1.0, copy_paste_prob=1.0, copy_paste_max=1, copy_paste_context_px=0))
    own = _fake_sample(1)
    donor = _fake_sample(1, base=7)
    own["target"][:] = 1   # the own instance covers the whole image, so any paste lands on top of it
    donor["target"][:] = 0
    donor["target"][0, 30:70, 30:100] = 1
    out, m, lab, att = tf(own["image"], own["target"], own["labels"], own["attrs"], sampler=lambda: donor)
    assert m.shape[0] == 2
    solid = m[1] > 0.9  # inside the pasted instance (its soft edge legitimately overlaps by up to (1-u)*u <= 0.25)
    assert solid.sum() > 500
    assert float(m[0][solid].max()) < 0.05  # the pasted instance carves a hole in the old one
    assert m[0].sum() < own["target"].sum() - 0.8 * solid.sum()  # and the old mass really went down


# ---------------------------------------------------------------------------------------------------------------------
# Datasets and train() wiring
# ---------------------------------------------------------------------------------------------------------------------


def _mini_coco(tmp_path, n_images=4, bbox_only=False, n_negatives=0):
    """n tiny images, each with a wide and a thin instance of different 'typology' (plus ``n_negatives`` images with
    no annotations at all); returns (img_dir, json)."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    anns, imgs, aid = [], [], 1
    rng = np.random.default_rng(0)
    for i in range(1, n_images + 1):
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(img_dir / f"im{i}.png")
        imgs.append({"id": i, "file_name": f"im{i}.png", "width": 160, "height": 120})
        for typ, (x0, y0, x1, y1) in ((0, (10, 10, 90, 60)), (1, (20, 80, 140, 86))):
            anns.append({"id": aid, "image_id": i, "category_id": 1, "iscrowd": 0, "bbox": [x0, y0, x1 - x0, y1 - y0],
                         "area": (x1 - x0) * (y1 - y0), "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                         "attributes": {"typology": typ}})
            aid += 1
    for i in range(n_images + 1, n_images + n_negatives + 1):
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(img_dir / f"im{i}.png")
        imgs.append({"id": i, "file_name": f"im{i}.png", "width": 160, "height": 120})
    coco = {"images": imgs, "categories": [{"id": 1, "name": "damage"}], "annotations": anns,
            "attributes": [{"name": "typology", "categories": [{"id": 0, "name": "blob"}, {"id": 1, "name": "line"}]}]}
    jf = tmp_path / "instances_train.json"
    jf.write_text(json.dumps(coco))
    return img_dir, jf


def test_dataset_defaults_are_the_same_as_the_trainers(tmp_path):
    """Regression: CocoInstanceSeg used to default to scale 0.5-1.0 while train() used 0.1-2.0."""
    import inspect

    from eomt.engine.train import train

    img_dir, jf = _mini_coco(tmp_path)
    inst, det = CocoInstanceSeg(img_dir, jf, imgsz=112), CocoDetection(img_dir, jf, imgsz=112)
    assert inst.aug == det.aug == AugConfig()
    assert isinstance(inst.transform, TrainAugment) and inst.transform.cfg == AugConfig()
    sig = inspect.signature(train).parameters
    assert sig["flip_prob"].default is None and sig["min_scale"].default is None and sig["max_scale"].default is None
    assert "aug" in sig and "train_transform" in sig


def test_keep_empty_keeps_negatives_as_zero_target_samples(tmp_path):
    img_dir, jf = _mini_coco(tmp_path, n_images=4, n_negatives=3)
    assert len(CocoInstanceSeg(img_dir, jf, imgsz=112)) == 4  # default: images without annotations are dropped
    aug = {"rotate_prob": 1.0, "shear_prob": 1.0, "perspective_prob": 1.0, "erasing_prob": 1.0,
           "instance_crop_prob": 1.0, "instance_crop_empty_prob": 0.5, "instance_crop_rarity_attr": "typology"}
    custom = lambda image, masks: build_val_transform(112)(image, masks)  # noqa: E731
    for ds in (CocoInstanceSeg(img_dir, jf, imgsz=112, aug=aug, keep_empty=True),
               CocoInstanceSeg(img_dir, jf, imgsz=112, transform=custom, keep_empty=True)):
        assert len(ds) == 7
        torch.manual_seed(0)
        for _ in range(5):
            for i in range(len(ds)):
                x, m, c, a = ds[i]
                assert x.shape == (3, 112, 112) and m.dtype == torch.float32
                if i < 4:  # positives are untouched by the option
                    assert m.shape[0] == c.shape[0] == a["typology"].shape[0] >= 1
                else:      # negatives: genuinely zero targets, not a fabricated placeholder instance
                    assert m.shape == (0, 112, 112) and c.shape == (0,) and a["typology"].shape == (0,)
                    assert c.dtype == torch.long and a["typology"].dtype == torch.long


def test_collate_and_train_handle_negatives(tmp_path):
    from eomt.data import collate_train

    img_dir, jf = _mini_coco(tmp_path, n_images=2, n_negatives=6)
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, keep_empty=True)
    px, masks, classes, aux = collate_train([ds[2], ds[3]])  # an all-negative batch
    assert px.shape == (2, 3, 112, 112) and [m.shape[0] for m in masks] == [0, 0] and aux["typology"][0].numel() == 0
    # the real trainer: batch 2 over 2 positives + 6 negatives (so all-negative micro-batches occur), with the aux head
    from eomt.engine.train import train

    res = train(train_images=str(img_dir), train_json=str(jf), size="s", imgsz=112, epochs=1, batch=2, accum=1, workers=0,
                device="cpu", amp=False, pretrained=False, ema=False, seed=0, keep_empty=True, project=str(tmp_path / "runs"), name="neg")
    args = yaml.safe_load((tmp_path / "runs" / "neg" / "args.yaml").read_text())
    assert args["keep_empty"] is True and res["last"].endswith("last.pt")
    with pytest.raises(ValueError, match="keep_empty"):
        train(train_images=str(img_dir), train_json=str(jf), size="s", family="detect", imgsz=112, epochs=1, batch=2, workers=0,
              device="cpu", amp=False, pretrained=False, keep_empty=True, project=str(tmp_path / "runs"), name="det")


def test_dataset_items_are_valid_with_the_default_pipeline(tmp_path):
    torch.manual_seed(0)
    img_dir, jf = _mini_coco(tmp_path)
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112)
    for i in range(len(ds)):
        for _ in range(5):
            x, m, cls, attrs = ds[i]
            assert x.shape == (3, 112, 112) and m.shape[1:] == (112, 112) and m.dtype == torch.float32
            assert m.shape[0] == cls.shape[0] == attrs["typology"].shape[0] >= 1
            assert (m.flatten(1).sum(1) >= ds.aug.min_mask_mass).all()


def test_explicit_keywords_beat_the_aug_mapping_in_the_dataset(tmp_path):
    img_dir, jf = _mini_coco(tmp_path)
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, flip_prob=0.0, aug={"flip_prob": 1.0, "min_scale": 0.7})
    assert ds.aug.flip_prob == 0.0 and ds.aug.min_scale == 0.7


def test_custom_transform_is_still_honoured(tmp_path):
    img_dir, jf = _mini_coco(tmp_path)
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, transform=build_val_transform(112))
    x, m, cls, attrs = ds[0]
    assert x.shape == (3, 112, 112) and m.shape[0] == 2 and not hasattr(ds.transform, "cfg")


@pytest.mark.parametrize("op", ["mosaic", "mixup", "copy_paste"])
def test_dataset_runs_each_multi_image_op_end_to_end(tmp_path, op):
    torch.manual_seed(0)
    img_dir, jf = _mini_coco(tmp_path)
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, aug={f"{op}_prob": 1.0})
    seen = set()
    for _ in range(6):
        x, m, cls, attrs = ds[0]
        assert m.shape[0] == cls.shape[0] == attrs["typology"].shape[0]
        seen.add(m.shape[0])
    assert max(seen) > 2  # instances really were added from other images


def test_instance_crop_weights_follow_the_rarest_attribute_value(tmp_path):
    img_dir, jf = _mini_coco(tmp_path)
    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, aug={"instance_crop_prob": 1.0, "instance_crop_rarity_attr": "typology"})
    assert ds._rarity is not None
    w = ds._instance_weights([0, 1, 1])
    assert w.shape == (3,) and (w > 0).all()
    # off by default: no weights are even built
    assert CocoInstanceSeg(img_dir, jf, imgsz=112)._rarity is None
    # a wrong attribute name warns and falls back to the primary class
    with pytest.warns(UserWarning, match="not an attribute"):
        CocoInstanceSeg(img_dir, jf, imgsz=112, aug={"instance_crop_prob": 1.0, "instance_crop_rarity_attr": "nope"})


def test_detection_dataset_uses_the_same_pipeline(tmp_path):
    torch.manual_seed(0)
    img_dir, jf = _mini_coco(tmp_path)
    ds = CocoDetection(img_dir, jf, imgsz=112, aug={"rotate_prob": 1.0, "instance_crop_prob": 1.0})
    for _ in range(8):
        x, boxes, cls, attrs = ds[0]
        assert x.shape == (3, 112, 112) and boxes.shape[1] == 4 and boxes.shape[0] == cls.shape[0] >= 1
        assert (boxes >= 0).all() and (boxes <= 1).all()


def _tiny_train(tmp_path, **kw):
    from eomt.engine.train import train

    img_dir, jf = _mini_coco(tmp_path)
    return train(train_images=str(img_dir), train_json=str(jf), size="s", imgsz=112, epochs=1, batch=1, accum=1, workers=0,
                 device="cpu", amp=False, pretrained=False, ema=False, seed=0, project=str(tmp_path / "runs"), name="t", **kw)


def test_train_records_the_resolved_augmentation_and_runs(tmp_path):
    res = _tiny_train(tmp_path, aug={"rotate_prob": 0.9, "instance_crop_prob": 0.5, "preset": "default"}, min_scale=0.5)
    args = yaml.safe_load((tmp_path / "runs" / "t" / "args.yaml").read_text())
    assert args["min_scale"] == 0.5 and args["aug"]["rotate_prob"] == 0.9 and args["aug"]["instance_crop_prob"] == 0.5
    assert args["aug"]["min_scale"] == 0.5 and args["aug"]["mask_resize"] == "area"
    assert res["last"].endswith("last.pt")


def test_train_accepts_a_custom_train_transform(tmp_path):
    calls = []
    base = build_val_transform(112)

    def custom(image, masks):
        calls.append(1)
        return base(image, masks)

    _tiny_train(tmp_path, train_transform=custom)
    args = yaml.safe_load((tmp_path / "runs" / "t" / "args.yaml").read_text())
    assert calls and args["aug"] == "custom train_transform"


def test_eomt_train_merges_the_yaml_train_aug_block(monkeypatch, tmp_path):
    from eomt import EoMT

    img_dir, jf = _mini_coco(tmp_path)
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({
        "path": str(tmp_path), "train_images": "images", "train_json": jf.name,
        "train_aug": {"rotate_prob": 0.9, "min_scale": 0.6, "preset": "legacy"},
    }))
    seen = {}
    monkeypatch.setattr("eomt.api._train", lambda **kw: seen.update(kw) or {"best": None, "last": None})
    EoMT("s", device="cpu", pretrained=False).train(data=str(data), aug={"blur_prob": 0.5, "min_scale": 0.8}, flip_prob=0.0)
    # code beats the YAML key by key; explicit keywords are forwarded untouched for train() to resolve last
    assert seen["aug"] == {"rotate_prob": 0.9, "min_scale": 0.8, "preset": "legacy", "blur_prob": 0.5}
    assert seen["flip_prob"] == 0.0
    # a malformed block is rejected early
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"path": str(tmp_path), "train_images": "images", "train_json": jf.name, "train_aug": [1, 2]}))
    with pytest.raises(ValueError, match="train_aug"):
        EoMT("s", device="cpu", pretrained=False).train(data=str(bad))


def test_cli_aug_parsing():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("train_cli", Path(__file__).resolve().parents[1] / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod._parse_aug(["rotate_prob=0.5", "blur_sigma=[0.3,1.0]", "erasing_avoid_instances=false", "instance_crop_rarity_attr=typology"], "legacy")
    assert out == {"preset": "legacy", "rotate_prob": 0.5, "blur_sigma": [0.3, 1.0], "erasing_avoid_instances": False,
                   "instance_crop_rarity_attr": "typology"}
    assert mod._parse_aug([], None) is None
    with pytest.raises(SystemExit):
        mod._parse_aug(["oops"], None)
