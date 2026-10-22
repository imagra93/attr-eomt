"""The run config: one YAML holds the data paths and how a model is trained on them and run (eomt.data.config); the
dataset holds only the data."""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from eomt import EoMT
from eomt.data import CocoInstanceSeg, load_data_config

from test_augment import _mini_coco  # noqa: E402  (tests/ is on sys.path)


def test_a_run_config_resolves_its_sections_and_paths_against_its_folder(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "run.yaml").write_text(yaml.safe_dump({
        "path": "../data", "train_images": "images", "train_json": "train.json",
        "attributes": {"side": {"flip_aware": {"left": "right"}}, "size": {"log": True, "crop": "scale"}},
        "aug": {"rotate_prob": 0.3},
        "train": {"epochs": 24, "init_weights": "../weights/best.pt", "project": "../runs", "seed": 0},
        "inference": {"conf_thres": 0.25},
    }))
    cfg = load_data_config(tmp_path / "configs" / "run.yaml", autodownload=False)
    assert cfg["train_json"] == str(tmp_path / "data" / "train.json")
    assert cfg["train"] == {"epochs": 24, "init_weights": str(tmp_path / "weights" / "best.pt"),
                            "project": str(tmp_path / "runs"), "seed": 0}
    assert cfg["attributes"] == {"side": {"flip_aware": {"left": "right"}}, "size": {"log": True, "crop": "scale"}}
    assert cfg["train_aug"] == {"rotate_prob": 0.3} and cfg["inference"] == {"conf_thres": 0.25}

    (tmp_path / "both.yaml").write_text(yaml.safe_dump({"aug": {"rotate_prob": 0.1}, "train_aug": {"blur_prob": 0.1}}))
    with pytest.raises(ValueError, match="same section"):
        load_data_config(tmp_path / "both.yaml", autodownload=False)
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({"train": {"train_json": "x.json", "size": "s"}}))
    with pytest.raises(ValueError, match="cannot set"):
        load_data_config(tmp_path / "paths.yaml", autodownload=False)


def test_the_shipped_template_builds_the_sample_dataset_heads():
    from pathlib import Path

    run = Path(__file__).resolve().parents[1] / "sample_data" / "run.yaml"
    cfg = load_data_config(run, autodownload=False)
    ds = CocoInstanceSeg(cfg["train_images"], cfg["train_json"], imgsz=112, attributes=cfg["attributes"])
    assert [s.name for s in ds.aux_specs] == ["scale", "viewpoint"]
    assert set(cfg["train"]) >= {"epochs", "imgsz", "batch"} and set(cfg["inference"]) == {"conf_thres", "mask_thresh", "max_det"}


def test_the_run_config_picks_the_attributes_that_train_and_how(tmp_path):
    img_dir, jf = _mini_coco(tmp_path)  # the data's attribute "kind" (blob / line)
    coco = json.loads(jf.read_text())
    coco["attributes"] += [{"name": "side", "categories": [{"id": 0, "name": "left"}, {"id": 1, "name": "right"}]},
                           {"name": "size", "type": "regression", "unit": "cm2"}]
    for i, a in enumerate(coco["annotations"]):
        a["attributes"].update(side=i % 2, size=10.0 * (i + 1))
    jf.write_text(json.dumps(coco))

    ds = CocoInstanceSeg(img_dir, jf, imgsz=112, attributes={"side": {"flip_aware": {"left": "right"}},
                                                             "size": {"log": True, "crop": "scale"}})
    specs = {s.name: s for s in ds.aux_specs}
    assert set(specs) == {"side", "size"}  # "kind" is in the data but not in the config: no head
    assert specs["side"].flip_aware == {0: 1, 1: 0} and specs["size"].log and specs["size"].crop == "scale"
    assert not CocoInstanceSeg(img_dir, jf, imgsz=112).aux_specs[1].flip_aware  # the data alone: no behavior
    with pytest.raises(ValueError, match="not defined in the data"):
        CocoInstanceSeg(img_dir, jf, imgsz=112, attributes={"color": None})
    with pytest.raises(ValueError, match="unknown keys"):
        CocoInstanceSeg(img_dir, jf, imgsz=112, attributes={"side": {"flipaware": {"left": "right"}}})


def test_a_run_from_its_yaml_records_it_and_predicts_with_its_inference_settings(tmp_path, monkeypatch):
    from eomt.engine.train import train

    with pytest.raises(ValueError, match="inference"):
        train(train_images="x", train_json="x", inference={"conf": 0.5})

    img_dir, jf = _mini_coco(tmp_path)
    run = tmp_path / "run.yaml"
    run.write_text(yaml.safe_dump({
        "path": str(tmp_path), "train_images": "images", "train_json": jf.name, "val_images": "images", "val_json": jf.name,
        "attributes": {"kind": None},
        "train": {"epochs": 1, "imgsz": 112, "batch": 1, "accum": 1, "workers": 0, "amp": False, "pretrained": False,
                  "ema": False, "seed": 0, "project": "runs", "name": "r", "warmup_steps": [1, 1]},
        "inference": {"conf_thres": 0.9, "max_det": 7},
    }))
    model = EoMT("s", device="cpu", pretrained=False)
    res = model.train(data=str(run))

    args = yaml.safe_load((tmp_path / "runs" / "r" / "args.yaml").read_text())  # `project` relative to the YAML
    assert args["attributes"] == {"kind": {"type": "categorical", "categories": ["blob", "line"]}}
    assert args["inference"] == {"conf_thres": 0.9, "max_det": 7} and args["epochs"] == 1
    ck = torch.load(res["best"], weights_only=False)
    assert ck["inference"] == {"conf_thres": 0.9, "max_det": 7} and ck["run_config"]["attributes"] == args["attributes"]

    seen = {}
    monkeypatch.setattr("eomt.api._predict", lambda model, source, **kw: seen.update(kw) or [])
    model.predict(str(img_dir))
    assert (seen["conf_thres"], seen["max_det"], seen["mask_thresh"]) == (0.9, 7, 0.5)  # saved, saved, default
    model.predict(str(img_dir), conf_thres=0.1)
    assert seen["conf_thres"] == 0.1  # the call wins
