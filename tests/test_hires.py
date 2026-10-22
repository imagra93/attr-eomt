"""The high-resolution final phase of ``train`` (``hires_epochs``): schedule, checkpoints per phase, resume."""

from __future__ import annotations

import csv

import pytest
import torch
import yaml

from eomt import build_model
from eomt.engine.train import HIRES_EMA_DECAY, _fit_batch, _state_at_imgsz, train
from eomt.serialization import load_model

from test_augment import _mini_coco  # noqa: E402  (tests/ is on sys.path)


def test_fit_batch_halves_down_to_what_fits():
    assert _fit_batch(4, lambda b: b <= 2) == 2
    assert _fit_batch(3, lambda b: b == 1) == 1
    assert _fit_batch(2, lambda b: False) == 0


def test_a_state_saved_at_another_size_predicts_as_the_model_run_at_that_size():
    low = build_model("s", nc=2, imgsz=112).eval()
    high = build_model("s", nc=2, imgsz=168).eval()
    high.load_state_dict(_state_at_imgsz(low.state_dict(), 112, 168, 14))
    x = torch.rand(1, 3, 168, 168)
    with torch.no_grad():
        a, b = low(x), high(x)
    for k in ("masks_queries_logits", "class_queries_logits"):
        assert torch.equal(a[k], b[k])


def test_hires_phase_trains_at_its_size_keeps_each_phase_best_and_resumes_inside_it(tmp_path):
    img_dir, jf = _mini_coco(tmp_path)
    kw = dict(train_images=str(img_dir), train_json=str(jf), val_images=str(img_dir), val_json=str(jf), size="s",
              imgsz=112, epochs=4, batch=2, accum=1, workers=0, device="cpu", amp=False, pretrained=False, ema=True,
              project=str(tmp_path / "runs"), name="h", warmup_steps=(1, 1), val_interval=5,
              hires_epochs=2, hires_imgsz=168, hires_batch=1)
    res = train(stop_after_epochs=3, seed=0, **kw)   # 2 epochs at 112, then the first at 168
    res = train(resume=res["last"], **kw)            # resumed inside the high-resolution phase
    run = tmp_path / "runs" / "h"

    rows = list(csv.DictReader((run / "metrics.csv").open()))
    assert [int(r["imgsz"]) for r in rows] == [112, 112, 168, 168]
    # validated at the end of each phase and every epoch of the high-resolution one, though val_interval is 5
    assert [r["val/segm/mAP"] != "nan" for r in rows] == [False, True, True, True]

    for imgsz in (112, 168):
        ck = torch.load(run / "weights" / f"best_{imgsz}.pt", weights_only=False)
        assert ck["imgsz"] == imgsz and "optimizer" not in ck
        assert ck["model"]["eomt.embeddings.position_embeddings.weight"].shape[0] == (imgsz // 14) ** 2
        assert load_model(run / "weights" / f"best_{imgsz}.pt", device="cpu").image_size == imgsz
    assert torch.load(res["best"], weights_only=False)["imgsz"] in (112, 168)

    last = torch.load(res["last"], weights_only=False)
    assert last["epoch"] == 3 and last["imgsz"] == 112 and last["hires_batch"] == 1  # resumes at the base size
    assert last["ema"]["decay"] == HIRES_EMA_DECAY
    head = next(g for g in last["optimizer"]["param_groups"] if not g["is_backbone"])
    assert head["initial_lr"] == pytest.approx(1e-4 / 5)  # the phase's own LR (lr0 / 5)
    args = yaml.safe_load((run / "args.yaml").read_text())
    assert args["hires_epochs"] == 2 and args["hires_imgsz"] == 168 and args["hires_batch"] == 1


def test_hires_arguments_are_checked():
    with pytest.raises(ValueError, match="hires_epochs"):
        train(train_images="x", train_json="x", epochs=2, hires_epochs=2, hires_imgsz=168)
    with pytest.raises(ValueError, match="hires_imgsz"):
        train(train_images="x", train_json="x", epochs=2, hires_epochs=1, hires_imgsz=160)
