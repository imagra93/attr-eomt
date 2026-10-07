"""Multi-GPU training (one process per GPU): needs two CUDA GPUs, skipped otherwise."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 CUDA GPUs")

IMGSZ = 140


def _shard_and_gather(device):
    """In each process: its shard of a 5-item dataset (gathered) and per-process counts (summed)."""
    from eomt.engine.distributed import gather, rank, shard, sum_over_processes

    items = gather(list(shard(list(range(5)))))
    counts = sum_over_processes(({"seen": 1}, {"head": {rank(): [rank(), 1]}}))
    return items, counts


def test_shard_gather_cover_every_item_once():
    """Evaluation shards are disjoint and unpadded (an odd count is not topped up); counts sum over the processes."""
    from eomt.engine.distributed import launch

    items, counts = launch(_shard_and_gather, [0, 1], {"device": None})
    assert items == [[0, 2, 4], [1, 3]]
    assert counts == ({"seen": 2}, {"head": {0: [0, 1], 1: [1, 1]}})


def _write_dataset(root):
    """A few images with one or two rectangles (one attribute head) and an empty negative image."""
    rng = np.random.default_rng(0)
    for split, n in (("train", 8), ("val", 3)):
        (root / split).mkdir(parents=True)
        images, anns = [], []
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)).save(root / split / f"{i}.png")
            images.append({"id": i + 1, "file_name": f"{i}.png", "width": IMGSZ, "height": IMGSZ})
            for k in range(i % 3):  # 0, 1 or 2 instances: some images are negatives
                x, y = 10 + 60 * k, 20
                anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": 1 + k, "iscrowd": 0,
                             "bbox": [x, y, 50, 60], "area": 3000, "segmentation": [[x, y, x + 50, y, x + 50, y + 60, x, y + 60]],
                             "attributes": {"tone": k}})
        coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
                "attributes": [{"name": "tone", "categories": [{"id": 0, "name": "dark"}, {"id": 1, "name": "light"}]}]}
        (root / f"{split}.json").write_text(json.dumps(coco))
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "train_images": "train", "train_json": "train.json",
                                    "val_images": "val", "val_json": "val.json"}))
    return data


def test_train_on_two_gpus(tmp_path):
    """``device="0,1"`` trains end to end: same effective batch, one metrics row with val metrics, weights, result."""
    from eomt import EoMT

    data = _write_dataset(tmp_path / "data")
    result = EoMT("s", device="0,1", pretrained=False).train(
        data=str(data), epochs=1, batch=1, nominal_batch=4, imgsz=IMGSZ, workers=1, keep_empty=True,
        pretrained=False, project=str(tmp_path / "runs"), name="ddp", seed=0,
    )
    run = tmp_path / "runs" / "ddp"
    args = yaml.safe_load(open(run / "args.yaml"))
    rows = list(csv.DictReader(open(run / "metrics.csv")))
    assert (args["gpus"], args["accum"], args["effective_batch"]) == (2, 2, 4)
    assert len(rows) == 1 and rows[0]["val/segm/mAP"] != "nan"  # the main process got the gathered val metrics
    assert result["last"] == str(run / "weights" / "last.pt") and (run / "weights" / "last.pt").is_file()
