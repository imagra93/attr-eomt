"""CPU, no-network smoke tests for the high-level :class:`eomt.EoMT` interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from eomt import EoMT, EoMTModel, load_raw, summarize_checkpoint

IMGSZ = 140  # 14 * 10 -> tiny patch grid keeps the test fast
NC = 3


def test_init_from_size_builds_lazily():
    """``EoMT(size)`` defers building (and the DINOv2 load) until the model is used."""
    m = EoMT("s", device="cpu", pretrained=False, nc=NC, imgsz=IMGSZ)
    assert m.size == "s"
    assert m._model is None  # not built yet
    model = m.model  # triggers the lazy build
    assert isinstance(model, EoMTModel) and model.nc == NC


def test_init_rejects_unknown_spec():
    with pytest.raises(ValueError):
        EoMT("xl")  # neither a known size nor an existing checkpoint


def test_save_reload_and_predict_roundtrip(tmp_path):
    """save() writes a self-describing ckpt; EoMT(path) reloads it and predict() runs."""
    torch.manual_seed(0)
    m = EoMT("s", device="cpu", pretrained=False, nc=NC, imgsz=IMGSZ)
    ckpt = tmp_path / "m.pt"
    m.save(ckpt)

    reloaded = EoMT(ckpt, device="cpu")
    assert reloaded.size == "s" and reloaded.model.nc == NC

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.fromarray(np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)).save(img_dir / "a.png")

    out_dir = tmp_path / "out"
    results = reloaded.predict(img_dir, plot=True, save=str(out_dir), conf_thres=0.0)
    assert len(results) == 1
    r = results[0]
    assert {"num_detections", "boxes", "scores", "classes", "masks"} <= set(r)
    assert "plot_path" in r and (out_dir / "a.png").is_file()
    # Inference reports per-image timing.
    assert "elapsed_ms" in r and r["elapsed_ms"] > 0


def test_checkpoint_carries_norm_metadata(tmp_path):
    """save() records normalization + patch size so preprocessing is reproducible."""
    m = EoMT("s", device="cpu", pretrained=False, nc=NC, imgsz=IMGSZ)
    ckpt = tmp_path / "m.pt"
    m.save(ckpt)

    raw = load_raw(ckpt)
    assert raw["patch_size"] == 14
    assert [round(x, 3) for x in raw["norm_mean"]] == [0.485, 0.456, 0.406]
    assert [round(x, 3) for x in raw["norm_std"]] == [0.229, 0.224, 0.225]

    summary = summarize_checkpoint(ckpt)
    assert summary["size"] == "s" and summary["nc"] == NC and summary["imgsz"] == IMGSZ
    assert summary["norm_mean"] and summary["num_tensors"] > 0

    # Reload restores the normalization onto the model.
    reloaded = EoMT(ckpt, device="cpu")
    assert tuple(round(x, 3) for x in reloaded.model.pixel_mean) == (0.485, 0.456, 0.406)


def _write_photos(tmp_path, sizes):
    """A folder of random photos of differing sizes (exercises letterbox + panels)."""
    import numpy as np
    from PIL import Image

    d = tmp_path / "photos"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i, (w, h) in enumerate(sizes):
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype="uint8"), "RGB").save(
            d / f"{i:02d}.png"
        )
    return d


def test_infer_match_roundtrip(tmp_path):
    import json

    from PIL import Image

    src = _write_photos(tmp_path, [(60, 80), (80, 60), (50, 50)])
    out = tmp_path / "run"
    model = EoMT("s", device="cpu", nc=3, imgsz=140, pretrained=False)
    result = model.infer_match(
        src, plot=True, save=str(out), conf_thres=0.0, max_det=4, panel_size=96
    )

    assert result["num_images"] == 3 and len(result["images"]) == 3
    for i, res in enumerate(result["images"]):
        n = res["num_detections"]
        assert res["image_index"] == i
        assert len(res["query_idx"]) == len(res["embed"]) == len(res["identity_ids"]) == n
        assert "masks" not in res  # keep_masks defaults off
        if n:
            # ``embed`` is the RAW per-query embedding: normalization (and optional
            # centering) belong to the similarity step, which needs the raw vectors.
            assert res["embed"].dtype is torch.float32
            assert (res["embed"].norm(dim=-1) > 0).all()
            from eomt.reid import l2_normalize
            assert torch.allclose(
                l2_normalize(res["embed"]).norm(dim=-1), torch.ones(n), atol=1e-4
            )

    # Every identity holds at most one instance per photo.
    for rec in result["identities"]:
        photos = [m["image_index"] for m in rec["members"]]
        assert len(photos) == len(set(photos))

    # The summary half must be JSON-able: no tensor may have leaked into it.
    summary = json.loads(Path(result["summary_path"]).read_text())
    assert summary["num_identities"] == result["num_identities"]
    assert summary["config"]["group_by"] == ["class"]
    assert set(summary) >= {"diagnostics", "identities", "matches", "config"}
    Image.open(result["plot_path"]).close()


def test_infer_match_rejects_unknown_group_by(tmp_path):
    src = _write_photos(tmp_path, [(40, 40), (40, 40)])
    model = EoMT("s", device="cpu", nc=3, imgsz=140, pretrained=False)
    # Must raise on the typo before any forward runs (this model has no aux heads,
    # which doubles as the "group_by on an aux-less checkpoint" case).
    with pytest.raises(ValueError, match="Unknown group_by key"):
        model.infer_match(src, plot=False, save=None, group_by=("class", "frontalty"))
    # "class" alone is always valid.
    model.infer_match(src, plot=False, save=None, conf_thres=0.0, max_det=2)


def test_chain_links_report_the_true_similarity_of_a_transitive_link():
    # An identity built from a-c and b-c puts a next to b in panel order, but that
    # cut was never matched directly. Defaulting it to 0.0 would clamp to hairline
    # under sim_range=(sim_thres, 1.0) and paint a confidently linked pair as a
    # barely-made match; the real cosine is what gets drawn.
    from eomt.engine.match import _links_for
    from eomt.reid import cluster, pairwise_matches

    sim = torch.tensor([[1.0, 0.50, 0.95], [0.50, 1.0, 0.90], [0.95, 0.90, 1.0]])
    photo = torch.tensor([0, 1, 2])
    pairs = pairwise_matches(sim, photo, [(0,)] * 3, sim_thres=0.6)
    identity, pairs = cluster(pairs, sim, photo, num_instances=3, sim_thres=0.6)
    assert len(set(identity.tolist())) == 1  # all three are one identity

    accepted = [p for p in pairs if p["accepted"]]
    assert {(p["a"], p["b"]) for p in accepted} == {(0, 2), (1, 2)}  # 0-1 never was
    origin = [(0, 0), (1, 0), (2, 0)]
    drawn = {
        (a["a_panel"], a["b_panel"]): a["similarity"]
        for a in _links_for("chain", accepted, origin, identity, sim)
    }
    assert drawn[(0, 1)] == pytest.approx(0.50, abs=1e-5)  # not 0.0
    assert drawn[(1, 2)] == pytest.approx(0.90, abs=1e-5)


def test_infer_match_single_photo_and_empty_source(tmp_path):
    src = _write_photos(tmp_path, [(48, 48)])
    model = EoMT("s", device="cpu", nc=3, imgsz=140, pretrained=False)
    with pytest.warns(UserWarning, match="single photo"):
        result = model.infer_match(src, plot=False, save=None, conf_thres=0.0, max_det=3)
    assert result["matches"] == []
    assert result["num_identities"] == result["images"][0]["num_detections"]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No images found"):
        model.infer_match(empty, plot=False, save=None)
