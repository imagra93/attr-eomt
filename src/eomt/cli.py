"""Command-line interface for libre-eomt.

    eomt train   --data configs/coco.yaml --size s --epochs 50 --batch 4
    eomt val     --weights runs/train/eomt/weights/best.pt --data configs/coco.yaml
    eomt predict --weights best.pt --source path/to/img_or_dir --out runs/predict
    eomt download --root datasets/coco
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="libre-eomt: EoMT instance segmentation.")


def _run_args(resume_path: str) -> dict:
    """Best-effort read of a run's ``args.yaml`` from a resume target.

    ``resume_path`` may be the run folder, its ``weights/`` subdir, or a ``.pt``
    inside either — we walk up to find the ``args.yaml`` train() wrote.
    """
    import yaml

    p = Path(resume_path)
    for d in (p, p.parent, p.parent.parent):
        f = d / "args.yaml"
        try:
            if f.is_file():
                return yaml.safe_load(f.read_text()) or {}
        except OSError:
            pass
    return {}


@app.command()
def train(
    data: Optional[str] = typer.Option(None, help="Dataset YAML (see configs/coco.yaml)."),
    train_images: Optional[str] = typer.Option(None, help="Train image dir (overrides --data)."),
    train_json: Optional[str] = typer.Option(None, help="Train COCO JSON (overrides --data)."),
    val_images: Optional[str] = typer.Option(None, help="Val image dir."),
    val_json: Optional[str] = typer.Option(None, help="Val COCO JSON."),
    size: str = typer.Option("l", help="Model size: s | b | l."),
    imgsz: int = typer.Option(644, help="Square input size (divisible by 14)."),
    epochs: int = typer.Option(50),
    batch: int = typer.Option(4, help="Micro-batch size (per optimizer micro-step)."),
    nominal_batch: int = typer.Option(
        16, help="Target effective batch via grad accumulation (0=off). EoMT recipe is 16."
    ),
    accum: int = typer.Option(0, help="Explicit grad-accum steps; overrides --nominal-batch when >0."),
    lr0: float = typer.Option(1e-4),
    weight_decay: float = typer.Option(0.05),
    backbone_lr_mult: float = typer.Option(0.1),
    llrd: float = typer.Option(0.85, help="Layer-wise LR decay for the backbone (1.0=off)."),
    warmup_epochs: float = typer.Option(1.0),
    ema: bool = typer.Option(True, help="Validate/export an EMA copy of the weights."),
    ema_decay: float = typer.Option(0.9999, help="EMA decay (ramped via --ema-tau)."),
    ema_tau: float = typer.Option(2000.0, help="EMA decay ramp time-constant (updates)."),
    mask_anneal: bool = typer.Option(True, help="Anneal masked attention off (EoMT recipe)."),
    mask_anneal_start: float = typer.Option(0.0, help="Anneal start, fraction of training."),
    mask_anneal_end: float = typer.Option(0.9, help="Anneal end (mask-free after), fraction."),
    aux_w: float = typer.Option(1.0, help="Weight for secondary attribute head losses."),
    aux_w_warmup: float = typer.Option(
        0.0, help="Ramp aux loss weight 0->1 over this fraction of training (0=off)."
    ),
    aux_w_head: Optional[str] = typer.Option(
        None, help="Per-head aux weights, e.g. 'laterality=0.5,severity=2.0'."
    ),
    aux_class_weights: bool = typer.Option(
        False, help="Inverse-sqrt-frequency CE weights per aux head (imbalance)."
    ),
    aux_iou_gate: float = typer.Option(
        0.5, help="Train aux heads only on matched queries with mask IoU >= this (0=off)."
    ),
    aux_class_gate: bool = typer.Option(
        True, help="Also require the matched query's predicted primary class to be correct."
    ),
    aux_head_layers: int = typer.Option(
        2, help="Aux head depth: 1=linear probe, >=2=MLP."
    ),
    aux_head_hidden: Optional[int] = typer.Option(
        None, help="Aux MLP hidden width (default: encoder hidden_size)."
    ),
    aux_head_dropout: float = typer.Option(0.0, help="Aux MLP dropout."),
    no_object_weight: float = typer.Option(
        0.1, help="CE weight on the null class for unmatched queries; lower => more "
        "detections / higher recall (try 0.05)."
    ),
    class_weight: float = typer.Option(2.0, help="Classification loss/matcher weight."),
    mask_weight: float = typer.Option(5.0, help="Per-pixel mask BCE loss/matcher weight."),
    dice_weight: float = typer.Option(
        5.0, help="Dice loss/matcher weight (scale-invariant; raise for small/thin masks, try 8)."
    ),
    train_num_points: float = typer.Option(
        12544, help="PointRend points sampled for the mask loss; more => sharper boundaries (try 24576)."
    ),
    oversample_ratio: float = typer.Option(3.0, help="PointRend oversampling factor."),
    importance_sample_ratio: float = typer.Option(0.75, help="PointRend uncertain-point fraction."),
    num_upscale_blocks: Optional[int] = typer.Option(
        None, help="Mask-head upscale blocks (default: size preset, 2). Each block doubles "
        "mask-logit resolution; 3 sharpens masks but breaks 1:1 warm-start of the head."
    ),
    clip_norm: float = typer.Option(0.01),
    workers: int = typer.Option(8),
    prefetch: int = typer.Option(4, help="DataLoader prefetch_factor per worker."),
    device: str = typer.Option("auto"),
    amp: bool = typer.Option(True),
    tf32: bool = typer.Option(True, help="Enable TF32 matmul/conv on Ampere+ GPUs."),
    compile: bool = typer.Option(False, help="torch.compile the model (experimental)."),
    seed: Optional[int] = typer.Option(None, help="Seed RNGs for reproducibility."),
    pretrained: bool = typer.Option(True, help="Init encoder from DINOv2."),
    flip_prob: float = typer.Option(
        0.5,
        help="Horizontal-flip probability (train aug; Mask2Former/EoMT default 0.5). "
        "NOTE: hflip mirrors the image without swapping any per-instance left/right "
        "'laterality' attribute, so set --flip-prob 0 when training a dataset that has "
        "a laterality head (e.g. the vehicle-parts data).",
    ),
    min_scale: float = typer.Option(0.1, help="LSJ min scale (legacy stretch-style: 0.5)."),
    max_scale: float = typer.Option(2.0, help="LSJ max scale (legacy stretch-style: 1.0)."),
    letterbox: bool = typer.Option(True, help="Aspect-preserving letterbox eval (vs square stretch)."),
    project: str = typer.Option("runs/train"),
    name: Optional[str] = typer.Option(None, help="Run name (default: eomt-{size})."),
    val_interval: int = typer.Option(1),
    logger: str = typer.Option("none", help="none | tensorboard | wandb."),
    resume: Optional[str] = typer.Option(
        None, help="Resume from a checkpoint or a run/weights folder (uses last.pt)."
    ),
    weights: Optional[str] = typer.Option(
        None,
        help="Warm-start (fine-tune) model weights from a checkpoint or run/weights "
        "folder (uses best.pt), then train fresh from epoch 0 with a new optimizer/LR "
        "schedule/EMA. Use this to retrain with changed settings (e.g. --flip-prob 0) "
        "while keeping the learned weights. Differs from --resume, which continues a run.",
    ),
):
    """Fine-tune EoMT on a COCO-format dataset."""
    from .data import load_data_config
    from .engine import train as run_train

    ti, tj, vi, vj = train_images, train_json, val_images, val_json
    if data:
        cfg = load_data_config(data)
        ti = ti or cfg["train_images"]
        tj = tj or cfg["train_json"]
        vi = vi or cfg["val_images"]
        vj = vj or cfg["val_json"]
    if (resume or weights) and not (ti and tj):
        # Recover the dataset paths recorded in the source run's args.yaml.
        run_cfg = _run_args(resume or weights)
        ti = ti or run_cfg.get("train_images")
        tj = tj or run_cfg.get("train_json")
        vi = vi or run_cfg.get("val_images")
        vj = vj or run_cfg.get("val_json")
    if not (ti and tj):
        raise typer.BadParameter(
            "Provide --data or --train-images/--train-json "
            "(or --resume a run with an args.yaml that records them)."
        )

    per_head = None
    if aux_w_head:
        per_head = {}
        for part in aux_w_head.split(","):
            part = part.strip()
            if not part:
                continue
            key, sep, val = part.partition("=")
            if not sep:
                raise typer.BadParameter(f"--aux-w-head expects 'name=weight', got {part!r}.")
            per_head[key.strip()] = float(val)

    result = run_train(
        train_images=ti,
        train_json=tj,
        val_images=vi,
        val_json=vj,
        size=size,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        nominal_batch=nominal_batch,
        accum=accum,
        lr0=lr0,
        weight_decay=weight_decay,
        backbone_lr_mult=backbone_lr_mult,
        llrd=llrd,
        warmup_epochs=warmup_epochs,
        ema=ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        mask_anneal=mask_anneal,
        mask_anneal_start=mask_anneal_start,
        mask_anneal_end=mask_anneal_end,
        aux_w=aux_w,
        aux_w_warmup=aux_w_warmup,
        aux_w_per_head=per_head,
        aux_class_weights=aux_class_weights,
        aux_iou_gate=aux_iou_gate,
        aux_class_gate=aux_class_gate,
        aux_head_layers=aux_head_layers,
        aux_head_hidden=aux_head_hidden,
        aux_head_dropout=aux_head_dropout,
        no_object_weight=no_object_weight,
        class_weight=class_weight,
        mask_weight=mask_weight,
        dice_weight=dice_weight,
        train_num_points=int(train_num_points),
        oversample_ratio=oversample_ratio,
        importance_sample_ratio=importance_sample_ratio,
        num_upscale_blocks=num_upscale_blocks,
        clip_norm=clip_norm,
        workers=workers,
        prefetch=prefetch,
        device=device,
        amp=amp,
        tf32=tf32,
        compile=compile,
        seed=seed,
        pretrained=pretrained,
        flip_prob=flip_prob,
        min_scale=min_scale,
        max_scale=max_scale,
        letterbox=letterbox,
        project=project,
        name=name,
        val_interval=val_interval,
        logger=logger,
        resume=resume,
        init_weights=weights,
    )
    if result["best_metric"] >= 0:
        typer.echo(
            f"[done] best segm mAP={result['best_metric']:.4f}; weights in {result['weights_dir']}"
        )
    else:
        typer.echo(f"[done] no validation set; weights in {result['weights_dir']}")


@app.command()
def val(
    weights: str = typer.Option(..., help="Checkpoint .pt or a run/weights folder."),
    data: Optional[str] = typer.Option(None, help="Dataset YAML."),
    val_images: Optional[str] = typer.Option(None),
    val_json: Optional[str] = typer.Option(None),
    batch: int = typer.Option(4),
    workers: int = typer.Option(4),
    device: str = typer.Option("auto"),
    conf_thres: float = typer.Option(0.0),
    max_det: int = typer.Option(100),
    letterbox: Optional[bool] = typer.Option(
        None, help="Override eval preprocessing; default follows how the checkpoint was trained."
    ),
):
    """Evaluate a checkpoint with COCO segm/bbox mAP."""
    from .data import CocoValImages, load_data_config
    from .engine import evaluate
    from .serialization import load_model

    vi, vj = val_images, val_json
    if data:
        cfg = load_data_config(data)
        vi = vi or cfg["val_images"]
        vj = vj or cfg["val_json"]
    if not (vi and vj):
        raise typer.BadParameter("Provide --data or --val-images/--val-json.")

    model = load_model(weights, device=device)
    dev = next(model.parameters()).device
    lb = letterbox if letterbox is not None else bool(getattr(model, "preprocess_letterbox", False))
    val_ds = CocoValImages(vi, vj, imgsz=int(model.image_size), letterbox=lb)
    metrics = evaluate(
        model, val_ds, device=dev, batch_size=batch, num_workers=workers,
        conf_thres=conf_thres, max_det=max_det,
    )
    for k, v in metrics.items():
        typer.echo(f"{k}: {v:.4f}")


def _parse_floats(spec: str, name: str) -> list[float]:
    """Parse a comma-separated numeric grid axis like ``"0,0.1,0.5"``."""
    try:
        vals = [float(x) for x in spec.split(",") if x.strip() != ""]
    except ValueError as e:
        raise typer.BadParameter(f"--{name} expects comma-separated numbers, got {spec!r}.") from e
    if not vals:
        raise typer.BadParameter(f"--{name} is empty.")
    return vals


@app.command()
def sweep(
    weights: str = typer.Option(..., help="Checkpoint .pt or a run/weights folder."),
    data: Optional[str] = typer.Option(None, help="Dataset YAML."),
    val_images: Optional[str] = typer.Option(None),
    val_json: Optional[str] = typer.Option(None),
    batch: int = typer.Option(4),
    workers: int = typer.Option(4),
    device: str = typer.Option("auto"),
    conf_thres: str = typer.Option("0,0.1,0.2,0.3,0.4,0.5", help="conf_thres grid axis."),
    mask_thresh: str = typer.Option("0.4,0.5,0.6", help="mask_thresh grid axis."),
    max_det: str = typer.Option("100,300", help="max_det grid axis."),
    min_area: str = typer.Option("0", help="min_mask_area grid axis (original-image pixels)."),
    out: Optional[str] = typer.Option(None, help="Write the full results table to this CSV."),
    letterbox: Optional[bool] = typer.Option(
        None, help="Override eval preprocessing; default follows how the checkpoint was trained."
    ),
):
    """Sweep inference knobs (conf/mask threshold, max_det, min area) — no retraining.

    Runs the model once per batch and re-scores the shared logits across the knob
    grid, reporting COCO segm metrics per operating point and recommending the best
    for overall mAP and small-object mAP. Map the winner onto ``eomt val/predict``
    via ``--conf-thres / --mask-thresh / --max-det``.
    """
    import csv as _csv
    import itertools

    from .data import CocoValImages, load_data_config
    from .engine import sweep as run_sweep
    from .serialization import load_model

    vi, vj = val_images, val_json
    if data:
        cfg = load_data_config(data)
        vi = vi or cfg["val_images"]
        vj = vj or cfg["val_json"]
    if not (vi and vj):
        raise typer.BadParameter("Provide --data or --val-images/--val-json.")

    confs = _parse_floats(conf_thres, "conf-thres")
    masks = _parse_floats(mask_thresh, "mask-thresh")
    dets = [int(x) for x in _parse_floats(max_det, "max-det")]
    areas = _parse_floats(min_area, "min-area")
    grid = [
        {"conf_thres": c, "mask_thresh": m, "max_det": d, "min_mask_area": a}
        for c, m, d, a in itertools.product(confs, masks, dets, areas)
    ]
    typer.echo(f"[sweep] {len(grid)} operating points over {len(confs)}x{len(masks)}x{len(dets)}x{len(areas)} grid")

    model = load_model(weights, device=device)
    dev = next(model.parameters()).device
    lb = letterbox if letterbox is not None else bool(getattr(model, "preprocess_letterbox", False))
    val_ds = CocoValImages(vi, vj, imgsz=int(model.image_size), letterbox=lb)

    rows = run_sweep(model, val_ds, device=dev, grid=grid, batch_size=batch, num_workers=workers)

    cols = ["conf_thres", "mask_thresh", "max_det", "min_mask_area",
            "segm/mAP", "segm/mAP50", "segm/mAP75", "segm/mAP_small", "segm/AR100"]
    # Printed markdown table.
    typer.echo("| " + " | ".join(cols) + " |")
    typer.echo("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        typer.echo("| " + " | ".join(f"{r.get(c, 0.0):.4g}" for c in cols) + " |")

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_cols = ["conf_thres", "mask_thresh", "max_det", "min_mask_area"] + \
            [k for k in rows[0] if k.startswith(("segm/", "bbox/"))]
        with out_path.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=all_cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in all_cols})
        typer.echo(f"[sweep] wrote {len(rows)} rows to {out_path}")

    def _best(metric: str):
        return max(rows, key=lambda r: r.get(metric, -1.0))

    for metric in ("segm/mAP", "segm/mAP_small"):
        b = _best(metric)
        typer.echo(
            f"[best {metric}={b.get(metric, 0.0):.4f}] conf_thres={b['conf_thres']} "
            f"mask_thresh={b['mask_thresh']} max_det={b['max_det']} min_area={b['min_mask_area']}"
        )


@app.command()
def predict(
    weights: str = typer.Option(..., help="Checkpoint .pt or a run/weights folder."),
    source: str = typer.Option(..., help="Image file or directory."),
    out: str = typer.Option("runs/predict", help="Output directory."),
    conf_thres: float = typer.Option(0.3),
    max_det: int = typer.Option(100),
    mask_thresh: float = typer.Option(0.5),
    device: str = typer.Option("auto"),
    alpha: float = typer.Option(0.5, help="Mask overlay opacity."),
    draw_boxes: bool = typer.Option(True),
):
    """Run inference and render masks/boxes onto images."""
    from .engine import predict as run_predict

    written = run_predict(
        weights, source, out_dir=out, conf_thres=conf_thres, max_det=max_det,
        mask_thresh=mask_thresh, device=device, alpha=alpha, draw_boxes=draw_boxes,
    )
    typer.echo(f"[done] wrote {len(written)} image(s) to {out}")


@app.command()
def download(
    root: str = typer.Option("datasets/coco", help="Destination root."),
    train: bool = typer.Option(True),
    val: bool = typer.Option(True),
):
    """Download COCO 2017 (images + annotations) into ROOT."""
    from .data import ensure_coco

    paths = ensure_coco(root, train=train, val=val)
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


if __name__ == "__main__":
    app()
