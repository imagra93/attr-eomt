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
    batch: int = typer.Option(4),
    lr0: float = typer.Option(1e-4),
    weight_decay: float = typer.Option(0.05),
    backbone_lr_mult: float = typer.Option(0.1),
    warmup_epochs: float = typer.Option(1.0),
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
    clip_norm: float = typer.Option(0.01),
    workers: int = typer.Option(4),
    device: str = typer.Option("auto"),
    amp: bool = typer.Option(True),
    pretrained: bool = typer.Option(True, help="Init encoder from DINOv2."),
    project: str = typer.Option("runs/train"),
    name: str = typer.Option("eomt"),
    val_interval: int = typer.Option(1),
    logger: str = typer.Option("none", help="none | tensorboard | wandb."),
    resume: Optional[str] = typer.Option(
        None, help="Resume from a checkpoint or a run/weights folder (uses last.pt)."
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
    if resume and not (ti and tj):
        # Recover the dataset paths recorded in the resumed run's args.yaml.
        run_cfg = _run_args(resume)
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
        lr0=lr0,
        weight_decay=weight_decay,
        backbone_lr_mult=backbone_lr_mult,
        warmup_epochs=warmup_epochs,
        mask_anneal=mask_anneal,
        mask_anneal_start=mask_anneal_start,
        mask_anneal_end=mask_anneal_end,
        aux_w=aux_w,
        aux_w_warmup=aux_w_warmup,
        aux_w_per_head=per_head,
        aux_class_weights=aux_class_weights,
        clip_norm=clip_norm,
        workers=workers,
        device=device,
        amp=amp,
        pretrained=pretrained,
        project=project,
        name=name,
        val_interval=val_interval,
        logger=logger,
        resume=resume,
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
    val_ds = CocoValImages(vi, vj, imgsz=int(model.image_size))
    metrics = evaluate(
        model, val_ds, device=dev, batch_size=batch, num_workers=workers,
        conf_thres=conf_thres, max_det=max_det,
    )
    for k, v in metrics.items():
        typer.echo(f"{k}: {v:.4f}")


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
