"""Command-line interface for libre-eomt.

    eomt train   --data configs/coco.yaml --size s --epochs 50 --batch 4
    eomt val     --weights runs/train/eomt/weights/best.pt --data configs/coco.yaml
    eomt predict --weights best.pt --source path/to/img_or_dir --out runs/predict
    eomt download --root datasets/coco
"""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="libre-eomt: EoMT instance segmentation.")


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
    clip_norm: float = typer.Option(0.01),
    workers: int = typer.Option(4),
    device: str = typer.Option("auto"),
    amp: bool = typer.Option(True),
    pretrained: bool = typer.Option(True, help="Init encoder from DINOv2."),
    project: str = typer.Option("runs/train"),
    name: str = typer.Option("eomt"),
    val_interval: int = typer.Option(1),
    logger: str = typer.Option("none", help="none | tensorboard | wandb."),
    resume: Optional[str] = typer.Option(None, help="Resume from a last.pt checkpoint."),
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
    if not (ti and tj):
        raise typer.BadParameter("Provide --data or --train-images/--train-json.")

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
    typer.echo(f"[done] best segm mAP={result['best_metric']:.4f}; weights in {result['weights_dir']}")


@app.command()
def val(
    weights: str = typer.Option(..., help="Path to a trained checkpoint."),
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
    weights: str = typer.Option(..., help="Path to a trained checkpoint."),
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
