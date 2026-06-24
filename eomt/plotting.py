"""Training-progress plots written alongside ``metrics.csv``.

Two figures, both overwritten every epoch so the run directory always holds the
latest view:

* ``metrics.png`` — loss + COCO segm mAP curves + aux-head accuracy, read straight
  from ``metrics.csv`` (no pandas; stdlib ``csv`` only).
* ``aux_per_class.png`` — secondary-head accuracy bucketed by **primary** class, a
  diagnostic for which primary classes the attribute is (in)accurate on. This one
  is *not* recorded in ``metrics.csv``.

``matplotlib`` is imported lazily (Agg backend) so the package still imports on a
box without it; the caller wraps these in try/except so a plotting failure never
interrupts training.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def _use_agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _read_csv(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Read ``metrics.csv`` into ``(fieldnames, {col: [float|nan, ...]})``."""
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        cols: dict[str, list[float]] = {k: [] for k in fields}
        for r in reader:
            for k in fields:
                v = r.get(k, "")
                try:
                    cols[k].append(float(v))
                except (TypeError, ValueError):
                    cols[k].append(float("nan"))
    return fields, cols


def _has_data(ys: list[float]) -> bool:
    return any(not math.isnan(y) for y in ys)


def _plot_series(ax, x, cols, keys_labels, title, ylabel):
    """Plot each ``(col_key, label)`` that exists and has non-NaN data."""
    plotted = False
    for key, label in keys_labels:
        if key in cols and _has_data(cols[key]):
            ax.plot(x, cols[key], marker=".", ms=3, label=label)
            plotted = True
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if plotted:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return plotted


def plot_metrics_csv(csv_path, out_png) -> None:
    """Render the run's metrics history to ``out_png`` (overwrites).

    The mAP panels follow the run's primary task: segmentation runs have
    ``val/segm/*`` columns and are plotted as segm mAP; detection runs only ever
    write ``val/bbox/*`` (see the trainer's CSV header), so those are plotted as
    bbox mAP instead — otherwise a detect run would show two empty segm panels.
    """
    csv_path, out_png = Path(csv_path), Path(out_png)
    fields, cols = _read_csv(csv_path)
    if "epoch" not in cols or not cols["epoch"]:
        return
    x = cols["epoch"]
    plt = _use_agg()

    # Primary eval task: segm when present (instance), else bbox (detect).
    task = "segm" if any(f.startswith("val/segm/") for f in fields) else "bbox"

    aux_keys = [k for k in fields if k.startswith(("train/aux_acc/", "val/aux_acc/"))]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _plot_series(axes[0, 0], x, cols, [("train/loss", "train loss")], "Loss", "loss")
    _plot_series(
        axes[0, 1], x, cols,
        [(f"val/{task}/mAP", "mAP"), (f"val/{task}/mAP50", "mAP50"),
         (f"val/{task}/mAP75", "mAP75")],
        f"Val {task} mAP", "mAP",
    )
    _plot_series(
        axes[1, 0], x, cols,
        [(f"val/{task}/mAP_small", "small"), (f"val/{task}/mAP_medium", "medium"),
         (f"val/{task}/mAP_large", "large")],
        f"Val {task} mAP by size", "mAP",
    )
    if aux_keys:
        _plot_series(
            axes[1, 1], x, cols,
            [(k, k.replace("/aux_acc/", " ").replace("train", "tr").replace("val", "va"))
             for k in aux_keys],
            "Aux head accuracy", "accuracy",
        )
    else:
        axes[1, 1].axis("off")

    fig.suptitle(f"{csv_path.parent.name} — through epoch {int(x[-1])}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_aux_per_class(per_class, class_names, out_png) -> None:
    """Bar chart of aux accuracy per primary class, one row per aux head (overwrites).

    ``per_class`` is ``{head: {primary_cls_id: (correct, total)}}``; ``class_names``
    maps ``primary_cls_id -> label``. Bars are labelled with the instance count, and
    classes with no instances are omitted.
    """
    out_png = Path(out_png)
    heads = [h for h, b in per_class.items() if b]
    if not heads:
        return
    plt = _use_agg()

    fig, axes = plt.subplots(
        len(heads), 1, figsize=(max(7, 0.5 * max(len(per_class[h]) for h in heads)), 3.2 * len(heads)),
        squeeze=False,
    )
    for ax, head in zip(axes[:, 0], heads):
        buckets = per_class[head]
        cls_ids = sorted(buckets)
        labels = [str(class_names.get(c, c)) for c in cls_ids]
        accs = [(buckets[c][0] / buckets[c][1] if buckets[c][1] else 0.0) for c in cls_ids]
        totals = [buckets[c][1] for c in cls_ids]
        bars = ax.bar(range(len(cls_ids)), accs, color="#4c78a8")
        for bar, n in zip(bars, totals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"n={n}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(cls_ids)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("accuracy")
        ax.set_title(f"aux '{head}' accuracy by primary class")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
