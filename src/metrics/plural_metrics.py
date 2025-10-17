from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


BASE_WEIGHTS = Path('/src/yolo12n.pt')
FT_WEIGHTS = Path('/src/v1/result/train_30k/weights/best.pt')
DATA_YAML = Path("/src/v1/data_megafall.yaml")
OUTPUT_ROOT = Path("/src/metrics")

DEFAULT_CONF = 0.6
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "mps"


def val(weights: Path, data_yaml: Path) -> Dict[str, float]:
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        classes=[1],
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        half=False,
        save=False,
    )
    summary = results.results_dict or {}

    precision = float(summary.get("metrics/precision(B)", 0.0))
    recall = float(summary.get("metrics/recall(B)", 0.0))
    map50 = float(summary.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(summary.get("metrics/mAP50-95(B)", 0.0))
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
    }


def run_val() -> Dict[str, Dict[str, float]]:
    """Validate baseline and fine-tuned checkpoints on the same dataset."""
    finetuned_path = FT_WEIGHTS.resolve()
    data_path = DATA_YAML.resolve()
    base_metrics = val(BASE_WEIGHTS.resolve(), data_path)
    finetuned_metrics = val(finetuned_path, data_path)
    return {"baseline": base_metrics, "finetuned": finetuned_metrics}


def save_benchmark_report(results: Dict[str, Dict[str, float]]) -> Path:
    title = "MegaFallV2 Benchmark"

    """Persist metric summary (JSON + plot) under a timestamped directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["precision", "recall", "f1", "map50", "map50_95"]
    labels = list(results.keys())
    values = np.array([[results[label][metric] for label in labels] for metric in metrics])

    x = np.arange(len(metrics))
    width = 0.35 if len(labels) == 2 else 0.6

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, values[:, idx], width, label=label)
        for bar, metric in zip(bars, values[:, idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{metric:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    upper_ylim = max(1.0, float(values.max()) + 0.05)
    ax.set_ylim(0, upper_ylim)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "metrics.png", dpi=200)
    plt.close(fig)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return run_dir


if __name__ == '__main__':
    result = run_val()
    save_benchmark_report(result)
