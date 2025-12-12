from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ============================================================================
# 설정: 평가할 모델들 (Instance Based Weights)
# ============================================================================
MODELS = [
    {
        "name": "100k",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/train_ins_100k/weights/best.pt'),
        "classes": [0],
    },
    {
        "name": "10k",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/train_ins_10k/weights/best.pt'),
        "classes": [0],
    },
    {
        "name": "5k",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/train_ins_5k/weights/best.pt'),
        "classes": [0],
    },
    {
        "name": "2k",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/train_ins_2k/weights/best.pt'),
        "classes": [0],
    },
    {
        "name": "1k",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/train_ins_1k/weights/best.pt'),
        "classes": [0],
    },
]

DATA_YAML = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls/data_kisa_person_val.yaml')
OUTPUT_ROOT = Path("/Users/jihunjang/workspace/ust/fall-detection/src/metrics/kisa_person_ins")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_CONF = 0.6
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "mps"

TITLE = "FT based on Instances - KISA Overseas Person"


def val(weights: Path, data_yaml: Path, classes: List[int]) -> Dict[str, float]:
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        classes=classes,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        half=False,
        save=False,
        cache=False,
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
    results = {}
    data_yaml = DATA_YAML.resolve()
    
    print(f"\n{'='*80}")
    print(f"Running validation on {len(MODELS)} models (Instance Weights)")
    print(f"{'='*80}\n")
    
    for i, model_config in enumerate(MODELS, 1):
        name = model_config["name"]
        weights = model_config["weights"].resolve()
        classes = model_config["classes"]
        
        print(f"[{i}/{len(MODELS)}] Evaluating: {name}")
        print(f"  Weights: {weights.name}")
        print(f"  Classes: {classes}")
        
        metrics = val(weights, data_yaml, classes)

        results[name] = metrics
        
        print(f"  Results: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        print()
    
    return results


def save_benchmark_report(results: Dict[str, Dict[str, float]]) -> Path:
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["precision", "recall", "f1"]
    labels = list(results.keys())
    values = np.array([[results[label][metric] for label in labels] for metric in metrics])

    x = np.arange(len(metrics))
    
    width = min(0.8 / len(labels), 0.35)
    fig_width = max(10, 6 + len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    value_fontsize = max(7, 10 - len(labels) * 0.4)
    
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
                fontsize=value_fontsize,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    upper_ylim = max(1.0, float(values.max()) + 0.05)
    ax.set_ylim(0, upper_ylim)
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title(TITLE, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "metrics.png", dpi=200)
    plt.close(fig)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    
    print(f"Saved results to {run_dir}")
    return run_dir


if __name__ == '__main__':
    result = run_val()
    save_benchmark_report(result)





