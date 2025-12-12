from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# ============================================================================
# 설정: 평가할 모델들
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
    {
        "name": "500",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/train_ins_500/weights/best.pt'),
        "classes": [0],
    },
    {
        "name": "250",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/train_ins_250/weights/best.pt'),
        "classes": [0],
    },
]

DATA_YAML = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls/data_kisa_person_val.yaml')
OUTPUT_ROOT = Path("/Users/jihunjang/workspace/ust/fall-detection/src/metrics")

DEFAULT_CONF = 0.6
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "mps"

TITLE = "Subset-based - Kisa Overseas Person"

# 라벨 디렉토리 (fall0class 교체용)
LABELS_BASE_DIR = Path("/Users/jihunjang/Downloads/ust/dataset/val/kisa-overseas-fall/labels")


@contextmanager
def use_fall0class_labels():
    """
    Context manager: train-fall0class 라벨을 임시로 train으로 교체
    
    자동으로 원복되므로 안전합니다.
    캐시 파일도 함께 삭제하여 오래된 경로 참조 방지합니다.
    """
    train_dir = LABELS_BASE_DIR / "train"
    fall0_dir = LABELS_BASE_DIR / "train-fall0class"
    backup_dir = LABELS_BASE_DIR / "train_backup"
    cache_file = LABELS_BASE_DIR / "train.cache"
    
    # 기존 캐시 삭제 (중요!)
    if cache_file.exists():
        print("  [Cache] Deleting train.cache")
        cache_file.unlink()
    
    # 폴더 교체
    print("  [Label Swap] train -> train_backup")
    train_dir.rename(backup_dir)
    print("  [Label Swap] train-fall0class -> train")
    fall0_dir.rename(train_dir)
    
    try:
        yield
    finally:
        # 원복 (에러가 나도 반드시 실행)
        print("  [Label Swap] Restoring original labels...")
        train_dir.rename(fall0_dir)
        backup_dir.rename(train_dir)
        
        # 생성된 캐시 삭제 (다음을 위해)
        if cache_file.exists():
            print("  [Cache] Deleting generated cache")
            cache_file.unlink()
        
        print("  [Label Swap] Restored!")


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
        cache=False,  # 캐시 비활성화 (라벨 폴더 교체 시 필수)
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
    print(f"Running validation on {len(MODELS)} models")
    print(f"{'='*80}\n")
    
    for i, model_config in enumerate(MODELS, 1):
        name = model_config["name"]
        weights = model_config["weights"].resolve()
        classes = model_config["classes"]
        use_fall0class = model_config.get("use_fall0class", False)
        
        print(f"[{i}/{len(MODELS)}] Evaluating: {name}")
        print(f"  Weights: {weights.name}")
        print(f"  Classes: {classes}")
        if use_fall0class:
            print(f"  Labels: train-fall0class (swapped)")
        
        # fall0class 사용 시 자동 교체/원복
        if use_fall0class:
            with use_fall0class_labels():
                metrics = val(weights, data_yaml, classes)
        else:
            metrics = val(weights, data_yaml, classes)
        
        results[name] = metrics
        
        print(f"  Results: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        print()
    
    return results


def save_benchmark_report(results: Dict[str, Dict[str, float]]) -> Path:
    """Persist metric summary (JSON + plot) under a timestamped directory."""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # metrics = ["precision", "recall", "f1", "map50", "map50_95"]
    metrics = ["precision", "recall", "f1"]
    labels = list(results.keys())
    values = np.array([[results[label][metric] for label in labels] for metric in metrics])

    x = np.arange(len(metrics))
    
    # 동적 width 계산: 라벨 개수에 반비례
    width = min(0.8 / len(labels), 0.35)
    
    # 동적 figure 크기: 라벨 많을수록 넓게
    fig_width = max(10, 6 + len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    
    # 동적 폰트 크기: 라벨 많을수록 작게
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

    return run_dir


if __name__ == '__main__':
    result = run_val()
    save_benchmark_report(result)
