"""
Finetuned 모델 평가 스크립트
- 5k~250 모델에 대해 fall, person 평가
- 예측 이미지 + 라벨 저장 (recall 원인 분석용)
"""

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
# 설정
# ============================================================================
MODELS = [
    {
        "name": "5k",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/gopr_5k/weights/best.pt'),
    },
    {
        "name": "1k",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/gopr_1k/weights/best.pt'),
    },
    {
        "name": "500",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/gopr_500/weights/best.pt'),
    },
    {
        "name": "250",
        "weights": Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/gopr_250/weights/best.pt'),
    },
]

DATA_YAML = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls/data_gopr_fall_val.yaml')
OUTPUT_ROOT = Path("/Users/jihunjang/workspace/ust/fall-detection/src/metrics")

EVAL_TYPES = {
    'person': {'classes': [0], 'use_fall0class': False},
    'fall':   {'classes': [1], 'use_fall0class': False},
}

DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "mps"

# GOPR test 이미지/라벨 경로
TEST_IMAGES_DIR = Path("/Users/jihunjang/Downloads/ust/dataset/train/gopr/images/test")
LABELS_BASE_DIR = Path("/Users/jihunjang/Downloads/ust/dataset/train/gopr/labels/test")


@contextmanager
def use_fall0class_labels():
    """GOPR test 라벨에서 class 0/1 스왑"""
    test_dir = LABELS_BASE_DIR
    backup_dir = test_dir.parent / "test_backup"
    fall0_dir = test_dir.parent / "test-fall0class"
    cache_file = test_dir.parent / "test.cache"

    if not fall0_dir.exists():
        print("  [Label] Creating fall0class labels...")
        fall0_dir.mkdir(parents=True)
        for label_file in test_dir.glob("*.txt"):
            lines = label_file.read_text().strip().split('\n')
            new_lines = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts[0] == '1':
                        parts[0] = '0'
                    elif parts[0] == '0':
                        parts[0] = '1'
                    new_lines.append(' '.join(parts))
            (fall0_dir / label_file.name).write_text('\n'.join(new_lines))
        print(f"  [Label] Created {len(list(fall0_dir.glob('*.txt')))} fall0class labels")

    if cache_file.exists():
        cache_file.unlink()

    print("  [Label Swap] test -> test_backup")
    test_dir.rename(backup_dir)
    print("  [Label Swap] test-fall0class -> test")
    fall0_dir.rename(test_dir)

    try:
        yield
    finally:
        print("  [Label Swap] Restoring original labels...")
        test_dir.rename(fall0_dir)
        backup_dir.rename(test_dir)
        if cache_file.exists():
            cache_file.unlink()
        print("  [Label Swap] Restored!")


def val(weights: Path, data_yaml: Path, classes: List[int],
        save_dir: str, run_name: str) -> Dict[str, float]:
    """model.val()로 메트릭 계산 + model.predict()로 개별 이미지 저장"""
    model = YOLO(str(weights))

    # 1. val: 메트릭 계산 + plots
    results = model.val(
        data=str(data_yaml),
        classes=classes,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        half=False,
        save=False,
        plots=True,
        cache=False,
        project=save_dir,
        name=run_name,
        exist_ok=True,
    )
    summary = results.results_dict or {}

    precision = float(summary.get("metrics/precision(B)", 0.0))
    recall = float(summary.get("metrics/recall(B)", 0.0))
    map50 = float(summary.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(summary.get("metrics/mAP50-95(B)", 0.0))
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom else 0.0

    # 2. predict: 모든 test 이미지에 prediction 박스 그려서 저장
    predict_dir = str(Path(save_dir) / f"{run_name}_predictions")
    model.predict(
        source=str(TEST_IMAGES_DIR),
        classes=classes,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=predict_dir,
        name="images",
        exist_ok=True,
    )
    print(f"    Predictions saved to: {predict_dir}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
    }


def save_bar_plot(results: Dict[str, Dict[str, float]], eval_type: str, save_dir: Path):
    """Bar plot 저장"""
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
        for bar, metric_val in zip(bars, values[:, idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{metric_val:.3f}",
                ha="center", va="bottom", fontsize=value_fontsize,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.set_ylim(0, max(1.0, float(values.max()) + 0.05))
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title(f"VFP290K {eval_type.capitalize()}", fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_dir / f"metrics_{eval_type}.png", dpi=200)
    plt.close(fig)


def main():
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_dir = OUTPUT_ROOT / f"finetuned_{timestamp}"
    data_yaml = DATA_YAML.resolve()

    print(f"{'='*80}")
    print(f"Finetuned Model Evaluation (fall + person)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {base_dir}")
    print(f"{'='*80}")

    all_results = {}

    for eval_type, eval_config in EVAL_TYPES.items():
        classes = eval_config['classes']
        use_fall0class = eval_config['use_fall0class']
        detect_dir = str(base_dir / eval_type)

        print(f"\n{'='*80}")
        print(f"  {eval_type.upper()} Evaluation (classes={classes})")
        if use_fall0class:
            print(f"  Using fall0class labels")
        print(f"{'='*80}")

        eval_results = {}

        for i, model_config in enumerate(MODELS, 1):
            name = model_config["name"]
            weights = model_config["weights"].resolve()

            print(f"\n  [{i}/{len(MODELS)}] {name}")

            if use_fall0class:
                with use_fall0class_labels():
                    metrics = val(weights, data_yaml, classes, detect_dir, name)
            else:
                metrics = val(weights, data_yaml, classes, detect_dir, name)

            eval_results[name] = metrics
            print(f"    P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

        all_results[eval_type] = eval_results

        # Bar plot 저장
        save_bar_plot(eval_results, eval_type, base_dir)

    # 전체 결과 JSON 저장
    with (base_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Summary
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    for eval_type, eval_results in all_results.items():
        print(f"\n[{eval_type.upper()}]")
        print(f"  {'Model':<10} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>8}")
        print(f"  {'-'*42}")
        for name, m in eval_results.items():
            print(f"  {name:<10} {m['precision']:>8.3f} {m['recall']:>8.3f} "
                  f"{m['f1']:>8.3f} {m['map50']:>8.3f}")

    print(f"\nAll results saved to: {base_dir}")


if __name__ == '__main__':
    main()
