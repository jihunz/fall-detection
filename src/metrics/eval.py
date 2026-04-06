"""
스케일별 모델 평가 스크립트
- person(class 0), fall(class 1) 각각 평가
- Precision, Recall, F1, mAP50, mAP50-95
- Scaling curve (log-scale X축) 시각화
- 결과 JSON 저장

Usage:
  python eval.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# ============================================================================
# 설정
# ============================================================================
EVAL_TARGET = 'outdoor'  # 평가 대상 ('indoor' / 'outdoor')

V2_RESULT_DIR = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v2/result')

MODELS = [
    {"name": "1percent",   "scale": 375,   "weights": V2_RESULT_DIR / 'indoor_1percent/weights/best.pt'},
    {"name": "5percent",   "scale": 1876,  "weights": V2_RESULT_DIR / 'indoor_5percent/weights/best.pt'},
    {"name": "10percent",  "scale": 3753,  "weights": V2_RESULT_DIR / 'indoor_10percent/weights/best.pt'},
    {"name": "25percent",  "scale": 9382,  "weights": V2_RESULT_DIR / 'indoor_25percent/weights/best.pt'},
    {"name": "50percent",  "scale": 18764, "weights": V2_RESULT_DIR / 'indoor_50percent/weights/best.pt'},
    {"name": "75percent",  "scale": 28145, "weights": V2_RESULT_DIR / 'indoor_75percent/weights/best.pt'},
    {"name": "100percent", "scale": 37527, "weights": V2_RESULT_DIR / 'indoor_100percent/weights/best.pt'},
]

DATA_YAML = Path(f'/Users/jihunjang/workspace/ust/fall-detection/src/v2/yamls/data_{EVAL_TARGET}_eval.yaml')
OUTPUT_ROOT = Path('/Users/jihunjang/workspace/ust/fall-detection/src/metrics')

EVAL_TYPES = {
    'person': {'classes': [0]},
    'fall':   {'classes': [1]},
}

DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "mps"


# ============================================================================
# 평가
# ============================================================================
def val(weights: Path, data_yaml: Path, classes: List[int],
        save_dir: str, run_name: str) -> Dict[str, float]:
    """model.val()로 메트릭 계산"""
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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
    }


# ============================================================================
# 시각화
# ============================================================================
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
    ax.set_title(f"{EVAL_TARGET.capitalize()} {eval_type.capitalize()}", fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_dir / f"metrics_{eval_type}.png", dpi=200)
    plt.close(fig)


def save_scaling_curve(all_results: Dict[str, Dict[str, Dict[str, float]]],
                       scales: List[int], save_dir: Path):
    """Scaling curve: mAP50 vs training data size (log scale)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, eval_type in zip(axes, ['person', 'fall']):
        if eval_type not in all_results:
            continue

        eval_results = all_results[eval_type]
        model_names = list(eval_results.keys())

        x = scales[:len(model_names)]
        map50_vals = [eval_results[name]['map50'] for name in model_names]
        f1_vals = [eval_results[name]['f1'] for name in model_names]

        ax.plot(x, map50_vals, 'o-', color='#FFA400', linewidth=2, markersize=8, label='mAP50')
        ax.plot(x, f1_vals, 's--', color='#404040', linewidth=2, markersize=7, label='F1')

        for xi, m50, f1 in zip(x, map50_vals, f1_vals):
            ax.annotate(f'{m50:.3f}', (xi, m50), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8, color='#FFA400')
            ax.annotate(f'{f1:.3f}', (xi, f1), textcoords="offset points",
                        xytext=(0, -15), ha='center', fontsize=8, color='#404040')

        ax.set_xscale('log')
        ax.set_xlabel('Training images (log scale)', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'Scaling Curve ({EVAL_TARGET.capitalize()}) —{eval_type.capitalize()}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(save_dir / "scaling_curve.png", dpi=200)
    plt.close(fig)
    print(f"Scaling curve saved to: {save_dir / 'scaling_curve.png'}")


# ============================================================================
# Main
# ============================================================================
def main():
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_dir = OUTPUT_ROOT / f"eval_{EVAL_TARGET}_{timestamp}"
    data_yaml = DATA_YAML.resolve()

    # 존재하는 모델만 필터
    available_models = [m for m in MODELS if m['weights'].exists()]
    if not available_models:
        print("ERROR: No trained models found. Run yolo_ft_indoor.py first.")
        return

    print(f"{'=' * 80}")
    print(f"{EVAL_TARGET.capitalize()} Scale Evaluation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {[m['name'] for m in available_models]}")
    print(f"Output: {base_dir}")
    print(f"{'=' * 80}")

    all_results = {}

    for eval_type, eval_config in EVAL_TYPES.items():
        classes = eval_config['classes']
        detect_dir = str(base_dir / eval_type)

        print(f"\n{'=' * 80}")
        print(f"  {eval_type.upper()} Evaluation (classes={classes})")
        print(f"{'=' * 80}")

        eval_results = {}

        for i, model_config in enumerate(available_models, 1):
            name = model_config["name"]
            weights = model_config["weights"].resolve()

            print(f"\n  [{i}/{len(available_models)}] {name}")

            metrics = val(weights, data_yaml, classes, detect_dir, name)
            eval_results[name] = metrics
            print(f"    P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                  f"F1={metrics['f1']:.3f}, mAP50={metrics['map50']:.3f}")

        all_results[eval_type] = eval_results

        # Bar plot
        save_bar_plot(eval_results, eval_type, base_dir)

    # Scaling curve
    scales = [m['scale'] for m in available_models]
    save_scaling_curve(all_results, scales, base_dir)

    # JSON
    with (base_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Summary
    print(f"\n{'=' * 80}")
    print("Results Summary")
    print(f"{'=' * 80}")
    for eval_type, eval_results in all_results.items():
        print(f"\n[{eval_type.upper()}]")
        print(f"  {'Model':<10} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>8} {'mAP50-95':>10}")
        print(f"  {'-' * 54}")
        for name, m in eval_results.items():
            print(f"  {name:<10} {m['precision']:>8.3f} {m['recall']:>8.3f} "
                  f"{m['f1']:>8.3f} {m['map50']:>8.3f} {m['map50_95']:>10.3f}")

    print(f"\nAll results saved to: {base_dir}")


if __name__ == '__main__':
    main()
