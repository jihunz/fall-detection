"""
Indoor Only 3K Fine-tuning + 기존 OOD 실험 결과와 합산 시각화
- Phase 1: Indoor Only 3K 훈련 (ood_mixed_ft.py 동일 하이퍼파라미터)
- Phase 2: OOD test셋 평가 (fall class only)
- Phase 3: 기존 summary.json 로드 + Indoor Only 결과 합산 → bar plot + booktabs table

Usage:
  python ood_indoor_only_ft.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ============================================================================
# MPS fallback patch (MacOS)
# ============================================================================
from ultralytics.utils.tal import TaskAlignedAssigner
import torch


def patch_task_aligned_assigner_for_mps():
    if not torch.backends.mps.is_available():
        return

    original_forward = TaskAlignedAssigner._forward

    def _forward_mps_safe(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        if pd_scores.device.type == "mps":
            tensors = [t.cpu() if torch.is_tensor(t) else t for t in
                       (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            outputs = original_forward(self, *tensors)
            return tuple(out.to("mps") if torch.is_tensor(out) else out for out in outputs)
        return original_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    TaskAlignedAssigner._forward = _forward_mps_safe


patch_task_aligned_assigner_for_mps()


# ============================================================================
# 설정
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / 'result'
YAMLS_DIR = BASE_DIR / 'yamls'
METRICS_DIR = BASE_DIR.parent / 'metrics'
DEFAULT_WEIGHTS = Path('/Users/jihunjang/workspace/ust/fall-detection/src/models/nano/yolo12n.pt')

OOD_TEST_IMAGES = Path('/Users/jihunjang/Downloads/ust/outdoor/outdoor-refined/test/images')

# 기존 OOD 실험 결과
PREV_SUMMARY = METRICS_DIR / 'ood_mixed_eval_260402_082630' / 'summary.json'

# Indoor Only 실험
EXP_NAME = 'indoor_only_3k'
EXP_YAML = YAMLS_DIR / 'data_indoor_only_3k.yaml'

# 전체 실험 display 이름 (순서: Indoor Only → 0% → 50% → 100%)
ALL_DISPLAY = {
    'indoor_only_3k':     'Indoor Only',
    'ood_hard_0pct_3k':   'OOD 0% Hard',
    'ood_hard_50pct_3k':  'OOD 50% Hard',
    'ood_hard_100pct_3k': 'OOD 100% Hard',
}

COMMON_ARGS = dict(
    imgsz=640,
    epochs=100,
    batch=32,
    freeze=10,
    lr0=1e-3,
    lrf=1.0,
    device='mps',
    workers=12,
    cache='ram',
    single_cls=False,
    save=True,
    save_period=1,
    verbose=True,
)

EVAL_CONF = 0.4
EVAL_IOU = 0.6
EVAL_IMGSZ = 640
EVAL_DEVICE = "mps"


# ============================================================================
# Phase 1: Fine-tuning
# ============================================================================
def get_last_epoch(name: str) -> int:
    results_csv = RESULT_DIR / name / 'results.csv'
    if not results_csv.exists():
        return -1
    lines = results_csv.read_text().strip().split('\n')
    if not lines:
        return -1
    try:
        return int(lines[-1].split(',')[0].strip())
    except (ValueError, IndexError):
        return -1


def is_completed(name: str) -> bool:
    best_pt = RESULT_DIR / name / 'weights' / 'best.pt'
    if not best_pt.exists():
        return False
    last_epoch = get_last_epoch(name)
    target_epoch = COMMON_ARGS['epochs'] - 1
    if last_epoch < target_epoch:
        print(f'  [INFO] {name}: best.pt exists but only reached epoch {last_epoch}/{target_epoch}')
        return False
    return True


def can_resume(name: str) -> bool:
    last_pt = RESULT_DIR / name / 'weights' / 'last.pt'
    return last_pt.exists()


def delete_labels_cache():
    datasets_dir = Path('/Users/jihunjang/Downloads/ust/hard-negative-exp')
    if datasets_dir.exists():
        for cache in datasets_dir.rglob('*.cache'):
            try:
                cache.unlink()
                print(f'  [Cache] Deleted: {cache}')
            except OSError:
                pass


def cleanup_failed_run(name: str):
    import shutil
    for p in RESULT_DIR.glob(f'{name}[0-9]*'):
        if p.is_dir():
            shutil.rmtree(p)
            print(f'  [Cleanup] Removed duplicate dir: {p}')

    run_dir = RESULT_DIR / name
    if not run_dir.exists():
        return

    best_pt = run_dir / 'weights' / 'best.pt'
    last_epoch = get_last_epoch(name)

    if not best_pt.exists() and last_epoch < 5:
        print(f'  [Cleanup] Removing failed run artifacts for {name} (last_epoch={last_epoch})')
        for item in run_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            print(f'  [Cleanup] Deleted: {item}')


def train(name: str, yaml_path: Path) -> Path:
    print(f"\n{'=' * 80}")
    print(f"Phase 1: Training — {name}")
    print(f"{'=' * 80}")

    best_pt = RESULT_DIR / name / 'weights' / 'best.pt'

    delete_labels_cache()

    if is_completed(name):
        print(f'  [SKIP] {name} — already completed')
        return best_pt

    cleanup_failed_run(name)

    if can_resume(name):
        print(f'  [RESUME] {name} — resuming from last.pt')
        last_pt = RESULT_DIR / name / 'weights' / 'last.pt'
        model = YOLO(str(last_pt))
        model.train(resume=True, device='mps')
        return best_pt

    if not DEFAULT_WEIGHTS.exists():
        raise FileNotFoundError(f'Weights not found: {DEFAULT_WEIGHTS}')

    model = YOLO(str(DEFAULT_WEIGHTS))
    model.train(
        data=str(yaml_path),
        project=str(RESULT_DIR),
        name=name,
        exist_ok=True,
        **COMMON_ARGS,
    )
    return best_pt


# ============================================================================
# Phase 2: 평가
# ============================================================================
def evaluate(weights: Path, name: str, output_dir: Path) -> Dict[str, float]:
    print(f"\n{'=' * 80}")
    print(f"Phase 2: Evaluation — {name}")
    print(f"{'=' * 80}")

    model = YOLO(str(weights))
    eval_yaml = YAMLS_DIR / 'data_outdoor_eval.yaml'

    val_dir = str(output_dir / name)
    results = model.val(
        data=str(eval_yaml),
        classes=[1],
        conf=EVAL_CONF,
        iou=EVAL_IOU,
        imgsz=EVAL_IMGSZ,
        device=EVAL_DEVICE,
        half=False,
        save=False,
        plots=True,
        cache=False,
        project=val_dir,
        name='val',
        exist_ok=True,
    )
    summary = results.results_dict or {}

    precision = float(summary.get("metrics/precision(B)", 0.0))
    recall = float(summary.get("metrics/recall(B)", 0.0))
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom else 0.0

    metrics = {"precision": precision, "recall": recall, "f1": f1}
    print(f"  P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    predict_dir = str(output_dir / name / 'predictions')
    model.predict(
        source=str(OOD_TEST_IMAGES),
        conf=EVAL_CONF,
        iou=EVAL_IOU,
        imgsz=EVAL_IMGSZ,
        device=EVAL_DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=predict_dir,
        name='images',
        exist_ok=True,
    )
    print(f"  Predictions saved to: {predict_dir}")

    return metrics


# ============================================================================
# Phase 3: 시각화 — 4개 모델 비교 (Indoor Only + OOD 0/50/100%)
# ============================================================================
def create_booktabs_table(all_metrics: Dict[str, Dict], save_path: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    })

    # 순서 보장
    order = [k for k in ALL_DISPLAY if k in all_metrics]
    display_names = [ALL_DISPLAY[k] for k in order]
    columns = ["precision", "recall", "f1"]
    col_display = ["Precision", "Recall", "F1"]

    best = {}
    for m in columns:
        vals = {k: all_metrics[k][m] for k in order}
        best[m] = max(vals, key=vals.get)

    col_width = 1.5
    row_height = 0.5
    model_col_width = 3.2
    table_width = model_col_width + col_width * len(columns)
    table_height = row_height * (len(order) + 1)

    fig_width = table_width + 1.0
    fig_height = table_height + 1.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, table_width)
    ax.set_ylim(0, table_height)
    ax.axis("off")

    header_y_top = table_height
    header_y_bot = table_height - row_height

    def _col_x(j):
        return 0 if j == 0 else model_col_width + (j - 1) * col_width

    thick = dict(color="black", linewidth=1.5, clip_on=False)
    thin = dict(color="black", linewidth=0.7, clip_on=False)
    ax.plot([0, table_width], [header_y_top, header_y_top], **thick)
    ax.plot([0, table_width], [header_y_bot, header_y_bot], **thin)
    ax.plot([0, table_width], [0, 0], **thick)

    header_labels = ["Training Data"] + col_display
    for j, label in enumerate(header_labels):
        x = _col_x(j)
        w = model_col_width if j == 0 else col_width
        cx = x + w / 2
        cy = (header_y_top + header_y_bot) / 2
        ax.text(cx, cy, label, ha="center", va="center", fontsize=11, fontweight="bold")

    for i, (key, display) in enumerate(zip(order, display_names)):
        yt = header_y_bot - i * row_height
        yb = yt - row_height
        cy = (yt + yb) / 2

        ax.text(_col_x(0) + 0.15, cy, display, ha="left", va="center", fontsize=10.5)

        for j, m in enumerate(columns):
            val = all_metrics[key][m]
            x = _col_x(j + 1)
            w = col_width
            cx = x + w / 2
            text = f"{val * 100:.2f}"
            is_best = (best[m] == key)
            ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if is_best else "normal")

    caption = (
        r"$\bf{Table.}$  OOD fall detection with varying hard negative ratios "
        f"(test={195} images, train=3K). "
        f"conf={EVAL_CONF}, NMS IoU={EVAL_IOU}. "
        r"$\bf{Bold}$ = best."
    )
    ax.text(table_width / 2, -0.45, caption,
            ha="center", va="top", fontsize=9.5, style="italic")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"  [Table] Saved: {save_path}")


def save_bar_plot(all_metrics: Dict[str, Dict], save_path: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    })

    order = [k for k in ALL_DISPLAY if k in all_metrics]
    display_names = [ALL_DISPLAY[k] for k in order]
    metrics = ["precision", "recall", "f1"]
    values = np.array([[all_metrics[k][m] for k in order] for m in metrics])

    x = np.arange(len(metrics))
    n_models = len(order)
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = ['#D4D4D4', '#A8D8A8', '#7CAEE8', '#E8927C']
    for idx, (key, display, color) in enumerate(zip(order, display_names, colors)):
        offset = (idx - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values[:, idx], width, label=display, color=color,
                      edgecolor='white', linewidth=0.8)
        for bar, metric_val in zip(bars, values[:, idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{metric_val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight='bold',
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.set_ylim(0, min(1.0, float(values.max()) + 0.12))
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("Hard Negative Ratio Ablation — OOD Fall Detection",
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=9.5, loc='lower right')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [Bar Plot] Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # Phase 1: Indoor Only 훈련
    best_pt = train(EXP_NAME, EXP_YAML)

    # Phase 2: 평가
    eval_dir = METRICS_DIR / f"ood_mixed_eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not best_pt.exists():
        print(f"ERROR: {EXP_NAME} weights not found: {best_pt}")
        return

    indoor_metrics = evaluate(best_pt, EXP_NAME, eval_dir)

    # Phase 3: 기존 결과 로드 + 합산
    prev_metrics = {}
    if PREV_SUMMARY.exists():
        with PREV_SUMMARY.open() as f:
            prev_metrics = json.load(f)
        print(f"\n  [Load] Previous results from: {PREV_SUMMARY}")
    else:
        print(f"\n  [WARN] Previous summary not found: {PREV_SUMMARY}")

    # 전체 결과 합산
    all_metrics = {EXP_NAME: indoor_metrics}
    all_metrics.update(prev_metrics)

    # JSON 저장
    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=float)

    # 시각화
    print(f"\n{'=' * 80}")
    print("Phase 3: Generating plots (4 models)...")
    print(f"{'=' * 80}")

    create_booktabs_table(all_metrics, eval_dir / "metrics_table.png")
    save_bar_plot(all_metrics, eval_dir / "metrics_bar.png")

    # Summary
    print(f"\n{'=' * 80}")
    print("Results Summary: Indoor Only + Hard Negative Ratio Ablation")
    print(f"{'=' * 80}")
    print(f"  {'Method':<28} {'P':>8} {'R':>8} {'F1':>8}")
    print(f"  {'-' * 52}")
    for key in ALL_DISPLAY:
        if key in all_metrics:
            m = all_metrics[key]
            display = ALL_DISPLAY[key]
            print(f"  {display:<28} {m['precision']:>8.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}")

    print(f"\nAll results saved to: {eval_dir}")


if __name__ == '__main__':
    main()
