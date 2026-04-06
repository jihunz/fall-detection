"""
YOLO Baseline 평가 스크립트 (GOPR / Indoor-Refined 전환 가능)
- Person, Fall 각각 평가
- Fall 평가 시 use_fall0class_labels() 적용 (class 0↔1 스왑)
- 학술 논문 스타일 표(PNG) + Bar Plot + JSON 저장

Usage:
  python yolo_id.py                  # DATASET 변수에 따라 실행
  python yolo_id.py --dataset indoor # CLI로 데이터셋 지정
  python yolo_id.py --dataset gopr
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# ============================================================================
# 데이터셋 선택
# ============================================================================
DATASET = 'indoor'  # 'gopr' 또는 'indoor'

MODELS_DIR = Path('/Users/jihunjang/workspace/ust/fall-detection/src/models')
V1_YAMLS_DIR = Path('/src/train-v1/yamls')
V2_YAMLS_DIR = Path('/src/train-v2/yamls')
OUTPUT_ROOT = Path('/Users/jihunjang/workspace/ust/fall-detection/src/metrics')

DATASET_CONFIG = {
    'gopr': {
        'labels_dir': Path('/Users/jihunjang/Downloads/ust/dataset/train/gopr/labels/test'),
        'data_yaml': V1_YAMLS_DIR / 'data_gopr_fall_val.yaml',
        'output_prefix': 'yolo-vfp',
        'eval_name': 'VFP290K',
    },
    'indoor': {
        'labels_dir': Path('/Users/jihunjang/Downloads/indoor-refined/val/labels'),
        'data_yaml': V2_YAMLS_DIR / 'data_indoor_eval.yaml',
        'output_prefix': 'yolo_id',
        'eval_name': 'Indoor-Refined',
    },
}

# 모델 정의
SCALES = {
    'nano': {
        'models': [
            {'name': 'YOLOv8n', 'weights': MODELS_DIR / 'nano/yolov8n.pt'},
            {'name': 'YOLO11n', 'weights': MODELS_DIR / 'nano/yolo11n.pt'},
            {'name': 'YOLO12n', 'weights': MODELS_DIR / 'nano/yolo12n.pt'},
            {'name': 'YOLO26n', 'weights': MODELS_DIR / 'nano/yolo26n.pt'},
        ]
    },
}

# 평가 설정
NUM_RUNS = 1
DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = 'mps'


# ============================================================================
# fall0class 라벨 스왑
# ============================================================================
@contextmanager
def use_fall0class_labels(labels_dir: Path):
    """
    라벨에서 class 0↔1 스왑하여 fall을 class 0으로 변환.
    COCO pretrained 모델이 fall(원래 class 1)을 person(class 0)으로 검출하는지 측정.
    """
    test_dir = labels_dir
    backup_dir = test_dir.parent / f"{test_dir.name}_backup"
    fall0_dir = test_dir.parent / f"{test_dir.name}-fall0class"
    cache_file = test_dir.parent / f"{test_dir.name}.cache"

    # fall0class 라벨이 없으면 생성
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
        print("  [Cache] Deleting cache")
        cache_file.unlink()

    print(f"  [Label Swap] {test_dir.name} -> {backup_dir.name}")
    test_dir.rename(backup_dir)
    print(f"  [Label Swap] {fall0_dir.name} -> {test_dir.name}")
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


# ============================================================================
# 평가
# ============================================================================
def val(weights: Path, data_yaml: Path, classes: List[int],
        save_dir: Path = None, model_name: str = None) -> Dict[str, float]:
    model = YOLO(str(weights))

    is_yolo26 = '26' in str(weights.name)
    conf = 0.001 if is_yolo26 else DEFAULT_CONF

    results = model.val(
        data=str(data_yaml),
        classes=classes,
        conf=conf,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        half=False,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(save_dir) if save_dir else None,
        name=model_name if model_name else 'val',
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


def run_evaluation(scale: str, eval_type: str, cfg: dict) -> Dict[str, Dict]:
    scale_config = SCALES[scale]
    data_yaml = cfg['data_yaml'].resolve()
    labels_dir = cfg['labels_dir']

    is_fall = (eval_type == 'fall')
    classes = [0] if is_fall else [0]  # fall0class 시 class=0으로 평가

    print(f"\n{'=' * 80}")
    print(f"Evaluating {scale.upper()} models for {eval_type.upper()} ({cfg['eval_name']})")
    if is_fall:
        print(f"Using fall0class labels (class=0)")
    print(f"{'=' * 80}")

    output_prefix = cfg['output_prefix']
    save_dir = OUTPUT_ROOT / output_prefix / scale / eval_type / 'predictions'

    results = {}
    for model_config in scale_config['models']:
        model_name = model_config['name']
        weights = model_config['weights'].resolve()

        print(f"\n[{model_name}] Starting {NUM_RUNS} runs...")

        run_results = {}
        for run_idx in range(1, NUM_RUNS + 1):
            print(f"  Run {run_idx}/{NUM_RUNS}...", end=" ")

            if is_fall:
                with use_fall0class_labels(labels_dir):
                    metrics = val(weights, data_yaml, classes, save_dir=save_dir, model_name=model_name)
            else:
                metrics = val(weights, data_yaml, classes, save_dir=save_dir, model_name=model_name)

            run_results[str(run_idx)] = metrics
            print(f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

        metrics_keys = ['precision', 'recall', 'f1', 'map50', 'map50_95']
        mean = {k: np.mean([run_results[str(i)][k] for i in range(1, NUM_RUNS + 1)]) for k in metrics_keys}
        std = {k: np.std([run_results[str(i)][k] for i in range(1, NUM_RUNS + 1)]) for k in metrics_keys}

        run_results['mean'] = mean
        run_results['std'] = std
        results[model_name] = run_results

        print(f"  → Mean: P={mean['precision']:.3f}, R={mean['recall']:.3f}, F1={mean['f1']:.3f}")

    return results


# ============================================================================
# 시각화
# ============================================================================
def save_results(results: Dict, scale: str, eval_type: str, cfg: dict):
    output_prefix = cfg['output_prefix']
    output_dir = OUTPUT_ROOT / output_prefix / scale / eval_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    raw_path = output_dir / 'raw_results.json'
    with raw_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=float)

    mean_data = {name: data['mean'] for name, data in results.items()}
    mean_path = output_dir / 'metrics_mean.json'
    with mean_path.open('w', encoding='utf-8') as f:
        json.dump(mean_data, f, indent=2, default=float)

    # Booktabs-style table
    create_booktabs_table(mean_data, output_dir, scale, eval_type, cfg)


def create_booktabs_table(mean_data: Dict, output_dir: Path, scale: str, eval_type: str, cfg: dict):
    """
    학술 논문 booktabs 스타일 표 (metrics_mean.png).
    세리프 폰트, 3-line rule, 최고값 볼드 처리.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    })

    models = list(mean_data.keys())
    metrics = ["precision", "recall", "f1", "map50", "map50_95"]
    metric_display = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "map50": r"$AP_{50}$",
        "map50_95": r"$AP_{50:95}$",
    }

    # 메트릭별 최고값 모델
    best = {}
    for m in metrics:
        vals = {model: mean_data[model][m] for model in models}
        best[m] = max(vals, key=vals.get)

    # 레이아웃 계산
    col_width = 1.3
    row_height = 0.45
    model_col_width = 1.6
    table_width = model_col_width + col_width * len(metrics)
    table_height = row_height * (len(models) + 1)

    fig_width = table_width + 1.0
    fig_height = table_height + 1.6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, table_width)
    ax.set_ylim(0, table_height)
    ax.axis("off")

    header_y_top = table_height
    header_y_bot = table_height - row_height

    def _row_y(i):
        return header_y_bot - i * row_height

    def _col_x(j):
        return 0 if j == 0 else model_col_width + (j - 1) * col_width

    # 수평선 (booktabs)
    thick = dict(color="black", linewidth=1.5, clip_on=False)
    thin = dict(color="black", linewidth=0.7, clip_on=False)
    ax.plot([0, table_width], [header_y_top, header_y_top], **thick)
    ax.plot([0, table_width], [header_y_bot, header_y_bot], **thin)
    ax.plot([0, table_width], [0, 0], **thick)

    # 헤더
    header_labels = ["Model"] + [metric_display[m] for m in metrics]
    for j, label in enumerate(header_labels):
        x = _col_x(j)
        w = model_col_width if j == 0 else col_width
        cx = x + w / 2
        cy = (header_y_top + header_y_bot) / 2
        ax.text(cx, cy, label, ha="center", va="center", fontsize=11, fontweight="bold")

    # 데이터 행
    for i, model in enumerate(models):
        yt = _row_y(i)
        yb = yt - row_height
        cy = (yt + yb) / 2

        ax.text(_col_x(0) + 0.15, cy, model, ha="left", va="center", fontsize=10.5)

        for j, m in enumerate(metrics):
            val = mean_data[model][m]
            x = _col_x(j + 1)
            w = col_width
            cx = x + w / 2
            text = f"{val * 100:.2f}"
            is_best = (best[m] == model)
            ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if is_best else "normal")

    # 캡션
    eval_label = eval_type.capitalize()
    caption = (
        rf"$\bf{{Table.}}$  Results on {cfg['eval_name']} ({eval_label}). "
        "Metrics are reported as percentages. "
        r"$\bf{Bold}$ indicates the best result per metric."
    )
    ax.text(table_width / 2, -0.45, caption,
            ha="center", va="top", fontsize=9.5, style="italic")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
    out_path = output_dir / 'metrics_mean.png'
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"  [Table] Saved: {out_path}")


def create_academic_table(all_results: Dict[str, Dict[str, Dict]], cfg: dict, save_path: Path):
    """
    학술 논문 booktabs 스타일 통합 표 PNG 생성.
    Fall / Person 섹션을 하나의 표에 midrule로 구분.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    })

    metrics = ["precision", "recall", "f1", "map50", "map50_95"]
    metric_display = ["Precision", "Recall", "F1", r"$AP_{50}$", r"$AP_{50:95}$"]

    # 섹션별 데이터 수집
    sections = []
    for eval_type, section_label in [('fall', 'Fall Detection'), ('person', 'Person Detection')]:
        if eval_type not in all_results:
            continue
        rows = []
        for scale_name in SCALES:
            if scale_name not in all_results[eval_type]:
                continue
            scale_results = all_results[eval_type][scale_name]
            for model_config in SCALES[scale_name]['models']:
                model_name = model_config['name']
                if model_name not in scale_results:
                    continue
                mean = scale_results[model_name]['mean']
                rows.append((model_name, {m: mean[m] for m in metrics}))
        if rows:
            sections.append((section_label, rows))

    if not sections:
        return

    # 메트릭별 섹션별 최고값
    section_bests = []
    for _, rows in sections:
        bests = {}
        for m in metrics:
            vals = {name: d[m] for name, d in rows}
            bests[m] = max(vals, key=vals.get) if vals else None
        section_bests.append(bests)

    # 레이아웃 계산
    col_width = 1.3
    row_height = 0.45
    model_col_width = 1.6
    section_label_col = 1.8
    table_width = section_label_col + model_col_width + col_width * len(metrics)

    total_data_rows = sum(len(rows) for _, rows in sections)
    midrule_gaps = (len(sections) - 1) * 0.15
    table_height = row_height * (total_data_rows + 1) + midrule_gaps

    fig_width = table_width + 1.0
    fig_height = table_height + 1.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, table_width)
    ax.set_ylim(0, table_height)
    ax.axis("off")

    def _col_x(j):
        """j=0: section label, j=1: model, j=2..n+1: metrics"""
        if j == 0:
            return 0
        elif j == 1:
            return section_label_col
        else:
            return section_label_col + model_col_width + (j - 2) * col_width

    thick = dict(color="black", linewidth=1.5, clip_on=False)
    thin = dict(color="black", linewidth=0.7, clip_on=False)

    header_y_top = table_height
    header_y_bot = table_height - row_height

    # 상단 굵은 선
    ax.plot([0, table_width], [header_y_top, header_y_top], **thick)
    # 헤더 아래 얇은 선
    ax.plot([0, table_width], [header_y_bot, header_y_bot], **thin)

    # 헤더
    header_labels = ["", "Model"] + metric_display
    col_widths = [section_label_col, model_col_width] + [col_width] * len(metrics)
    for j, (label, w) in enumerate(zip(header_labels, col_widths)):
        cx = _col_x(j) + w / 2
        cy = (header_y_top + header_y_bot) / 2
        ax.text(cx, cy, label, ha="center", va="center", fontsize=11, fontweight="bold")

    # 데이터 행
    y_cursor = header_y_bot
    for sec_idx, ((section_label, rows), bests) in enumerate(zip(sections, section_bests)):
        if sec_idx > 0:
            # midrule 사이 간격
            ax.plot([0, table_width], [y_cursor, y_cursor], **thin)

        for row_idx, (model_name, vals) in enumerate(rows):
            yt = y_cursor - row_idx * row_height
            yb = yt - row_height
            cy = (yt + yb) / 2

            # 섹션 라벨 (첫 행에만, 세로 중앙)
            if row_idx == 0:
                section_center_y = y_cursor - (len(rows) * row_height) / 2
                ax.text(_col_x(0) + section_label_col / 2, section_center_y,
                        section_label, ha="center", va="center",
                        fontsize=10.5, style="italic")

            # 모델명
            ax.text(_col_x(1) + 0.15, cy, model_name,
                    ha="left", va="center", fontsize=10.5)

            # 메트릭 값
            for j, m in enumerate(metrics):
                val = vals[m]
                cx = _col_x(j + 2) + col_width / 2
                text = f"{val * 100:.2f}"
                is_best = (bests[m] == model_name)
                ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5,
                        fontweight="bold" if is_best else "normal")

        y_cursor -= len(rows) * row_height + (0.15 if sec_idx < len(sections) - 1 else 0)

    # 하단 굵은 선
    ax.plot([0, table_width], [0, 0], **thick)

    # 캡션
    caption = (
        rf"$\bf{{Table.}}$  Baseline results on {cfg['eval_name']} dataset. "
        "Metrics are reported as percentages. "
        r"$\bf{Bold}$ indicates the best result per metric within each section."
    )
    ax.text(table_width / 2, -0.45, caption,
            ha="center", va="top", fontsize=9.5, style="italic")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"  [Academic Table] Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='YOLO Baseline Evaluation')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['gopr', 'indoor'],
                        help='데이터셋 선택 (기본: 스크립트 상단 DATASET 변수)')
    args = parser.parse_args()

    dataset = args.dataset or DATASET
    cfg = DATASET_CONFIG[dataset]

    print(f"{'=' * 80}")
    print(f"YOLO Baseline Evaluation — {cfg['eval_name']}")
    print(f"Dataset: {dataset}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    all_results = {}

    for eval_type in ['person', 'fall']:
        all_results[eval_type] = {}
        for scale in ['nano']:
            results = run_evaluation(scale, eval_type, cfg)
            save_results(results, scale, eval_type, cfg)
            all_results[eval_type][scale] = results

    # 학술 논문 스타일 통합 표 생성
    table_path = OUTPUT_ROOT / cfg['output_prefix'] / 'benchmark_academic.png'
    create_academic_table(all_results, cfg, table_path)

    # Summary
    print(f"\n{'=' * 80}")
    print("Results Summary")
    print(f"{'=' * 80}")
    for eval_type in ['fall', 'person']:
        print(f"\n[{eval_type.upper()}]")
        for scale, scale_results in all_results.get(eval_type, {}).items():
            for model_name, data in scale_results.items():
                m = data['mean']
                print(f"  {model_name:<12} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    print(f"\nAll results saved to: {OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
