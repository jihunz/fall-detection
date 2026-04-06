"""
OOD Hard Negative 수집 스크립트
- outdoor-refined train셋에서 fall(class 1) 미검출(FN) 이미지 수집
- 인스턴스 단위 IoU 매칭 → GT fall bbox 중 1개라도 미매칭 시 해당 이미지를 hard negative로 판정
- 모델별 hard/ 폴더에 이미지 + 라벨 복사
- Booktabs 스타일 결과 표 + Bar plot + Scaling curve + JSON 저장

Usage:
  python ood_fp_fn.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# ============================================================================
# 설정
# ============================================================================
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

EVAL_IMAGES_DIR = Path('/Users/jihunjang/Downloads/ust/outdoor/outdoor-refined/train/images')
EVAL_LABELS_DIR = Path('/Users/jihunjang/Downloads/ust/outdoor/outdoor-refined/train/labels')
OUTPUT_ROOT = Path('/Users/jihunjang/workspace/ust/fall-detection/src/metrics')

TARGET_CLASS = 1        # fall
MATCH_IOU_THRESH = 0.5  # IoU threshold for GT-pred matching
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "mps"


# ============================================================================
# IoU 계산 및 매칭
# ============================================================================
def xywh_to_xyxy(box: List[float], img_w: int = 1, img_h: int = 1) -> List[float]:
    """YOLO normalized xywh → pixel xyxy"""
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """xyxy 형식 두 박스의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def load_gt_boxes(label_path: Path, target_class: int = 1) -> List[List[float]]:
    """GT 라벨에서 target_class bbox만 로드 (normalized xywh)"""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        cls_id = int(parts[0])
        if cls_id == target_class:
            boxes.append([float(x) for x in parts[1:5]])
    return boxes


def match_predictions(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_thresh: float = 0.5,
) -> Tuple[int, int, int]:
    """
    GT와 예측 bbox를 IoU greedy matching.
    Returns: (tp, fp, fn)
    """
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    # normalized xywh → 1000x1000 pixel xyxy (비율 유지용)
    gt_xyxy = [xywh_to_xyxy(b, 1000, 1000) for b in gt_boxes]
    pred_xyxy = [xywh_to_xyxy(b, 1000, 1000) for b in pred_boxes]

    # IoU 행렬
    iou_matrix = np.zeros((len(gt_xyxy), len(pred_xyxy)))
    for i, gb in enumerate(gt_xyxy):
        for j, pb in enumerate(pred_xyxy):
            iou_matrix[i, j] = compute_iou(gb, pb)

    # Greedy matching (IoU 내림차순)
    gt_matched = set()
    pred_matched = set()

    while True:
        max_iou = iou_matrix.max()
        if max_iou < iou_thresh:
            break
        gi, pi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        gt_matched.add(gi)
        pred_matched.add(pi)
        iou_matrix[gi, :] = 0
        iou_matrix[:, pi] = 0

    tp = len(gt_matched)
    fp = len(pred_boxes) - len(pred_matched)
    fn = len(gt_boxes) - len(gt_matched)

    return tp, fp, fn


# ============================================================================
# 모델별 Hard Negative 수집
# ============================================================================
def evaluate_model(
    weights: Path,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    model_name: str,
) -> Dict:
    """
    단일 모델 평가: predict → GT 매칭 → FN 이미지(hard negative) 수집
    """
    model = YOLO(str(weights))

    # predict
    results = model.predict(
        source=str(images_dir),
        classes=[TARGET_CLASS],
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        save=False,
        verbose=False,
    )

    hard_dir = output_dir / model_name / 'hard'
    hard_dir.mkdir(parents=True, exist_ok=True)

    total_tp, total_fp, total_fn = 0, 0, 0
    hard_images = []
    total_gt_fall = 0

    for result in results:
        img_path = Path(result.path)
        img_name = img_path.stem
        label_path = labels_dir / f"{img_name}.txt"

        # GT fall boxes
        gt_boxes = load_gt_boxes(label_path, TARGET_CLASS)
        total_gt_fall += len(gt_boxes)

        # GT에 fall이 없는 이미지는 FN 판정 대상이 아님 → skip
        if not gt_boxes:
            continue

        # Predicted fall boxes (normalized xywh)
        pred_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if cls_id == TARGET_CLASS:
                    xywhn = box.xywhn[0].tolist()
                    pred_boxes.append(xywhn)

        tp, fp, fn = match_predictions(gt_boxes, pred_boxes, MATCH_IOU_THRESH)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # FN: GT fall 중 매칭 안 된 것이 1개라도 있으면 → hard negative
        if fn > 0:
            hard_images.append(img_path.name)
            shutil.copy2(img_path, hard_dir / img_path.name)
            # 라벨도 함께 복사
            if label_path.exists():
                shutil.copy2(label_path, hard_dir / label_path.name)

    # 메트릭 계산
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_images": len(results),
        "total_gt_fall_images": total_gt_fall,
        "tp": total_tp,
        "fn": total_fn,
        "hard_images": len(hard_images),
        "hard_ratio": len(hard_images) / len(results) if len(results) > 0 else 0.0,
        "recall": recall,
        "miss_rate": 1.0 - recall,
    }


# ============================================================================
# 시각화 — Booktabs 학술 논문 스타일
# ============================================================================
def create_booktabs_table(all_metrics: Dict[str, Dict], save_path: Path):
    """
    Booktabs 스타일 결과 표.
    세리프 폰트, 3-line rule, 최고값 볼드 처리.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    })

    models = list(all_metrics.keys())

    # 컬럼 정의: (key, display_name, format, higher_is_better)
    col_defs = [
        ("recall",      "Recall",       "pct",   True),
        ("miss_rate",   "Miss Rate",    "pct",   False),
        ("hard_images", "Hard Neg.",     "int",   False),
        ("hard_ratio",  "Hard Ratio",   "pct",   False),
        ("tp",          "TP",           "int",   True),
        ("fn",          "FN",           "int",   False),
    ]

    col_keys = [c[0] for c in col_defs]
    col_display = [c[1] for c in col_defs]

    # 메트릭별 최고값
    best = {}
    for key, _, _, higher in col_defs:
        vals = {model: all_metrics[model][key] for model in models}
        best[key] = max(vals, key=vals.get) if higher else min(vals, key=vals.get)

    # 레이아웃 계산
    col_width = 1.3
    row_height = 0.45
    model_col_width = 1.8
    table_width = model_col_width + col_width * len(col_defs)
    table_height = row_height * (len(models) + 1)

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

    # 수평선 (booktabs)
    thick = dict(color="black", linewidth=1.5, clip_on=False)
    thin = dict(color="black", linewidth=0.7, clip_on=False)
    ax.plot([0, table_width], [header_y_top, header_y_top], **thick)
    ax.plot([0, table_width], [header_y_bot, header_y_bot], **thin)
    ax.plot([0, table_width], [0, 0], **thick)

    # 헤더
    header_labels = ["Model"] + col_display
    for j, label in enumerate(header_labels):
        x = _col_x(j)
        w = model_col_width if j == 0 else col_width
        cx = x + w / 2
        cy = (header_y_top + header_y_bot) / 2
        ax.text(cx, cy, label, ha="center", va="center", fontsize=11, fontweight="bold")

    # 데이터 행
    for i, model in enumerate(models):
        yt = header_y_bot - i * row_height
        yb = yt - row_height
        cy = (yt + yb) / 2

        ax.text(_col_x(0) + 0.15, cy, model, ha="left", va="center", fontsize=10.5)

        for j, (key, _, fmt, _) in enumerate(col_defs):
            val = all_metrics[model][key]
            x = _col_x(j + 1)
            w = col_width
            cx = x + w / 2
            if fmt == "pct":
                text = f"{val * 100:.1f}"
            else:
                text = f"{val}"
            is_best = (best[key] == model)
            ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if is_best else "normal")

    # 캡션
    total_img = all_metrics[models[0]]["total_images"]
    caption = (
        r"$\bf{Table.}$  Hard negative collection on OOD train set "
        f"(Outdoor-Refined, {total_img} images). "
        f"conf={DEFAULT_CONF}, IoU≥{MATCH_IOU_THRESH}. "
        r"$\bf{Bold}$ = best per column."
    )
    ax.text(table_width / 2, -0.45, caption,
            ha="center", va="top", fontsize=9.5, style="italic")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"  [Table] Saved: {save_path}")


def save_hard_neg_bar(all_metrics: Dict[str, Dict], save_path: Path):
    """Hard negative 이미지 수 모델별 비교 bar chart"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    })

    models = list(all_metrics.keys())
    hard_counts = [all_metrics[m]["hard_images"] for m in models]
    total_images = all_metrics[models[0]]["total_images"]

    x = np.arange(len(models))
    width = 0.5
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 5))

    bars = ax.bar(x, hard_counts, width, color='#7CAEE8', edgecolor='#4A7FB5', linewidth=0.8)

    for bar, count in zip(bars, hard_counts):
        h = bar.get_height()
        ratio = count / total_images * 100
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f'{count}\n({ratio:.1f}%)', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.axhline(y=total_images, color='#E8927C', linestyle='--', linewidth=1.2,
               label=f'Total images ({total_images})')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Hard Negative Images", fontsize=12, fontweight='bold')
    ax.set_title("Hard Negative Samples per Model (FN on OOD Train)",
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [Hard Neg Bar] Saved: {save_path}")


def save_scaling_curve(all_metrics: Dict[str, Dict], scales: List[int], save_path: Path):
    """Scaling curve: Recall / Miss Rate vs training data size (log scale)"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    })

    models = list(all_metrics.keys())
    x = scales[:len(models)]

    recall_vals = [all_metrics[m]['recall'] for m in models]
    miss_vals = [all_metrics[m]['miss_rate'] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, recall_vals, 'o-', color='#2CA02C', linewidth=2, markersize=8, label='Recall')
    ax.plot(x, miss_vals, 's--', color='#D62728', linewidth=2, markersize=7, label='Miss Rate')

    for xi, r, m in zip(x, recall_vals, miss_vals):
        ax.annotate(f'{r:.3f}', (xi, r), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8, color='#2CA02C')
        ax.annotate(f'{m:.3f}', (xi, m), textcoords="offset points",
                    xytext=(0, -15), ha='center', fontsize=8, color='#D62728')

    ax.set_xscale('log')
    ax.set_xlabel('Indoor training images (log scale)', fontsize=11)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_title('OOD Fall Detection — Recall & Miss Rate Scaling', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [Scaling Curve] Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_dir = OUTPUT_ROOT / f"ood_hard_neg_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    # 존재하는 모델만 필터
    available_models = [m for m in MODELS if m['weights'].exists()]
    if not available_models:
        print("ERROR: No trained models found.")
        return

    print(f"{'=' * 80}")
    print(f"OOD Hard Negative Collection — Fall Detection (FN only)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Eval set: {EVAL_IMAGES_DIR} ({len(list(EVAL_IMAGES_DIR.iterdir()))} images)")
    print(f"Models: {[m['name'] for m in available_models]}")
    print(f"Settings: conf={DEFAULT_CONF}, iou={DEFAULT_IOU}, match_iou={MATCH_IOU_THRESH}")
    print(f"Output: {base_dir}")
    print(f"{'=' * 80}")

    all_metrics = {}

    for i, model_config in enumerate(available_models, 1):
        name = model_config["name"]
        weights = model_config["weights"].resolve()

        print(f"\n[{i}/{len(available_models)}] {name}")
        print(f"  Weights: {weights}")

        metrics = evaluate_model(
            weights=weights,
            images_dir=EVAL_IMAGES_DIR,
            labels_dir=EVAL_LABELS_DIR,
            output_dir=base_dir,
            model_name=name,
        )

        all_metrics[name] = metrics
        print(f"  GT fall instances: {metrics['total_gt_fall_images']}")
        print(f"  TP={metrics['tp']}, FN={metrics['fn']}")
        print(f"  Hard negative images: {metrics['hard_images']} "
              f"({metrics['hard_ratio'] * 100:.1f}%)")
        print(f"  Recall={metrics['recall']:.3f}, Miss Rate={metrics['miss_rate']:.3f}")

    # 시각화
    print(f"\n{'=' * 80}")
    print("Generating plots...")
    print(f"{'=' * 80}")

    scales = [m['scale'] for m in available_models]

    create_booktabs_table(all_metrics, base_dir / "metrics_table.png")
    save_hard_neg_bar(all_metrics, base_dir / "hard_neg_bar.png")
    save_scaling_curve(all_metrics, scales, base_dir / "scaling_curve.png")

    # JSON 저장
    with (base_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=float)

    # Summary
    print(f"\n{'=' * 80}")
    print("Results Summary")
    print(f"{'=' * 80}")
    print(f"  {'Model':<12} {'Recall':>8} {'Miss':>8} {'Hard':>8} {'Ratio':>8}")
    print(f"  {'-' * 44}")
    for name, m in all_metrics.items():
        print(f"  {name:<12} {m['recall']:>8.3f} {m['miss_rate']:>8.3f} "
              f"{m['hard_images']:>8} {m['hard_ratio'] * 100:>7.1f}%")

    print(f"\nAll results saved to: {base_dir}")


if __name__ == '__main__':
    main()
