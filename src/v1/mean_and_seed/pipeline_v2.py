'''
V2 전체 파이프라인: 데이터셋 생성 → 훈련 → 평가
- 100k_v2의 서브셋으로 10k~250 v2 데이터셋 생성
- 250 → 500 → 1k → 2k → 5k → 10k → 100k 순차 훈련
- Fall/Person 평가
'''

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
YAMLS_DIR = BASE_DIR / 'yamls'
DATA_CFG = YAMLS_DIR / 'data_megafall.yaml'
DEFAULT_WEIGHTS = Path('/src/models/yolo12n.pt')
METRICS_DIR = Path('/src/metrics')

# 100k_v2 소스
TRAIN_100K_V2 = YAMLS_DIR / 'train_ins_100k_v2.txt'
VAL_100K_V2 = YAMLS_DIR / 'val_ins_100k_v2.txt'

# 데이터셋 설정: (이름, fall_person 목표)
DATASETS = [
    ('250', 250),
    ('500', 500),
    ('1k', 1000),
    ('2k', 2000),
    ('5k', 5000),
    ('10k', 10000),
    ('100k', 100000),
]

# ============================================================
# MPS 패치 (yolo_ft_megafall.py 기반)
# ============================================================
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


# ============================================================
# Phase 1: 데이터셋 생성
# ============================================================
def count_fall_person_instances(label_path: Path) -> int:
    """라벨 파일에서 fall_person (class 1) 인스턴스 수"""
    if not label_path.exists():
        return 0
    count = 0
    for line in label_path.read_text().strip().split('\n'):
        if line and line.split()[0] == '1':
            count += 1
    return count


def create_datasets():
    """100k_v2에서 서브셋으로 10k~250 v2 데이터셋 생성"""
    print("\n" + "=" * 60)
    print("[Phase 1] 데이터셋 생성 (100k_v2의 서브셋)")
    print("=" * 60)
    
    # 100k_v2 로드
    train_100k = TRAIN_100K_V2.read_text().strip().split('\n')
    val_100k = VAL_100K_V2.read_text().strip().split('\n')
    
    print(f"  100k_v2 Train: {len(train_100k)}개 이미지")
    print(f"  100k_v2 Val: {len(val_100k)}개 이미지")
    
    # 이미지별 fall_person 인스턴스 수 계산
    print("\n  Train 이미지별 fall_person 계산 중...")
    train_fall_counts = []
    for img_path in train_100k:
        label_path = Path(img_path.replace('/images/', '/labels/').replace('.jpg', '.txt'))
        fall_count = count_fall_person_instances(label_path)
        train_fall_counts.append((img_path, fall_count))
    
    print("  Val 이미지별 fall_person 계산 중...")
    val_fall_counts = []
    for img_path in val_100k:
        label_path = Path(img_path.replace('/images/', '/labels/').replace('.jpg', '.txt'))
        fall_count = count_fall_person_instances(label_path)
        val_fall_counts.append((img_path, fall_count))
    
    # 누적 슬라이싱으로 서브셋 생성
    print("\n  서브셋 생성 중...")
    
    for name, target in DATASETS[:-1]:  # 100k 제외 (이미 존재)
        # Train
        train_images = []
        train_fall = 0
        for img_path, fall_count in train_fall_counts:
            if train_fall >= target:
                break
            train_images.append(img_path)
            train_fall += fall_count
        
        # Val (0.2 비율)
        val_target = target // 5
        val_images = []
        val_fall = 0
        for img_path, fall_count in val_fall_counts:
            if val_fall >= val_target:
                break
            val_images.append(img_path)
            val_fall += fall_count
        
        # 저장
        train_path = YAMLS_DIR / f'train_ins_{name}_v2.txt'
        val_path = YAMLS_DIR / f'val_ins_{name}_v2.txt'
        train_path.write_text('\n'.join(train_images))
        val_path.write_text('\n'.join(val_images))
        
        print(f"    {name}_v2: Train={len(train_images)} ({train_fall} fall), Val={len(val_images)} ({val_fall} fall)")
    
    print("\n  데이터셋 생성 완료!")


# ============================================================
# Phase 2: 훈련 (yolo_ft_megafall.py 기반)
# ============================================================
def update_data_yaml(train_txt: str, val_txt: str):
    """data_megafall.yaml 업데이트"""
    yaml_content = f"""path: /Users/jihunjang/Downloads/ust/dataset/train/megafallv2
train: /Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls/{train_txt}
val: /Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls/{val_txt}
names:
  0: person
  1: fall_person
  2: bicycle
  3: car
  4: motorcycle
  5: airplane
  6: bus
  7: train
  8: truck
  9: boat
  10: traffic light
  11: fire hydrant
"""
    DATA_CFG.write_text(yaml_content)


def run_train(name: str) -> None:
    """단일 모델 훈련 (yolo_ft_megafall.py run_train 기반)"""
    print(f"\n{'='*60}")
    print(f"[Train] {name}_v2 훈련 시작")
    print(f"{'='*60}")
    
    result_name = f'train_ins_{name}_v2'
    
    model = YOLO(str(DEFAULT_WEIGHTS))
    model.train(
        data=str(DATA_CFG),
        imgsz=640,
        epochs=100,
        patience=20,
        batch=-1,
        save=True,
        save_period=1,
        single_cls=False,
        freeze=10,
        lr0=1e-3,
        device='mps',
        workers=12,
        cache='ram',
        project=str(BASE_DIR / 'result'),
        name=result_name,
        verbose=True,
    )
    
    print(f"[Train] {name}_v2 훈련 완료!")


def train_all():
    """모든 모델 순차 훈련"""
    print("\n" + "=" * 60)
    print("[Phase 2] 모델 훈련 (250 → 500 → 1k → 2k → 5k → 10k → 100k)")
    print("=" * 60)
    
    for name, _ in DATASETS:
        # YAML 업데이트
        update_data_yaml(f'train_ins_{name}_v2.txt', f'val_ins_{name}_v2.txt')
        # 훈련
        run_train(name)


# ============================================================
# Phase 3: 평가 (finetuned.py 기반)
# ============================================================
def val_model(weights: Path, data_yaml: Path, classes: list) -> dict:
    """단일 모델 평가 (finetuned.py val 함수 기반)"""
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        classes=classes,
        conf=0.6,
        iou=0.6,
        imgsz=640,
        device='mps',
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
    
    return {"precision": precision, "recall": recall, "f1": f1, "map50": map50, "map50_95": map50_95}


def save_eval_report(results: dict, title: str, prefix: str) -> Path:
    """평가 결과 저장 (finetuned.py save_benchmark_report 기반)"""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = METRICS_DIR / f"{prefix}_{timestamp}"
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
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{metric:.3f}", ha="center", va="bottom", fontsize=value_fontsize)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.set_ylim(0, max(1.0, float(values.max()) + 0.05))
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "metrics.png", dpi=200)
    plt.close(fig)
    
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    
    return run_dir


def evaluate_all():
    """Fall/Person 평가"""
    print("\n" + "=" * 60)
    print("[Phase 3] 평가 (Fall + Person)")
    print("=" * 60)
    
    # 모델 경로
    models = [(name, BASE_DIR / f'result/train_ins_{name}_v2/weights/best.pt') for name, _ in DATASETS]
    
    # ========== Fall 평가 ==========
    print("\n  [Fall 평가] KISA Overseas Fall")
    fall_yaml = YAMLS_DIR / 'data_kisa_fall_val.yaml'
    fall_results = {}
    
    for name, weights in models:
        print(f"    Evaluating {name}_v2...")
        fall_results[f"{name}_v2"] = val_model(weights, fall_yaml, classes=[1])
        print(f"      F1={fall_results[f'{name}_v2']['f1']:.3f}")
    
    fall_dir = save_eval_report(fall_results, "V2 Subset-based - Kisa Overseas Fall", "v2_fall")
    print(f"  Fall 결과 저장: {fall_dir.name}")
    
    # ========== Person 평가 ==========
    print("\n  [Person 평가] KISA Overseas Person")
    person_yaml = YAMLS_DIR / 'data_kisa_person_val.yaml'
    person_results = {}
    
    for name, weights in models:
        print(f"    Evaluating {name}_v2...")
        person_results[f"{name}_v2"] = val_model(weights, person_yaml, classes=[0])
        print(f"      F1={person_results[f'{name}_v2']['f1']:.3f}")
    
    person_dir = save_eval_report(person_results, "V2 Subset-based - Kisa Overseas Person", "v2_person")
    print(f"  Person 결과 저장: {person_dir.name}")
    
    return fall_results, person_results


# ============================================================
# 메인 실행
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("V2 전체 파이프라인: 데이터셋 → 훈련 → 평가")
    print("100k_v2의 서브셋으로 구성 (Seed 42)")
    print("=" * 60)
    
    # 1. 데이터셋 생성
    create_datasets()
    
    # 2. 훈련
    train_all()
    
    # 3. 평가
    fall_results, person_results = evaluate_all()
    
    # 최종 결과 출력
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    
    print("\n[Fall F1]")
    for name, metrics in fall_results.items():
        print(f"  {name}: {metrics['f1']:.3f}")
    
    print("\n[Person F1]")
    for name, metrics in person_results.items():
        print(f"  {name}: {metrics['f1']:.3f}")
    
    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)


