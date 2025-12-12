'''
전체 파이프라인: 데이터셋 구성 → 훈련 → 평가
- Train: 100k의 진정한 서브셋 (Seed 42, 누적 슬라이싱)
- Val: Train 제외 이미지에서 별도 시드로 0.2 비율 샘플링
'''

from __future__ import annotations

import random
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
YAMLS_DIR = BASE_DIR / 'yamls'
DATA_CFG = YAMLS_DIR / 'data_megafall.yaml'
DEFAULT_WEIGHTS = Path('/Users/jihunjang/workspace/ust/fall-detection/src/models/yolo12n.pt')
METRICS_DIR = Path('/Users/jihunjang/workspace/ust/fall-detection/src/metrics')

# 100k 데이터셋 (소스)
TRAIN_100K = YAMLS_DIR / 'train_ins_100k.txt'

# 시드 설정
TRAIN_SEED = 42
VAL_SEED = 100

# 데이터셋 설정: (이름, fall_person 목표)
DATASETS = [
    ('250', 250),
    ('500', 500),
    ('1k', 1000),
    ('2k', 2000),
    ('5k', 5000),
    ('10k', 10000),
]

# ============================================================
# MPS 패치
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
# 데이터셋 생성 함수
# ============================================================
def count_fall_person_instances(label_path: Path) -> int:
    """라벨 파일에서 fall_person (class 1) 인스턴스 수를 세기"""
    if not label_path.exists():
        return 0
    count = 0
    for line in label_path.read_text().strip().split('\n'):
        if line and line.split()[0] == '1':
            count += 1
    return count


def create_all_datasets():
    """모든 데이터셋 생성 (서브셋 관계 보장)"""
    print("\n" + "=" * 60)
    print("[Phase 1] 데이터셋 생성 (서브셋 관계 보장)")
    print("=" * 60)
    
    # 100k 이미지 목록 로드
    all_images = TRAIN_100K.read_text().strip().split('\n')
    
    # 이미지별 fall_person 인스턴스 수 계산
    image_fall_counts = []
    print("  이미지별 fall_person 인스턴스 수 계산 중...")
    for img_path in all_images:
        label_path = Path(img_path.replace('/images/', '/labels/').replace('.jpg', '.txt'))
        fall_count = count_fall_person_instances(label_path)
        if fall_count > 0:
            image_fall_counts.append((img_path, fall_count))
    
    print(f"  fall_person 포함 이미지: {len(image_fall_counts)}개")
    
    # ========== Train 데이터셋 생성 (Seed 42) ==========
    print(f"\n  [Train] Seed {TRAIN_SEED}로 셔플 및 누적 슬라이싱")
    random.seed(TRAIN_SEED)
    random.shuffle(image_fall_counts)
    
    # 누적 슬라이싱으로 서브셋 생성
    train_datasets = {}
    current_fall = 0
    current_idx = 0
    
    for name, target in DATASETS:
        selected = []
        fall_sum = 0
        
        # 이전 데이터셋 끝에서 시작 (누적)
        for i in range(current_idx, len(image_fall_counts)):
            if current_fall + fall_sum >= target:
                break
            img_path, fall_count = image_fall_counts[i]
            selected.append(img_path)
            fall_sum += fall_count
            current_idx = i + 1
        
        current_fall += fall_sum
        train_datasets[name] = {
            'images': [img for img, _ in image_fall_counts[:current_idx]],
            'fall_count': current_fall,
        }
        
        print(f"    {name}: {len(train_datasets[name]['images'])} images, {current_fall} fall_person")
    
    # Train에 사용된 이미지 집합
    train_used_images = set(train_datasets['10k']['images'])
    
    # ========== Val 데이터셋 생성 (별도 시드) ==========
    print(f"\n  [Val] Seed {VAL_SEED}로 별도 샘플링 (Train 제외, 0.2 비율)")
    
    # Train에 포함되지 않은 이미지만 필터링
    val_candidates = [(img, count) for img, count in image_fall_counts if img not in train_used_images]
    print(f"    Val 후보 이미지: {len(val_candidates)}개")
    
    random.seed(VAL_SEED)
    random.shuffle(val_candidates)
    
    # Val도 누적 슬라이싱
    val_datasets = {}
    current_val_fall = 0
    current_val_idx = 0
    
    for name, target in DATASETS:
        val_target = target // 5  # 0.2 비율
        selected = []
        fall_sum = 0
        
        for i in range(current_val_idx, len(val_candidates)):
            if current_val_fall + fall_sum >= val_target:
                break
            img_path, fall_count = val_candidates[i]
            selected.append(img_path)
            fall_sum += fall_count
            current_val_idx = i + 1
        
        current_val_fall += fall_sum
        val_datasets[name] = {
            'images': [img for img, _ in val_candidates[:current_val_idx]],
            'fall_count': current_val_fall,
        }
        
        print(f"    {name}: {len(val_datasets[name]['images'])} images, {current_val_fall} fall_person")
    
    # ========== 파일 저장 ==========
    print("\n  [저장] txt 파일 생성")
    for name, _ in DATASETS:
        train_path = YAMLS_DIR / f'train_ins_{name}.txt'
        val_path = YAMLS_DIR / f'val_ins_{name}.txt'
        
        train_path.write_text('\n'.join(train_datasets[name]['images']))
        val_path.write_text('\n'.join(val_datasets[name]['images']))
        
        print(f"    {name}: train={len(train_datasets[name]['images'])}, val={len(val_datasets[name]['images'])}")
    
    return train_datasets, val_datasets


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


# ============================================================
# 훈련 함수
# ============================================================
def run_train(name: str) -> None:
    """모델 훈련"""
    print(f"\n{'='*60}")
    print(f"[{name}] 훈련 시작")
    print(f"{'='*60}")
    
    result_name = f'train_ins_{name}'
    
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
    
    print(f"[{name}] 훈련 완료!")


def train_all():
    """모든 모델 훈련"""
    print("\n" + "=" * 60)
    print("[Phase 2] 모델 훈련 (250 → 500 → 1k → 2k → 5k → 10k)")
    print("=" * 60)
    
    for name, _ in DATASETS:
        # YAML 업데이트
        update_data_yaml(f'train_ins_{name}.txt', f'val_ins_{name}.txt')
        # 훈련
        run_train(name)


# ============================================================
# 평가 함수
# ============================================================
def val_model(weights: Path, data_yaml: Path, classes: list) -> dict:
    """단일 모델 평가"""
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
    """평가 결과 저장"""
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
    models = [(name, BASE_DIR / f'result/train_ins_{name}/weights/best.pt') for name, _ in DATASETS]
    
    # ========== Fall 평가 ==========
    print("\n  [Fall 평가] KISA Overseas Fall")
    fall_yaml = YAMLS_DIR / 'data_kisa_fall_val.yaml'
    fall_results = {}
    
    for name, weights in models:
        print(f"    Evaluating {name}...")
        fall_results[name] = val_model(weights, fall_yaml, classes=[1])
        print(f"      F1={fall_results[name]['f1']:.3f}")
    
    fall_dir = save_eval_report(fall_results, "Subset-based Instances - Kisa Overseas Fall", "ins_subset_fall")
    print(f"  Fall 결과 저장: {fall_dir.name}")
    
    # ========== Person 평가 ==========
    print("\n  [Person 평가] KISA Overseas Person")
    person_yaml = YAMLS_DIR / 'data_kisa_person_val.yaml'
    person_results = {}
    
    for name, weights in models:
        print(f"    Evaluating {name}...")
        person_results[name] = val_model(weights, person_yaml, classes=[0])
        print(f"      F1={person_results[name]['f1']:.3f}")
    
    person_dir = save_eval_report(person_results, "Subset-based Instances - Kisa Overseas Person", "ins_subset_person")
    print(f"  Person 결과 저장: {person_dir.name}")
    
    return fall_results, person_results


# ============================================================
# 메인 실행
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("전체 파이프라인: 데이터셋 → 훈련 → 평가")
    print("Train: Seed 42 (서브셋 관계 보장)")
    print("Val: Seed 100 (별도 샘플링, 0.2 비율)")
    print("=" * 60)
    
    # 1. 데이터셋 생성
    create_all_datasets()
    
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



