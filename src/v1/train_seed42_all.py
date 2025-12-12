'''
Seed 42 기반 데이터셋 재생성 + 2k → 5k → 10k 연속 훈련 스크립트
'''

from __future__ import annotations

import random
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
YAMLS_DIR = BASE_DIR / 'yamls'
DATA_CFG = YAMLS_DIR / 'data_megafall.yaml'
DEFAULT_WEIGHTS = Path('/Users/jihunjang/workspace/ust/fall-detection/src/models/yolo12n.pt')

# 100k 데이터셋 (소스)
TRAIN_100K = YAMLS_DIR / 'train_ins_100k.txt'

# MPS 패치
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
# 데이터셋 생성 함수 (V2 규칙: fall_person만 목표 수, 나머지는 자연 포함)
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


def create_dataset_seed42(target_fall_instances: int, name: str) -> tuple[Path, Path]:
    """
    Seed 42로 데이터셋 생성
    V2 규칙: fall_person만 목표 수치, 나머지는 자연 포함
    """
    print(f"\n{'='*60}")
    print(f"[{name}] Seed 42 데이터셋 생성 (목표 fall_person: {target_fall_instances})")
    print(f"{'='*60}")
    
    # Seed 42 고정
    random.seed(42)
    
    # 100k 이미지 목록 로드
    all_images = TRAIN_100K.read_text().strip().split('\n')
    
    # 이미지별 fall_person 인스턴스 수 계산
    image_fall_counts = []
    for img_path in all_images:
        label_path = Path(img_path.replace('/images/', '/labels/').replace('.jpg', '.txt'))
        fall_count = count_fall_person_instances(label_path)
        if fall_count > 0:
            image_fall_counts.append((img_path, fall_count))
    
    # 셔플
    random.shuffle(image_fall_counts)
    
    # fall_person 인스턴스 목표까지 이미지 선택
    selected_images = []
    current_fall_instances = 0
    
    for img_path, fall_count in image_fall_counts:
        if current_fall_instances >= target_fall_instances:
            break
        selected_images.append(img_path)
        current_fall_instances += fall_count
    
    # Train/Val 분리 (80/20)
    random.shuffle(selected_images)
    val_size = max(1, len(selected_images) // 5)
    val_images = selected_images[:val_size]
    train_images = selected_images[val_size:]
    
    # 파일 저장
    train_path = YAMLS_DIR / f'train_ins_{name}.txt'
    val_path = YAMLS_DIR / f'val_ins_{name}.txt'
    
    train_path.write_text('\n'.join(train_images))
    val_path.write_text('\n'.join(val_images))
    
    # 통계 출력
    train_fall = sum(count_fall_person_instances(
        Path(img.replace('/images/', '/labels/').replace('.jpg', '.txt'))
    ) for img in train_images)
    
    val_fall = sum(count_fall_person_instances(
        Path(img.replace('/images/', '/labels/').replace('.jpg', '.txt'))
    ) for img in val_images)
    
    print(f"  Train: {len(train_images)} images, {train_fall} fall_person instances")
    print(f"  Val:   {len(val_images)} images, {val_fall} fall_person instances")
    print(f"  Total: {len(selected_images)} images, {current_fall_instances} fall_person instances")
    print(f"  저장: {train_path.name}, {val_path.name}")
    
    return train_path, val_path


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
    print(f"  YAML 업데이트: {train_txt}, {val_txt}")


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


# ============================================================
# 메인 실행
# ============================================================
if __name__ == '__main__':
    # 데이터셋 설정: (이름, 목표 fall_person 인스턴스 수)
    DATASETS = [
        ('2k', 2000),
        ('5k', 5000),
        ('10k', 10000),
    ]
    
    print("=" * 60)
    print("Seed 42 기반 데이터셋 재생성 + 연속 훈련")
    print("순서: 2k → 5k → 10k")
    print("=" * 60)
    
    # 1. 모든 데이터셋 재생성
    print("\n[Phase 1] 데이터셋 재생성 (Seed 42)")
    for name, target in DATASETS:
        create_dataset_seed42(target, name)
    
    # 2. 연속 훈련
    print("\n[Phase 2] 연속 훈련")
    for name, _ in DATASETS:
        # YAML 업데이트
        update_data_yaml(f'train_ins_{name}.txt', f'val_ins_{name}.txt')
        
        # 훈련
        run_train(name)
    
    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)



