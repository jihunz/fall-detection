'''
5개 시드별 1k, 500, 250 데이터셋 학습

실행: env PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims python src/train-v1/train_seed_experiments.py
'''

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner

BASE_DIR = Path(__file__).resolve().parent
YAMLS_DIR = BASE_DIR / 'yamls'
RESULT_DIR = BASE_DIR / 'result'
DEFAULT_WEIGHTS = Path('/src/models/yolo12n.pt')

# 데이터셋 경로 (megafallv2)
DATASET_PATH = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2')

# 시드 목록
SEEDS = ['seed32', 'seed123', 'seed456', 'seed789', 'seed1004']

# 데이터셋 크기 목록
DATASET_SIZES = ['1k', '500', '250']

# 클래스 정의 (megafallv2 기준)
CLASS_NAMES = '''names:
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
  11: fire hydrant'''


def patch_task_aligned_assigner_for_mps():
    """MPS 백엔드 불리언 인덱싱 버그 패치"""
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


def create_temp_yaml(train_txt: Path, val_txt: Path) -> Path:
    """임시 data.yaml 파일 생성"""
    yaml_content = f'''path: {DATASET_PATH}
train: {train_txt}
val: {val_txt}
{CLASS_NAMES}'''
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    temp_file.write(yaml_content)
    temp_file.close()
    return Path(temp_file.name)


def run_single_train(data_yaml: Path, output_name: str) -> None:
    """단일 학습 실행"""
    model = YOLO(str(DEFAULT_WEIGHTS))
    
    model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=100,
        batch=32,
        save=True,
        save_period=1,
        single_cls=False,
        freeze=10,
        lr0=1e-3,
        lrf=1.0,
        device='mps',
        workers=12,
        cache='ram',
        project=str(RESULT_DIR),
        name=output_name,
        verbose=True,
    )


def main():
    patch_task_aligned_assigner_for_mps()
    
    # 학습할 조합 생성
    train_configs = []
    for seed in SEEDS:
        for size in DATASET_SIZES:
            train_txt = YAMLS_DIR / f'train_ins_{size}_{seed}.txt'
            val_txt = YAMLS_DIR / f'val_ins_{size}_{seed}.txt'
            output_name = f'train_ins_{size}_{seed}'
            train_configs.append((seed, size, train_txt, val_txt, output_name))
    
    total_runs = len(train_configs)
    
    print(f"\n{'='*80}")
    print(f"Seed Experiment Training")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {DATASET_SIZES}")
    print(f"Total training runs: {total_runs}")
    print(f"{'='*80}\n")
    
    for i, (seed, size, train_txt, val_txt, output_name) in enumerate(train_configs, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{total_runs}] Training: {output_name}")
        print(f"  Seed: {seed}")
        print(f"  Dataset size: {size}")
        print(f"  Train: {train_txt.name}")
        print(f"  Val: {val_txt.name}")
        print(f"{'='*80}\n")
        
        # 파일 존재 확인
        if not train_txt.exists():
            print(f"  ❌ Train file not found: {train_txt}")
            continue
        if not val_txt.exists():
            print(f"  ❌ Val file not found: {val_txt}")
            continue
        
        # 임시 yaml 생성
        temp_yaml = create_temp_yaml(train_txt, val_txt)
        
        try:
            run_single_train(temp_yaml, output_name)
            print(f"\n✅ Completed: {output_name}")
        except Exception as e:
            print(f"\n❌ Failed: {output_name}")
            print(f"   Error: {e}")
        finally:
            # 임시 파일 삭제
            temp_yaml.unlink(missing_ok=True)
    
    print(f"\n{'='*80}")
    print(f"All training completed!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

