'''
1k, 500, 250 데이터셋을 각각 5번씩 학습하여 평균 성능 측정용 가중치 생성

실행: env PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims python src/train-v1/train_mean_experiments.py
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

# 학습할 데이터셋 목록 (이름, train_txt, val_txt)
DATASETS = [
    ('1k', YAMLS_DIR / 'train_ins_1k_v2.txt', YAMLS_DIR / 'val_ins_1k_v2.txt'),
    ('500', YAMLS_DIR / 'train_ins_500_v2.txt', YAMLS_DIR / 'val_ins_500_v2.txt'),
    ('250', YAMLS_DIR / 'train_ins_250_v2.txt', YAMLS_DIR / 'val_ins_250_v2.txt'),
]

# 반복 횟수
NUM_REPEATS = 5

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
    
    total_runs = len(DATASETS) * NUM_REPEATS
    current_run = 0
    
    print(f"\n{'='*80}")
    print(f"Mean Experiment Training")
    print(f"Datasets: {[d[0] for d in DATASETS]}")
    print(f"Repeats per dataset: {NUM_REPEATS}")
    print(f"Total training runs: {total_runs}")
    print(f"{'='*80}\n")
    
    for dataset_name, train_txt, val_txt in DATASETS:
        for repeat_num in range(1, NUM_REPEATS + 1):
            current_run += 1
            output_name = f"train_ins_{dataset_name}_v2_mean{repeat_num}"
            
            print(f"\n{'='*80}")
            print(f"[{current_run}/{total_runs}] Training: {output_name}")
            print(f"  Dataset: {dataset_name}")
            print(f"  Repeat: {repeat_num}/{NUM_REPEATS}")
            print(f"  Train: {train_txt.name}")
            print(f"  Val: {val_txt.name}")
            print(f"{'='*80}\n")
            
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

