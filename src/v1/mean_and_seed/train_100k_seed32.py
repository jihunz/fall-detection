'''
Seed 32 - 100k, 10k, 5k 순차 훈련
env PYTORCH_ENABLE_MPS_FALLBACK=1 python src/v1/mean_and_seed/train_100k_seed32.py
'''

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
YAMLS_DIR = BASE_DIR / 'yamls'
RESULT_DIR = BASE_DIR / 'result'

# 기본 가중치
DEFAULT_WEIGHTS = Path('/src/models/nano/yolo12n.pt')

# 훈련 순서: 100k → 10k → 5k
TRAIN_CONFIGS = [
    {
        'name': 'train_ins_100k_seed32',
        'data_yaml': YAMLS_DIR / 'data_megafall_100k_seed32.yaml',
    },
    {
        'name': 'train_ins_10k_seed32',
        'data_yaml': YAMLS_DIR / 'data_megafall_10k_seed32.yaml',
    },
    {
        'name': 'train_ins_5k_seed32',
        'data_yaml': YAMLS_DIR / 'data_megafall_5k_seed32.yaml',
    },
]

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


def check_already_trained(name: str) -> bool:
    """이미 훈련 완료된 모델인지 확인"""
    result_path = RESULT_DIR / name / 'weights' / 'best.pt'
    return result_path.exists()


def get_resume_weights(name: str) -> Path | None:
    """Resume 가능한 weights 반환"""
    last_pt = RESULT_DIR / name / 'weights' / 'last.pt'
    args_yaml = RESULT_DIR / name / 'args.yaml'
    if last_pt.exists() and args_yaml.exists():
        return last_pt
    return None


def train_model(config: dict) -> None:
    name = config['name']
    data_yaml = config['data_yaml']
    
    # 이미 완료된 경우 스킵
    if check_already_trained(name):
        print(f"\n⏭️  [{name}] 이미 훈련 완료됨, 스킵")
        return
    
    # Resume 확인
    resume_weights = get_resume_weights(name)
    
    if resume_weights:
        print(f"\n🔄 [{name}] Resume 모드로 훈련 재개")
        model = YOLO(str(resume_weights))
        model.train(resume=True, device='mps')
    else:
        print(f"\n🚀 [{name}] 새로운 훈련 시작")
        print(f"   Data: {data_yaml.name}")
        
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
            name=name,
            verbose=True,
        )
    
    print(f"\n✅ [{name}] 훈련 완료!")


def main():
    print("=" * 70)
    print("Seed 32 - 100k, 10k, 5k 순차 훈련")
    print("=" * 70)
    
    for i, config in enumerate(TRAIN_CONFIGS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(TRAIN_CONFIGS)}] {config['name']}")
        print("=" * 70)
        train_model(config)
    
    print("\n" + "=" * 70)
    print("🎉 모든 훈련 완료!")
    print("=" * 70)


if __name__ == '__main__':
    main()
