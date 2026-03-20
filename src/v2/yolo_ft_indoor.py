"""
Indoor-Refined 데이터셋 스케일별 YOLO 훈련 스크립트
- config dict로 6개 스케일 순차 실행 (작은 스케일 → 큰 스케일)
- 이미 완료된 스케일은 자동 skip
- 37527(100%)는 주석 처리 상태

Usage:
  python yolo_ft_indoor.py
  python yolo_ft_indoor.py --scale indoor_10percent  # 특정 스케일만
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# MPS fallback 환경 변수 자동 설정 (import 전에 선언해야 적용됨)
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

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
RESULT_DIR = str(BASE_DIR / 'result')
DEFAULT_WEIGHTS = Path('/Users/jihunjang/workspace/ust/fall-detection/src/models/nano/yolo12n.pt')

# 스케일별 config — 작은 순서대로 정렬
SCALES = {
    'indoor_1percent':   str(BASE_DIR / 'yamls/data_indoor_375.yaml'),
    'indoor_5percent':   str(BASE_DIR / 'yamls/data_indoor_1876.yaml'),
    'indoor_10percent':  str(BASE_DIR / 'yamls/data_indoor_3753.yaml'),
    'indoor_25percent':  str(BASE_DIR / 'yamls/data_indoor_9382.yaml'),
    'indoor_50percent':  str(BASE_DIR / 'yamls/data_indoor_18764.yaml'),
    # 'indoor_100percent': str(BASE_DIR / 'yamls/data_indoor_37527.yaml'),  # 100% — 필요시 주석 해제
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


# ============================================================================
# 훈련 로직
# ============================================================================
def is_completed(name: str) -> bool:
    """이미 완료된 스케일인지 확인 (best.pt 존재 여부)"""
    best_pt = Path(RESULT_DIR) / name / 'weights' / 'best.pt'
    return best_pt.exists()


def can_resume(name: str) -> bool:
    """중단된 학습을 이어서 할 수 있는지 확인"""
    last_pt = Path(RESULT_DIR) / name / 'weights' / 'last.pt'
    args_yaml = Path(RESULT_DIR) / name / 'args.yaml'
    return last_pt.exists() and args_yaml.exists()


def train_scale(name: str, data_yaml: str) -> None:
    """단일 스케일 훈련"""
    if is_completed(name):
        print(f'\n[SKIP] {name} — already completed (best.pt exists)')
        return

    if can_resume(name):
        print(f'\n[RESUME] {name} — resuming from last.pt')
        last_pt = Path(RESULT_DIR) / name / 'weights' / 'last.pt'
        model = YOLO(str(last_pt))
        model.train(resume=True, device='mps')
        return

    print(f'\n{"=" * 60}')
    print(f'[TRAIN] {name}')
    print(f'  Data: {data_yaml}')
    print(f'{"=" * 60}')

    if not DEFAULT_WEIGHTS.exists():
        raise FileNotFoundError(f'기본 가중치를 찾을 수 없습니다: {DEFAULT_WEIGHTS}')

    model = YOLO(str(DEFAULT_WEIGHTS))
    model.train(
        data=data_yaml,
        project=RESULT_DIR,
        name=name,
        **COMMON_ARGS,
    )


def main():
    parser = argparse.ArgumentParser(description='Indoor-Refined scale training')
    parser.add_argument('--scale', type=str, default=None,
                        help='특정 스케일만 훈련 (e.g., indoor_3753)')
    args = parser.parse_args()

    if args.scale:
        if args.scale not in SCALES:
            print(f'ERROR: Unknown scale "{args.scale}". Available: {list(SCALES.keys())}')
            return
        train_scale(args.scale, SCALES[args.scale])
    else:
        # 전체 순차 실행
        for name, data_yaml in SCALES.items():
            train_scale(name, data_yaml)

    print('\n' + '=' * 60)
    print('All training complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
