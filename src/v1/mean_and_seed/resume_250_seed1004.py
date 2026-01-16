'''
train_ins_250_seed1004 학습 재개

실행: env PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dims python src/v1/resume_250_seed1004.py
'''

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner


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


patch_task_aligned_assigner_for_mps()


RESUME_WEIGHTS = Path('/src/v1/result/train_ins_250_seed1004/weights/last.pt')


def main():
    print("=" * 60)
    print("Resuming: train_ins_250_seed1004")
    print(f"From: {RESUME_WEIGHTS}")
    print("=" * 60)
    
    model = YOLO(str(RESUME_WEIGHTS))
    model.train(resume=True, device='mps')
    
    print("\n✅ Training completed!")


if __name__ == '__main__':
    main()

