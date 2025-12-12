'''
10k 모델 훈련 재개 스크립트
'''

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
FINETUNED_WEIGHT_DIR = 'train_ins_10k'
RESUME_WEIGHTS = Path(f'/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/{FINETUNED_WEIGHT_DIR}/weights/last.pt')

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


def run_train() -> None:
    print(f"Resuming training from: {RESUME_WEIGHTS}")
    model = YOLO(str(RESUME_WEIGHTS))
    model.train(resume=True, device='mps')
    print("Training completed!")


if __name__ == '__main__':
    run_train()



