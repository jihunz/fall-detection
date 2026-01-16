'''
100k_v2 훈련 재개
'''

from pathlib import Path
from ultralytics import YOLO

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

# Resume
RESUME_WEIGHTS = Path('/src/v1/result/train_ins_100k_v2/weights/last.pt')

if __name__ == '__main__':
    print("=" * 60)
    print("100k_v2 훈련 재개")
    print(f"Resume from: {RESUME_WEIGHTS}")
    print("=" * 60)
    
    model = YOLO(str(RESUME_WEIGHTS))
    model.train(resume=True, device='mps')
    
    print("=" * 60)
    print("100k_v2 훈련 완료!")
    print("=" * 60)


