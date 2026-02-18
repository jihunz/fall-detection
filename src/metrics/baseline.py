"""
YOLO Nano/Small 스케일별 평가 스크립트
- Person, Fall 각각 평가
- Fall 평가 시 use_fall0class_labels() 적용 (class=0으로 평가)
- 10회 반복 → Mean 계산
- 스케일별 결과 저장 (JSON, PNG)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ============================================================================
# 설정
# ============================================================================
MODELS_DIR = Path('/Users/jihunjang/workspace/ust/fall-detection/src/models')
YAMLS_DIR = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls')
OUTPUT_ROOT = Path('/Users/jihunjang/workspace/ust/fall-detection/src/metrics')

# 라벨 디렉토리 (fall0class 교체용)
LABELS_BASE_DIR = Path("/Users/jihunjang/Downloads/ust/dataset/train/gopr/labels/test")

# 스케일별 모델 정의
SCALES = {
    'nano': {
        'models': [
            {'name': 'YOLOv8n', 'weights': MODELS_DIR / 'nano/yolov8n.pt'},
            {'name': 'YOLO11n', 'weights': MODELS_DIR / 'nano/yolo11n.pt'},
            {'name': 'YOLO12n', 'weights': MODELS_DIR / 'nano/yolo12n.pt'},
            {'name': 'YOLO26n', 'weights': MODELS_DIR / 'nano/yolo26n.pt'},
        ]
    },
}

# 평가 타입별 설정
EVAL_TYPES = {
    'person': {
        'data_yaml': YAMLS_DIR / 'data_gopr_fall_val.yaml',
        'classes': [0],
        'output_dir': 'yolo-vfp-person',
        'use_fall0class': False,
    },
    'fall': {
        'data_yaml': YAMLS_DIR / 'data_gopr_fall_val.yaml',
        'classes': [0],  # fall0class 라벨 사용 시 class=0으로 평가
        'output_dir': 'yolo-vfp-fall',
        'use_fall0class': True,
    }
}

# 평가 설정
NUM_RUNS = 1
DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = 'mps'


@contextmanager
def use_fall0class_labels():
    """
    Context manager: GOPR test 라벨에서 class 1(fall_person)을 class 0으로 임시 변환
    
    GOPR 구조: gopr/labels/test/ (train 하위 폴더 없음)
    - test/ → test_backup/
    - test-fall0class/ → test/
    
    자동으로 원복되므로 안전합니다.
    """
    test_dir = LABELS_BASE_DIR  # gopr/labels/test
    backup_dir = test_dir.parent / "test_backup"
    fall0_dir = test_dir.parent / "test-fall0class"
    cache_file = test_dir.parent / "test.cache"
    
    # fall0class 라벨이 없으면 생성
    if not fall0_dir.exists():
        print("  [Label] Creating fall0class labels...")
        fall0_dir.mkdir(parents=True)
        for label_file in test_dir.glob("*.txt"):
            lines = label_file.read_text().strip().split('\n')
            new_lines = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts[0] == '1':      # fall_person → class 0
                        parts[0] = '0'
                    elif parts[0] == '0':    # person → class 1
                        parts[0] = '1'
                    new_lines.append(' '.join(parts))
            (fall0_dir / label_file.name).write_text('\n'.join(new_lines))
        print(f"  [Label] Created {len(list(fall0_dir.glob('*.txt')))} fall0class labels")
    
    # 기존 캐시 삭제 (중요!)
    if cache_file.exists():
        print("  [Cache] Deleting test.cache")
        cache_file.unlink()
    
    # 폴더 교체
    print("  [Label Swap] test -> test_backup")
    test_dir.rename(backup_dir)
    print("  [Label Swap] test-fall0class -> test")
    fall0_dir.rename(test_dir)
    
    try:
        yield
    finally:
        # 원복 (에러가 나도 반드시 실행)
        print("  [Label Swap] Restoring original labels...")
        test_dir.rename(fall0_dir)
        backup_dir.rename(test_dir)
        
        # 생성된 캐시 삭제 (다음을 위해)
        if cache_file.exists():
            print("  [Cache] Deleting generated cache")
            cache_file.unlink()
        
        print("  [Label Swap] Restored!")


def val(weights: Path, data_yaml: Path, classes: List[int],
        save_dir: Path = None, model_name: str = None) -> Dict[str, float]:
    """단일 모델 평가"""
    model = YOLO(str(weights))
    
    # YOLO26은 end-to-end (NMS-free) 모델이므로 val 기본 conf 사용
    is_yolo26 = '26' in str(weights.name)
    conf = 0.001 if is_yolo26 else DEFAULT_CONF
    
    results = model.val(
        data=str(data_yaml),
        classes=classes,
        conf=conf,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        half=False,
        save=True,           # prediction 이미지 저장
        save_txt=True,       # 예측 라벨 저장
        save_conf=True,      # confidence 포함
        project=str(save_dir) if save_dir else None,
        name=model_name if model_name else 'val',
        cache=False,
    )
    summary = results.results_dict or {}

    precision = float(summary.get("metrics/precision(B)", 0.0))
    recall = float(summary.get("metrics/recall(B)", 0.0))
    map50 = float(summary.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(summary.get("metrics/mAP50-95(B)", 0.0))
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
    }


def run_evaluation(scale: str, eval_type: str) -> Dict[str, Dict]:
    scale_config = SCALES[scale]
    eval_config = EVAL_TYPES[eval_type]
    
    results = {}
    data_yaml = eval_config['data_yaml'].resolve()
    classes = eval_config['classes']
    use_fall0class = eval_config.get('use_fall0class', False)
    
    print(f"\n{'='*80}")
    print(f"Evaluating {scale.upper()} models for {eval_type.upper()}")
    if use_fall0class:
        print(f"Using fall0class labels (class=0)")
    print(f"{'='*80}")
    
    # prediction 저장 경로
    save_dir = OUTPUT_ROOT / eval_config['output_dir'] / scale / 'predictions'
    
    for model_config in scale_config['models']:
        model_name = model_config['name']
        weights = model_config['weights'].resolve()
        
        print(f"\n[{model_name}] Starting {NUM_RUNS} runs...")
        
        run_results = {}
        for run_idx in range(1, NUM_RUNS + 1):
            print(f"  Run {run_idx}/{NUM_RUNS}...", end=" ")
            
            # fall 평가 시 fall0class 라벨 사용
            if use_fall0class:
                with use_fall0class_labels():
                    metrics = val(weights, data_yaml, classes, save_dir=save_dir, model_name=model_name)
            else:
                metrics = val(weights, data_yaml, classes, save_dir=save_dir, model_name=model_name)
            
            run_results[str(run_idx)] = metrics
            print(f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        # Mean 계산
        metrics_keys = ['precision', 'recall', 'f1', 'map50', 'map50_95']
        mean = {k: np.mean([run_results[str(i)][k] for i in range(1, NUM_RUNS + 1)]) for k in metrics_keys}
        std = {k: np.std([run_results[str(i)][k] for i in range(1, NUM_RUNS + 1)]) for k in metrics_keys}
        
        run_results['mean'] = mean
        run_results['std'] = std
        results[model_name] = run_results
        
        print(f"  → Mean: P={mean['precision']:.3f}, R={mean['recall']:.3f}, F1={mean['f1']:.3f}")
    
    return results


def save_results(results: Dict, scale: str, eval_type: str):
    """결과 저장 (JSON + Bar Plot)"""
    eval_config = EVAL_TYPES[eval_type]
    output_dir = OUTPUT_ROOT / eval_config['output_dir'] / scale
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Raw 결과 저장
    raw_path = output_dir / 'raw_results.json'
    with raw_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved raw results to {raw_path}")
    
    # 2. Mean 표 저장
    mean_data = {name: data['mean'] for name, data in results.items()}
    mean_path = output_dir / 'metrics_mean.json'
    with mean_path.open('w', encoding='utf-8') as f:
        json.dump(mean_data, f, indent=2, default=float)
    print(f"Saved mean metrics to {mean_path}")
    
    # 3. Bar Plot 생성
    create_bar_plot(mean_data, output_dir, scale, eval_type)


def create_bar_plot(mean_data: Dict, output_dir: Path, scale: str, eval_type: str):
    """Bar Plot 생성"""
    metrics = ["precision", "recall", "f1"]
    labels = list(mean_data.keys())
    values = np.array([[mean_data[label][metric] for label in labels] for metric in metrics])

    x = np.arange(len(metrics))
    width = min(0.8 / len(labels), 0.35)
    
    fig_width = max(10, 6 + len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    
    value_fontsize = max(7, 10 - len(labels) * 0.4)
    
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, values[:, idx], width, label=label)
        for bar, metric_val in zip(bars, values[:, idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{metric_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    upper_ylim = max(1.0, float(values.max()) + 0.05)
    ax.set_ylim(0, upper_ylim)
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    
    title = f"YOLO {scale.capitalize()} - VFP290K {eval_type.capitalize()}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    
    plot_path = output_dir / 'metrics_mean.png'
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved bar plot to {plot_path}")


def main():
    """전체 평가 실행"""
    print(f"Starting YOLO Scale Evaluation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for eval_type in ['person', 'fall']:
        for scale in ['nano']:
            results = run_evaluation(scale, eval_type)
            save_results(results, scale, eval_type)
    
    print("\n" + "="*80)
    print("All evaluations completed!")
    print("="*80)


if __name__ == '__main__':
    main()
