from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


DATA_YAML = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls/data_kisa_person_val.yaml')
OUTPUT_ROOT = Path("/Users/jihunjang/workspace/ust/fall-detection/src/metrics")

DEFAULT_CONF = 0.6
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "mps"

TITLE = "Train iteration - Kisa Overseas Person"

# 라벨 디렉토리 (fall0class 교체용)
LABELS_BASE_DIR = Path("/Users/jihunjang/Downloads/ust/dataset/val/kisa-overseas-fall/labels")


@contextmanager
def use_fall0class_labels():
    """
    Context manager: train-fall0class 라벨을 임시로 train으로 교체
    
    자동으로 원복되므로 안전합니다.
    캐시 파일도 함께 삭제하여 오래된 경로 참조 방지합니다.
    """
    train_dir = LABELS_BASE_DIR / "train"
    fall0_dir = LABELS_BASE_DIR / "train-fall0class"
    backup_dir = LABELS_BASE_DIR / "train_backup"
    cache_file = LABELS_BASE_DIR / "train.cache"
    
    # 기존 캐시 삭제 (중요!)
    if cache_file.exists():
        print("  [Cache] Deleting train.cache")
        cache_file.unlink()
    
    # 폴더 교체
    print("  [Label Swap] train -> train_backup")
    train_dir.rename(backup_dir)
    print("  [Label Swap] train-fall0class -> train")
    fall0_dir.rename(train_dir)
    
    try:
        yield
    finally:
        # 원복 (에러가 나도 반드시 실행)
        print("  [Label Swap] Restoring original labels...")
        train_dir.rename(fall0_dir)
        backup_dir.rename(train_dir)
        
        # 생성된 캐시 삭제 (다음을 위해)
        if cache_file.exists():
            print("  [Cache] Deleting generated cache")
            cache_file.unlink()
        
        print("  [Label Swap] Restored!")


def val(weights: Path, data_yaml: Path, classes: List[int], model_name: str, run_dir: Path) -> Dict[str, float]:
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        classes=classes,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        half=False,
        save=True,  # 예측 결과 이미지 저장
        save_txt=True,  # 예측 라벨 txt 저장
        save_conf=True,  # 신뢰도 저장
        cache=False,  # 캐시 비활성화 (라벨 폴더 교체 시 필수)
        project=str(run_dir / "detect_results"),  # 벤치마크 폴더 하위에 저장
        name=model_name,  # 모델별 폴더 이름
        exist_ok=True,  # 기존 폴더 덮어쓰기
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


def run_val():
    results = {}
    w_list = []
    data_yaml = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls/data_kisa_person_val.yaml').resolve()

    # 타임스탬프 폴더 미리 생성
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)


    for item in ['1k', '500', '250']:
        for i in range(1, 6):
            name = f'{item}_{i}'
            model = {
                "name": name,
                "weights": Path(
                    f"/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/train_ins_{item}_v2_mean{i}/weights/best.pt"),
                "classes": [0],
            }
            name = model["name"]
            weights = model["weights"].resolve()
            classes = model["classes"]
            metrics = val(weights, data_yaml, classes, name, run_dir)
            w_list.append(weights)
            results.setdefault(item, []).append(metrics)

    return results, run_dir, w_list


def compute_mean_results(results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """각 스케일 그룹의 5개 시드 결과를 평균 계산"""
    mean_results = {}
    
    for scale, metrics_list in results.items():
        # 각 metric별 평균 계산
        mean_metrics = {}
        for metric in ["precision", "recall", "f1", "map50", "map50_95"]:
            values = [m[metric] for m in metrics_list]
            mean_metrics[metric] = sum(values) / len(values)
        mean_results[scale] = mean_metrics
    
    return mean_results


def save_benchmark_report(results: Dict[str, Dict[str, float]], run_dir: Path, raw_results: Dict[str, List[Dict[str, float]]] = None) -> Path:
    """Persist metric summary (JSON + plot) under the given run directory."""

    metrics = ["precision", "recall", "f1"]
    labels = list(results.keys())
    values = np.array([[results[label][metric] for label in labels] for metric in metrics])

    x = np.arange(len(metrics))
    
    # 동적 width 계산: 라벨 개수에 반비례
    width = min(0.8 / len(labels), 0.35)
    
    # 동적 figure 크기: 라벨 많을수록 넓게
    fig_width = max(10, 6 + len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    
    # 동적 폰트 크기: 라벨 많을수록 작게
    value_fontsize = max(7, 10 - len(labels) * 0.4)
    
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, values[:, idx], width, label=label)
        for bar, metric in zip(bars, values[:, idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{metric:.3f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    upper_ylim = max(1.0, float(values.max()) + 0.05)
    ax.set_ylim(0, upper_ylim)
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title(f"{TITLE} (Mean of 5 Trains)", fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "metrics_mean.png", dpi=200)
    plt.close(fig)

    # JSON 저장: 평균 결과 + 원본 결과
    save_data = {
        "mean": results,
        "raw": raw_results if raw_results else {}
    }
    with (run_dir / "metrics_mean.json").open("w", encoding="utf-8") as fh:
        json.dump(save_data, fh, indent=2)

    return run_dir


if __name__ == '__main__':
    # 1. 평가 실행
    raw_results, run_dir, w_list = run_val()
    
    # 2. 평균 계산
    mean_results = compute_mean_results(raw_results)
    
    # 3. 결과 출력
    print("\n" + "=" * 60)
    print("Mean Results (5 Seeds)")
    print("=" * 60)
    for scale, metrics in mean_results.items():
        print(f"  {scale}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    # 4. Plot 저장
    save_benchmark_report(mean_results, run_dir, raw_results)
    print(f"\n✅ All results saved to: {run_dir}")
