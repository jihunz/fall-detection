"""
OOD Variation 플롯: summary.json → Bar Plot
- eval.py의 save_bar_plot 스타일 추출
- 범례: Indoor Only → ID 100%, OOD 0% Hard → OOD Random 100%

Usage:
  python plot_ood_variation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# 설정
# ============================================================================
SUMMARY_JSON = Path(__file__).resolve().parent / 'ood_mixed_eval_260402_220704' / 'summary.json'
OUTPUT_DIR = SUMMARY_JSON.parent

# key → display name 매핑
DISPLAY_NAMES = {
    'indoor_only_3k':     'ID 100%',
    'ood_hard_0pct_3k':   'OOD Random 100%',
    'ood_hard_50pct_3k':  'OOD Hard 50%',
    'ood_hard_100pct_3k': 'OOD Hard 100%',
}


# ============================================================================
# 시각화 (eval.py save_bar_plot 스타일)
# ============================================================================
def save_bar_plot(results: dict, save_path: Path):
    metrics = ["precision", "recall", "f1"]
    keys = [k for k in DISPLAY_NAMES if k in results]
    labels = [DISPLAY_NAMES[k] for k in keys]
    values = np.array([[results[k][m] for k in keys] for m in metrics])

    x = np.arange(len(metrics))
    width = min(0.8 / len(keys), 0.35)
    fig_width = max(10, 6 + len(keys) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    value_fontsize = max(7, 10 - len(keys) * 0.4)

    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, values[:, idx], width, label=label)
        for bar, metric_val in zip(bars, values[:, idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{metric_val:.3f}",
                ha="center", va="bottom", fontsize=value_fontsize,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.set_ylim(0, max(1.0, float(values.max()) + 0.05))
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("Fall Detection: ID vs. OOD Variation", fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    if not SUMMARY_JSON.exists():
        print(f"ERROR: {SUMMARY_JSON} not found")
        return

    with SUMMARY_JSON.open() as f:
        results = json.load(f)

    save_path = OUTPUT_DIR / "ood_variation_bar.png"
    save_bar_plot(results, save_path)


if __name__ == '__main__':
    main()
