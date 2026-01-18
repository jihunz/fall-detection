"""
YOLO 벤치마크 결과를 학술 논문 스타일 테이블로 시각화
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

# 폰트 설정
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# 경로 설정
METRICS_ROOT = Path(__file__).parent


def load_data():
    """metrics_mean.json 파일들 로드"""
    return {
        'fall': {
            'nano': json.loads((METRICS_ROOT / 'yolo-fall/nano/metrics_mean.json').read_text()),
            'small': json.loads((METRICS_ROOT / 'yolo-fall/small/metrics_mean.json').read_text()),
        },
        'person': {
            'nano': json.loads((METRICS_ROOT / 'yolo-person/nano/metrics_mean.json').read_text()),
            'small': json.loads((METRICS_ROOT / 'yolo-person/small/metrics_mean.json').read_text()),
        }
    }


def create_academic_table(ax, category_data, title, columns):
    """학술 논문 스타일 테이블 생성"""
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15, family='serif')
    
    # 테이블 데이터 준비
    table_rows = []
    for scale_key, scale_name in [('nano', 'Nano'), ('small', 'Small')]:
        metrics = category_data[scale_key]
        models = list(metrics.keys())
        for i, model in enumerate(models):
            m = metrics[model]
            row = [
                scale_name if i == 0 else '',
                model,
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1']:.3f}"
            ]
            table_rows.append(row)
    
    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.0)
    
    n_rows = len(table_rows)
    
    for key, cell in table.get_celld().items():
        cell.set_text_props(family='serif')
        row, col = key
        
        if row == 0:  # 헤더
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('white')
            cell.visible_edges = 'BT'
            cell.set_linewidth(1.5)
        else:
            cell.set_facecolor('white')
            # Small 시작 (5번째 행)
            if row == 5:
                cell.visible_edges = 'T'
                cell.set_linewidth(0.8)
            # 마지막 행
            elif row == n_rows:
                cell.visible_edges = 'B'
                cell.set_linewidth(1.5)
            else:
                cell.visible_edges = ''
                cell.set_linewidth(0)


def main():
    data = load_data()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5))
    columns = ['Scale', 'Model', 'Precision', 'Recall', 'F1-Score']
    
    create_academic_table(ax1, data['fall'], '(a) Fall Detection', columns)
    create_academic_table(ax2, data['person'], '(b) Person Detection', columns)
    
    plt.tight_layout()
    output_path = METRICS_ROOT / 'yolo_benchmark_academic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
