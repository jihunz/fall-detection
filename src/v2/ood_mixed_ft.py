"""
OOD Mixed Fine-tuning 실험: Hard Negative 비율별 성능 비교
- 0% / 50% / 100% hard negative (OOD 313장 고정, Indoor 2687장 공통, 총 3000장)
- Phase 1: YOLO fine-tuning (yolo_ft_indoor.py 동일 하이퍼파라미터)
- Phase 2: OOD test셋 평가 (model.val → P/R/F1, model.predict → 전체 예측 이미지 저장)
- Phase 3: Booktabs 학술 논문 스타일 플롯 생성

사전 준비: python prepare_ood_datasets.py
  → datasets/ood_hard_{0pct,50pct,100pct}_3k/train/{images,labels}/ (3000장)
  → datasets/ood_hard_{0pct,50pct,100pct}_3k/val/{images,labels}/ (→indoor val symlink)
  → yamls/data_ood_hard_{0pct,50pct,100pct}_3k.yaml (디렉토리 기반)
                      7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
# OOD test셋
OOD_TEST_IMAGES = Path('/Users/jihunjang/Downloads/ust/outdoor/outdoor-refined/test/images')

# 실험 정의
EXPERIMENTS = {
    'ood_hard_0pct_3k':   {'display': '0% Hard (Random Only)',   'yaml': YAMLS_DIR / 'data_ood_hard_0pct_3k.yaml'},
    'ood_hard_50pct_3k':  {'display': '50% Hard + 50% Random',  'yaml': YAMLS_DIR / 'data_ood_hard_50pct_3k.yaml'},
    'ood_hard_100pct_3k': {'display': '100% Hard',               'yaml': YAMLS_DIR / 'data_ood_hard_100pct_3k.yaml'},
}

# 훈련 하이퍼파라미터 (yolo_ft_indoor.py 동일)
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

# 평가 설정 (eval.py 기반)
EVAL_CONF = 0.4
EVAL_IOU = 0.6
EVAL_IMGSZ = 640
EVAL_DEVICE = "mps"


# ============================================================================
# Phase 1: Fine-tuning
# ============================================================================
def get_last_epoch(name: str) -> int:
    results_csv = RESULT_DIR / name / 'results.csv'
    if not results_csv.exists():
        return -1
    lines = results_csv.read_text().strip().split('\n')
    if not lines:
        return -1
    try:
        return int(lines[-1].split(',')[0].strip())
    except (ValueError, IndexError):
        return -1


def is_completed(name: str) -> bool:
    best_pt = RESULT_DIR / name / 'weights' / 'best.pt'
    if not best_pt.exists():
        return False
    last_epoch = get_last_epoch(name)
    target_epoch = COMMON_ARGS['epochs'] - 1
    if last_epoch < target_epoch:
        print(f'  [INFO] {name}: best.pt exists but only reached epoch {last_epoch}/{target_epoch}')
        return False
    return True


def can_resume(name: str) -> bool:
    last_pt = RESULT_DIR / name / 'weights' / 'last.pt'
    return last_pt.exists()


def delete_labels_cache():
    """labels.cache 삭제 — datasets_v2/ 하위"""
    datasets_dir = BASE_DIR / 'datasets_v2'
    if datasets_dir.exists():
        for cache in datasets_dir.rglob('*.cache'):
            try:
                cache.unlink()
                print(f'  [Cache] Deleted: {cache}')
            except OSError:
                pass


def cleanup_failed_run(name: str):
    """이전 실패한 실행 정리 (false resume 방지 + 번호 붙은 디렉토리 삭제)"""
    import shutil

    # 번호 붙은 중복 디렉토리 삭제 (e.g., ood_hard_0pct_3k2, 3k3, ...)
    for p in RESULT_DIR.glob(f'{name}[0-9]*'):
        if p.is_dir():
            shutil.rmtree(p)
            print(f'  [Cleanup] Removed duplicate dir: {p}')

    run_dir = RESULT_DIR / name
    if not run_dir.exists():
        return

    last_pt = run_dir / 'weights' / 'last.pt'
    best_pt = run_dir / 'weights' / 'best.pt'
    last_epoch = get_last_epoch(name)

    # best.pt가 없고 epoch이 작으면 실패한 실행
    if not best_pt.exists() and last_epoch < 5:
        print(f'  [Cleanup] Removing failed run artifacts for {name} (last_epoch={last_epoch})')
        # YOLO 학습 산출물만 삭제 (weights, args.yaml, results.csv 등)
        for item in run_dir.iterdir():
            if item.name == 'dataset':  # dataset 보존 (혹시 있으면)
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            print(f'  [Cleanup] Deleted: {item}')


def train(name: str, yaml_path: Path) -> Path:
    """단일 실험 훈련. Returns best.pt path."""
    print(f"\n{'=' * 80}")
    print(f"Phase 1: Training — {name}")
    print(f"{'=' * 80}")

    best_pt = RESULT_DIR / name / 'weights' / 'best.pt'

    # ★ 항상 먼저 cache 삭제 (모든 분기 전에 실행)
    delete_labels_cache()

    if is_completed(name):
        print(f'  [SKIP] {name} — already completed')
        return best_pt

    # 이전 실패한 실행 정리 (epoch < 5이고 best.pt 없으면 삭제)
    cleanup_failed_run(name)

    if can_resume(name):
        print(f'  [RESUME] {name} — resuming from last.pt')
        last_pt = RESULT_DIR / name / 'weights' / 'last.pt'
        model = YOLO(str(last_pt))
        model.train(resume=True, device='mps')
        return best_pt

    if not DEFAULT_WEIGHTS.exists():
        raise FileNotFoundError(f'Weights not found: {DEFAULT_WEIGHTS}')

    model = YOLO(str(DEFAULT_WEIGHTS))
    model.train(
        data=str(yaml_path),
        project=str(RESULT_DIR),
        name=name,
        exist_ok=True,
        **COMMON_ARGS,
    )
    return best_pt


# ============================================================================
# Phase 2: 평가
# ============================================================================
def evaluate(weights: Path, name: str, output_dir: Path) -> Dict[str, float]:
    """
    model.val() → P/R/F1 (fall class만)
    model.predict(save=True) → 전체 예측 이미지 저장
    """
    print(f"\n{'=' * 80}")
    print(f"Phase 2: Evaluation — {name}")
    print(f"{'=' * 80}")

    model = YOLO(str(weights))
    eval_yaml = YAMLS_DIR / 'data_outdoor_eval.yaml'

    # 1. val: fall class 메트릭
    val_dir = str(output_dir / name)
    results = model.val(
        data=str(eval_yaml),
        classes=[1],  # fall only
        conf=EVAL_CONF,
        iou=EVAL_IOU,
        imgsz=EVAL_IMGSZ,
        device=EVAL_DEVICE,
        half=False,
        save=False,
        plots=True,
        cache=False,
        project=val_dir,
        name='val',
        exist_ok=True,
    )
    summary = results.results_dict or {}

    precision = float(summary.get("metrics/precision(B)", 0.0))
    recall = float(summary.get("metrics/recall(B)", 0.0))
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    print(f"  P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # 2. predict: 전체 test 이미지에 예측 bbox 그려서 저장
    predict_dir = str(output_dir / name / 'predictions')
    model.predict(
        source=str(OOD_TEST_IMAGES),
        conf=EVAL_CONF,
        iou=EVAL_IOU,
        imgsz=EVAL_IMGSZ,
        device=EVAL_DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=predict_dir,
        name='images',
        exist_ok=True,
    )
    print(f"  Predictions saved to: {predict_dir}")

    return metrics


# ============================================================================
# Phase 3: 시각화 — Booktabs 학술 논문 스타일
# ============================================================================
def create_booktabs_table(all_metrics: Dict[str, Dict], save_path: Path):
    """Booktabs 스타일 결과 비교 표"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    })

    models = list(all_metrics.keys())
    display_names = [EXPERIMENTS[m]['display'] for m in models]
    columns = ["precision", "recall", "f1"]
    col_display = ["Precision", "Recall", "F1"]

    # 최고값
    best = {}
    for m in columns:
        vals = {model: all_metrics[model][m] for model in models}
        best[m] = max(vals, key=vals.get)

    # 레이아웃
    col_width = 1.5
    row_height = 0.5
    model_col_width = 3.2
    table_width = model_col_width + col_width * len(columns)
    table_height = row_height * (len(models) + 1)

    fig_width = table_width + 1.0
    fig_height = table_height + 1.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, table_width)
    ax.set_ylim(0, table_height)
    ax.axis("off")

    header_y_top = table_height
    header_y_bot = table_height - row_height

    def _col_x(j):
        return 0 if j == 0 else model_col_width + (j - 1) * col_width

    # Booktabs lines
    thick = dict(color="black", linewidth=1.5, clip_on=False)
    thin = dict(color="black", linewidth=0.7, clip_on=False)
    ax.plot([0, table_width], [header_y_top, header_y_top], **thick)
    ax.plot([0, table_width], [header_y_bot, header_y_bot], **thin)
    ax.plot([0, table_width], [0, 0], **thick)

    # 헤더
    header_labels = ["OOD Sampling"] + col_display
    for j, label in enumerate(header_labels):
        x = _col_x(j)
        w = model_col_width if j == 0 else col_width
        cx = x + w / 2
        cy = (header_y_top + header_y_bot) / 2
        ax.text(cx, cy, label, ha="center", va="center", fontsize=11, fontweight="bold")

    # 데이터 행
    for i, (model, display) in enumerate(zip(models, display_names)):
        yt = header_y_bot - i * row_height
        yb = yt - row_height
        cy = (yt + yb) / 2

        ax.text(_col_x(0) + 0.15, cy, display, ha="left", va="center", fontsize=10.5)

        for j, m in enumerate(columns):
            val = all_metrics[model][m]
            x = _col_x(j + 1)
            w = col_width
            cx = x + w / 2
            text = f"{val * 100:.2f}"
            is_best = (best[m] == model)
            ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if is_best else "normal")

    # 캡션
    caption = (
        r"$\bf{Table.}$  Hard negative ratio ablation on OOD test set "
        f"(195 images, 3K mixed training). "
        f"conf={EVAL_CONF}, NMS IoU={EVAL_IOU}. "
        r"$\bf{Bold}$ = best."
    )
    ax.text(table_width / 2, -0.45, caption,
            ha="center", va="top", fontsize=9.5, style="italic")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"  [Table] Saved: {save_path}")


def save_bar_plot(all_metrics: Dict[str, Dict], save_path: Path):
    """Bar plot: 0% / 50% / 100% Hard — Precision / Recall / F1"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "text.usetex": False,
    })

    metrics = ["precision", "recall", "f1"]
    models = list(all_metrics.keys())
    display_names = [EXPERIMENTS[m]['display'] for m in models]
    values = np.array([[all_metrics[m][metric] for m in models] for metric in metrics])

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ['#A8D8A8', '#7CAEE8', '#E8927C']
    for idx, (model, display, color) in enumerate(zip(models, display_names, colors)):
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values[:, idx], width, label=display, color=color,
                      edgecolor='white', linewidth=0.8)
        for bar, metric_val in zip(bars, values[:, idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{metric_val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight='bold',
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.set_ylim(0, max(1.0, float(values.max()) + 0.15))
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("Hard Negative Ratio Ablation — OOD Fall Detection",
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  [Bar Plot] Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='OOD Mixed Fine-tuning')
    parser.add_argument('--exp', type=str, default=None,
                        help='특정 실험만 실행 (e.g., ood_hard_100pct_3k)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # 실험 대상 결정
    if args.exp:
        if args.exp not in EXPERIMENTS:
            print(f'ERROR: Unknown experiment "{args.exp}". Available: {list(EXPERIMENTS.keys())}')
            return
        target_exps = {args.exp: EXPERIMENTS[args.exp]}
    else:
        target_exps = EXPERIMENTS

    print(f"{'=' * 80}")
    print(f"OOD Mixed Fine-tuning — Hard Negative Ratio Ablation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments: {list(target_exps.keys())}")
    print(f"{'=' * 80}")

    # Phase 1: 훈련
    trained_weights = {}
    for exp_name, exp_cfg in target_exps.items():
        best_pt = train(exp_name, exp_cfg['yaml'])
        trained_weights[exp_name] = best_pt

    # Phase 2: 평가
    eval_dir = METRICS_DIR / f"ood_mixed_eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for exp_name, weights in trained_weights.items():
        if not weights.exists():
            print(f"  [WARN] {exp_name} weights not found: {weights}")
            continue
        metrics = evaluate(weights, exp_name, eval_dir)
        all_metrics[exp_name] = metrics

    if not all_metrics:
        print("ERROR: No models evaluated.")
        return

    # Phase 3: 시각화 (2개 이상 모델 평가 시에만)
    if len(all_metrics) >= 2:
        print(f"\n{'=' * 80}")
        print("Phase 3: Generating plots...")
        print(f"{'=' * 80}")

        create_booktabs_table(all_metrics, eval_dir / "metrics_table.png")
        save_bar_plot(all_metrics, eval_dir / "metrics_bar.png")

    # JSON 저장
    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=float)

    # Summary
    print(f"\n{'=' * 80}")
    print("Results Summary: Hard Negative Ratio Ablation")
    print(f"{'=' * 80}")
    print(f"  {'Method':<28} {'P':>8} {'R':>8} {'F1':>8}")
    print(f"  {'-' * 52}")
    for name, m in all_metrics.items():
        display = EXPERIMENTS[name]['display']
        print(f"  {display:<28} {m['precision']:>8.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}")

    print(f"\nAll results saved to: {eval_dir}")


if __name__ == '__main__':
    main()
