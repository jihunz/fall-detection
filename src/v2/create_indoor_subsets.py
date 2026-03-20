"""
indoor-refined/train에서 stratified sampling으로 스케일별 서브셋 추출.
- fall 포함 이미지 / non-fall 이미지를 각각 비율 유지하며 추출
- test/val은 전체 고정 (서브셋 없음)
- seed=42 고정

산출물:
  yamls/train_indoor_{scale}.txt   (6개)
  yamls/val_indoor.txt             (1개, 공용)
  yamls/test_indoor.txt            (1개, 공용)
  yamls/data_indoor_{scale}.yaml   (6개)
"""

from __future__ import annotations

import os
import random
from pathlib import Path

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
YAMLS_DIR = BASE_DIR / 'yamls'

# Host paths (for yaml/txt output — used by ultralytics on host machine)
HOST_INDOOR_REFINED = Path('/Users/jihunjang/Downloads/indoor-refined')
HOST_YAMLS_DIR = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v2/yamls')

# VM paths (for actual file I/O during script execution)
VM_INDOOR_REFINED = Path('/sessions/amazing-determined-shannon/mnt/Downloads/indoor-refined')

SCALES = {
    '375':   375,
    '1876':  1876,
    '3753':  3753,
    '9382':  9382,
    '18764': 18764,
    '37527': 37527,
}


def classify_images(labels_dir: Path, images_dir: Path):
    """이미지를 fall 포함 / non-fall로 분류"""
    fall_images = []      # fall 인스턴스가 1개 이상인 이미지
    nonfall_images = []   # fall 인스턴스가 없는 이미지

    for label_file in sorted(labels_dir.glob('*.txt')):
        stem = label_file.stem
        # 대응 이미지 찾기
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        has_fall = False
        for line in label_file.read_text().strip().split('\n'):
            parts = line.strip().split()
            if parts and int(parts[0]) == 1:
                has_fall = True
                break

        if has_fall:
            fall_images.append(vm_to_host(str(img_path)))
        else:
            nonfall_images.append(vm_to_host(str(img_path)))

    return fall_images, nonfall_images


def stratified_sample(fall_images: list, nonfall_images: list, n: int, seed: int):
    """fall 비율을 유지하며 n장 샘플링"""
    total = len(fall_images) + len(nonfall_images)
    fall_ratio = len(fall_images) / total
    n_fall = round(n * fall_ratio)
    n_nonfall = n - n_fall

    rng = random.Random(seed)
    sampled_fall = rng.sample(fall_images, min(n_fall, len(fall_images)))
    sampled_nonfall = rng.sample(nonfall_images, min(n_nonfall, len(nonfall_images)))

    return sorted(sampled_fall + sampled_nonfall)


def collect_all_images(images_dir: Path):
    """디렉토리 내 모든 이미지 경로를 정렬 반환"""
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sorted(str(p) for p in images_dir.iterdir() if p.suffix.lower() in exts)


def write_txt(path: Path, lines: list):
    path.write_text('\n'.join(lines) + '\n')
    print(f'  Written: {path.name} ({len(lines)} lines)')


def write_yaml(path: Path, scale_name: str):
    content = f"""path: {HOST_INDOOR_REFINED}
train: {HOST_YAMLS_DIR / f'train_indoor_{scale_name}.txt'}
val: {HOST_YAMLS_DIR / 'test_indoor.txt'}
names:
  0: person
  1: fall
"""
    path.write_text(content)
    print(f'  Written: {path.name}')


def host_to_vm(host_path: str) -> str:
    """Host 경로를 VM 경로로 변환"""
    return host_path.replace(
        str(HOST_INDOOR_REFINED), str(VM_INDOOR_REFINED)
    )


def count_instances(image_paths: list, labels_dir: Path):
    """이미지 리스트(host 경로)에 대응하는 라벨에서 인스턴스 수 집계"""
    c0 = c1 = 0
    for img_path in image_paths:
        stem = Path(img_path).stem
        label_file = labels_dir / (stem + '.txt')  # labels_dir is VM path
        if not label_file.exists():
            continue
        for line in label_file.read_text().strip().split('\n'):
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls == 0:
                c0 += 1
            elif cls == 1:
                c1 += 1
    return c0, c1


def vm_to_host(vm_path: str) -> str:
    """VM 경로를 host 경로로 변환"""
    return vm_path.replace(
        str(VM_INDOOR_REFINED), str(HOST_INDOOR_REFINED)
    )


def main():
    YAMLS_DIR.mkdir(parents=True, exist_ok=True)

    # VM paths for actual file I/O
    train_images_dir = VM_INDOOR_REFINED / 'train' / 'images'
    train_labels_dir = VM_INDOOR_REFINED / 'train' / 'labels'
    val_images_dir = VM_INDOOR_REFINED / 'val' / 'images'
    test_images_dir = VM_INDOOR_REFINED / 'test' / 'images'

    # 1. val / test txt (공용) — host 경로로 변환하여 저장
    print('[1] Val/Test txt 생성')
    val_imgs_vm = collect_all_images(val_images_dir)
    test_imgs_vm = collect_all_images(test_images_dir)
    val_imgs = [vm_to_host(p) for p in val_imgs_vm]
    test_imgs = [vm_to_host(p) for p in test_imgs_vm]
    write_txt(YAMLS_DIR / 'val_indoor.txt', val_imgs)
    write_txt(YAMLS_DIR / 'test_indoor.txt', test_imgs)

    # 2. train 이미지 분류
    print('\n[2] Train 이미지 분류 (fall / non-fall)')
    fall_images, nonfall_images = classify_images(train_labels_dir, train_images_dir)
    print(f'  Fall images: {len(fall_images)}, Non-fall images: {len(nonfall_images)}')
    print(f'  Total: {len(fall_images) + len(nonfall_images)}')

    # 3. 스케일별 서브셋 생성
    print('\n[3] 스케일별 서브셋 생성')
    stats = {}
    for scale_name, n in SCALES.items():
        print(f'\n  --- Scale {scale_name} ({n} images) ---')
        if n >= len(fall_images) + len(nonfall_images):
            # 100%: 전체 사용
            sampled = sorted(fall_images + nonfall_images)
        else:
            sampled = stratified_sample(fall_images, nonfall_images, n, SEED)

        # txt 작성
        write_txt(YAMLS_DIR / f'train_indoor_{scale_name}.txt', sampled)
        # yaml 작성
        write_yaml(YAMLS_DIR / f'data_indoor_{scale_name}.yaml', scale_name)

        # 통계
        c0, c1 = count_instances(sampled, train_labels_dir)
        total_inst = c0 + c1
        fall_pct = c1 / total_inst * 100 if total_inst else 0
        stats[scale_name] = {
            'images': len(sampled),
            'instances': total_inst,
            'c0': c0,
            'c1': c1,
            'fall_pct': fall_pct,
        }
        print(f'  Images: {len(sampled)}, Instances: {total_inst}, '
              f'c0: {c0}, c1: {c1}, Fall: {fall_pct:.1f}%')

    # 4. 통계 출력
    print('\n' + '=' * 70)
    print('서브셋 통계 요약')
    print('=' * 70)
    print(f"{'Scale':>7} {'Images':>8} {'Inst':>8} {'c0':>8} {'c1':>8} {'Fall%':>7}")
    print('-' * 50)
    for s, st in stats.items():
        print(f"{s:>7} {st['images']:>8} {st['instances']:>8} "
              f"{st['c0']:>8} {st['c1']:>8} {st['fall_pct']:>6.1f}%")

    print(f"\nTest images: {len(test_imgs)}")
    print(f"Val images:  {len(val_imgs)}")
    print('\nDone!')


if __name__ == '__main__':
    main()
