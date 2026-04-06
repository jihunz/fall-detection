'''
5개의 시드로 1k, 500, 250 데이터셋 생성
- 각 시드별로 독립적인 Train/Val 샘플링
- 500은 1k의 서브셋, 250은 500의 서브셋

실행: python src/train-v1/create_seed_datasets.py
'''

import random
from pathlib import Path
from typing import List, Tuple

# 경로 설정
LABELS_DIR = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/labels/train')
IMAGES_DIR = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/images/train')
YAMLS_DIR = Path('/src/train-v1/yamls')

# 시드 설정 (Train 시드, Val 시드)
SEEDS = [
    ('seed32', 32, 132),
    ('seed123', 123, 223),
    ('seed456', 456, 556),
    ('seed789', 789, 889),
    ('seed1004', 1004, 1104),
]

# 데이터셋 목표 (이름, Train fall_person, Val fall_person)
DATASET_TARGETS = [
    ('1k', 1000, 200),
    ('500', 500, 100),
    ('250', 250, 50),
]


def count_fall_person(label_path: Path) -> int:
    """라벨 파일에서 fall_person (class 1) 인스턴스 수"""
    if not label_path.exists():
        return 0
    count = 0
    for line in label_path.read_text().strip().split('\n'):
        if line and line.split()[0] == '1':
            count += 1
    return count


def sample_images(
    candidates: List[Tuple[str, int]], 
    target_instances: int, 
    seed: int
) -> Tuple[List[str], int]:
    """이미지 샘플링하여 목표 인스턴스 수 달성"""
    random.seed(seed)
    shuffled = candidates.copy()
    random.shuffle(shuffled)
    
    selected = []
    total_instances = 0
    
    for img_path, fall_count in shuffled:
        if total_instances >= target_instances:
            break
        selected.append(img_path)
        total_instances += fall_count
    
    return selected, total_instances


def create_subset(
    parent_images: List[str], 
    image_fall_map: dict,
    target_instances: int, 
    seed: int
) -> Tuple[List[str], int]:
    """부모 데이터셋에서 서브셋 추출"""
    parent_with_counts = [(img, image_fall_map[img]) for img in parent_images]
    return sample_images(parent_with_counts, target_instances, seed)


def main():
    print("=" * 70)
    print("Multi-Seed Dataset Generation")
    print("Seeds:", [s[0] for s in SEEDS])
    print("Datasets:", [d[0] for d in DATASET_TARGETS])
    print("=" * 70)
    
    # ========== Step 1: 모든 라벨 스캔 ==========
    print("\n[1] 라벨 파일 스캔 중...")
    label_files = list(LABELS_DIR.glob('*.txt'))
    print(f"    총 라벨 파일: {len(label_files)}개")
    
    # fall_person 포함 이미지 추출
    print("\n[2] fall_person 포함 이미지 추출 중...")
    all_image_fall_counts = []
    image_fall_map = {}  # 이미지 -> fall_person 수 매핑
    
    for i, label_path in enumerate(label_files):
        if i % 50000 == 0:
            print(f"    진행: {i}/{len(label_files)}")
        
        fall_count = count_fall_person(label_path)
        if fall_count > 0:
            img_name = label_path.stem + '.jpg'
            img_path = IMAGES_DIR / img_name
            if img_path.exists():
                img_str = str(img_path)
                all_image_fall_counts.append((img_str, fall_count))
                image_fall_map[img_str] = fall_count
    
    print(f"    fall_person 포함 이미지: {len(all_image_fall_counts)}개")
    total_fall = sum(count for _, count in all_image_fall_counts)
    print(f"    총 fall_person 인스턴스: {total_fall}개")
    
    # ========== Step 2: 각 시드별 데이터셋 생성 ==========
    summary = []
    
    for seed_name, train_seed, val_seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"[{seed_name}] Train Seed: {train_seed}, Val Seed: {val_seed}")
        print("="*70)
        
        # 1k Train 생성
        train_1k, train_1k_fall = sample_images(
            all_image_fall_counts, 
            DATASET_TARGETS[0][1],  # 1000
            train_seed
        )
        train_1k_set = set(train_1k)
        
        # 1k Val 생성 (Train 제외)
        val_candidates = [(img, count) for img, count in all_image_fall_counts 
                         if img not in train_1k_set]
        val_1k, val_1k_fall = sample_images(
            val_candidates, 
            DATASET_TARGETS[0][2],  # 200
            val_seed
        )
        
        print(f"  1k Train: {len(train_1k)} images, {train_1k_fall} fall_person")
        print(f"  1k Val: {len(val_1k)} images, {val_1k_fall} fall_person")
        
        # 500 생성 (1k의 서브셋)
        train_500, train_500_fall = create_subset(
            train_1k, image_fall_map, 
            DATASET_TARGETS[1][1],  # 500
            train_seed
        )
        val_500, val_500_fall = create_subset(
            val_1k, image_fall_map, 
            DATASET_TARGETS[1][2],  # 100
            val_seed
        )
        
        print(f"  500 Train: {len(train_500)} images, {train_500_fall} fall_person")
        print(f"  500 Val: {len(val_500)} images, {val_500_fall} fall_person")
        
        # 250 생성 (500의 서브셋)
        train_250, train_250_fall = create_subset(
            train_500, image_fall_map, 
            DATASET_TARGETS[2][1],  # 250
            train_seed
        )
        val_250, val_250_fall = create_subset(
            val_500, image_fall_map, 
            DATASET_TARGETS[2][2],  # 50
            val_seed
        )
        
        print(f"  250 Train: {len(train_250)} images, {train_250_fall} fall_person")
        print(f"  250 Val: {len(val_250)} images, {val_250_fall} fall_person")
        
        # 파일 저장
        datasets_to_save = [
            ('1k', train_1k, val_1k, train_1k_fall, val_1k_fall),
            ('500', train_500, val_500, train_500_fall, val_500_fall),
            ('250', train_250, val_250, train_250_fall, val_250_fall),
        ]
        
        for ds_name, train_imgs, val_imgs, train_fall, val_fall in datasets_to_save:
            train_path = YAMLS_DIR / f'train_ins_{ds_name}_{seed_name}.txt'
            val_path = YAMLS_DIR / f'val_ins_{ds_name}_{seed_name}.txt'
            
            train_path.write_text('\n'.join(train_imgs))
            val_path.write_text('\n'.join(val_imgs))
            
            summary.append({
                'seed': seed_name,
                'dataset': ds_name,
                'train_images': len(train_imgs),
                'train_fall': train_fall,
                'val_images': len(val_imgs),
                'val_fall': val_fall,
            })
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("생성 완료! 요약:")
    print("=" * 70)
    print(f"{'Seed':<12} {'Dataset':<8} {'Train Img':<12} {'Train Fall':<12} {'Val Img':<10} {'Val Fall':<10}")
    print("-" * 70)
    
    for s in summary:
        print(f"{s['seed']:<12} {s['dataset']:<8} {s['train_images']:<12} {s['train_fall']:<12} {s['val_images']:<10} {s['val_fall']:<10}")
    
    print("=" * 70)
    print(f"총 생성 파일: {len(summary) * 2}개 (train + val)")


if __name__ == '__main__':
    main()

