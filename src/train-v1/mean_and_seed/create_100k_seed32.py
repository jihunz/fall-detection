'''
Seed 32로 100k, 10k, 5k 데이터셋 생성
- Train: Seed 32
- Val: Seed 132 (32 + 100)
- 서브셋 관계: 5k ⊂ 10k ⊂ 100k

실행: python src/train-v1/mean_and_seed/create_100k_seed32.py
'''

import random
from pathlib import Path
from typing import List, Tuple

# 경로 설정
LABELS_DIR = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/labels/train')
IMAGES_DIR = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/images/train')
YAMLS_DIR = Path('/src/train-v1/yamls')

# 시드 설정
TRAIN_SEED = 32
VAL_SEED = 132  # TRAIN_SEED + 100

# 데이터셋 목표 (이름, Train fall_person, Val fall_person)
DATASET_TARGETS = [
    ('100k', 100000, 20000),
    ('10k', 10000, 2000),
    ('5k', 5000, 1000),
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


def main():
    print("=" * 70)
    print("Seed 32 - 100k, 10k, 5k 데이터셋 생성")
    print(f"Train Seed: {TRAIN_SEED}, Val Seed: {VAL_SEED}")
    print("서브셋 관계: 5k ⊂ 10k ⊂ 100k")
    print("=" * 70)
    
    # ========== Step 1: 모든 라벨 스캔 ==========
    print("\n[1] 라벨 파일 스캔 중...")
    label_files = list(LABELS_DIR.glob('*.txt'))
    print(f"    총 라벨 파일: {len(label_files)}개")
    
    # fall_person 포함 이미지 추출
    print("\n[2] fall_person 포함 이미지 추출 중...")
    all_image_fall_counts = []
    image_fall_map = {}
    
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
    
    # ========== Step 2: 100k Train 생성 (기준) ==========
    print(f"\n[3] 100k Train 생성 (Seed {TRAIN_SEED}, 목표: 100,000 fall_person)")
    random.seed(TRAIN_SEED)
    shuffled_images = all_image_fall_counts.copy()
    random.shuffle(shuffled_images)
    
    train_100k = []
    train_100k_fall = 0
    
    for img_path, fall_count in shuffled_images:
        if train_100k_fall >= DATASET_TARGETS[0][1]:  # 100000
            break
        train_100k.append(img_path)
        train_100k_fall += fall_count
    
    print(f"    100k Train: {len(train_100k)} images, {train_100k_fall} fall_person")
    
    # ========== Step 3: 10k, 5k Train 생성 (100k의 서브셋) ==========
    print(f"\n[4] 10k, 5k Train 생성 (100k의 서브셋)")
    
    # 10k: 100k 순서 유지하면서 처음 10k fall_person까지
    train_10k = []
    train_10k_fall = 0
    for img_path in train_100k:
        if train_10k_fall >= DATASET_TARGETS[1][1]:  # 10000
            break
        train_10k.append(img_path)
        train_10k_fall += image_fall_map[img_path]
    
    print(f"    10k Train: {len(train_10k)} images, {train_10k_fall} fall_person")
    
    # 5k: 10k 순서 유지하면서 처음 5k fall_person까지
    train_5k = []
    train_5k_fall = 0
    for img_path in train_10k:
        if train_5k_fall >= DATASET_TARGETS[2][1]:  # 5000
            break
        train_5k.append(img_path)
        train_5k_fall += image_fall_map[img_path]
    
    print(f"    5k Train: {len(train_5k)} images, {train_5k_fall} fall_person")
    
    # ========== Step 4: Val 생성 (Train 100k 제외) ==========
    print(f"\n[5] Val 데이터셋 생성 (Seed {VAL_SEED})")
    
    train_100k_set = set(train_100k)
    val_candidates = [(img, count) for img, count in all_image_fall_counts 
                      if img not in train_100k_set]
    print(f"    Val 후보: {len(val_candidates)}개 이미지")
    
    random.seed(VAL_SEED)
    random.shuffle(val_candidates)
    
    # 100k Val
    val_100k = []
    val_100k_fall = 0
    for img_path, fall_count in val_candidates:
        if val_100k_fall >= DATASET_TARGETS[0][2]:  # 20000
            break
        val_100k.append(img_path)
        val_100k_fall += fall_count
    
    print(f"    100k Val: {len(val_100k)} images, {val_100k_fall} fall_person")
    
    # 10k Val (100k Val의 서브셋)
    val_10k = []
    val_10k_fall = 0
    for img_path in val_100k:
        if val_10k_fall >= DATASET_TARGETS[1][2]:  # 2000
            break
        val_10k.append(img_path)
        val_10k_fall += image_fall_map[img_path]
    
    print(f"    10k Val: {len(val_10k)} images, {val_10k_fall} fall_person")
    
    # 5k Val (10k Val의 서브셋)
    val_5k = []
    val_5k_fall = 0
    for img_path in val_10k:
        if val_5k_fall >= DATASET_TARGETS[2][2]:  # 1000
            break
        val_5k.append(img_path)
        val_5k_fall += image_fall_map[img_path]
    
    print(f"    5k Val: {len(val_5k)} images, {val_5k_fall} fall_person")
    
    # ========== Step 5: 파일 저장 ==========
    print("\n[6] 파일 저장")
    
    datasets = [
        ('100k_seed32', train_100k, val_100k, train_100k_fall, val_100k_fall),
        ('10k_seed32', train_10k, val_10k, train_10k_fall, val_10k_fall),
        ('5k_seed32', train_5k, val_5k, train_5k_fall, val_5k_fall),
    ]
    
    for ds_name, train_imgs, val_imgs, train_fall, val_fall in datasets:
        train_path = YAMLS_DIR / f'train_ins_{ds_name}.txt'
        val_path = YAMLS_DIR / f'val_ins_{ds_name}.txt'
        
        train_path.write_text('\n'.join(train_imgs))
        val_path.write_text('\n'.join(val_imgs))
        
        print(f"    {ds_name}: Train {len(train_imgs)} imgs ({train_fall} fall), Val {len(val_imgs)} imgs ({val_fall} fall)")
    
    # ========== Step 6: data.yaml 파일 생성 ==========
    print("\n[7] data.yaml 파일 생성")
    
    yaml_template = '''path: /Users/jihunjang/Downloads/ust/dataset/train/megafallv2
train: {train_txt}
val: {val_txt}
names:
  0: person
  1: fall_person
  2: bicycle
  3: car
  4: motorcycle
  5: airplane
  6: bus
  7: train
  8: truck
  9: boat
  10: traffic light
  11: fire hydrant
'''
    
    for ds_name, _, _, _, _ in datasets:
        yaml_path = YAMLS_DIR / f'data_megafall_{ds_name}.yaml'
        train_txt = YAMLS_DIR / f'train_ins_{ds_name}.txt'
        val_txt = YAMLS_DIR / f'val_ins_{ds_name}.txt'
        
        yaml_content = yaml_template.format(
            train_txt=str(train_txt),
            val_txt=str(val_txt)
        )
        yaml_path.write_text(yaml_content)
        print(f"    생성: {yaml_path.name}")
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("✅ 생성 완료!")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Train Img':<12} {'Train Fall':<12} {'Val Img':<10} {'Val Fall':<10}")
    print("-" * 70)
    
    for ds_name, train_imgs, val_imgs, train_fall, val_fall in datasets:
        print(f"{ds_name:<15} {len(train_imgs):<12} {train_fall:<12} {len(val_imgs):<10} {val_fall:<10}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
