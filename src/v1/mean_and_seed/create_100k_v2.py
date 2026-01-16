'''
100k_v2 데이터셋 생성
- Train: Seed 42, fall_person 100,000개 목표
- Val: Seed 100, Train 제외, 0.2 비율 (20,000 fall_person)
'''

import random
from pathlib import Path

# 경로 설정
LABELS_DIR = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/labels/train')
IMAGES_DIR = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/images/train')
YAMLS_DIR = Path('/src/v1/yamls')

# 시드 설정
TRAIN_SEED = 42
VAL_SEED = 100

# 목표
TRAIN_TARGET = 100000
VAL_TARGET = 20000  # 0.2 비율


def count_fall_person(label_path: Path) -> int:
    """라벨 파일에서 fall_person (class 1) 인스턴스 수"""
    if not label_path.exists():
        return 0
    count = 0
    for line in label_path.read_text().strip().split('\n'):
        if line and line.split()[0] == '1':
            count += 1
    return count


def main():
    print("=" * 60)
    print("100k_v2 데이터셋 생성")
    print("=" * 60)
    
    # 모든 라벨 파일 스캔
    print("\n[1] 라벨 파일 스캔 중...")
    label_files = list(LABELS_DIR.glob('*.txt'))
    print(f"    총 라벨 파일: {len(label_files)}개")
    
    # fall_person 포함 이미지 추출
    print("\n[2] fall_person 포함 이미지 추출 중...")
    image_fall_counts = []
    
    for i, label_path in enumerate(label_files):
        if i % 50000 == 0:
            print(f"    진행: {i}/{len(label_files)}")
        
        fall_count = count_fall_person(label_path)
        if fall_count > 0:
            # 이미지 경로 구성
            img_name = label_path.stem + '.jpg'
            img_path = IMAGES_DIR / img_name
            if img_path.exists():
                image_fall_counts.append((str(img_path), fall_count))
    
    print(f"    fall_person 포함 이미지: {len(image_fall_counts)}개")
    total_fall = sum(count for _, count in image_fall_counts)
    print(f"    총 fall_person 인스턴스: {total_fall}개")
    
    # ========== Train 데이터셋 생성 ==========
    print(f"\n[3] Train 데이터셋 생성 (Seed {TRAIN_SEED}, 목표: {TRAIN_TARGET})")
    random.seed(TRAIN_SEED)
    random.shuffle(image_fall_counts)
    
    train_images = []
    train_fall = 0
    
    for img_path, fall_count in image_fall_counts:
        if train_fall >= TRAIN_TARGET:
            break
        train_images.append(img_path)
        train_fall += fall_count
    
    print(f"    Train 이미지: {len(train_images)}개")
    print(f"    Train fall_person: {train_fall}개")
    
    # Train에 사용된 이미지 집합
    train_set = set(train_images)
    
    # ========== Val 데이터셋 생성 ==========
    print(f"\n[4] Val 데이터셋 생성 (Seed {VAL_SEED}, 목표: {VAL_TARGET})")
    
    # Train에 포함되지 않은 이미지만
    val_candidates = [(img, count) for img, count in image_fall_counts if img not in train_set]
    print(f"    Val 후보 이미지: {len(val_candidates)}개")
    
    random.seed(VAL_SEED)
    random.shuffle(val_candidates)
    
    val_images = []
    val_fall = 0
    
    for img_path, fall_count in val_candidates:
        if val_fall >= VAL_TARGET:
            break
        val_images.append(img_path)
        val_fall += fall_count
    
    print(f"    Val 이미지: {len(val_images)}개")
    print(f"    Val fall_person: {val_fall}개")
    
    # ========== 파일 저장 ==========
    print("\n[5] 파일 저장")
    
    train_path = YAMLS_DIR / 'train_ins_100k_v2.txt'
    val_path = YAMLS_DIR / 'val_ins_100k_v2.txt'
    
    train_path.write_text('\n'.join(train_images))
    val_path.write_text('\n'.join(val_images))
    
    print(f"    {train_path.name}: {len(train_images)}개 이미지")
    print(f"    {val_path.name}: {len(val_images)}개 이미지")
    
    # ========== 요약 ==========
    print("\n" + "=" * 60)
    print("생성 완료!")
    print("=" * 60)
    print(f"Train: {len(train_images)} images, {train_fall} fall_person")
    print(f"Val: {len(val_images)} images, {val_fall} fall_person")
    print("=" * 60)


if __name__ == '__main__':
    main()


