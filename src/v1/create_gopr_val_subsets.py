"""
GOPR Val 서브셋 생성 스크립트
- Val도 Train과 동일한 비율로 서브셋 구성
- 5k: 50영상×20장 = 1,000장
- 2k: 50영상×8장 = 400장
- 1k: 50영상×4장 = 200장
- 500: 50영상×2장 = 100장
- 250: 25영상×2장 = 50장
- Seed 32 사용
"""

import os
import random
from pathlib import Path
from collections import defaultdict
import re

# 설정
SEED = 32

# 경로
DST_BASE = Path('/Users/jihunjang/Downloads/ust/dataset/train/gopr')
YAMLS_DIR = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls')
VAL_IMAGES_DIR = DST_BASE / 'images' / 'val'


def extract_video_number(filename):
    """파일명에서 GOPR 영상 번호 추출"""
    match = re.search(r'GOPR(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def create_txt_file(image_paths, txt_path):
    """이미지 경로 리스트를 txt 파일로 저장"""
    with open(txt_path, 'w') as f:
        for img_path in sorted(image_paths):
            f.write(f"{img_path}\n")


def update_yaml_file(yaml_path, train_txt, val_txt):
    """YAML 설정 파일 업데이트"""
    content = f"""path: {DST_BASE}
train: {train_txt}
val: {val_txt}
names:
  0: person
  1: fall_person
"""
    with open(yaml_path, 'w') as f:
        f.write(content)


def main():
    random.seed(SEED)
    
    print("=" * 60)
    print("GOPR Val 서브셋 생성 (Seed 32)")
    print("=" * 60)
    
    # 1. Val 이미지를 영상별로 그룹화
    print("\n1. Val 이미지 분석 중...")
    video_images = defaultdict(list)
    
    for img_file in VAL_IMAGES_DIR.iterdir():
        video_num = extract_video_number(img_file.name)
        if video_num:
            video_images[video_num].append(img_file.name)
    
    video_list = list(video_images.keys())
    print(f"   Val 영상 수: {len(video_list)}개")
    print(f"   Val 총 이미지: {sum(len(imgs) for imgs in video_images.values())}장")
    
    # 2. 영상 셔플 (서브셋 관계 유지를 위해)
    random.shuffle(video_list)
    
    # 각 영상의 이미지도 셔플
    for video_num in video_images:
        random.shuffle(video_images[video_num])
    
    # 3. Val 서브셋 정의: (이름, 영상 수, 영상당 이미지 수)
    val_subsets = [
        ('5k', 50, 20),   # 1,000장
        ('2k', 50, 8),    # 400장
        ('1k', 50, 4),    # 200장
        ('500', 50, 2),   # 100장
        ('250', 25, 2),   # 50장
    ]
    
    print("\n2. Val 서브셋 txt 파일 생성 중...")
    
    for name, num_videos, imgs_per_video in val_subsets:
        videos_to_use = video_list[:num_videos]
        
        subset_images = []
        for video_num in videos_to_use:
            subset_images.extend(video_images[video_num][:imgs_per_video])
        
        # 이미지 경로 생성 (상대 경로)
        image_paths = [f"images/val/{img}" for img in subset_images]
        
        txt_path = YAMLS_DIR / f"val_gopr_{name}.txt"
        create_txt_file(image_paths, txt_path)
        print(f"   val_gopr_{name}.txt: {len(subset_images)}장")
    
    # 4. YAML 파일 업데이트
    print("\n3. YAML 파일 업데이트 중...")
    
    for name, _, _ in val_subsets:
        yaml_path = YAMLS_DIR / f"data_gopr_{name}.yaml"
        train_txt = YAMLS_DIR / f"train_gopr_{name}.txt"
        val_txt = YAMLS_DIR / f"val_gopr_{name}.txt"
        update_yaml_file(yaml_path, train_txt, val_txt)
        print(f"   data_gopr_{name}.yaml 업데이트")
    
    # 5. 서브셋 관계 검증
    print("\n4. 서브셋 관계 검증...")
    
    prev_set = None
    for name, _, _ in val_subsets:
        txt_path = YAMLS_DIR / f"val_gopr_{name}.txt"
        with open(txt_path, 'r') as f:
            current_set = set(line.strip() for line in f)
        
        if prev_set is not None:
            if current_set.issubset(prev_set):
                print(f"   val_{name} ⊂ val_{prev_name} ✓")
            else:
                diff = current_set - prev_set
                print(f"   val_{name} ⊄ val_{prev_name} ✗ (차이: {len(diff)}개)")
        
        prev_set = current_set
        prev_name = name
    
    # 6. 최종 요약
    print("\n" + "=" * 60)
    print("Val 서브셋 생성 완료!")
    print("=" * 60)
    
    print(f"\n{'데이터셋':<10} {'Train':<15} {'Val':<15}")
    print("-" * 45)
    
    train_subsets = [
        ('5k', 5000),
        ('2k', 2000),
        ('1k', 1000),
        ('500', 500),
        ('250', 250),
    ]
    
    val_counts = [1000, 400, 200, 100, 50]
    
    for (name, train_count), val_count in zip(train_subsets, val_counts):
        print(f"{name:<10} {train_count:<15} {val_count:<15}")


if __name__ == '__main__':
    main()
