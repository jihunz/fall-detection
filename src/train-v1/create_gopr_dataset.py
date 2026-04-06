"""
GOPR 데이터셋 생성 스크립트
- Train: 100개 영상 × 50장 = 5,000장
- Test: 50개 영상 × 20장 = 1,000장
- 영상 간 겹침 없음
- Seed 32 사용
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
import re

# 설정
SEED = 32
TRAIN_VIDEOS = 100
TRAIN_IMAGES_PER_VIDEO = 50
TEST_VIDEOS = 50
TEST_IMAGES_PER_VIDEO = 20
MIN_IMAGES_FOR_TRAIN = 50  # Train용 영상 최소 이미지 수

# 경로
SRC_IMAGES = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/images/train')
SRC_LABELS = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/labels/train')
DST_BASE = Path('/Users/jihunjang/Downloads/ust/dataset/train/gopr')

DST_TRAIN_IMAGES = DST_BASE / 'images' / 'train'
DST_TEST_IMAGES = DST_BASE / 'images' / 'test'
DST_TRAIN_LABELS = DST_BASE / 'labels' / 'train'
DST_TEST_LABELS = DST_BASE / 'labels' / 'test'


def extract_video_number(filename):
    """파일명에서 GOPR 영상 번호 추출"""
    match = re.search(r'GOPR(\d+)', filename)
    if match:
        return match.group(1)
    return None


def main():
    random.seed(SEED)
    
    # 1. GOPR 이미지 수집 및 영상별 그룹화
    print("1. GOPR 이미지 수집 중...")
    video_images = defaultdict(list)
    
    for img_file in SRC_IMAGES.iterdir():
        if 'GOPR' in img_file.name.upper():
            video_num = extract_video_number(img_file.name)
            if video_num:
                video_images[video_num].append(img_file.name)
    
    print(f"   총 {len(video_images)}개 영상 발견")
    
    # 2. 50장 이상 보유 영상 필터링
    eligible_videos = [v for v, imgs in video_images.items() if len(imgs) >= MIN_IMAGES_FOR_TRAIN]
    print(f"   {MIN_IMAGES_FOR_TRAIN}장 이상 보유 영상: {len(eligible_videos)}개")
    
    # 3. 영상 셔플 (Seed 32)
    random.shuffle(eligible_videos)
    print(f"   Seed {SEED}로 셔플 완료")
    
    # 4. Train/Test 영상 분배
    train_videos = eligible_videos[:TRAIN_VIDEOS]
    test_videos = eligible_videos[TRAIN_VIDEOS:TRAIN_VIDEOS + TEST_VIDEOS]
    
    print(f"   Train 영상: {len(train_videos)}개")
    print(f"   Test 영상: {len(test_videos)}개")
    
    # 5. 출력 디렉토리 생성
    for d in [DST_TRAIN_IMAGES, DST_TEST_IMAGES, DST_TRAIN_LABELS, DST_TEST_LABELS]:
        d.mkdir(parents=True, exist_ok=True)
    print("2. 출력 디렉토리 생성 완료")
    
    # 6. Train 이미지/라벨 복사
    print("3. Train 데이터 복사 중...")
    train_count = 0
    for video_num in train_videos:
        images = video_images[video_num]
        selected = random.sample(images, TRAIN_IMAGES_PER_VIDEO)
        
        for img_name in selected:
            # 이미지 복사
            src_img = SRC_IMAGES / img_name
            dst_img = DST_TRAIN_IMAGES / img_name
            shutil.copy2(src_img, dst_img)
            
            # 라벨 복사
            label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            src_label = SRC_LABELS / label_name
            dst_label = DST_TRAIN_LABELS / label_name
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            
            train_count += 1
    
    print(f"   Train 완료: {train_count}장")
    
    # 7. Test 이미지/라벨 복사
    print("4. Test 데이터 복사 중...")
    test_count = 0
    for video_num in test_videos:
        images = video_images[video_num]
        selected = random.sample(images, TEST_IMAGES_PER_VIDEO)
        
        for img_name in selected:
            # 이미지 복사
            src_img = SRC_IMAGES / img_name
            dst_img = DST_TEST_IMAGES / img_name
            shutil.copy2(src_img, dst_img)
            
            # 라벨 복사
            label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            src_label = SRC_LABELS / label_name
            dst_label = DST_TEST_LABELS / label_name
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            
            test_count += 1
    
    print(f"   Test 완료: {test_count}장")
    
    # 8. 결과 요약
    print("\n" + "="*50)
    print("데이터셋 생성 완료!")
    print("="*50)
    print(f"Train: {train_count}장 ({TRAIN_VIDEOS}개 영상 × {TRAIN_IMAGES_PER_VIDEO}장)")
    print(f"Test:  {test_count}장 ({TEST_VIDEOS}개 영상 × {TEST_IMAGES_PER_VIDEO}장)")
    print(f"\n출력 경로: {DST_BASE}")
    print(f"  - images/train: {train_count}장")
    print(f"  - images/test:  {test_count}장")
    print(f"  - labels/train: {len(list(DST_TRAIN_LABELS.iterdir()))}개")
    print(f"  - labels/test:  {len(list(DST_TEST_LABELS.iterdir()))}개")
    
    # 9. 선택된 영상 번호 저장 (재현성)
    with open(DST_BASE / 'dataset_info.txt', 'w') as f:
        f.write(f"Seed: {SEED}\n")
        f.write(f"Train videos ({len(train_videos)}): {sorted(train_videos)}\n")
        f.write(f"Test videos ({len(test_videos)}): {sorted(test_videos)}\n")
    print(f"\n영상 정보 저장: {DST_BASE / 'dataset_info.txt'}")


if __name__ == '__main__':
    main()
