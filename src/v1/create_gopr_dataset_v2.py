"""
GOPR 데이터셋 생성 스크립트 v2
- Train: 100개 영상 × 50장 = 5,000장 (megafallv2/train)
- Val: 50개 영상 × 20장 = 1,000장 (megafallv2/train, Train과 다른 영상)
- Test: 18개 영상 × 56장 = 1,008장 (megafallv2/test)
- Train 서브셋: 5k, 2k, 1k, 500, 250
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

# 경로
SRC_TRAIN_IMAGES = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/images/train')
SRC_TRAIN_LABELS = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/labels/train')
SRC_TEST_IMAGES = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/images/test')
SRC_TEST_LABELS = Path('/Users/jihunjang/Downloads/ust/dataset/train/megafallv2/labels/test')

DST_BASE = Path('/Users/jihunjang/Downloads/ust/dataset/train/gopr')
YAMLS_DIR = Path('/Users/jihunjang/workspace/ust/fall-detection/src/v1/yamls')


def extract_video_number(filename):
    """파일명에서 GOPR 영상 번호 추출"""
    match = re.search(r'GOPR(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def get_gopr_videos(src_path, min_images):
    """GOPR 영상별 이미지 그룹화"""
    video_images = defaultdict(list)
    for f in os.listdir(src_path):
        if 'GOPR' in f.upper():
            video_num = extract_video_number(f)
            if video_num:
                video_images[video_num].append(f)
    return {v: imgs for v, imgs in video_images.items() if len(imgs) >= min_images}


def copy_images_and_labels(image_names, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """이미지와 라벨 복사"""
    for img_name in image_names:
        # 이미지 복사
        src_img = src_img_dir / img_name
        dst_img = dst_img_dir / img_name
        shutil.copy2(src_img, dst_img)
        
        # 라벨 복사
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        src_label = src_lbl_dir / label_name
        dst_label = dst_lbl_dir / label_name
        if src_label.exists():
            shutil.copy2(src_label, dst_label)


def create_txt_file(image_paths, txt_path):
    """이미지 경로 리스트를 txt 파일로 저장"""
    with open(txt_path, 'w') as f:
        for img_path in sorted(image_paths):
            f.write(f"{img_path}\n")


def create_yaml_file(yaml_path, train_txt, val_txt):
    """YAML 설정 파일 생성"""
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
    print("GOPR 데이터셋 생성 시작 (Seed 32)")
    print("=" * 60)
    
    # 1. 출력 디렉토리 생성
    dirs = [
        DST_BASE / 'images' / 'train',
        DST_BASE / 'images' / 'val',
        DST_BASE / 'images' / 'test',
        DST_BASE / 'labels' / 'train',
        DST_BASE / 'labels' / 'val',
        DST_BASE / 'labels' / 'test',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("\n1. 출력 디렉토리 생성 완료")
    
    # 2. Train 소스에서 영상 분배 (Train 100개, Val 50개)
    print("\n2. 영상 분배 중...")
    train_source_videos = get_gopr_videos(SRC_TRAIN_IMAGES, 50)
    video_list = list(train_source_videos.keys())
    random.shuffle(video_list)
    
    train_video_nums = video_list[:100]
    val_video_nums = video_list[100:150]
    
    print(f"   Train용 영상: {len(train_video_nums)}개")
    print(f"   Val용 영상: {len(val_video_nums)}개")
    
    # 3. Test 소스에서 영상
    test_source_videos = get_gopr_videos(SRC_TEST_IMAGES, 56)
    test_video_nums = list(test_source_videos.keys())
    print(f"   Test용 영상: {len(test_video_nums)}개")
    
    # 4. Train 이미지 선택 및 복사 (영상당 50장)
    print("\n3. Train 데이터 복사 중...")
    train_all_images = []  # 5k용 (전체)
    train_images_by_video = {}  # 서브셋 생성용
    
    for video_num in train_video_nums:
        images = train_source_videos[video_num]
        selected = random.sample(images, 50)
        train_images_by_video[video_num] = selected
        train_all_images.extend(selected)
    
    copy_images_and_labels(
        train_all_images,
        SRC_TRAIN_IMAGES, SRC_TRAIN_LABELS,
        DST_BASE / 'images' / 'train', DST_BASE / 'labels' / 'train'
    )
    print(f"   Train 완료: {len(train_all_images)}장")
    
    # 5. Val 이미지 선택 및 복사 (영상당 20장)
    print("\n4. Val 데이터 복사 중...")
    val_all_images = []
    
    for video_num in val_video_nums:
        images = train_source_videos[video_num]
        selected = random.sample(images, 20)
        val_all_images.extend(selected)
    
    copy_images_and_labels(
        val_all_images,
        SRC_TRAIN_IMAGES, SRC_TRAIN_LABELS,
        DST_BASE / 'images' / 'val', DST_BASE / 'labels' / 'val'
    )
    print(f"   Val 완료: {len(val_all_images)}장")
    
    # 6. Test 이미지 선택 및 복사 (영상당 56장)
    print("\n5. Test 데이터 복사 중...")
    test_all_images = []
    
    for video_num in test_video_nums:
        images = test_source_videos[video_num]
        selected = random.sample(images, 56)
        test_all_images.extend(selected)
    
    copy_images_and_labels(
        test_all_images,
        SRC_TEST_IMAGES, SRC_TEST_LABELS,
        DST_BASE / 'images' / 'test', DST_BASE / 'labels' / 'test'
    )
    print(f"   Test 완료: {len(test_all_images)}장")
    
    # 7. Train 서브셋 생성 (영상 수 고정 방식)
    print("\n6. Train 서브셋 txt 파일 생성 중...")
    
    # 서브셋 정의: (이름, 영상 수, 영상당 이미지 수)
    subsets = [
        ('5k', 100, 50),
        ('2k', 100, 20),
        ('1k', 100, 10),
        ('500', 100, 5),
        ('250', 50, 5),
    ]
    
    # 각 영상의 이미지를 한 번 셔플 (서브셋 관계 유지를 위해)
    for video_num in train_images_by_video:
        random.shuffle(train_images_by_video[video_num])
    
    # 250용 영상 50개 선택 (서브셋 관계 유지)
    videos_for_250 = train_video_nums[:50]
    
    for name, num_videos, imgs_per_video in subsets:
        if name == '250':
            videos_to_use = videos_for_250
        else:
            videos_to_use = train_video_nums[:num_videos]
        
        subset_images = []
        for video_num in videos_to_use:
            subset_images.extend(train_images_by_video[video_num][:imgs_per_video])
        
        # 이미지 경로 생성 (상대 경로)
        image_paths = [f"images/train/{img}" for img in subset_images]
        
        txt_path = YAMLS_DIR / f"train_gopr_{name}.txt"
        create_txt_file(image_paths, txt_path)
        print(f"   train_gopr_{name}.txt: {len(subset_images)}장")
    
    # 8. Val, Test txt 파일 생성
    print("\n7. Val/Test txt 파일 생성 중...")
    
    val_paths = [f"images/val/{img}" for img in val_all_images]
    create_txt_file(val_paths, YAMLS_DIR / "val_gopr.txt")
    print(f"   val_gopr.txt: {len(val_paths)}장")
    
    test_paths = [f"images/test/{img}" for img in test_all_images]
    create_txt_file(test_paths, YAMLS_DIR / "test_gopr.txt")
    print(f"   test_gopr.txt: {len(test_paths)}장")
    
    # 9. YAML 파일 생성
    print("\n8. YAML 파일 생성 중...")
    
    for name, _, _ in subsets:
        yaml_path = YAMLS_DIR / f"data_gopr_{name}.yaml"
        train_txt = YAMLS_DIR / f"train_gopr_{name}.txt"
        val_txt = YAMLS_DIR / "val_gopr.txt"
        create_yaml_file(yaml_path, train_txt, val_txt)
        print(f"   data_gopr_{name}.yaml")
    
    # 10. 데이터셋 정보 저장
    print("\n9. 데이터셋 정보 저장 중...")
    
    with open(DST_BASE / 'dataset_info.txt', 'w') as f:
        f.write(f"Seed: {SEED}\n\n")
        f.write(f"Train videos ({len(train_video_nums)}): {sorted(train_video_nums)}\n\n")
        f.write(f"Val videos ({len(val_video_nums)}): {sorted(val_video_nums)}\n\n")
        f.write(f"Test videos ({len(test_video_nums)}): {sorted(test_video_nums)}\n")
    
    # 11. 최종 요약
    print("\n" + "=" * 60)
    print("데이터셋 생성 완료!")
    print("=" * 60)
    
    print(f"\n[이미지/라벨 폴더]")
    print(f"  경로: {DST_BASE}")
    print(f"  - images/train: {len(list((DST_BASE / 'images' / 'train').iterdir()))}장")
    print(f"  - images/val:   {len(list((DST_BASE / 'images' / 'val').iterdir()))}장")
    print(f"  - images/test:  {len(list((DST_BASE / 'images' / 'test').iterdir()))}장")
    print(f"  - labels/train: {len(list((DST_BASE / 'labels' / 'train').iterdir()))}개")
    print(f"  - labels/val:   {len(list((DST_BASE / 'labels' / 'val').iterdir()))}개")
    print(f"  - labels/test:  {len(list((DST_BASE / 'labels' / 'test').iterdir()))}개")
    
    print(f"\n[txt/yaml 파일]")
    print(f"  경로: {YAMLS_DIR}")
    for name, _, _ in subsets:
        print(f"  - train_gopr_{name}.txt, data_gopr_{name}.yaml")
    print(f"  - val_gopr.txt, test_gopr.txt")


if __name__ == '__main__':
    main()
