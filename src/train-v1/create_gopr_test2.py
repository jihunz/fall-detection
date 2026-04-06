"""
GOPR test2 데이터셋 생성
- 1854, 1860 영상 전체 제외
- 지정된 15장 제외
- 나머지 영상은 50장에 맞추기 위해 무작위(Seed 32) 제외
- 결과: 16개 영상 x 50장 = 800장
"""

import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 32

SRC_IMAGES = Path('/Users/jihunjang/Downloads/ust/dataset/train/gopr/images/test')
SRC_LABELS = Path('/Users/jihunjang/Downloads/ust/dataset/train/gopr/labels/test')
DST_IMAGES = Path('/Users/jihunjang/Downloads/ust/dataset/train/gopr/images/test2')
DST_LABELS = Path('/Users/jihunjang/Downloads/ust/dataset/train/gopr/labels/test2')

EXCLUDE_VIDEOS = {'1854', '1860'}
EXCLUDE_FILES = {
    'images_GOPR2096_00350.jpg', 'images_GOPR2096_02171.jpg',
    'images_GOPR2096_02619.jpg', 'images_GOPR2096_07491.jpg',
    'images_GOPR2710_00926.jpg', 'images_GOPR2710_01020.jpg',
    'images_GOPR2710_01021.jpg', 'images_GOPR2710_01067.jpg',
    'images_GOPR2710_01489.jpg', 'images_GOPR2710_01560.jpg',
    'images_GOPR2712_01108.jpg', 'images_GOPR2712_01353.jpg',
    'images_GOPR2712_01368.jpg', 'images_GOPR2712_01533.jpg',
    'images_GOPR2712_07447.jpg',
}

TARGET_PER_VIDEO = 50


def main():
    random.seed(SEED)

    DST_IMAGES.mkdir(parents=True, exist_ok=True)
    DST_LABELS.mkdir(parents=True, exist_ok=True)

    # 1. 영상별 이미지 수집 (제외 대상 필터링)
    video_images = defaultdict(list)
    for f in sorted(os.listdir(SRC_IMAGES)):
        match = re.search(r'GOPR(\d+)', f)
        if not match:
            continue
        vid = match.group(1)
        if vid in EXCLUDE_VIDEOS or f in EXCLUDE_FILES:
            continue
        video_images[vid].append(f)

    print(f"대상 영상: {len(video_images)}개")

    # 2. 영상당 50장 선택 후 복사
    total = 0
    for vid in sorted(video_images.keys()):
        images = video_images[vid]
        if len(images) > TARGET_PER_VIDEO:
            selected = random.sample(images, TARGET_PER_VIDEO)
        else:
            selected = images

        for img_name in selected:
            shutil.copy2(SRC_IMAGES / img_name, DST_IMAGES / img_name)
            label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            src_label = SRC_LABELS / label_name
            if src_label.exists():
                shutil.copy2(src_label, DST_LABELS / label_name)

        total += len(selected)
        print(f"  GOPR{vid}: {len(images)}장 → {len(selected)}장")

    print(f"\n완료: {total}장 (images/test2, labels/test2)")


if __name__ == '__main__':
    main()
