from ultralytics import YOLO

model = YOLO('/Users/jihunjang/workspace/ust/fall-detection/src/v1/result/gopr_250/weights/best.pt')  # 검증/추론 모델 로드

results = model.predict(
    source="/Users/jihunjang/Downloads/ust/dataset/train/gopr/images/test",  # 평가할 이미지 폴더/파일
    save=True,                    # annotate 이미지 저장
    save_txt=True,                # txt로 bbox 저장
    save_crop=True                # 잘린 객체 이미지 저장
)