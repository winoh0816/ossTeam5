import cv2
import easyocr
import os
import numpy as np
import torch
import json


def extract_text_ultimate_performance(video_path, output_file, batch_size=16):

    if not os.path.exists(video_path):
        print(f"파일 없음: {video_path}")
        return

    # GPU 사용 여부 확인
    use_gpu = torch.cuda.is_available()
    print(f"GPU 활성화 상태: {use_gpu}")

    # EasyOCR Reader 초기화
    reader = easyocr.Reader(['en'], gpu=use_gpu)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"전 프레임 조사 시작 (Total: {total_frames} frames)")

    all_results = []

    frames_batch = []
    frame_ids = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # ROI 설정 (하단 40%)
        h, w, _ = frame.shape
        roi = frame

        # RGB 변환 (EasyOCR 권장)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        frames_batch.append(roi_rgb)
        frame_ids.append(i)

        # 배치 실행 조건
        if len(frames_batch) == batch_size or i == total_frames - 1:

            try:
                results_batch = reader.readtext_batched(
                    frames_batch,
                    detail=1  # bbox, text, confidence 반환
                )

                for idx, results in enumerate(results_batch):
                    current_frame_id = frame_ids[idx]
                    timestamp = current_frame_id / fps

                    frame_texts = []

                    for bbox, text, conf in results:
                        clean_text = text.strip()

                        # 최소 길이 필터
                        if len(clean_text) >= 1:
                            continue

                        frame_texts.append({
                            "content": clean_text,
                            "confidence": float(round(conf, 4))
                        })

                        print(f"[Frame {current_frame_id}] {clean_text} ({conf:.2f})")

                    if frame_texts:
                        all_results.append({
                            "timestamp": round(timestamp, 3),
                            "texts": frame_texts
                        })

            except Exception as e:
                print(f"Batch 처리 중 에러 발생: {e}")

            # 배치 초기화
            frames_batch = []
            frame_ids = []

        if i % 500 == 0 and total_frames > 0:
            print(f"진행률: {(i / total_frames) * 100:.1f}% 처리 중...")

    cap.release()

    # JSON 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n[완료] OCR 결과 저장: {output_file}")
    print(f"총 OCR 기록 개수: {len(all_results)}")


# ==============================
# 실행 부분
# ==============================

input_video = '/home/work/bigdata/yubin/yubin/ocr/data/sample_video.mp4'
output_json = '/home/work/bigdata/yubin/yubin/ocr/output/OCR_output.json'

if __name__ == "__main__":
    extract_text_ultimate_performance(input_video, output_json, batch_size=16)