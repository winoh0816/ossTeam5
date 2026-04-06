import cv2
import easyocr
import os
import numpy as np
import torch
import json

# ==============================
# 1. 전처리 함수: 한글 인식률 향상
# ==============================
def preprocess_receipt(img):
    """
    그레이스케일 변환, 노이즈 제거, 적응형 이진화를 통해 
    영수증의 글자를 배경과 명확히 분리합니다.
    """
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 노이즈 제거 (흐릿한 글자 보정)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 3. 적응형 이진화 (불균일한 조명 해결)
    # 주변 픽셀과의 차이를 이용해 글자를 추출합니다.
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary

# ==============================
# 2. 메인 OCR 처리 함수
# ==============================
def extract_receipt_text(image_path, output_file, reader):
    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"파일 없음: {image_path}")
        return

    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return
    
    # [개선] 전처리 적용
    # 전처리된 이미지는 한글 획을 더 뚜렷하게 만들어줍니다.
    processed_img = preprocess_receipt(img)

    print(f"OCR 분석 시작: {os.path.basename(image_path)}")

    # [개선] OCR 실행 파라미터 최적화
    # mag_ratio=2.0: 이미지를 2배 확대하여 작은 글씨 인식률 향상
    results = reader.readtext(processed_img, detail=1, mag_ratio=2.0)

    all_texts = []
    
    # 결과 파싱
    for (bbox, text, conf) in results:
        clean_text = text.strip()
        
        if len(clean_text) > 0:
            # bbox 좌표를 JSON 저장 가능한 형태로 변환
            box_coords = [list(map(float, point)) for point in bbox]
            
            all_texts.append({
                "content": clean_text,
                "confidence": float(round(conf, 4)),
                "bbox": box_coords
            })
            print(f"[Detected] {clean_text} (Conf: {conf:.2f})")

    # 결과 데이터 구조화
    output_data = {
        "file_name": os.path.basename(image_path),
        "total_count": len(all_texts),
        "results": all_texts
    }

    # JSON 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"[완료] OCR 결과 저장: {output_file}")


# ==============================
# 3. 실행 부분 (테스트용 10개)
# ==============================
if __name__ == "__main__":
    # 경로 설정
    input_dir = 'yubin/ossTeam5/receiptIMG'
    output_dir = 'yubin/ossTeam5/receipt_output'
    
    os.makedirs(output_dir, exist_ok=True)

    # GPU 사용 여부 확인 및 Reader 초기화 (루프 밖에서 한 번만 실행)
    use_gpu = torch.cuda.is_available()
    print(f"GPU 활성화 상태: {use_gpu}")
    # 한글(ko)과 영어(en) 동시 인식 설정
    reader = easyocr.Reader(['ko', 'en'], gpu=use_gpu)

    # 이미지 파일 목록 확보
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
    
    # 상위 10개만 테스트 슬라이싱
    image_files = all_files[:10] 

    if not image_files:
        print(f"'{input_dir}' 폴더에 이미지 파일이 없습니다.")
    else:
        print(f"테스트 시작: 총 {len(all_files)}개 중 {len(image_files)}개 처리 예정")

        for i, filename in enumerate(image_files):
            img_path = os.path.join(input_dir, filename)
            json_name = os.path.splitext(filename)[0] + "_OCR.json"
            json_path = os.path.join(output_dir, json_name)

            print(f"\n[{i+1}/{len(image_files)}] 처리 중: {filename}")
            
            try:
                extract_receipt_text(img_path, json_path, reader)
            except Exception as e:
                print(f"에러 발생 ({filename}): {e}")

        print("\n" + "="*40)
        print(f"테스트 완료! 결과 저장 위치: {output_dir}")