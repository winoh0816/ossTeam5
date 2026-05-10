import os
from pathlib import Path

from google.api_core.exceptions import ResourceExhausted
import google.generativeai as genai
import PIL.Image


# 1. 구글 Gemini API 키 설정
# PowerShell 예시:
# $env:GEMINI_API_KEY="여기에_API_키"
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit(
        "GEMINI_API_KEY 환경변수가 없습니다.\n"
        'PowerShell에서 $env:GEMINI_API_KEY="여기에_API_키" 를 먼저 실행해주세요.'
    )

genai.configure(api_key=api_key)


# 2. Gemini 2.5 Flash 모델 설정
model = genai.GenerativeModel("gemini-2.5-flash")


# 3. 테스트용 영수증 이미지 불러오기
# OCR이 잘 되는지 확인할 이미지 한 장만 지정합니다.
# 다른 사진을 테스트하려면 아래 파일명만 바꿔주세요.
image_path = Path("영수증 이미지") / "112.jpg"

if not image_path.exists():
    raise FileNotFoundError(f"영수증 이미지가 없습니다: {image_path}")

img = PIL.Image.open(image_path)


# 4. 프롬프트 전달: 영수증 인식 + 컬럼 추출 요청
prompt = """
이 영수증 사진을 읽고 지출 분석용 데이터만 추출해주세요.
긴 분석이나 조언은 하지 말고, 아래 컬럼에 맞는 JSON만 출력해주세요.

추출할 컬럼:
1. store_name: 상호명
2. purchased_at: 구매 일시
3. item_name: 항목명
4. price: 항목 가격
5. category: 항목 카테고리
6. total: 영수증 총 합계 금액

카테고리는 아래 중 가장 가까운 값 하나만 선택해주세요.
- 카페
- 편의점
- 식사
- 마트
- 교통
- 생활용품
- 의료
- 문화/여가
- 기타

반드시 아래 JSON 형식으로만 출력해주세요.
읽기 어려운 값은 null로 표시해주세요.
금액은 쉼표 없이 숫자로만 작성해주세요.

{
  "store_name": "상호명 또는 null",
  "purchased_at": "구매 일시 또는 null",
  "total": 0,
  "rows": [
    {
      "item_name": "항목명",
      "price": 0,
      "category": "카테고리"
    }
  ]
}
"""


# 5. 결과 확인
print(f"테스트 이미지: {image_path}")

try:
    response = model.generate_content([prompt, img])
    print(response.text)
except ResourceExhausted:
    print(
        "Gemini API 무료 사용량 또는 분당 요청 제한을 초과했습니다.\n"
        "잠시 후 다시 실행하거나, Google AI Studio에서 결제/할당량 설정을 확인해주세요."
    )
