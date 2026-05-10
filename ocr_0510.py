import json
import os
import re
import sqlite3
from pathlib import Path

from google.api_core.exceptions import GoogleAPIError, ResourceExhausted
import google.generativeai as genai
import PIL.Image


DB_PATH = Path("nutrition.db")
PROMPT = """
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


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("JSON 형식을 찾지 못했습니다.")

    return json.loads(text[start : end + 1])


def to_int(value):
    if value is None:
        return None

    if isinstance(value, int):
        return value

    text = str(value)
    numbers = re.sub(r"[^0-9]", "", text)
    return int(numbers) if numbers else None


def clean_item_name(item_name):
    text = item_name or ""
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"(스타벅스|투썸|이디야|메가커피|컴포즈|빽다방|CU|GS25|세븐일레븐)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^0-9A-Za-z가-힣 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_nutrition_match(conn, item_name):
    clean_name = clean_item_name(item_name)
    if not clean_name:
        return None

    keywords = [clean_name]
    words = [word for word in clean_name.split() if len(word) >= 2]
    keywords.extend(words)

    for keyword in keywords:
        row = conn.execute(
            """
            SELECT id
            FROM nutrition_items
            WHERE food_name LIKE ? OR representative_name LIKE ?
            ORDER BY
                CASE
                    WHEN food_name = ? THEN 0
                    WHEN representative_name = ? THEN 1
                    ELSE 2
                END,
                LENGTH(food_name)
            LIMIT 1
            """,
            (f"%{keyword}%", f"%{keyword}%", keyword, keyword),
        ).fetchone()

        if row:
            return row[0]

    return None


def save_receipt_result(conn, image_path, result):
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO receipts (image_file, store_name, purchased_at, total)
        VALUES (?, ?, ?, ?)
        """,
        (
            str(image_path),
            result.get("store_name"),
            result.get("purchased_at"),
            to_int(result.get("total")),
        ),
    )
    receipt_id = cur.lastrowid

    for item in result.get("rows", []):
        item_name = item.get("item_name")
        if not item_name:
            continue

        matched_nutrition_id = find_nutrition_match(conn, item_name)

        cur.execute(
            """
            INSERT INTO receipt_items (
                receipt_id,
                item_name,
                price,
                category,
                matched_nutrition_id
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                item_name,
                to_int(item.get("price")),
                item.get("category"),
                matched_nutrition_id,
            ),
        )

    conn.commit()
    return receipt_id


def main():
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

    # 3. 실제 처리할 영수증 이미지 목록 불러오기
    # '영수증 이미지' 폴더 안의 jpg 파일을 모두 읽습니다.
    image_folder = Path("영수증 이미지")
    image_files = sorted(image_folder.glob("*.jpg"))

    if not image_files:
        raise FileNotFoundError(f"영수증 이미지가 없습니다: {image_folder}")

    if not DB_PATH.exists():
        raise FileNotFoundError("nutrition.db가 없습니다. 먼저 create_db.py와 import_nutrition.py를 실행해주세요.")

    # 4. 전체 영수증 추출 결과 확인
    print(f"총 {len(image_files)}개의 영수증 이미지를 추출합니다.\n")

    conn = sqlite3.connect(DB_PATH)

    for index, image_path in enumerate(image_files, start=1):
        print("=" * 60)
        print(f"[{index}/{len(image_files)}] 추출 이미지: {image_path}")

        try:
            img = PIL.Image.open(image_path)
            response = model.generate_content([PROMPT, img])
            result = extract_json(response.text)
            receipt_id = save_receipt_result(conn, image_path, result)

            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"DB 저장 완료: receipt_id={receipt_id}")
        except ResourceExhausted:
            print(
                "Gemini API 무료 사용량 또는 분당 요청 제한을 초과했습니다.\n"
                "잠시 후 다시 실행하거나, Google AI Studio에서 결제/할당량 설정을 확인해주세요."
            )
            break
        except (json.JSONDecodeError, ValueError) as error:
            print(f"JSON 변환에 실패했습니다: {error}")
            print(response.text)
        except (OSError, GoogleAPIError) as error:
            print(f"이 이미지는 처리하지 못했습니다: {error}")

    conn.close()


if __name__ == "__main__":
    main()
