import json


DEFAULT_MODEL = "qwen2.5"

SYSTEM_PROMPT = """
너는 영수증 기반 소비 습관 분석가이자 영양 코치야.
주어진 영수증 분석 결과를 바탕으로 JSON 형식으로만 답해.

요구사항:
1. 구매 품목과 카테고리를 보고 소비 습관을 짧게 분석해.
2. 음식, 음료, 카페, 편의점, 식사, 마트 품목은 영양 관점에서 평가해.
3. 칼로리, 당, 나트륨, 지방이 높아 보이는 품목이 있으면 주의점을 알려줘.
4. 내일 먹으면 좋을 식단을 추천해.

반드시 아래 JSON 형식으로만 답해.
{
  "habit_analysis": "소비 습관 분석",
  "nutrition_analysis": "영양 분석",
  "diet_recommendation": "내일 추천 식단",
  "warnings": ["주의할 점"]
}
"""


def receipt_json_to_text(receipt_result):
    # OCR 결과 JSON을 LLM이 읽기 쉬운 문장 형태로 변환
    parts = []

    store_name = receipt_result.get("store_name")
    purchased_at = receipt_result.get("purchased_at")
    total = receipt_result.get("total")

    if store_name:
        parts.append(f"상호명: {store_name}")

    if purchased_at:
        parts.append(f"구매일시: {purchased_at}")

    if total is not None:
        parts.append(f"총액: {total}원")

    for row in receipt_result.get("rows", []):
        item_name = row.get("item_name", "")
        price = row.get("price", "")
        category = row.get("category", "")
        parts.append(f"품목: {item_name}, 가격: {price}원, 카테고리: {category}")

    return "\n".join(parts).strip()


def extract_response_json(text):
    # 모델이 코드블록이나 설명을 섞어도 JSON 부분만 최대한 추출
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "", 1).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("LLM 응답에서 JSON을 찾을 수 없습니다.")

    return json.loads(text[start : end + 1])


def analyze_receipt_with_llm(receipt_result, model_name=DEFAULT_MODEL):
    # Ollama의 Qwen 모델로 OCR 결과를 소비/영양 관점에서 분석
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("ollama 패키지가 설치되어 있지 않습니다. pip install ollama 를 실행해주세요.") from exc

    raw_text = receipt_json_to_text(receipt_result)
    if not raw_text:
        raise ValueError("분석할 영수증 내용이 없습니다.")

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"다음 영수증 분석 결과를 보고 평가해줘:\n{raw_text}"},
        ],
        format="json",
    )

    return extract_response_json(response["message"]["content"])


def analyze_receipt_file(input_file, output_file, model_name=DEFAULT_MODEL):
    # 파일로 저장된 OCR 결과를 분석하고 결과 JSON 파일로 저장
    with open(input_file, "r", encoding="utf-8") as f:
        receipt_result = json.load(f)

    result = analyze_receipt_with_llm(receipt_result, model_name)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return result
