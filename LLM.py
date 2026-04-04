import os
import subprocess
import sys
import importlib
import json

# 프로그램 확인 및 환경 자동 세팅
def ensure_setup():
    #ollama 유무 확인 및 자동 설치
    try:
        importlib.import_module('ollama')
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])

    # 기본 입출력 폴더 생성성
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for folder in ['data', 'output']:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


    # qwen 다운로드 및 실행 점검
    try:
        subprocess.run(
            ['ollama', 'pull', 'qwen2.5'],
            check=True,

            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        print("\n ollama가 실행 중이지 않거나 설치되지 않음")
        sys.exit(1)

# 환경 세팅 우선 실행
ensure_setup()

# 환경 세팅 완료 후 로드
import ollama

# 영수증 텍스트 전달 후 영양 분석 및 식단 추천
def analyze_receipt_local_llm(input_file, output_file, model_name='qwen2.5'):

    # 입력 파일 확인
    if not os.path.exists(input_file):
        print(f"파일 없음 : {input_file}")
        return
    
    # ocr 결과 JSON 데이터 읽기
    with open(input_file, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    # 텍스트 데이터 문장으로 병합
    extracted_texts = []
    for frame_data in ocr_data:
        for text_info in frame_data.get("texts", []):
            extracted_texts.append(text_info.get("content", ""))

    raw_text = " ".join(extracted_texts).strip()

    # 텍스트 가 비어있으면 분석 중단
    if not raw_text:
        print("추출된 글자가 없어 분석 불가")
        return
    
    # AI 프롬프트
    system_prompt = """
    너는 가계부 분석 및 전문 영양사야.
    주어진 영수증 텍스트를 분석해서 아래 요구사항에 맞게 JSON 형식으로만 대답해.

    1. 품목 분류: 각 품목의 '이름'과 '가격'을 찾고, 카테고리를 [음식, 취미, 의류, 생필품, 기타] 중 하나로 정확히 분류해.
    2. 영양 분석: 카테고리가 '음식'인 경우 대략적인 칼로리(kcal), 당(g), 나트륨(mg), 지방(g)을 추정해. (음식이 아니면 모두 0으로 작성)
    3. 소비 습관 분석: 구매한 전체 품목과 카테고리 지출 비율을 바탕으로 사용자의 현재 소비 습관을 평가하고 조언해.
    4. 식단 추천: 분석된 음식의 영양 성분(칼로리, 당, 나트륨, 지방 등)을 바탕으로 내일 먹으면 좋을 건강한 식단을 추천해.

    결과는 무조건 아래 JSON 형식으로만 대답해야 해.
    {
        "items": [
            {
                "name": "품목명",
                "price": 1000,
                "category": "음식",
                "nutrition": {
                    "calories": 500,
                    "sugar": 10,
                    "sodium": 800,
                    "fat": 15
                }
            }
        ],
        "habit_analysis": "생필품보다 외식 지출 비율이 높습니다. 충동구매 여부를 점검해보는 것이 좋습니다.",
        "diet_recommendation": "나트륨과 지방 섭취가 많은 편입니다. 내일은 칼륨이 풍부한 채소 샐러드 위주의 저염 식단을 추천합니다."
    }
    """
    print(f"가게부 분류 및 종합 분석 중...")
    print("대기중\n")

    # 영수증 텍스트트
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"다음 영수증 텍스트를 분석해줘:\n{raw_text}"}
            ],
            format='json'
        )
        # json으로 객체 변환
        result_text = response['message']['content']
        final_data = json.loads(result_text)

        # 최종 결과물을 지정된 파일로 저장
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
            
        print(f"[완료] 결과 저장: {output_file}")
        
        # 분석된 주요 내용(소비 습관, 식단 추천)을 터미널에 깔끔하게 출력
        print("-" * 50)
        print("[소비 습관 분석]")
        print(final_data.get('habit_analysis', '분석 내용 없음'))
        print("\n[AI 영양사 식단 추천]")
        print(final_data.get('diet_recommendation', '추천 내용 없음'))
        print("-" * 50)

    except Exception as e:
        print(f"분석 중 에러 발생: {e}")
    

# 입출력 파일 경로 설정 및 분석 함수 실행
input_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'OCR_output.json')
output_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'FINAL_Qwen_output.json')

if __name__ == "__main__":
    analyze_receipt_local_llm(input_json, output_json)







