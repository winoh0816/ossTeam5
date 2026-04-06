# https://github.com/langchain-ai/langchain.git #LLM과 구축한 DB(Pandas DataFrame 등)를 연결할 때 사용하는 프레임워크의 GITHUB 코드페이지
# 아래 코드는 만일 위 모델을 사용한다 가정하에 만든 임의의 코드 (참고용으로만 작성했습니다!)
!pip install langchain langchain-openai langchain-experimental pandas python-dotenv
import os
import pandas as pd
from dotenv import load_dotenv # API 키에 대한 보안 설정 라이브러리의 함수
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. 환경 변수 로드 (API KEY)
load_dotenv()

def analyze_receipt_data(csv_path, user_query):
    # 2. 데이터 로드 (OCR 결과물)
    df = pd.read_csv(csv_path)
    
    # 3. 모델 설정 (GPT-4o 등 권장)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 4. 에이전트 생성
    # allow_dangerous_code=True: 데이터 분석을 위해 파이썬 코드를 실행하도록 허용
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True
    )
    
    # 5. 질문 실행 및 답변 반환
    response = agent.invoke(user_query) # 실행 코드
    return response['output']
