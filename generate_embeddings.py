import os
import json
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import openai

# 환경 변수 로드
load_dotenv()

# 데이터베이스 연결 정보
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# OpenAI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
openai.api_key = OPENAI_API_KEY

def connect_to_db():
    """데이터베이스 연결 함수"""
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

def preprocess_text(text):
    """텍스트 데이터 전처리 함수"""
    if not text:
        return ""
    
    # 기본적인 전처리 (공백 정리, 줄바꿈 처리 등)
    text = text.strip()
    text = ' '.join(text.split())
    
    # 최대 토큰 수 제한 (OpenAI 임베딩 모델 제한 고려)
    # text-embedding-3-small 모델은 약 8191 토큰까지 처리 가능
    # 간단한 휴리스틱: 평균적으로 한 토큰은 약 4글자로 가정
    max_chars = 8000 * 4
    if len(text) > max_chars:
        text = text[:max_chars]
    
    return text

def create_embedding(text, model="text-embedding-3-small"):
    """OpenAI API를 사용하여 텍스트 임베딩 생성 함수"""
    if not text or text.strip() == "":
        return None
    
    try:
        # 텍스트 전처리
        processed_text = preprocess_text(text)
        
        # OpenAI API 호출하여 임베딩 생성
        response = openai.Embedding.create(
            input=processed_text,
            model=model
        )
        
        # 임베딩 벡터 추출
        embedding = response.data[0].embedding
        
        return embedding
    
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return None

def create_rnd_tech_embedding(conn, tech_id, embedding_type="description"):
    """공공 R&D 기술 임베딩 생성 및 저장 함수"""
    cursor = conn.cursor()
    
    # 기술 데이터 조회
    cursor.execute("SELECT tech_title, tech_description, tech_category FROM rnd_technologies WHERE tech_id = %s", (tech_id,))
    result = cursor.fetchone()
    
    if not result:
        print(f"기술 ID {tech_id}에 해당하는 데이터를 찾을 수 없습니다.")
        return False
    
    tech_title, tech_description, tech_category = result
    
    # 임베딩 생성 대상 텍스트 선택
    if embedding_type == "title":
        text = tech_title
    elif embedding_type == "description":
        text = tech_description
    elif embedding_type == "combined":
        text = f"제목: {tech_title}\n카테고리: {tech_category}\n내용: {tech_description}"
    else:
        print(f"지원하지 않는 임베딩 타입: {embedding_type}")
        return False
    
    # 임베딩 생성
    embedding = create_embedding(text)
    
    if embedding:
        # 임베딩 저장
        cursor.execute("""
        INSERT INTO rnd_embeddings (tech_id, embedding_type, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (tech_id, embedding_type) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            created_at = CURRENT_TIMESTAMP
        """, (tech_id, embedding_type, embedding))
        
        conn.commit()
        return True
    else:
        print(f"기술 ID {tech_id}의 임베딩 생성에 실패했습니다.")
        return False

def create_company_embedding(conn, company_id, embedding_type="profile"):
    """기업 프로파일 임베딩 생성 및 저장 함수"""
    cursor = conn.cursor()
    
    # 기업 데이터 조회
    cursor.execute("""
    SELECT company_name, industry_name, company_size, business_purpose, 
           array_to_string(main_products, ', ') as products
    FROM companies WHERE company_id = %s
    """, (company_id,))
    result = cursor.fetchone()
    
    if not result:
        print(f"기업 ID {company_id}에 해당하는 데이터를 찾을 수 없습니다.")
        return False
    
    company_name, industry_name, company_size, business_purpose, products = result
    
    # 임베딩 생성 대상 텍스트 선택
    if embedding_type == "name":
        text = company_name
    elif embedding_type == "purpose":
        text = business_purpose
    elif embedding_type == "profile":
        text = f"기업명: {company_name}\n산업: {industry_name}\n규모: {company_size}\n주요제품: {products}\n사업목적: {business_purpose}"
    else:
        print(f"지원하지 않는 임베딩 타입: {embedding_type}")
        return False
    
    # 임베딩 생성
    embedding = create_embedding(text)
    
    if embedding:
        # 임베딩 저장
        cursor.execute("""
        INSERT INTO company_embeddings (company_id, embedding_type, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (company_id, embedding_type) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            created_at = CURRENT_TIMESTAMP
        """, (company_id, embedding_type, embedding))
        
        conn.commit()
        return True
    else:
        print(f"기업 ID {company_id}의 임베딩 생성에 실패했습니다.")
        return False

def generate_all_embeddings(batch_size=10):
    """모든 데이터에 대한 임베딩 생성 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 공공 R&D 기술 데이터 임베딩 생성
    print("공공 R&D 기술 데이터 임베딩 생성 중...")
    cursor.execute("SELECT tech_id FROM rnd_technologies")
    tech_ids = [row[0] for row in cursor.fetchall()]
    
    total_techs = len(tech_ids)
    success_count = 0
    
    for i, tech_id in enumerate(tech_ids):
        if i % batch_size == 0:
            print(f"기술 데이터 임베딩 진행 중: {i}/{total_techs}")
        
        success = create_rnd_tech_embedding(conn, tech_id, "combined")
        if success:
            success_count += 1
    
    print(f"공공 R&D 기술 데이터 임베딩 생성 완료: {success_count}/{total_techs}")
    
    # 기업 프로파일 데이터 임베딩 생성
    print("\n기업 프로파일 데이터 임베딩 생성 중...")
    cursor.execute("SELECT company_id FROM companies")
    company_ids = [row[0] for row in cursor.fetchall()]
    
    total_companies = len(company_ids)
    success_count = 0
    
    for i, company_id in enumerate(company_ids):
        if i % batch_size == 0:
            print(f"기업 데이터 임베딩 진행 중: {i}/{total_companies}")
        
        success = create_company_embedding(conn, company_id, "profile")
        if success:
            success_count += 1
    
    print(f"기업 프로파일 데이터 임베딩 생성 완료: {success_count}/{total_companies}")
    
    # 연결 종료
    cursor.close()
    conn.close()
    print("데이터베이스 연결이 종료되었습니다.")

def test_embedding_generation():
    """임베딩 생성 테스트 함수"""
    print("임베딩 생성 테스트 중...")
    
    # 테스트 텍스트
    test_text = "인공지능 기반 스마트팩토리 최적화 시스템"
    
    # 임베딩 생성
    embedding = create_embedding(test_text)
    
    if embedding:
        print(f"임베딩 생성 성공! 벡터 차원: {len(embedding)}")
        print(f"임베딩 벡터 일부: {embedding[:5]}...")
        return True
    else:
        print("임베딩 생성 실패!")
        return False

if __name__ == "__main__":
    # OpenAI API 연결 테스트
    print("OpenAI API 연결 테스트 중...")
    test_result = test_embedding_generation()
    
    if test_result:
        # 사용자 입력 받기
        user_input = input("모든 데이터에 대한 임베딩을 생성하시겠습니까? (y/n): ")
        
        if user_input.lower() == 'y':
            generate_all_embeddings()
        else:
            print("임베딩 생성을 건너뜁니다.")
    else:
        print("OpenAI API 연결 테스트에 실패했습니다. 환경 변수와 API 키를 확인하세요.")
