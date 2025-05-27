import os
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

def get_active_weight_settings():
    """현재 활성화된 가중치 설정 조회 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, setting_name, text_similarity_weight, quantitative_weight, 
           category_weight, region_weight, size_weight
    FROM weight_settings
    WHERE is_active = TRUE
    LIMIT 1
    """)
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if result:
        return {
            "id": result[0],
            "setting_name": result[1],
            "text_similarity_weight": result[2],
            "quantitative_weight": result[3],
            "category_weight": result[4],
            "region_weight": result[5],
            "size_weight": result[6]
        }
    else:
        # 활성화된 설정이 없으면 기본값 반환
        return {
            "id": None,
            "setting_name": "default",
            "text_similarity_weight": 0.7,
            "quantitative_weight": 0.3,
            "category_weight": 0.5,
            "region_weight": 0.2,
            "size_weight": 0.3
        }

def update_weight_settings(setting_name, text_similarity_weight=0.7, quantitative_weight=0.3, 
                          category_weight=0.5, region_weight=0.2, size_weight=0.3):
    """가중치 설정 업데이트 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 기존 설정 비활성화
    cursor.execute("UPDATE weight_settings SET is_active = FALSE")
    
    # 새 설정 추가 또는 업데이트
    cursor.execute("""
    INSERT INTO weight_settings 
    (setting_name, description, text_similarity_weight, quantitative_weight, 
     category_weight, region_weight, size_weight, is_active)
    VALUES 
    (%s, %s, %s, %s, %s, %s, %s, TRUE)
    ON CONFLICT (setting_name) DO UPDATE SET
        text_similarity_weight = EXCLUDED.text_similarity_weight,
        quantitative_weight = EXCLUDED.quantitative_weight,
        category_weight = EXCLUDED.category_weight,
        region_weight = EXCLUDED.region_weight,
        size_weight = EXCLUDED.size_weight,
        is_active = TRUE,
        updated_at = CURRENT_TIMESTAMP
    """, (
        setting_name, 
        f"{setting_name} 가중치 설정", 
        text_similarity_weight, 
        quantitative_weight,
        category_weight,
        region_weight,
        size_weight
    ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"가중치 설정 '{setting_name}'이(가) 업데이트되었습니다.")
    return True

def recommend_rnd_tech_for_company(company_id, top_n=10):
    """기업 프로파일에 적합한 공공 R&D 기술 추천 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 가중치 설정 조회
    weights = get_active_weight_settings()
    
    # 기업 정보 조회
    cursor.execute("""
    SELECT c.company_id, c.company_name, c.industry_code, c.industry_name, 
           c.company_size, c.region, c.annual_revenue, c.employees,
           ce.embedding
    FROM companies c
    JOIN company_embeddings ce ON c.company_id = ce.company_id
    WHERE c.company_id = %s AND ce.embedding_type = 'profile'
    """, (company_id,))
    
    company_info = cursor.fetchone()
    
    if not company_info:
        print(f"기업 ID {company_id}에 해당하는 데이터를 찾을 수 없습니다.")
        cursor.close()
        conn.close()
        return []
    
    company_embedding = company_info[8]
    company_region = company_info[5]
    company_size = company_info[4]
    
    # 1. 텍스트 유사도 기반 추천 (PGVector 활용)
    cursor.execute("""
    WITH similarity_results AS (
        SELECT 
            t.tech_id,
            t.tech_title,
            t.tech_category,
            t.tech_subcategory,
            t.rnd_stage,
            t.region,
            1 - (e.embedding <=> %s) AS text_similarity
        FROM 
            rnd_technologies t
        JOIN 
            rnd_embeddings e ON t.tech_id = e.tech_id
        WHERE 
            e.embedding_type = 'combined'
        ORDER BY 
            text_similarity DESC
        LIMIT 100
    )
    SELECT * FROM similarity_results
    """, (company_embedding,))
    
    similarity_results = cursor.fetchall()
    
    # 결과를 DataFrame으로 변환
    columns = ['tech_id', 'tech_title', 'tech_category', 'tech_subcategory', 
               'rnd_stage', 'region', 'text_similarity']
    df = pd.DataFrame(similarity_results, columns=columns)
    
    # 2. 지역 유사도 계산 (같은 지역이면 1, 다르면 0)
    df['region_similarity'] = df['region'].apply(lambda x: 1 if x == company_region else 0)
    
    # 3. 카테고리 가중치 적용 (산업 코드와 기술 카테고리 간의 관계 매핑)
    industry_category_mapping = {
        'A': ['스마트팩토리', '신재생에너지', '바이오헬스'],  # 농업, 임업 및 어업
        'B': ['나노소재', '신재생에너지'],  # 광업
        'C': ['스마트팩토리', '로봇공학', '나노소재', '반도체', '디스플레이'],  # 제조업
        'D': ['신재생에너지', '스마트팩토리'],  # 전기, 가스, 증기 및 공기 조절 공급업
        'E': ['신재생에너지', '바이오헬스'],  # 수도, 하수 및 폐기물 처리, 원료 재생업
        'F': ['스마트팩토리', '로봇공학'],  # 건설업
        'G': ['빅데이터', '인공지능', '블록체인'],  # 도매 및 소매업
        'J': ['인공지능', '빅데이터', '클라우드', '사물인터넷', '블록체인', '통신기술', '보안기술'],  # 정보통신업
        'K': ['빅데이터', '인공지능', '블록체인', '보안기술'],  # 금융 및 보험업
        'M': ['인공지능', '빅데이터', '클라우드', '사물인터넷', '블록체인', '로봇공학', '자율주행']  # 전문, 과학 및 기술 서비스업
    }
    
    # 기업의 산업 코드
    industry_code = company_info[2]
    
    # 산업 코드에 해당하는 적합한 기술 카테고리 목록
    suitable_categories = industry_category_mapping.get(industry_code, [])
    
    # 카테고리 유사도 계산 (적합한 카테고리면 1, 아니면 0)
    df['category_similarity'] = df['tech_category'].apply(
        lambda x: 1 if x in suitable_categories else 0
    )
    
    # 4. 최종 점수 계산 (가중치 적용)
    df['final_score'] = (
        weights['text_similarity_weight'] * df['text_similarity'] +
        weights['category_weight'] * df['category_similarity'] +
        weights['region_weight'] * df['region_similarity']
    )
    
    # 최종 점수로 정렬
    df = df.sort_values('final_score', ascending=False)
    
    # 상위 N개 결과 반환
    top_results = df.head(top_n)
    
    # 추천 결과 저장
    for idx, row in top_results.iterrows():
        cursor.execute("""
        INSERT INTO recommendations
        (source_id, target_id, recommendation_type, similarity_score, rank)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source_id, target_id, recommendation_type) DO UPDATE SET
            similarity_score = EXCLUDED.similarity_score,
            rank = EXCLUDED.rank,
            created_at = CURRENT_TIMESTAMP
        """, (
            company_id,
            row['tech_id'],
            'company_to_rnd',
            float(row['final_score']),
            idx + 1
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return top_results.to_dict('records')

def recommend_companies_for_rnd_tech(tech_id, top_n=10):
    """공공 R&D 기술에 적합한 수요 기업 추천 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 가중치 설정 조회
    weights = get_active_weight_settings()
    
    # 기술 정보 조회
    cursor.execute("""
    SELECT t.tech_id, t.tech_title, t.tech_category, t.tech_subcategory, 
           t.rnd_stage, t.region, e.embedding
    FROM rnd_technologies t
    JOIN rnd_embeddings e ON t.tech_id = e.tech_id
    WHERE t.tech_id = %s AND e.embedding_type = 'combined'
    """, (tech_id,))
    
    tech_info = cursor.fetchone()
    
    if not tech_info:
        print(f"기술 ID {tech_id}에 해당하는 데이터를 찾을 수 없습니다.")
        cursor.close()
        conn.close()
        return []
    
    tech_embedding = tech_info[6]
    tech_region = tech_info[5]
    tech_category = tech_info[2]
    
    # 1. 텍스트 유사도 기반 추천 (PGVector 활용)
    cursor.execute("""
    WITH similarity_results AS (
        SELECT 
            c.company_id,
            c.company_name,
            c.industry_code,
            c.industry_name,
            c.company_size,
            c.region,
            c.annual_revenue,
            c.employees,
            c.total_assets,
            c.total_capital,
            array_to_string(c.main_products, ', ') as main_products,
            1 - (e.embedding <=> %s) AS text_similarity
        FROM 
            companies c
        JOIN 
            company_embeddings e ON c.company_id = e.company_id
        WHERE 
            e.embedding_type = 'profile'
        ORDER BY 
            text_similarity DESC
        LIMIT 100
    )
    SELECT * FROM similarity_results
    """, (tech_embedding,))
    
    similarity_results = cursor.fetchall()
    
    # 결과를 DataFrame으로 변환
    columns = ['company_id', 'company_name', 'industry_code', 'industry_name', 
               'company_size', 'region', 'annual_revenue', 'employees', 
               'total_assets', 'total_capital', 'main_products', 'text_similarity']
    df = pd.DataFrame(similarity_results, columns=columns)
    
    # 2. 지역 유사도 계산 (같은 지역이면 1, 다르면 0)
    df['region_similarity'] = df['region'].apply(lambda x: 1 if x == tech_region else 0)
    
    # 3. 산업 카테고리 유사도 계산
    industry_category_mapping = {
        'A': ['스마트팩토리', '신재생에너지', '바이오헬스'],  # 농업, 임업 및 어업
        'B': ['나노소재', '신재생에너지'],  # 광업
        'C': ['스마트팩토리', '로봇공학', '나노소재', '반도체', '디스플레이'],  # 제조업
        'D': ['신재생에너지', '스마트팩토리'],  # 전기, 가스, 증기 및 공기 조절 공급업
        'E': ['신재생에너지', '바이오헬스'],  # 수도, 하수 및 폐기물 처리, 원료 재생업
        'F': ['스마트팩토리', '로봇공학'],  # 건설업
        'G': ['빅데이터', '인공지능', '블록체인'],  # 도매 및 소매업
        'J': ['인공지능', '빅데이터', '클라우드', '사물인터넷', '블록체인', '통신기술', '보안기술'],  # 정보통신업
        'K': ['빅데이터', '인공지능', '블록체인', '보안기술'],  # 금융 및 보험업
        'M': ['인공지능', '빅데이터', '클라우드', '사물인터넷', '블록체인', '로봇공학', '자율주행']  # 전문, 과학 및 기술 서비스업
    }
    
    # 각 산업 코드별로 기술 카테고리와의 적합성 계산
    df['category_similarity'] = df['industry_code'].apply(
        lambda x: 1 if tech_category in industry_category_mapping.get(x, []) else 0
    )
    
    # 4. 기업 규모에 따른 가중치 적용
    size_weights = {
        '대기업': 0.8,
        '중견기업': 1.0,
        '중소기업': 0.9,
        '스타트업': 0.7
    }
    
    df['size_factor'] = df['company_size'].apply(lambda x: size_weights.get(x, 0.5))
    
    # 5. 최종 점수 계산 (가중치 적용)
    df['final_score'] = (
        weights['text_similarity_weight'] * df['text_similarity'] +
        weights['category_weight'] * df['category_similarity'] +
        weights['region_weight'] * df['region_similarity'] +
        weights['size_weight'] * df['size_factor']
    )
    
    # 최종 점수로 정렬
    df = df.sort_values('final_score', ascending=False)
    
    # 상위 N개 결과 반환
    top_results = df.head(top_n)
    
    # 추천 결과 저장
    for idx, row in top_results.iterrows():
        cursor.execute("""
        INSERT INTO recommendations
        (source_id, target_id, recommendation_type, similarity_score, rank)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source_id, target_id, recommendation_type) DO UPDATE SET
            similarity_score = EXCLUDED.similarity_score,
            rank = EXCLUDED.rank,
            created_at = CURRENT_TIMESTAMP
        """, (
            tech_id,
            row['company_id'],
            'rnd_to_company',
            float(row['final_score']),
            idx + 1
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return top_results.to_dict('records')

def get_recommendation_results(source_id, recommendation_type, limit=10):
    """저장된 추천 결과 조회 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    if recommendation_type == 'company_to_rnd':
        # 기업에 대한 R&D 기술 추천 결과 조회
        cursor.execute("""
        SELECT r.rank, r.similarity_score, 
               t.tech_id, t.tech_title, t.tech_category, t.tech_subcategory, 
               t.rnd_stage, t.region
        FROM recommendations r
        JOIN rnd_technologies t ON r.target_id = t.tech_id
        WHERE r.source_id = %s AND r.recommendation_type = %s
        ORDER BY r.rank
        LIMIT %s
        """, (source_id, recommendation_type, limit))
        
        columns = ['rank', 'similarity_score', 'tech_id', 'tech_title', 
                  'tech_category', 'tech_subcategory', 'rnd_stage', 'region']
    
    elif recommendation_type == 'rnd_to_company':
        # R&D 기술에 대한 기업 추천 결과 조회
        cursor.execute("""
        SELECT r.rank, r.similarity_score, 
               c.company_id, c.company_name, c.industry_name, c.company_size, 
               c.region, c.annual_revenue, c.employees, c.total_assets, c.total_capital
        FROM recommendations r
        JOIN companies c ON r.target_id = c.company_id
        WHERE r.source_id = %s AND r.recommendation_type = %s
        ORDER BY r.rank
        LIMIT %s
        """, (source_id, recommendation_type, limit))
        
        columns = ['rank', 'similarity_score', 'company_id', 'company_name', 
                  'industry_name', 'company_size', 'region', 'annual_revenue', 
                  'employees', 'total_assets', 'total_capital']
    
    else:
        print(f"지원하지 않는 추천 유형: {recommendation_type}")
        cursor.close()
        conn.close()
        return []
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results, columns=columns)
    
    return df.to_dict('records')

def test_recommendation_engine():
    """추천 엔진 테스트 함수"""
    print("추천 엔진 테스트 중...")
    
    # 가중치 설정 테스트
    print("\n1. 가중치 설정 테스트")
    update_weight_settings(
        "test_weights", 
        text_similarity_weight=0.6, 
        quantitative_weight=0.4,
        category_weight=0.4,
        region_weight=0.3,
        size_weight=0.3
    )
    
    weights = get_active_weight_settings()
    print(f"현재 활성화된 가중치 설정: {weights}")
    
    # 기업에 대한 R&D 기술 추천 테스트
    print("\n2. 기업에 대한 R&D 기술 추천 테스트")
    company_id = "COMP-00001"  # 테스트용 기업 ID
    rnd_recommendations = recommend_rnd_tech_for_company(company_id, top_n=5)
    
    print(f"기업 {company_id}에 대한 R&D 기술 추천 결과:")
    for i, rec in enumerate(rnd_recommendations, 1):
        print(f"{i}. {rec['tech_title']} (유사도: {rec['final_score']:.4f})")
    
    # R&D 기술에 대한 기업 추천 테스트
    print("\n3. R&D 기술에 대한 기업 추천 테스트")
    tech_id = "RND-00001"  # 테스트용 기술 ID
    company_recommendations = recommend_companies_for_rnd_tech(tech_id, top_n=5)
    
    print(f"R&D 기술 {tech_id}에 대한 기업 추천 결과:")
    for i, rec in enumerate(company_recommendations, 1):
        print(f"{i}. {rec['company_name']} (유사도: {rec['final_score']:.4f})")
    
    # 저장된 추천 결과 조회 테스트
    print("\n4. 저장된 추천 결과 조회 테스트")
    saved_recommendations = get_recommendation_results(company_id, "company_to_rnd", limit=3)
    
    print(f"기업 {company_id}에 대한 저장된 R&D 기술 추천 결과:")
    for i, rec in enumerate(saved_recommendations, 1):
        print(f"{i}. {rec['tech_title']} (순위: {rec['rank']}, 유사도: {rec['similarity_score']:.4f})")
    
    return True

if __name__ == "__main__":
    test_recommendation_engine()
