import os
import json
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

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
client = OpenAI(api_key=OPENAI_API_KEY)

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

def get_company_details(company_id):
    """회사 상세 정보 조회 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT 
        company_id, company_name, industry_code, industry_name, 
        company_size, region, annual_revenue, employees, 
        total_assets, total_capital, main_products, business_purpose
    FROM companies
    WHERE company_id = %s
    """, (company_id,))
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not result:
        return None
    
    return {
        "company_id": result[0],
        "company_name": result[1],
        "industry_code": result[2],
        "industry_name": result[3],
        "company_size": result[4],
        "region": result[5],
        "annual_revenue": result[6],
        "employees": result[7],
        "total_assets": result[8],
        "total_capital": result[9],
        "main_products": result[10],
        "business_purpose": result[11]
    }

def get_rnd_tech_details(tech_id):
    """R&D 기술 상세 정보 조회 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT 
        tech_id, tech_title, tech_category, tech_subcategory, 
        rnd_stage, region, tech_description, research_budget
    FROM rnd_technologies
    WHERE tech_id = %s
    """, (tech_id,))
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not result:
        return None
    
    return {
        "tech_id": result[0],
        "tech_title": result[1],
        "tech_category": result[2],
        "tech_subcategory": result[3],
        "rnd_stage": result[4],
        "region": result[5],
        "tech_description": result[6],
        "research_budget": result[7]
    }

def get_matching_details(company_id, tech_id):
    """회사와 R&D 기술 간의 매칭 세부 정보 조회 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 가중치 설정 조회
    cursor.execute("""
    SELECT 
        setting_name, text_similarity_weight, quantitative_weight, 
        category_weight, region_weight, size_weight
    FROM weight_settings
    WHERE is_active = TRUE
    LIMIT 1
    """)
    
    weight_result = cursor.fetchone()
    
    if not weight_result:
        weights = {
            "setting_name": "default",
            "text_similarity_weight": 0.7,
            "quantitative_weight": 0.3,
            "category_weight": 0.5,
            "region_weight": 0.2,
            "size_weight": 0.3
        }
    else:
        weights = {
            "setting_name": weight_result[0],
            "text_similarity_weight": weight_result[1],
            "quantitative_weight": weight_result[2],
            "category_weight": weight_result[3],
            "region_weight": weight_result[4],
            "size_weight": weight_result[5]
        }
    
    # 회사 정보 조회
    cursor.execute("""
    SELECT 
        c.company_id, c.company_name, c.industry_code, c.industry_name, 
        c.company_size, c.region, c.annual_revenue, c.employees, 
        c.total_assets, c.total_capital, array_to_string(c.main_products, ', ') as main_products,
        c.business_purpose
    FROM companies c
    WHERE c.company_id = %s
    """, (company_id,))
    
    company_result = cursor.fetchone()
    
    if not company_result:
        cursor.close()
        conn.close()
        return None
    
    company_info = {
        "company_id": company_result[0],
        "company_name": company_result[1],
        "industry_code": company_result[2],
        "industry_name": company_result[3],
        "company_size": company_result[4],
        "region": company_result[5],
        "annual_revenue": company_result[6],
        "employees": company_result[7],
        "total_assets": company_result[8],
        "total_capital": company_result[9],
        "main_products": company_result[10],
        "business_purpose": company_result[11]
    }
    
    # R&D 기술 정보 조회
    cursor.execute("""
    SELECT 
        t.tech_id, t.tech_title, t.tech_category, t.tech_subcategory, 
        t.rnd_stage, t.region, t.tech_description, t.research_budget
    FROM rnd_technologies t
    WHERE t.tech_id = %s
    """, (tech_id,))
    
    tech_result = cursor.fetchone()
    
    if not tech_result:
        cursor.close()
        conn.close()
        return None
    
    tech_info = {
        "tech_id": tech_result[0],
        "tech_title": tech_result[1],
        "tech_category": tech_result[2],
        "tech_subcategory": tech_result[3],
        "rnd_stage": tech_result[4],
        "region": tech_result[5],
        "tech_description": tech_result[6],
        "research_budget": tech_result[7]
    }
    
    # 텍스트 유사도 조회
    cursor.execute("""
    WITH company_embedding AS (
        SELECT embedding FROM company_embeddings 
        WHERE company_id = %s AND embedding_type = 'profile'
    ),
    tech_embedding AS (
        SELECT embedding FROM rnd_embeddings 
        WHERE tech_id = %s AND embedding_type = 'combined'
    )
    SELECT 1 - (c.embedding <=> t.embedding) AS text_similarity
    FROM company_embedding c, tech_embedding t
    """, (company_id, tech_id))
    
    similarity_result = cursor.fetchone()
    text_similarity = similarity_result[0] if similarity_result else 0.0
    
    # 지역 유사도 계산
    region_similarity = 1 if company_info["region"] == tech_info["region"] else 0
    
    # 산업-기술 카테고리 매핑
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
    
    # 카테고리 유사도 계산
    suitable_categories = industry_category_mapping.get(company_info["industry_code"], [])
    category_similarity = 1 if tech_info["tech_category"] in suitable_categories else 0
    
    # 기업 규모 가중치
    size_weights = {
        '대기업': 0.8,
        '중견기업': 1.0,
        '중소기업': 0.9,
        '스타트업': 0.7
    }
    
    size_factor = size_weights.get(company_info["company_size"], 0.5)
    
    # 최종 점수 계산
    final_score = (
        weights["text_similarity_weight"] * text_similarity +
        weights["category_weight"] * category_similarity +
        weights["region_weight"] * region_similarity +
        weights["size_weight"] * size_factor
    )
    
    # 매칭 세부 정보 반환
    matching_details = {
        "company_info": company_info,
        "tech_info": tech_info,
        "weights": weights,
        "scores": {
            "text_similarity": float(text_similarity),
            "category_similarity": category_similarity,
            "region_similarity": region_similarity,
            "size_factor": size_factor,
            "final_score": float(final_score)
        }
    }
    
    cursor.close()
    conn.close()
    
    return matching_details

def generate_matching_explanation(company_id, tech_id):
    """회사와 R&D 기술 간의 매칭 설명 생성 함수"""
    # 매칭 세부 정보 조회
    matching_details = get_matching_details(company_id, tech_id)
    
    if not matching_details:
        return {
            "success": False,
            "message": "회사 또는 R&D 기술 정보를 찾을 수 없습니다."
        }
    
    # 매칭 점수 및 세부 정보 추출
    company_info = matching_details["company_info"]
    tech_info = matching_details["tech_info"]
    scores = matching_details["scores"]
    
    # OpenAI API를 사용하여 매칭 설명 생성
    try:
        # 프롬프트 구성
        prompt = f"""
        당신은 기업과 R&D 기술 간의 매칭을 분석하고 설명하는 전문가입니다. 
        다음 정보를 바탕으로 이 기업이 해당 R&D 기술과 매칭되는 이유를 상세하게 분석해주세요.
        
        ## 기업 정보
        - 기업명: {company_info['company_name']}
        - 산업: {company_info['industry_name']} (코드: {company_info['industry_code']})
        - 규모: {company_info['company_size']}
        - 지역: {company_info['region']}
        - 직원 수: {company_info['employees']}명
        - 연간 매출: {company_info['annual_revenue']:,}원
        - 총자산: {company_info['total_assets']:,}원
        - 자본금: {company_info['total_capital']:,}원
        - 주요 제품: {company_info['main_products']}
        - 사업 목적: {company_info['business_purpose']}
        
        ## R&D 기술 정보
        - 기술명: {tech_info['tech_title']}
        - 기술 분야: {tech_info['tech_category']} / {tech_info['tech_subcategory']}
        - 연구 단계: {tech_info['rnd_stage']}
        - 지역: {tech_info['region']}
        - 지원 금액: {tech_info['research_budget']:,}원
        - 기술 설명: {tech_info['tech_description']}
        
        ## 매칭 점수
        - 텍스트 유사도: {scores['text_similarity']:.4f}
        - 카테고리 적합성: {scores['category_similarity']}
        - 지역 일치도: {scores['region_similarity']}
        - 기업 규모 적합성: {scores['size_factor']:.2f}
        - 최종 점수: {scores['final_score']:.4f}
        
        다음 항목을 포함하여 분석해주세요:
        1. 매칭 요약: 이 기업과 R&D 기술이 얼마나 잘 매칭되는지 전반적인 평가
        2. 산업 및 제품 적합성: 기업의 산업과 주요 제품이 R&D 기술과 어떻게 연관되는지
        3. 지역적 요소: 지역 일치 여부가 매칭에 미치는 영향
        4. 기업 규모 및 재무 상태: 기업의 규모와 재무 상태가 이 R&D 기술을 수행하기에 적합한지
        5. 강점 및 약점: 이 매칭의 주요 강점과 약점
        6. 종합 평가: 최종 점수를 바탕으로 한 종합적인 매칭 평가
        
        분석은 객관적이고 데이터에 기반한 내용이어야 하며, 기업과 R&D 기술의 특성을 모두 고려해야 합니다.
        """
        
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 기업과 R&D 기술 매칭을 분석하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        explanation = response.choices[0].message.content
        
        # 결과 반환
        return {
            "success": True,
            "company_id": company_id,
            "company_name": company_info["company_name"],
            "tech_id": tech_id,
            "tech_title": tech_info["tech_title"],
            "matching_score": float(scores["final_score"]),
            "explanation": explanation,
            "detailed_scores": scores
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"매칭 설명 생성 중 오류 발생: {str(e)}"
        }

def save_matching_explanation(company_id, tech_id, explanation_data):
    """매칭 설명 저장 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        INSERT INTO matching_explanations
        (company_id, tech_id, matching_score, explanation, detailed_scores)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (company_id, tech_id) DO UPDATE SET
            matching_score = EXCLUDED.matching_score,
            explanation = EXCLUDED.explanation,
            detailed_scores = EXCLUDED.detailed_scores,
            updated_at = CURRENT_TIMESTAMP
        """, (
            company_id,
            tech_id,
            explanation_data["matching_score"],
            explanation_data["explanation"],
            json.dumps(explanation_data["detailed_scores"])
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        conn.rollback()
        cursor.close()
        conn.close()
        
        print(f"매칭 설명 저장 중 오류 발생: {str(e)}")
        return False

def get_saved_explanation(company_id, tech_id):
    """저장된 매칭 설명 조회 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT 
        e.company_id, c.company_name, 
        e.tech_id, t.tech_title,
        e.matching_score, e.explanation, e.detailed_scores,
        e.created_at
    FROM matching_explanations e
    JOIN companies c ON e.company_id = c.company_id
    JOIN rnd_technologies t ON e.tech_id = t.tech_id
    WHERE e.company_id = %s AND e.tech_id = %s
    """, (company_id, tech_id))
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not result:
        return None
    
    return {
        "company_id": result[0],
        "company_name": result[1],
        "tech_id": result[2],
        "tech_title": result[3],
        "matching_score": result[4],
        "explanation": result[5],
        "detailed_scores": result[6],
        "created_at": result[7]
    }

def test_matching_explanation():
    """매칭 설명 생성 테스트 함수"""
    print("매칭 설명 생성 테스트 중...")
    
    # 테스트용 회사 ID와 기술 ID
    company_id = "COMP-00001"
    tech_id = "RND-00001"
    
    # 매칭 설명 생성
    explanation_data = generate_matching_explanation(company_id, tech_id)
    
    if explanation_data["success"]:
        print(f"\n회사 '{explanation_data['company_name']}'와 기술 '{explanation_data['tech_title']}' 간의 매칭 분석:")
        print(f"매칭 점수: {explanation_data['matching_score']:.4f}")
        print("\n매칭 설명:")
        print(explanation_data["explanation"])
        
        # 매칭 설명 저장
        save_result = save_matching_explanation(company_id, tech_id, explanation_data)
        print(f"\n매칭 설명 저장 결과: {'성공' if save_result else '실패'}")
        
        # 저장된 매칭 설명 조회
        saved_explanation = get_saved_explanation(company_id, tech_id)
        if saved_explanation:
            print(f"\n저장된 매칭 설명 조회 성공 (생성 시간: {saved_explanation['created_at']})")
        else:
            print("\n저장된 매칭 설명 조회 실패")
        
        return True
    else:
        print(f"매칭 설명 생성 실패: {explanation_data['message']}")
        return False

if __name__ == "__main__":
    test_matching_explanation()
