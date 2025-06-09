"""
LangChain 기반 데이터베이스 관리 모듈

이 모듈은 LangChain을 활용한 R&D 기술 추천 시스템의 데이터베이스 스키마 생성 및 관리 기능을 제공합니다.
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_config import APIConfig

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = APIConfig.OPENAI_API_KEY

class LangChainDatabaseManager:
    """
    LangChain 기반 데이터베이스 관리 클래스
    """
    
    def __init__(self):
        """데이터베이스 매니저 초기화"""
        self.db_params = APIConfig.get_db_connection_params()
        self.embeddings = OpenAIEmbeddings(
            model=APIConfig.OPENAI_EMBEDDING_MODEL,
            openai_api_key=APIConfig.OPENAI_API_KEY
        )
    
    def connect_to_db(self):
        """데이터베이스 연결"""
        return psycopg2.connect(**self.db_params)
    
    def create_database_schema(self):
        """데이터베이스 스키마 생성"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # PGVector 확장 활성화
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 산업 분류 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS industries (
                industry_id VARCHAR(20) PRIMARY KEY,
                industry_name VARCHAR(100) NOT NULL,
                industry_description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 기업 정보 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                company_id VARCHAR(20) PRIMARY KEY,
                company_name VARCHAR(100) NOT NULL,
                industry_id VARCHAR(20) REFERENCES industries(industry_id),
                company_size VARCHAR(20),
                region VARCHAR(50),
                employees INTEGER,
                annual_revenue BIGINT,
                total_assets BIGINT,
                total_capital BIGINT,
                main_products TEXT,
                business_purpose TEXT,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # R&D 기술 정보 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS rnd_technologies (
                tech_id VARCHAR(20) PRIMARY KEY,
                tech_title VARCHAR(200) NOT NULL,
                tech_category VARCHAR(100),
                tech_description TEXT,
                research_institution VARCHAR(100),
                tech_readiness_level INTEGER,
                expected_effect TEXT,
                application_field TEXT,
                keywords TEXT,
                embedding vector(1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 가중치 설정 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS weight_settings (
                id SERIAL PRIMARY KEY,
                setting_name VARCHAR(50) NOT NULL,
                text_similarity_weight FLOAT DEFAULT 0.5,
                quantitative_weight FLOAT DEFAULT 0.3,
                category_weight FLOAT DEFAULT 0.4,
                region_weight FLOAT DEFAULT 0.3,
                size_weight FLOAT DEFAULT 0.3,
                is_active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 추천 결과 로그 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_logs (
                id SERIAL PRIMARY KEY,
                recommendation_type VARCHAR(20) NOT NULL,
                source_id VARCHAR(20) NOT NULL,
                target_id VARCHAR(20) NOT NULL,
                similarity_score FLOAT,
                final_score FLOAT,
                weight_setting_id INTEGER REFERENCES weight_settings(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_companies_embedding ON companies USING ivfflat (embedding vector_cosine_ops);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rnd_technologies_embedding ON rnd_technologies USING ivfflat (embedding vector_cosine_ops);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies(industry_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_companies_region ON companies(region);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_companies_size ON companies(company_size);")
            
            # 기본 가중치 설정 추가
            cursor.execute("""
            INSERT INTO weight_settings (setting_name, text_similarity_weight, quantitative_weight, 
                                       category_weight, region_weight, size_weight, is_active)
            VALUES ('default', %s, %s, %s, %s, %s, TRUE)
            ON CONFLICT DO NOTHING;
            """, (
                APIConfig.DEFAULT_TEXT_SIMILARITY_WEIGHT,
                APIConfig.DEFAULT_QUANTITATIVE_WEIGHT,
                APIConfig.DEFAULT_CATEGORY_WEIGHT,
                APIConfig.DEFAULT_REGION_WEIGHT,
                APIConfig.DEFAULT_SIZE_WEIGHT
            ))
            
            conn.commit()
            print("데이터베이스 스키마가 성공적으로 생성되었습니다.")
            
        except Exception as e:
            print(f"스키마 생성 오류: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def create_sample_data(self):
        """샘플 데이터 생성"""
        # 산업 분류 샘플 데이터
        industries_data = [
            ("IND-001", "제조업", "제품 제조 및 생산 관련 산업"),
            ("IND-002", "정보통신업", "IT, 소프트웨어, 통신 관련 산업"),
            ("IND-003", "바이오헬스", "생명과학, 의료, 제약 관련 산업"),
            ("IND-004", "에너지", "신재생에너지, 전력 관련 산업"),
            ("IND-005", "화학", "화학물질, 소재 관련 산업"),
            ("IND-006", "기계", "기계 제조 및 자동화 관련 산업"),
            ("IND-007", "전자", "전자부품, 반도체 관련 산업"),
            ("IND-008", "자동차", "자동차 제조 및 부품 관련 산업"),
            ("IND-009", "건설", "건설, 건축 관련 산업"),
            ("IND-010", "서비스업", "각종 서비스 관련 산업")
        ]
        
        # 기업 샘플 데이터
        companies_data = [
            ("COMP-001", "테크이노베이션", "IND-002", "중소기업", "서울", 150, 50000000000, 30000000000, 15000000000, 
             "AI 기반 데이터 분석 솔루션, 클라우드 서비스", "인공지능과 빅데이터를 활용한 기업 디지털 전환 솔루션 제공"),
            ("COMP-002", "바이오메드", "IND-003", "중견기업", "경기", 300, 80000000000, 60000000000, 25000000000,
             "의료기기, 진단키트, 바이오 센서", "첨단 의료기술을 통한 질병 진단 및 치료 솔루션 개발"),
            ("COMP-003", "그린에너지", "IND-004", "중소기업", "부산", 80, 30000000000, 20000000000, 10000000000,
             "태양광 패널, 풍력 발전기, 에너지 저장 시스템", "신재생에너지 기술을 통한 지속가능한 에너지 솔루션 제공"),
            ("COMP-004", "스마트팩토리", "IND-006", "중견기업", "대구", 250, 70000000000, 50000000000, 20000000000,
             "산업용 로봇, 자동화 시스템, IoT 센서", "제조업 스마트화를 위한 자동화 및 IoT 솔루션 제공"),
            ("COMP-005", "나노소재", "IND-005", "중소기업", "대전", 120, 40000000000, 25000000000, 12000000000,
             "나노 복합소재, 기능성 코팅, 첨단 화학소재", "나노기술 기반 첨단 소재 개발 및 제조")
        ]
        
        # R&D 기술 샘플 데이터
        technologies_data = [
            ("RND-001", "AI 기반 예측 분석 플랫폼", "인공지능", 
             "머신러닝과 딥러닝을 활용한 고정밀 예측 분석 기술로, 다양한 산업 분야의 수요 예측, 품질 관리, 이상 탐지 등에 활용 가능",
             "한국과학기술원", 7, "생산성 30% 향상, 품질 불량률 50% 감소", "제조업, IT서비스업", "AI, 머신러닝, 예측분석"),
            ("RND-002", "차세대 바이오 센서 기술", "바이오기술",
             "나노기술과 바이오기술을 융합한 초고감도 바이오센서로, 질병 조기 진단 및 환경 모니터링에 활용 가능한 혁신 기술",
             "서울대학교", 6, "진단 정확도 95% 이상, 검사 시간 80% 단축", "의료, 환경", "바이오센서, 나노기술, 진단"),
            ("RND-003", "고효율 태양광 발전 시스템", "신재생에너지",
             "페로브스카이트 소재를 활용한 차세대 태양광 셀 기술로, 기존 대비 효율성 30% 향상 및 제조비용 40% 절감 가능",
             "한국에너지기술연구원", 8, "발전 효율 25% 향상, 설치비용 30% 절감", "에너지, 건설", "태양광, 페로브스카이트, 신재생에너지"),
            ("RND-004", "스마트 제조 자동화 시스템", "제조기술",
             "AI와 IoT를 결합한 지능형 제조 시스템으로, 실시간 품질 관리 및 예측 정비를 통한 생산성 극대화 기술",
             "한국생산기술연구원", 7, "생산성 40% 향상, 불량률 60% 감소", "제조업, 자동차", "스마트팩토리, IoT, 자동화"),
            ("RND-005", "친환경 나노 복합소재", "소재기술",
             "바이오 기반 나노 복합소재로 기존 플라스틱을 대체할 수 있는 친환경 소재 기술, 생분해성과 고강도를 동시에 구현",
             "한국화학연구원", 6, "환경 부하 70% 감소, 강도 20% 향상", "화학, 포장", "나노소재, 친환경, 생분해성")
        ]
        
        return industries_data, companies_data, technologies_data
    
    def load_sample_data_to_db(self):
        """샘플 데이터를 데이터베이스에 로드"""
        industries_data, companies_data, technologies_data = self.create_sample_data()
        
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # 산업 분류 데이터 삽입
            for industry in industries_data:
                cursor.execute("""
                INSERT INTO industries (industry_id, industry_name, industry_description)
                VALUES (%s, %s, %s)
                ON CONFLICT (industry_id) DO NOTHING;
                """, industry)
            
            # 기업 데이터 삽입 (임베딩 제외)
            for company in companies_data:
                cursor.execute("""
                INSERT INTO companies (company_id, company_name, industry_id, company_size, region,
                                     employees, annual_revenue, total_assets, total_capital,
                                     main_products, business_purpose)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (company_id) DO NOTHING;
                """, company)
            
            # R&D 기술 데이터 삽입 (임베딩 제외)
            for tech in technologies_data:
                cursor.execute("""
                INSERT INTO rnd_technologies (tech_id, tech_title, tech_category, tech_description,
                                            research_institution, tech_readiness_level, expected_effect,
                                            application_field, keywords)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tech_id) DO NOTHING;
                """, tech)
            
            conn.commit()
            print("샘플 데이터가 성공적으로 로드되었습니다.")
            
        except Exception as e:
            print(f"샘플 데이터 로드 오류: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def generate_embeddings_for_companies(self):
        """기업 데이터에 대한 임베딩 생성"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # 임베딩이 없는 기업 데이터 조회
            cursor.execute("""
            SELECT c.company_id, c.company_name, i.industry_name, c.company_size,
                   c.region, c.main_products, c.business_purpose
            FROM companies c
            JOIN industries i ON c.industry_id = i.industry_id
            WHERE c.embedding IS NULL
            """)
            
            companies = cursor.fetchall()
            
            for company in companies:
                # 기업 정보를 텍스트로 구성
                company_text = f"""
                기업명: {company[1]}
                산업분야: {company[2]}
                기업규모: {company[3]}
                지역: {company[4]}
                주요제품: {company[5]}
                사업목적: {company[6]}
                """
                
                # LangChain을 사용하여 임베딩 생성
                embedding = self.embeddings.embed_query(company_text.strip())
                
                # 임베딩을 데이터베이스에 저장
                cursor.execute("""
                UPDATE companies SET embedding = %s WHERE company_id = %s
                """, (embedding, company[0]))
                
                print(f"기업 {company[1]} 임베딩 생성 완료")
            
            conn.commit()
            print("모든 기업 임베딩 생성이 완료되었습니다.")
            
        except Exception as e:
            print(f"기업 임베딩 생성 오류: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def generate_embeddings_for_technologies(self):
        """R&D 기술 데이터에 대한 임베딩 생성"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # 임베딩이 없는 기술 데이터 조회
            cursor.execute("""
            SELECT tech_id, tech_title, tech_category, tech_description,
                   research_institution, expected_effect, application_field, keywords
            FROM rnd_technologies
            WHERE embedding IS NULL
            """)
            
            technologies = cursor.fetchall()
            
            for tech in technologies:
                # 기술 정보를 텍스트로 구성
                tech_text = f"""
                기술명: {tech[1]}
                기술분야: {tech[2]}
                기술설명: {tech[3]}
                연구기관: {tech[4]}
                기대효과: {tech[5]}
                적용분야: {tech[6]}
                키워드: {tech[7]}
                """
                
                # LangChain을 사용하여 임베딩 생성
                embedding = self.embeddings.embed_query(tech_text.strip())
                
                # 임베딩을 데이터베이스에 저장
                cursor.execute("""
                UPDATE rnd_technologies SET embedding = %s WHERE tech_id = %s
                """, (embedding, tech[0]))
                
                print(f"기술 {tech[1]} 임베딩 생성 완료")
            
            conn.commit()
            print("모든 R&D 기술 임베딩 생성이 완료되었습니다.")
            
        except Exception as e:
            print(f"기술 임베딩 생성 오류: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def setup_complete_database(self):
        """전체 데이터베이스 설정 완료"""
        print("LangChain 기반 R&D 추천 시스템 데이터베이스 설정을 시작합니다...")
        
        # 1. 스키마 생성
        print("1. 데이터베이스 스키마 생성 중...")
        self.create_database_schema()
        
        # 2. 샘플 데이터 로드
        print("2. 샘플 데이터 로드 중...")
        self.load_sample_data_to_db()
        
        # 3. 기업 임베딩 생성
        print("3. 기업 데이터 임베딩 생성 중...")
        self.generate_embeddings_for_companies()
        
        # 4. 기술 임베딩 생성
        print("4. R&D 기술 데이터 임베딩 생성 중...")
        self.generate_embeddings_for_technologies()
        
        print("LangChain 기반 R&D 추천 시스템 데이터베이스 설정이 완료되었습니다!")

# 편의 함수들
def create_database_schema():
    """데이터베이스 스키마 생성 편의 함수"""
    manager = LangChainDatabaseManager()
    manager.create_database_schema()

def load_sample_data():
    """샘플 데이터 로드 편의 함수"""
    manager = LangChainDatabaseManager()
    manager.load_sample_data_to_db()

def generate_all_embeddings():
    """모든 임베딩 생성 편의 함수"""
    manager = LangChainDatabaseManager()
    manager.generate_embeddings_for_companies()
    manager.generate_embeddings_for_technologies()

def setup_complete_database():
    """전체 데이터베이스 설정 편의 함수"""
    manager = LangChainDatabaseManager()
    manager.setup_complete_database()

if __name__ == "__main__":
    setup_complete_database()

