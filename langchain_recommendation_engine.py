"""
LangChain 기반 R&D 기술 추천 엔진

이 모듈은 LangChain의 PGVector와 OpenAIEmbeddings를 활용하여 
기업 프로파일에 적합한 공공 R&D 기술을 추천하고, 
보유 기술을 사업화할 수 있는 적합한 수요 기업을 예측하는 기능을 제공합니다.
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain.vectorstores import PGVector
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from langchain_config import APIConfig

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = APIConfig.OPENAI_API_KEY

class LangChainRecommendationEngine:
    """
    LangChain 기반 R&D 기술 추천 엔진 클래스
    """
    
    def __init__(self):
        """추천 엔진 초기화"""
        # 데이터베이스 연결 정보
        self.db_params = APIConfig.get_db_connection_params()
        self.connection_string = APIConfig.get_connection_string()
        
        # LangChain 컴포넌트 초기화
        self.embeddings = OpenAIEmbeddings(
            model=APIConfig.OPENAI_EMBEDDING_MODEL,
            openai_api_key=APIConfig.OPENAI_API_KEY
        )
        
        # LLM 초기화 (추천 사유 생성용)
        self.llm = OpenAI(
            model_name=APIConfig.OPENAI_COMPLETION_MODEL,
            temperature=0,
            openai_api_key=APIConfig.OPENAI_API_KEY
        )
        
        # 벡터스토어 (필요시 초기화)
        self.tech_vectorstore = None
        self.company_vectorstore = None
    
    def connect_to_db(self):
        """데이터베이스 연결 함수"""
        return psycopg2.connect(**self.db_params)
    
    def get_active_weight_settings(self) -> Dict[str, Any]:
        """현재 활성화된 가중치 설정 조회"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            SELECT id, setting_name, text_similarity_weight, quantitative_weight, 
                   category_weight, region_weight, size_weight
            FROM weight_settings
            WHERE is_active = TRUE
            LIMIT 1
            """)
            
            result = cursor.fetchone()
            
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
                # 기본 가중치 반환
                return {
                    "id": None,
                    "setting_name": "default",
                    "text_similarity_weight": APIConfig.DEFAULT_TEXT_SIMILARITY_WEIGHT,
                    "quantitative_weight": APIConfig.DEFAULT_QUANTITATIVE_WEIGHT,
                    "category_weight": APIConfig.DEFAULT_CATEGORY_WEIGHT,
                    "region_weight": APIConfig.DEFAULT_REGION_WEIGHT,
                    "size_weight": APIConfig.DEFAULT_SIZE_WEIGHT
                }
        finally:
            cursor.close()
            conn.close()
    
    def update_weight_settings(self, setting_name: str, **weights) -> bool:
        """가중치 설정 업데이트"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # 기존 활성 설정 비활성화
            cursor.execute("UPDATE weight_settings SET is_active = FALSE")
            
            # 새 설정 추가
            cursor.execute("""
            INSERT INTO weight_settings 
            (setting_name, text_similarity_weight, quantitative_weight, 
             category_weight, region_weight, size_weight, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE)
            """, (
                setting_name,
                weights.get('text_similarity_weight', APIConfig.DEFAULT_TEXT_SIMILARITY_WEIGHT),
                weights.get('quantitative_weight', APIConfig.DEFAULT_QUANTITATIVE_WEIGHT),
                weights.get('category_weight', APIConfig.DEFAULT_CATEGORY_WEIGHT),
                weights.get('region_weight', APIConfig.DEFAULT_REGION_WEIGHT),
                weights.get('size_weight', APIConfig.DEFAULT_SIZE_WEIGHT)
            ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"가중치 설정 업데이트 오류: {e}")
            conn.rollback()
            return False
        finally:
            cursor.close()
            conn.close()
    
    def _get_tech_vectorstore(self):
        """R&D 기술 벡터스토어 가져오기"""
        if self.tech_vectorstore is None:
            try:
                self.tech_vectorstore = PGVector(
                    collection_name="rnd_technologies",
                    connection_string=self.connection_string,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"기술 벡터스토어 연결 실패: {e}")
                return None
        return self.tech_vectorstore
    
    def _get_company_vectorstore(self):
        """기업 벡터스토어 가져오기"""
        if self.company_vectorstore is None:
            try:
                self.company_vectorstore = PGVector(
                    collection_name="companies",
                    connection_string=self.connection_string,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"기업 벡터스토어 연결 실패: {e}")
                return None
        return self.company_vectorstore
    
    def recommend_rnd_tech_for_company(self, company_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        기업에 적합한 R&D 기술 추천
        
        Args:
            company_id (str): 기업 ID
            top_k (int): 추천할 기술 수
            
        Returns:
            List[Dict]: 추천 기술 목록
        """
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # 기업 정보 조회
            cursor.execute("""
            SELECT c.company_id, c.company_name, c.industry_id, i.industry_name,
                   c.company_size, c.region, c.employees, c.annual_revenue,
                   c.main_products, c.business_purpose, c.embedding
            FROM companies c
            JOIN industries i ON c.industry_id = i.industry_id
            WHERE c.company_id = %s
            """, (company_id,))
            
            company_data = cursor.fetchone()
            if not company_data:
                return []
            
            # 기업 임베딩 벡터
            company_embedding = company_data[10]
            
            # 가중치 설정 가져오기
            weights = self.get_active_weight_settings()
            
            # 유사한 R&D 기술 검색 (벡터 유사도 기반)
            cursor.execute("""
            SELECT t.tech_id, t.tech_title, t.tech_category, t.tech_description,
                   t.research_institution, t.tech_readiness_level, t.expected_effect,
                   t.embedding,
                   (1 - (t.embedding <=> %s)) as similarity_score
            FROM rnd_technologies t
            ORDER BY t.embedding <=> %s
            LIMIT %s
            """, (company_embedding, company_embedding, top_k * 3))  # 더 많이 가져와서 필터링
            
            tech_results = cursor.fetchall()
            
            # 최종 스코어 계산 및 정렬
            recommendations = []
            for tech in tech_results:
                # 카테고리 매칭 보너스
                category_bonus = self._calculate_category_bonus(
                    company_data[2], tech[2], weights['category_weight']
                )
                
                # 최종 스코어 계산
                final_score = (
                    tech[8] * weights['text_similarity_weight'] +  # 텍스트 유사도
                    category_bonus  # 카테고리 보너스
                )
                
                recommendations.append({
                    'tech_id': tech[0],
                    'tech_title': tech[1],
                    'tech_category': tech[2],
                    'tech_description': tech[3],
                    'research_institution': tech[4],
                    'tech_readiness_level': tech[5],
                    'expected_effect': tech[6],
                    'similarity_score': tech[8],
                    'final_score': final_score
                })
            
            # 최종 스코어로 정렬하고 상위 k개 반환
            recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            print(f"R&D 기술 추천 오류: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def recommend_companies_for_rnd_tech(self, tech_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        R&D 기술에 적합한 기업 추천
        
        Args:
            tech_id (str): 기술 ID
            top_k (int): 추천할 기업 수
            
        Returns:
            List[Dict]: 추천 기업 목록
        """
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # R&D 기술 정보 조회
            cursor.execute("""
            SELECT tech_id, tech_title, tech_category, tech_description,
                   research_institution, tech_readiness_level, expected_effect, embedding
            FROM rnd_technologies
            WHERE tech_id = %s
            """, (tech_id,))
            
            tech_data = cursor.fetchone()
            if not tech_data:
                return []
            
            # 기술 임베딩 벡터
            tech_embedding = tech_data[7]
            
            # 가중치 설정 가져오기
            weights = self.get_active_weight_settings()
            
            # 유사한 기업 검색 (벡터 유사도 기반)
            cursor.execute("""
            SELECT c.company_id, c.company_name, c.industry_id, i.industry_name,
                   c.company_size, c.region, c.employees, c.annual_revenue,
                   c.total_assets, c.total_capital, c.main_products, c.business_purpose,
                   c.embedding,
                   (1 - (c.embedding <=> %s)) as similarity_score
            FROM companies c
            JOIN industries i ON c.industry_id = i.industry_id
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """, (tech_embedding, tech_embedding, top_k * 3))
            
            company_results = cursor.fetchall()
            
            # 최종 스코어 계산 및 정렬
            recommendations = []
            for company in company_results:
                # 카테고리 매칭 보너스
                category_bonus = self._calculate_category_bonus(
                    company[2], tech_data[2], weights['category_weight']
                )
                
                # 지역 매칭 보너스
                region_bonus = self._calculate_region_bonus(
                    company[5], weights['region_weight']
                )
                
                # 기업 규모 보너스
                size_bonus = self._calculate_size_bonus(
                    company[4], company[6], weights['size_weight']
                )
                
                # 최종 스코어 계산
                final_score = (
                    company[13] * weights['text_similarity_weight'] +  # 텍스트 유사도
                    category_bonus +  # 카테고리 보너스
                    region_bonus +    # 지역 보너스
                    size_bonus        # 규모 보너스
                )
                
                recommendations.append({
                    'company_id': company[0],
                    'company_name': company[1],
                    'industry_id': company[2],
                    'industry_name': company[3],
                    'company_size': company[4],
                    'region': company[5],
                    'employees': company[6],
                    'annual_revenue': company[7],
                    'total_assets': company[8],
                    'total_capital': company[9],
                    'main_products': company[10],
                    'business_purpose': company[11],
                    'similarity_score': company[13],
                    'final_score': final_score
                })
            
            # 최종 스코어로 정렬하고 상위 k개 반환
            recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            print(f"기업 추천 오류: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def generate_recommendation_reason(self, company_data: Dict, tech_data: Dict) -> str:
        """
        LangChain을 사용하여 추천 사유 생성
        
        Args:
            company_data (dict): 기업 정보
            tech_data (dict): 기술 정보
            
        Returns:
            str: 추천 사유 (200자 이내)
        """
        try:
            # 프롬프트 템플릿 정의
            prompt_template = PromptTemplate(
                input_variables=["company_name", "industry", "main_products", "tech_title", "tech_category", "expected_effect"],
                template="""
다음 기업과 R&D 기술의 매칭 사유를 200자 이내로 간결하게 설명해주세요.

기업 정보:
- 기업명: {company_name}
- 산업분야: {industry}
- 주요제품: {main_products}

R&D 기술 정보:
- 기술명: {tech_title}
- 기술분야: {tech_category}
- 기대효과: {expected_effect}

매칭 사유 (200자 이내):
"""
            )
            
            # LLM 체인 생성
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            
            # 추천 사유 생성
            reason = chain.run(
                company_name=company_data.get('company_name', ''),
                industry=company_data.get('industry_name', ''),
                main_products=company_data.get('main_products', ''),
                tech_title=tech_data.get('tech_title', ''),
                tech_category=tech_data.get('tech_category', ''),
                expected_effect=tech_data.get('expected_effect', '')
            )
            
            # 200자 제한
            return reason.strip()[:200]
            
        except Exception as e:
            print(f"추천 사유 생성 오류: {e}")
            return "기업의 사업 분야와 기술의 특성이 잘 매칭되어 시너지 효과가 기대됩니다."
    
    def _calculate_category_bonus(self, company_industry: str, tech_category: str, weight: float) -> float:
        """카테고리 매칭 보너스 계산"""
        # 간단한 카테고리 매칭 로직 (실제로는 더 정교한 매칭 필요)
        if company_industry and tech_category:
            # 키워드 기반 매칭
            company_keywords = company_industry.lower().split()
            tech_keywords = tech_category.lower().split()
            
            common_keywords = set(company_keywords) & set(tech_keywords)
            if common_keywords:
                return weight * 0.5  # 매칭 시 보너스
        
        return 0.0
    
    def _calculate_region_bonus(self, region: str, weight: float) -> float:
        """지역 매칭 보너스 계산"""
        # 수도권 기업에 약간의 보너스 (예시)
        if region and "서울" in region or "경기" in region:
            return weight * 0.3
        return 0.0
    
    def _calculate_size_bonus(self, company_size: str, employees: int, weight: float) -> float:
        """기업 규모 보너스 계산"""
        # 중견기업 이상에 보너스 (예시)
        if company_size in ["중견기업", "대기업"] or employees >= 300:
            return weight * 0.4
        return 0.0

# 편의 함수들
def get_recommendation_engine():
    """추천 엔진 인스턴스 반환"""
    return LangChainRecommendationEngine()

def recommend_tech_for_company(company_id: str, top_k: int = 10):
    """기업에 대한 R&D 기술 추천 편의 함수"""
    engine = get_recommendation_engine()
    return engine.recommend_rnd_tech_for_company(company_id, top_k)

def recommend_companies_for_tech(tech_id: str, top_k: int = 10):
    """R&D 기술에 대한 기업 추천 편의 함수"""
    engine = get_recommendation_engine()
    return engine.recommend_companies_for_rnd_tech(tech_id, top_k)

