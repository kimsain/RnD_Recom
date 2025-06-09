"""
LangChain 기반 추천 시스템 평가 지표 모듈

이 모듈은 LangChain을 활용한 R&D 기술 추천 시스템의 성능을 평가하는 다양한 지표를 제공합니다.
"""

import os
import psycopg2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings
from langchain_config import APIConfig
from langchain_recommendation_engine import LangChainRecommendationEngine

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = APIConfig.OPENAI_API_KEY

class LangChainEvaluationMetrics:
    """
    LangChain 기반 추천 시스템 평가 지표 클래스
    """
    
    def __init__(self):
        """평가 지표 클래스 초기화"""
        self.db_params = APIConfig.get_db_connection_params()
        self.recommendation_engine = LangChainRecommendationEngine()
        self.embeddings = OpenAIEmbeddings(
            model=APIConfig.OPENAI_EMBEDDING_MODEL,
            openai_api_key=APIConfig.OPENAI_API_KEY
        )
    
    def connect_to_db(self):
        """데이터베이스 연결"""
        return psycopg2.connect(**self.db_params)
    
    def calculate_coverage(self, recommendation_type: str, sample_size: int = 10) -> float:
        """
        커버리지 계산: 추천을 받은 항목 수 / 전체 항목 수
        
        Args:
            recommendation_type (str): 추천 타입 ('company_to_rnd' 또는 'rnd_to_company')
            sample_size (int): 평가 샘플 크기
            
        Returns:
            float: 커버리지 점수
        """
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            if recommendation_type == "company_to_rnd":
                # 기업에 대한 R&D 기술 추천 커버리지
                cursor.execute("SELECT COUNT(*) FROM companies")
                total_companies = cursor.fetchone()[0]
                
                cursor.execute("SELECT company_id FROM companies ORDER BY RANDOM() LIMIT %s", (sample_size,))
                sample_companies = cursor.fetchall()
                
                recommended_techs = set()
                for company in sample_companies:
                    recommendations = self.recommendation_engine.recommend_rnd_tech_for_company(company[0], 10)
                    for rec in recommendations:
                        recommended_techs.add(rec['tech_id'])
                
                cursor.execute("SELECT COUNT(*) FROM rnd_technologies")
                total_techs = cursor.fetchone()[0]
                
                coverage = len(recommended_techs) / total_techs if total_techs > 0 else 0
                
            else:  # rnd_to_company
                # R&D 기술에 대한 기업 추천 커버리지
                cursor.execute("SELECT COUNT(*) FROM rnd_technologies")
                total_techs = cursor.fetchone()[0]
                
                cursor.execute("SELECT tech_id FROM rnd_technologies ORDER BY RANDOM() LIMIT %s", (sample_size,))
                sample_techs = cursor.fetchall()
                
                recommended_companies = set()
                for tech in sample_techs:
                    recommendations = self.recommendation_engine.recommend_companies_for_rnd_tech(tech[0], 10)
                    for rec in recommendations:
                        recommended_companies.add(rec['company_id'])
                
                cursor.execute("SELECT COUNT(*) FROM companies")
                total_companies = cursor.fetchone()[0]
                
                coverage = len(recommended_companies) / total_companies if total_companies > 0 else 0
            
            return coverage
            
        except Exception as e:
            print(f"커버리지 계산 오류: {e}")
            return 0.0
        finally:
            cursor.close()
            conn.close()
    
    def calculate_diversity(self, recommendations: List[Dict[str, Any]], recommendation_type: str) -> float:
        """
        다양성 계산: 추천 결과의 카테고리 다양성
        
        Args:
            recommendations (list): 추천 결과 목록
            recommendation_type (str): 추천 타입
            
        Returns:
            float: 다양성 점수
        """
        if not recommendations:
            return 0.0
        
        try:
            if recommendation_type == "company_to_rnd":
                # R&D 기술 카테고리 다양성
                categories = [rec.get('tech_category', '') for rec in recommendations]
            else:
                # 기업 산업 분야 다양성
                categories = [rec.get('industry_name', '') for rec in recommendations]
            
            # 고유 카테고리 수 / 전체 추천 수
            unique_categories = len(set(categories))
            total_recommendations = len(recommendations)
            
            diversity = unique_categories / total_recommendations if total_recommendations > 0 else 0
            return diversity
            
        except Exception as e:
            print(f"다양성 계산 오류: {e}")
            return 0.0
    
    def calculate_relevance(self, source_id: str, recommendations: List[Dict[str, Any]], 
                          recommendation_type: str) -> float:
        """
        관련성 계산: 추천 결과와 소스 항목 간의 관련성
        
        Args:
            source_id (str): 소스 항목 ID
            recommendations (list): 추천 결과 목록
            recommendation_type (str): 추천 타입
            
        Returns:
            float: 관련성 점수
        """
        if not recommendations:
            return 0.0
        
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            if recommendation_type == "company_to_rnd":
                # 기업과 R&D 기술 간의 관련성
                cursor.execute("""
                SELECT c.industry_id, i.industry_name
                FROM companies c
                JOIN industries i ON c.industry_id = i.industry_id
                WHERE c.company_id = %s
                """, (source_id,))
                
                company_info = cursor.fetchone()
                if not company_info:
                    return 0.0
                
                company_industry = company_info[1].lower()
                
                # 추천된 기술들의 카테고리와 매칭 점수 계산
                relevance_scores = []
                for rec in recommendations:
                    tech_category = rec.get('tech_category', '').lower()
                    # 간단한 키워드 매칭 기반 관련성 계산
                    common_words = set(company_industry.split()) & set(tech_category.split())
                    relevance_score = len(common_words) / max(len(tech_category.split()), 1)
                    relevance_scores.append(relevance_score)
                
            else:  # rnd_to_company
                # R&D 기술과 기업 간의 관련성
                cursor.execute("""
                SELECT tech_category, application_field
                FROM rnd_technologies
                WHERE tech_id = %s
                """, (source_id,))
                
                tech_info = cursor.fetchone()
                if not tech_info:
                    return 0.0
                
                tech_category = tech_info[0].lower()
                application_field = tech_info[1].lower() if tech_info[1] else ""
                
                # 추천된 기업들의 산업 분야와 매칭 점수 계산
                relevance_scores = []
                for rec in recommendations:
                    industry_name = rec.get('industry_name', '').lower()
                    # 기술 카테고리와 산업 분야 간의 관련성 계산
                    tech_words = set(tech_category.split()) | set(application_field.split())
                    industry_words = set(industry_name.split())
                    common_words = tech_words & industry_words
                    relevance_score = len(common_words) / max(len(industry_words), 1)
                    relevance_scores.append(relevance_score)
            
            # 평균 관련성 점수 반환
            return np.mean(relevance_scores) if relevance_scores else 0.0
            
        except Exception as e:
            print(f"관련성 계산 오류: {e}")
            return 0.0
        finally:
            cursor.close()
            conn.close()
    
    def calculate_similarity(self, recommendations: List[Dict[str, Any]]) -> float:
        """
        유사도 계산: 추천 결과 간의 평균 유사도
        
        Args:
            recommendations (list): 추천 결과 목록
            
        Returns:
            float: 평균 유사도 점수
        """
        if len(recommendations) < 2:
            return 0.0
        
        try:
            # 추천 결과들의 유사도 점수 추출
            similarity_scores = [rec.get('similarity_score', 0) for rec in recommendations]
            
            # 평균 유사도 계산
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            print(f"유사도 계산 오류: {e}")
            return 0.0
    
    def evaluate_recommendation_system(self, recommendation_type: str, sample_size: int = 5) -> Dict[str, Any]:
        """
        추천 시스템 종합 평가
        
        Args:
            recommendation_type (str): 추천 타입 ('company_to_rnd' 또는 'rnd_to_company')
            sample_size (int): 평가 샘플 크기
            
        Returns:
            dict: 평가 결과
        """
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            print(f"LangChain 기반 추천 시스템 평가 시작 (타입: {recommendation_type})")
            
            # 1. 커버리지 계산
            print("1. 커버리지 계산 중...")
            coverage = self.calculate_coverage(recommendation_type, sample_size)
            
            # 2. 샘플 데이터로 다양성, 관련성, 유사도 계산
            print("2. 다양성, 관련성, 유사도 계산 중...")
            
            diversity_scores = []
            relevance_scores = []
            similarity_scores = []
            
            if recommendation_type == "company_to_rnd":
                cursor.execute("SELECT company_id FROM companies ORDER BY RANDOM() LIMIT %s", (sample_size,))
                samples = cursor.fetchall()
                
                for sample in samples:
                    recommendations = self.recommendation_engine.recommend_rnd_tech_for_company(sample[0], 10)
                    if recommendations:
                        diversity_scores.append(self.calculate_diversity(recommendations, recommendation_type))
                        relevance_scores.append(self.calculate_relevance(sample[0], recommendations, recommendation_type))
                        similarity_scores.append(self.calculate_similarity(recommendations))
            
            else:  # rnd_to_company
                cursor.execute("SELECT tech_id FROM rnd_technologies ORDER BY RANDOM() LIMIT %s", (sample_size,))
                samples = cursor.fetchall()
                
                for sample in samples:
                    recommendations = self.recommendation_engine.recommend_companies_for_rnd_tech(sample[0], 10)
                    if recommendations:
                        diversity_scores.append(self.calculate_diversity(recommendations, recommendation_type))
                        relevance_scores.append(self.calculate_relevance(sample[0], recommendations, recommendation_type))
                        similarity_scores.append(self.calculate_similarity(recommendations))
            
            # 평균 점수 계산
            avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
            avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            
            results = {
                "recommendation_type": recommendation_type,
                "coverage": coverage,
                "diversity": avg_diversity,
                "relevance": avg_relevance,
                "similarity": avg_similarity,
                "sample_size": sample_size
            }
            
            print("LangChain 기반 추천 시스템 평가 완료")
            return results
            
        except Exception as e:
            print(f"추천 시스템 평가 오류: {e}")
            return {
                "recommendation_type": recommendation_type,
                "coverage": 0.0,
                "diversity": 0.0,
                "relevance": 0.0,
                "similarity": 0.0,
                "sample_size": sample_size
            }
        finally:
            cursor.close()
            conn.close()
    
    def compare_with_baseline(self, recommendation_type: str, sample_size: int = 5) -> Dict[str, Any]:
        """
        기준선(랜덤 추천)과 비교 평가
        
        Args:
            recommendation_type (str): 추천 타입
            sample_size (int): 평가 샘플 크기
            
        Returns:
            dict: 비교 평가 결과
        """
        # LangChain 기반 추천 시스템 평가
        langchain_results = self.evaluate_recommendation_system(recommendation_type, sample_size)
        
        # 랜덤 추천 기준선 평가 (간단한 구현)
        baseline_results = {
            "recommendation_type": f"{recommendation_type}_baseline",
            "coverage": 0.1,  # 랜덤 추천의 예상 커버리지
            "diversity": 0.8,  # 랜덤 추천은 다양성이 높음
            "relevance": 0.1,  # 랜덤 추천은 관련성이 낮음
            "similarity": 0.3,  # 랜덤 추천의 평균 유사도
            "sample_size": sample_size
        }
        
        # 개선 정도 계산
        improvements = {
            "coverage_improvement": langchain_results["coverage"] - baseline_results["coverage"],
            "relevance_improvement": langchain_results["relevance"] - baseline_results["relevance"],
            "similarity_improvement": langchain_results["similarity"] - baseline_results["similarity"]
        }
        
        return {
            "langchain_results": langchain_results,
            "baseline_results": baseline_results,
            "improvements": improvements
        }

# 편의 함수들
def evaluate_company_to_rnd(sample_size: int = 5):
    """기업-R&D 기술 추천 평가 편의 함수"""
    evaluator = LangChainEvaluationMetrics()
    return evaluator.evaluate_recommendation_system("company_to_rnd", sample_size)

def evaluate_rnd_to_company(sample_size: int = 5):
    """R&D 기술-기업 추천 평가 편의 함수"""
    evaluator = LangChainEvaluationMetrics()
    return evaluator.evaluate_recommendation_system("rnd_to_company", sample_size)

def compare_with_baseline(recommendation_type: str, sample_size: int = 5):
    """기준선 비교 평가 편의 함수"""
    evaluator = LangChainEvaluationMetrics()
    return evaluator.compare_with_baseline(recommendation_type, sample_size)

if __name__ == "__main__":
    # 테스트 실행
    print("LangChain 기반 추천 시스템 평가 테스트")
    
    # 기업-R&D 기술 추천 평가
    results1 = evaluate_company_to_rnd(3)
    print("\n기업-R&D 기술 추천 평가 결과:")
    for key, value in results1.items():
        print(f"{key}: {value}")
    
    # R&D 기술-기업 추천 평가
    results2 = evaluate_rnd_to_company(3)
    print("\nR&D 기술-기업 추천 평가 결과:")
    for key, value in results2.items():
        print(f"{key}: {value}")

