import os
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

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

def calculate_coverage(recommendation_type, limit=100):
    """추천 시스템의 커버리지 계산 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    try:
        if recommendation_type == 'company_to_rnd':
            # 기업에 대한 R&D 기술 추천 커버리지
            cursor.execute("""
            SELECT COUNT(DISTINCT source_id) AS covered_companies
            FROM recommendations
            WHERE recommendation_type = 'company_to_rnd'
            """)
            covered_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM companies")
            total_count = cursor.fetchone()[0]
        
        elif recommendation_type == 'rnd_to_company':
            # R&D 기술에 대한 기업 추천 커버리지
            cursor.execute("""
            SELECT COUNT(DISTINCT source_id) AS covered_techs
            FROM recommendations
            WHERE recommendation_type = 'rnd_to_company'
            """)
            covered_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM rnd_technologies")
            total_count = cursor.fetchone()[0]
        
        else:
            print(f"지원하지 않는 추천 유형: {recommendation_type}")
            cursor.close()
            conn.close()
            return 0
        
        # 커버리지 계산 (추천을 받은 항목 수 / 전체 항목 수)
        coverage = covered_count / total_count if total_count > 0 else 0
        
        return coverage
    except Exception as e:
        print(f"커버리지 계산 중 오류 발생: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()

def calculate_diversity(recommendation_type, source_id, limit=10):
    """추천 결과의 다양성 계산 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    try:
        if recommendation_type == 'company_to_rnd':
            # 기업에 대한 R&D 기술 추천 다양성
            cursor.execute("""
            SELECT t.tech_category
            FROM recommendations r
            JOIN rnd_technologies t ON r.target_id = t.tech_id
            WHERE r.source_id = %s AND r.recommendation_type = %s
            ORDER BY r.rank
            LIMIT %s
            """, (source_id, recommendation_type, limit))
            
            categories = [row[0] for row in cursor.fetchall()]
            
        elif recommendation_type == 'rnd_to_company':
            # R&D 기술에 대한 기업 추천 다양성
            cursor.execute("""
            SELECT c.industry_code
            FROM recommendations r
            JOIN companies c ON r.target_id = c.company_id
            WHERE r.source_id = %s AND r.recommendation_type = %s
            ORDER BY r.rank
            LIMIT %s
            """, (source_id, recommendation_type, limit))
            
            categories = [row[0] for row in cursor.fetchall()]
        
        else:
            print(f"지원하지 않는 추천 유형: {recommendation_type}")
            cursor.close()
            conn.close()
            return 0
        
        # 다양성 계산 (고유 카테고리 수 / 전체 추천 수)
        if not categories:
            return 0
            
        unique_categories = len(set(categories))
        diversity = unique_categories / len(categories) if len(categories) > 0 else 0
        
        return diversity
    except Exception as e:
        print(f"다양성 계산 중 오류 발생: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()

def calculate_relevance(recommendation_type, source_id, limit=10):
    """추천 결과의 관련성 계산 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    try:
        if recommendation_type == 'company_to_rnd':
            # 기업 정보 조회
            cursor.execute("""
            SELECT c.industry_code
            FROM companies c
            WHERE c.company_id = %s
            """, (source_id,))
            
            result = cursor.fetchone()
            if not result:
                print(f"기업 ID {source_id}에 해당하는 데이터를 찾을 수 없습니다.")
                cursor.close()
                conn.close()
                return 0
                
            industry_code = result[0]
            
            # 산업 코드에 적합한 기술 카테고리 매핑
            industry_category_mapping = {
                'A': ['스마트팩토리', '신재생에너지', '바이오헬스'],
                'B': ['나노소재', '신재생에너지'],
                'C': ['스마트팩토리', '로봇공학', '나노소재', '반도체', '디스플레이'],
                'D': ['신재생에너지', '스마트팩토리'],
                'E': ['신재생에너지', '바이오헬스'],
                'F': ['스마트팩토리', '로봇공학'],
                'G': ['빅데이터', '인공지능', '블록체인'],
                'J': ['인공지능', '빅데이터', '클라우드', '사물인터넷', '블록체인', '통신기술', '보안기술'],
                'K': ['빅데이터', '인공지능', '블록체인', '보안기술'],
                'M': ['인공지능', '빅데이터', '클라우드', '사물인터넷', '블록체인', '로봇공학', '자율주행']
            }
            
            suitable_categories = industry_category_mapping.get(industry_code, [])
            
            # 추천된 기술 카테고리 조회
            cursor.execute("""
            SELECT t.tech_category
            FROM recommendations r
            JOIN rnd_technologies t ON r.target_id = t.tech_id
            WHERE r.source_id = %s AND r.recommendation_type = %s
            ORDER BY r.rank
            LIMIT %s
            """, (source_id, recommendation_type, limit))
            
            recommended_categories = [row[0] for row in cursor.fetchall()]
            
            if not recommended_categories:
                return 0
                
            # 관련성 있는 추천 수 계산
            relevant_count = sum(1 for cat in recommended_categories if cat in suitable_categories)
            
            # 관련성 계산 (관련성 있는 추천 수 / 전체 추천 수)
            relevance = relevant_count / len(recommended_categories) if len(recommended_categories) > 0 else 0
            
        elif recommendation_type == 'rnd_to_company':
            # 기술 정보 조회
            cursor.execute("""
            SELECT t.tech_category
            FROM rnd_technologies t
            WHERE t.tech_id = %s
            """, (source_id,))
            
            result = cursor.fetchone()
            if not result:
                print(f"기술 ID {source_id}에 해당하는 데이터를 찾을 수 없습니다.")
                cursor.close()
                conn.close()
                return 0
                
            tech_category = result[0]
            
            # 기술 카테고리에 적합한 산업 코드 매핑 (역매핑)
            category_industry_mapping = {
                '인공지능': ['G', 'J', 'K', 'M'],
                '빅데이터': ['G', 'J', 'K', 'M'],
                '클라우드': ['J', 'M'],
                '사물인터넷': ['J', 'M'],
                '블록체인': ['G', 'J', 'K', 'M'],
                '로봇공학': ['C', 'F', 'M'],
                '자율주행': ['M'],
                '스마트팩토리': ['A', 'C', 'D', 'F'],
                '신재생에너지': ['A', 'B', 'D', 'E'],
                '바이오헬스': ['A', 'E'],
                '나노소재': ['B', 'C'],
                '반도체': ['C'],
                '디스플레이': ['C'],
                '통신기술': ['J'],
                '보안기술': ['J', 'K']
            }
            
            suitable_industries = category_industry_mapping.get(tech_category, [])
            
            # 추천된 기업 산업 코드 조회
            cursor.execute("""
            SELECT c.industry_code
            FROM recommendations r
            JOIN companies c ON r.target_id = c.company_id
            WHERE r.source_id = %s AND r.recommendation_type = %s
            ORDER BY r.rank
            LIMIT %s
            """, (source_id, recommendation_type, limit))
            
            recommended_industries = [row[0] for row in cursor.fetchall()]
            
            if not recommended_industries:
                return 0
                
            # 관련성 있는 추천 수 계산
            relevant_count = sum(1 for ind in recommended_industries if ind in suitable_industries)
            
            # 관련성 계산 (관련성 있는 추천 수 / 전체 추천 수)
            relevance = relevant_count / len(recommended_industries) if len(recommended_industries) > 0 else 0
        
        else:
            print(f"지원하지 않는 추천 유형: {recommendation_type}")
            cursor.close()
            conn.close()
            return 0
        
        return relevance
    except Exception as e:
        print(f"관련성 계산 중 오류 발생: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()

def parse_embedding(embedding_str):
    """문자열 형태의 임베딩을 numpy 배열로 변환하는 함수"""
    try:
        # 문자열이 아닌 경우 그대로 반환
        if not isinstance(embedding_str, str):
            return embedding_str
            
        # 문자열에서 대괄호 제거
        embedding_str = embedding_str.strip('[]')
        
        # 쉼표로 구분된 문자열을 float 배열로 변환
        embedding_array = np.array([float(x.strip()) for x in embedding_str.split(',')])
        
        return embedding_array
    except Exception as e:
        print(f"임베딩 파싱 중 오류 발생: {e}")
        return np.zeros(1536)  # 오류 발생 시 기본 임베딩 반환

def calculate_average_similarity(recommendation_type, source_id, limit=10):
    """추천 결과 간의 평균 유사도 계산 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    try:
        if recommendation_type == 'company_to_rnd':
            # 추천된 기술 임베딩 조회
            cursor.execute("""
            SELECT e.embedding
            FROM recommendations r
            JOIN rnd_embeddings e ON r.target_id = e.tech_id
            WHERE r.source_id = %s AND r.recommendation_type = %s AND e.embedding_type = 'combined'
            ORDER BY r.rank
            LIMIT %s
            """, (source_id, recommendation_type, limit))
            
        elif recommendation_type == 'rnd_to_company':
            # 추천된 기업 임베딩 조회
            cursor.execute("""
            SELECT e.embedding
            FROM recommendations r
            JOIN company_embeddings e ON r.target_id = e.company_id
            WHERE r.source_id = %s AND r.recommendation_type = %s AND e.embedding_type = 'profile'
            ORDER BY r.rank
            LIMIT %s
            """, (source_id, recommendation_type, limit))
        
        else:
            print(f"지원하지 않는 추천 유형: {recommendation_type}")
            cursor.close()
            conn.close()
            return 0
        
        embeddings_raw = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if len(embeddings_raw) < 2:
            return 0
        
        # 임베딩 문자열을 numpy 배열로 변환
        embeddings = [parse_embedding(row[0]) for row in embeddings_raw]
        
        # 임베딩 배열 형태 확인 및 변환
        valid_embeddings = []
        for emb in embeddings:
            if emb is not None and len(emb) > 0:
                valid_embeddings.append(emb)
        
        if len(valid_embeddings) < 2:
            return 0
            
        # 임베딩 간의 코사인 유사도 계산
        embeddings_array = np.array(valid_embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # 대각선 요소 제외 (자기 자신과의 유사도)
        np.fill_diagonal(similarity_matrix, 0)
        
        # 평균 유사도 계산
        avg_similarity = np.sum(similarity_matrix) / (similarity_matrix.size - len(valid_embeddings))
        
        return avg_similarity
    except Exception as e:
        print(f"유사도 계산 중 오류 발생: {e}")
        return 0

def evaluate_recommendation_system(recommendation_type, sample_size=10):
    """추천 시스템 종합 평가 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    try:
        # 평가 대상 샘플 선택
        if recommendation_type == 'company_to_rnd':
            cursor.execute("""
            SELECT DISTINCT source_id
            FROM recommendations
            WHERE recommendation_type = 'company_to_rnd'
            LIMIT %s
            """, (sample_size,))
        else:
            cursor.execute("""
            SELECT DISTINCT source_id
            FROM recommendations
            WHERE recommendation_type = 'rnd_to_company'
            LIMIT %s
            """, (sample_size,))
        
        sample_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        if not sample_ids:
            print(f"평가할 {recommendation_type} 추천 결과가 없습니다.")
            return {
                'recommendation_type': recommendation_type,
                'coverage': 0,
                'diversity': 0,
                'relevance': 0,
                'similarity': 0,
                'sample_size': 0
            }
        
        # 각 지표 계산
        coverage = calculate_coverage(recommendation_type)
        
        diversity_scores = []
        relevance_scores = []
        similarity_scores = []
        
        for source_id in sample_ids:
            try:
                diversity_scores.append(calculate_diversity(recommendation_type, source_id))
                relevance_scores.append(calculate_relevance(recommendation_type, source_id))
                similarity_scores.append(calculate_average_similarity(recommendation_type, source_id))
            except Exception as e:
                print(f"ID {source_id}에 대한 평가 지표 계산 중 오류 발생: {e}")
                continue
        
        # 평균 점수 계산
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        # 결과 반환
        results = {
            'recommendation_type': recommendation_type,
            'coverage': coverage,
            'diversity': avg_diversity,
            'relevance': avg_relevance,
            'similarity': avg_similarity,
            'sample_size': len(sample_ids)
        }
        
        return results
    except Exception as e:
        print(f"추천 시스템 평가 중 오류 발생: {e}")
        return {
            'recommendation_type': recommendation_type,
            'coverage': 0,
            'diversity': 0,
            'relevance': 0,
            'similarity': 0,
            'sample_size': 0
        }

def visualize_evaluation_results(results):
    """평가 결과 시각화 함수"""
    if not results:
        print("시각화할 평가 결과가 없습니다.")
        return
    
    try:
        # 결과 데이터 준비
        metrics = ['Coverage', 'Diversity', 'Relevance', 'Similarity']
        values = [results['coverage'], results['diversity'], results['relevance'], results['similarity']]
        
        # 그래프 생성
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        
        # 그래프 스타일 설정
        plt.ylim(0, 1.0)
        plt.title(f"Recommendation System Evaluation: {results['recommendation_type']}")
        plt.ylabel('Score (0-1)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 그래프 저장
        plt.tight_layout()
        plt.savefig(f"evaluation_{results['recommendation_type']}.png")
        plt.close()
        
        print(f"평가 결과 시각화가 'evaluation_{results['recommendation_type']}.png'에 저장되었습니다.")
    except Exception as e:
        print(f"평가 결과 시각화 중 오류 발생: {e}")

def compare_weight_settings(company_id, tech_id, weight_settings):
    """가중치 설정 비교 함수"""
    try:
        from recommendation_engine import update_weight_settings, recommend_rnd_tech_for_company, recommend_companies_for_rnd_tech
        
        results = []
        
        for setting in weight_settings:
            # 가중치 설정 업데이트
            update_weight_settings(
                setting['name'],
                text_similarity_weight=setting['text_similarity_weight'],
                quantitative_weight=setting['quantitative_weight'],
                category_weight=setting['category_weight'],
                region_weight=setting['region_weight'],
                size_weight=setting['size_weight']
            )
            
            # 기업에 대한 R&D 기술 추천
            company_recommendations = recommend_rnd_tech_for_company(company_id, top_n=10)
            
            # R&D 기술에 대한 기업 추천
            tech_recommendations = recommend_companies_for_rnd_tech(tech_id, top_n=10)
            
            # 평가 지표 계산
            company_eval = evaluate_recommendation_system('company_to_rnd', sample_size=5)
            tech_eval = evaluate_recommendation_system('rnd_to_company', sample_size=5)
            
            # 결과 저장
            results.append({
                'setting_name': setting['name'],
                'weights': setting,
                'company_to_rnd_eval': company_eval,
                'rnd_to_company_eval': tech_eval
            })
        
        return results
    except Exception as e:
        print(f"가중치 설정 비교 중 오류 발생: {e}")
        return []

def test_evaluation_metrics():
    """평가 지표 테스트 함수"""
    print("평가 지표 테스트 중...")
    
    try:
        # 기업에 대한 R&D 기술 추천 평가
        print("\n1. 기업에 대한 R&D 기술 추천 평가")
        company_results = evaluate_recommendation_system('company_to_rnd', sample_size=3)
        
        if company_results:
            print(f"커버리지: {company_results['coverage']:.4f}")
            print(f"다양성: {company_results['diversity']:.4f}")
            print(f"관련성: {company_results['relevance']:.4f}")
            print(f"유사도: {company_results['similarity']:.4f}")
            
            # 결과 시각화
            visualize_evaluation_results(company_results)
        
        # R&D 기술에 대한 기업 추천 평가
        print("\n2. R&D 기술에 대한 기업 추천 평가")
        tech_results = evaluate_recommendation_system('rnd_to_company', sample_size=3)
        
        if tech_results:
            print(f"커버리지: {tech_results['coverage']:.4f}")
            print(f"다양성: {tech_results['diversity']:.4f}")
            print(f"관련성: {tech_results['relevance']:.4f}")
            print(f"유사도: {tech_results['similarity']:.4f}")
            
            # 결과 시각화
            visualize_evaluation_results(tech_results)
        
        return True
    except Exception as e:
        print(f"평가 지표 테스트 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    test_evaluation_metrics()
