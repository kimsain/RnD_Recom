import os
import sys
import pandas as pd
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 필요한 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import (
    connect_to_db,
    get_active_weight_settings,
    update_weight_settings,
    recommend_rnd_tech_for_company,
    recommend_companies_for_rnd_tech
)
from evaluation_metrics import evaluate_recommendation_system

def run_system_test():
    """전체 시스템 통합 테스트 함수"""
    print("=" * 80)
    print("R&D 기술 추천 시스템 통합 테스트 시작")
    print("=" * 80)
    
    # 1. 데이터베이스 연결 테스트
    print("\n1. 데이터베이스 연결 테스트")
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rnd_technologies")
        rnd_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM companies")
        company_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM rnd_embeddings")
        rnd_embedding_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM company_embeddings")
        company_embedding_count = cursor.fetchone()[0]
        
        print(f"  - 데이터베이스 연결 성공")
        print(f"  - 공공 R&D 기술 데이터: {rnd_count}개")
        print(f"  - 기업 프로파일 데이터: {company_count}개")
        print(f"  - R&D 기술 임베딩: {rnd_embedding_count}개")
        print(f"  - 기업 프로파일 임베딩: {company_embedding_count}개")
        
        cursor.close()
        conn.close()
        print("  ✓ 데이터베이스 연결 테스트 성공")
    except Exception as e:
        print(f"  ✗ 데이터베이스 연결 테스트 실패: {e}")
        return False
    
    # 2. 가중치 설정 테스트
    print("\n2. 가중치 설정 테스트")
    try:
        # 기존 가중치 설정 확인
        original_weights = get_active_weight_settings()
        print(f"  - 기존 가중치 설정: {original_weights['setting_name']}")
        
        # 테스트용 가중치 설정 업데이트
        update_weight_settings(
            "test_weights", 
            text_similarity_weight=0.5, 
            quantitative_weight=0.5,
            category_weight=0.3,
            region_weight=0.3,
            size_weight=0.4
        )
        
        # 업데이트된 가중치 설정 확인
        updated_weights = get_active_weight_settings()
        print(f"  - 업데이트된 가중치 설정: {updated_weights['setting_name']}")
        print(f"    텍스트 유사도 가중치: {updated_weights['text_similarity_weight']}")
        print(f"    정량적 데이터 가중치: {updated_weights['quantitative_weight']}")
        print(f"    카테고리 가중치: {updated_weights['category_weight']}")
        print(f"    지역 가중치: {updated_weights['region_weight']}")
        print(f"    기업 규모 가중치: {updated_weights['size_weight']}")
        
        # 원래 가중치로 복원
        update_weight_settings(
            original_weights['setting_name'],
            text_similarity_weight=original_weights['text_similarity_weight'],
            quantitative_weight=original_weights['quantitative_weight'],
            category_weight=original_weights['category_weight'],
            region_weight=original_weights['region_weight'],
            size_weight=original_weights['size_weight']
        )
        
        print("  ✓ 가중치 설정 테스트 성공")
    except Exception as e:
        print(f"  ✗ 가중치 설정 테스트 실패: {e}")
        return False
    
    # 3. 기업에 대한 R&D 기술 추천 테스트
    print("\n3. 기업에 대한 R&D 기술 추천 테스트")
    try:
        company_id = "COMP-00001"
        recommendations = recommend_rnd_tech_for_company(company_id, top_n=5)
        
        print(f"  - 기업 ID '{company_id}'에 대한 R&D 기술 추천 결과:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec['tech_title']} (유사도: {rec['final_score']:.4f})")
        
        print("  ✓ 기업에 대한 R&D 기술 추천 테스트 성공")
    except Exception as e:
        print(f"  ✗ 기업에 대한 R&D 기술 추천 테스트 실패: {e}")
        return False
    
    # 4. R&D 기술에 대한 기업 추천 테스트
    print("\n4. R&D 기술에 대한 기업 추천 테스트")
    try:
        tech_id = "RND-00001"
        recommendations = recommend_companies_for_rnd_tech(tech_id, top_n=5)
        
        print(f"  - R&D 기술 ID '{tech_id}'에 대한 기업 추천 결과:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec['company_name']} (유사도: {rec['final_score']:.4f})")
        
        print("  ✓ R&D 기술에 대한 기업 추천 테스트 성공")
    except Exception as e:
        print(f"  ✗ R&D 기술에 대한 기업 추천 테스트 실패: {e}")
        return False
    
    # 5. 평가 지표 테스트
    print("\n5. 평가 지표 테스트")
    try:
        # 기업에 대한 R&D 기술 추천 평가
        company_results = evaluate_recommendation_system('company_to_rnd', sample_size=3)
        
        print(f"  - 기업에 대한 R&D 기술 추천 평가 결과:")
        print(f"    커버리지: {company_results['coverage']:.4f}")
        print(f"    다양성: {company_results['diversity']:.4f}")
        print(f"    관련성: {company_results['relevance']:.4f}")
        print(f"    유사도: {company_results['similarity']:.4f}")
        
        # R&D 기술에 대한 기업 추천 평가
        tech_results = evaluate_recommendation_system('rnd_to_company', sample_size=3)
        
        print(f"  - R&D 기술에 대한 기업 추천 평가 결과:")
        print(f"    커버리지: {tech_results['coverage']:.4f}")
        print(f"    다양성: {tech_results['diversity']:.4f}")
        print(f"    관련성: {tech_results['relevance']:.4f}")
        print(f"    유사도: {tech_results['similarity']:.4f}")
        
        print("  ✓ 평가 지표 테스트 성공")
    except Exception as e:
        print(f"  ✗ 평가 지표 테스트 실패: {e}")
        return False
    
    # 테스트 결과 요약
    print("\n" + "=" * 80)
    print("R&D 기술 추천 시스템 통합 테스트 완료")
    print("모든 테스트가 성공적으로 완료되었습니다.")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    run_system_test()
