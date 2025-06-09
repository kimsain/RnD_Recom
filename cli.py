import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from recommendation_engine import (
    connect_to_db, 
    get_active_weight_settings,
    update_weight_settings,
    recommend_rnd_tech_for_company,
    recommend_companies_for_rnd_tech,
    get_recommendation_results
)
from evaluation_metrics import (
    evaluate_recommendation_system,
    calculate_coverage,
    calculate_diversity,
    calculate_relevance
)

# 환경 변수 로드
load_dotenv()

def format_currency(value):
    """통화 형식으로 포맷팅하는 함수"""
    return f"{value:,}원"

def display_rnd_tech_recommendations(recommendations):
    """R&D 기술 추천 결과 출력 함수"""
    if not recommendations:
        print("추천 결과가 없습니다.")
        return
    
    print("\n" + "="*80)
    print(f"{'순위':<5}{'기술 ID':<12}{'기술 제목':<40}{'기술 분야':<15}{'유사도':<10}")
    print("-"*80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:<5}{rec['tech_id']:<12}{rec['tech_title'][:38]:<40}{rec['tech_category']:<15}{rec['final_score']:.4f}")
    
    print("="*80)

def display_company_recommendations(recommendations):
    """기업 추천 결과 출력 함수"""
    if not recommendations:
        print("추천 결과가 없습니다.")
        return
    
    print("\n" + "="*100)
    print(f"{'순위':<5}{'기업 ID':<12}{'기업명':<20}{'산업':<20}{'규모':<10}{'지역':<8}{'유사도':<10}")
    print("-"*100)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:<5}{rec['company_id']:<12}{rec['company_name'][:18]:<20}{rec['industry_name'][:18]:<20}{rec['company_size']:<10}{rec['region']:<8}{rec['final_score']:.4f}")
    
    print("="*100)
    
    # 상세 정보 출력
    print("\n[상위 3개 기업 상세 정보]")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n{i}. {rec['company_name']} ({rec['company_id']})")
        print(f"   산업: {rec['industry_name']}")
        print(f"   규모: {rec['company_size']} (직원 수: {rec['employees']:,}명)")
        print(f"   지역: {rec['region']}")
        print(f"   연간 매출: {format_currency(rec['annual_revenue'])}")
        print(f"   총자산: {format_currency(rec['total_assets'])}")
        print(f"   자본금: {format_currency(rec['total_capital'])}")
        print(f"   주요 제품: {rec['main_products']}")

def display_evaluation_results(results):
    """평가 결과 출력 함수"""
    if not results:
        print("평가 결과가 없습니다.")
        return
    
    print("\n" + "="*50)
    print(f"추천 시스템 평가 결과: {results['recommendation_type']}")
    print("-"*50)
    print(f"커버리지: {results['coverage']:.4f}")
    print(f"다양성: {results['diversity']:.4f}")
    print(f"관련성: {results['relevance']:.4f}")
    print(f"유사도: {results['similarity']:.4f}")
    print(f"샘플 크기: {results['sample_size']}")
    print("="*50)

def display_weight_settings(weights):
    """가중치 설정 출력 함수"""
    print("\n" + "="*50)
    print(f"현재 가중치 설정: {weights['setting_name']}")
    print("-"*50)
    print(f"텍스트 유사도 가중치: {weights['text_similarity_weight']:.2f}")
    print(f"정량적 데이터 가중치: {weights['quantitative_weight']:.2f}")
    print(f"카테고리 가중치: {weights['category_weight']:.2f}")
    print(f"지역 가중치: {weights['region_weight']:.2f}")
    print(f"기업 규모 가중치: {weights['size_weight']:.2f}")
    print("="*50)

def list_items(item_type, limit=10):
    """항목 목록 출력 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    if item_type == 'companies':
        cursor.execute("""
        SELECT company_id, company_name, industry_name, company_size, region
        FROM companies
        ORDER BY company_id
        LIMIT %s
        """, (limit,))
        
        columns = ['기업 ID', '기업명', '산업', '규모', '지역']
        
    elif item_type == 'technologies':
        cursor.execute("""
        SELECT tech_id, tech_title, tech_category, rnd_stage, region
        FROM rnd_technologies
        ORDER BY tech_id
        LIMIT %s
        """, (limit,))
        
        columns = ['기술 ID', '기술 제목', '기술 분야', '연구 단계', '지역']
    
    else:
        print(f"지원하지 않는 항목 유형: {item_type}")
        cursor.close()
        conn.close()
        return
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not results:
        print(f"{item_type} 목록이 비어 있습니다.")
        return
    
    # 결과 출력
    print("\n" + "="*100)
    print(" | ".join(f"{col}" for col in columns))
    print("-"*100)
    
    for row in results:
        print(" | ".join(f"{str(val)}" for val in row))
    
    print("="*100)

def search_items(item_type, keyword, limit=10):
    """항목 검색 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    if item_type == 'companies':
        cursor.execute("""
        SELECT company_id, company_name, industry_name, company_size, region
        FROM companies
        WHERE 
            company_name ILIKE %s OR
            industry_name ILIKE %s OR
            region ILIKE %s
        ORDER BY company_id
        LIMIT %s
        """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit))
        
        columns = ['기업 ID', '기업명', '산업', '규모', '지역']
        
    elif item_type == 'technologies':
        cursor.execute("""
        SELECT tech_id, tech_title, tech_category, rnd_stage, region
        FROM rnd_technologies
        WHERE 
            tech_title ILIKE %s OR
            tech_category ILIKE %s OR
            tech_subcategory ILIKE %s OR
            region ILIKE %s
        ORDER BY tech_id
        LIMIT %s
        """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit))
        
        columns = ['기술 ID', '기술 제목', '기술 분야', '연구 단계', '지역']
    
    else:
        print(f"지원하지 않는 항목 유형: {item_type}")
        cursor.close()
        conn.close()
        return
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not results:
        print(f"'{keyword}' 검색 결과가 없습니다.")
        return
    
    # 결과 출력
    print(f"\n'{keyword}' 검색 결과:")
    print("="*100)
    print(" | ".join(f"{col}" for col in columns))
    print("-"*100)
    
    for row in results:
        print(" | ".join(f"{str(val)}" for val in row))
    
    print("="*100)

def main():
    parser = argparse.ArgumentParser(description='R&D 기술 추천 시스템')
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # 기업에 대한 R&D 기술 추천 명령어
    recommend_tech_parser = subparsers.add_parser('recommend-tech', help='기업에 적합한 R&D 기술 추천')
    recommend_tech_parser.add_argument('company_id', help='기업 ID')
    recommend_tech_parser.add_argument('--top', type=int, default=10, help='상위 N개 결과 (기본값: 10)')
    
    # R&D 기술에 대한 기업 추천 명령어
    recommend_company_parser = subparsers.add_parser('recommend-company', help='R&D 기술에 적합한 기업 추천')
    recommend_company_parser.add_argument('tech_id', help='기술 ID')
    recommend_company_parser.add_argument('--top', type=int, default=10, help='상위 N개 결과 (기본값: 10)')
    
    # 가중치 설정 명령어
    weights_parser = subparsers.add_parser('weights', help='가중치 설정 관리')
    weights_parser.add_argument('--show', action='store_true', help='현재 가중치 설정 표시')
    weights_parser.add_argument('--update', action='store_true', help='가중치 설정 업데이트')
    weights_parser.add_argument('--name', default='custom', help='가중치 설정 이름')
    weights_parser.add_argument('--text', type=float, help='텍스트 유사도 가중치')
    weights_parser.add_argument('--quant', type=float, help='정량적 데이터 가중치')
    weights_parser.add_argument('--category', type=float, help='카테고리 가중치')
    weights_parser.add_argument('--region', type=float, help='지역 가중치')
    weights_parser.add_argument('--size', type=float, help='기업 규모 가중치')
    
    # 평가 명령어
    evaluate_parser = subparsers.add_parser('evaluate', help='추천 시스템 평가')
    evaluate_parser.add_argument('--type', choices=['company_to_rnd', 'rnd_to_company'], 
                               required=True, help='평가할 추천 유형')
    evaluate_parser.add_argument('--sample', type=int, default=5, help='평가 샘플 크기 (기본값: 5)')
    
    # 목록 조회 명령어
    list_parser = subparsers.add_parser('list', help='항목 목록 조회')
    list_parser.add_argument('--type', choices=['companies', 'technologies'], 
                           required=True, help='조회할 항목 유형')
    list_parser.add_argument('--limit', type=int, default=10, help='최대 항목 수 (기본값: 10)')
    
    # 검색 명령어
    search_parser = subparsers.add_parser('search', help='항목 검색')
    search_parser.add_argument('--type', choices=['companies', 'technologies'], 
                             required=True, help='검색할 항목 유형')
    search_parser.add_argument('--keyword', required=True, help='검색 키워드')
    search_parser.add_argument('--limit', type=int, default=10, help='최대 결과 수 (기본값: 10)')
    
    args = parser.parse_args()
    
    # 명령어 처리
    if args.command == 'recommend-tech':
        print(f"\n기업 ID '{args.company_id}'에 적합한 R&D 기술 추천 중...")
        recommendations = recommend_rnd_tech_for_company(args.company_id, top_n=args.top)
        display_rnd_tech_recommendations(recommendations)
    
    elif args.command == 'recommend-company':
        print(f"\nR&D 기술 ID '{args.tech_id}'에 적합한 기업 추천 중...")
        recommendations = recommend_companies_for_rnd_tech(args.tech_id, top_n=args.top)
        display_company_recommendations(recommendations)
    
    elif args.command == 'weights':
        if args.show:
            weights = get_active_weight_settings()
            display_weight_settings(weights)
        
        elif args.update:
            if not all([args.text, args.quant, args.category, args.region, args.size]):
                print("모든 가중치 값을 지정해야 합니다.")
                return
            
            update_weight_settings(
                args.name,
                text_similarity_weight=args.text,
                quantitative_weight=args.quant,
                category_weight=args.category,
                region_weight=args.region,
                size_weight=args.size
            )
            
            print(f"가중치 설정 '{args.name}'이(가) 업데이트되었습니다.")
            weights = get_active_weight_settings()
            display_weight_settings(weights)
        
        else:
            print("--show 또는 --update 옵션을 지정해야 합니다.")
    
    elif args.command == 'evaluate':
        print(f"\n추천 시스템 평가 중 (유형: {args.type}, 샘플 크기: {args.sample})...")
        results = evaluate_recommendation_system(args.type, sample_size=args.sample)
        display_evaluation_results(results)
    
    elif args.command == 'list':
        list_items(args.type, limit=args.limit)
    
    elif args.command == 'search':
        search_items(args.type, args.keyword, limit=args.limit)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
