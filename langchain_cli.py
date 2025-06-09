"""
LangChain 기반 R&D 기술 추천 시스템 CLI

이 모듈은 LangChain을 활용한 R&D 기술 추천 시스템의 명령줄 인터페이스를 제공합니다.
"""

import os
import argparse
import pandas as pd
from typing import List, Dict, Any
from langchain_recommendation_engine import LangChainRecommendationEngine
from langchain_evaluation_metrics import LangChainEvaluationMetrics
from langchain_config import APIConfig

def format_currency(value):
    """통화 형식으로 포맷팅하는 함수"""
    if value is None:
        return "정보 없음"
    return f"{value:,}원"

def display_rnd_tech_recommendations(recommendations: List[Dict[str, Any]]):
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
    
    # 상위 3개 기술 상세 정보 출력
    print("\n[상위 3개 기술 상세 정보]")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n{i}. {rec['tech_title']} ({rec['tech_id']})")
        print(f"   기술분야: {rec['tech_category']}")
        print(f"   연구기관: {rec['research_institution']}")
        print(f"   기술성숙도: {rec['tech_readiness_level']}")
        print(f"   기대효과: {rec['expected_effect'][:100]}...")
        print(f"   유사도 점수: {rec['similarity_score']:.4f}")
        print(f"   최종 점수: {rec['final_score']:.4f}")

def display_company_recommendations(recommendations: List[Dict[str, Any]]):
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
        print(f"   주요 제품: {rec['main_products'][:100]}...")
        print(f"   유사도 점수: {rec['similarity_score']:.4f}")
        print(f"   최종 점수: {rec['final_score']:.4f}")

def display_evaluation_results(results: Dict[str, Any]):
    """평가 결과 출력 함수"""
    if not results:
        print("평가 결과가 없습니다.")
        return
    
    print("\n" + "="*50)
    print(f"LangChain 추천 시스템 평가 결과: {results['recommendation_type']}")
    print("-"*50)
    print(f"커버리지: {results['coverage']:.4f}")
    print(f"다양성: {results['diversity']:.4f}")
    print(f"관련성: {results['relevance']:.4f}")
    print(f"유사도: {results['similarity']:.4f}")
    print(f"샘플 크기: {results['sample_size']}")
    print("="*50)

def display_weight_settings(weights: Dict[str, Any]):
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

def list_items(item_type: str, limit: int = 10):
    """항목 목록 출력 함수"""
    engine = LangChainRecommendationEngine()
    conn = engine.connect_to_db()
    cursor = conn.cursor()
    
    try:
        if item_type == "companies":
            cursor.execute("""
            SELECT c.company_id, c.company_name, i.industry_name, c.company_size, c.region
            FROM companies c
            JOIN industries i ON c.industry_id = i.industry_id
            LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            print(f"\n기업 목록 (상위 {limit}개)")
            print("-" * 80)
            print(f"{'기업 ID':<12}{'기업명':<25}{'산업':<20}{'규모':<10}{'지역'}")
            print("-" * 80)
            
            for row in results:
                print(f"{row[0]:<12}{row[1][:23]:<25}{row[2][:18]:<20}{row[3]:<10}{row[4]}")
                
        elif item_type == "technologies":
            cursor.execute("""
            SELECT tech_id, tech_title, tech_category, research_institution
            FROM rnd_technologies
            LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            print(f"\nR&D 기술 목록 (상위 {limit}개)")
            print("-" * 80)
            print(f"{'기술 ID':<12}{'기술명':<30}{'분야':<15}{'연구기관'}")
            print("-" * 80)
            
            for row in results:
                print(f"{row[0]:<12}{row[1][:28]:<30}{row[2]:<15}{row[3]}")
        
    except Exception as e:
        print(f"목록 조회 오류: {e}")
    finally:
        cursor.close()
        conn.close()

def search_items(item_type: str, keyword: str, limit: int = 10):
    """항목 검색 함수"""
    engine = LangChainRecommendationEngine()
    conn = engine.connect_to_db()
    cursor = conn.cursor()
    
    try:
        if item_type == "companies":
            cursor.execute("""
            SELECT c.company_id, c.company_name, i.industry_name, c.company_size, c.region
            FROM companies c
            JOIN industries i ON c.industry_id = i.industry_id
            WHERE c.company_name ILIKE %s OR i.industry_name ILIKE %s OR c.main_products ILIKE %s
            LIMIT %s
            """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit))
            
            results = cursor.fetchall()
            print(f"\n기업 검색 결과: '{keyword}' (상위 {limit}개)")
            print("-" * 80)
            print(f"{'기업 ID':<12}{'기업명':<25}{'산업':<20}{'규모':<10}{'지역'}")
            print("-" * 80)
            
            for row in results:
                print(f"{row[0]:<12}{row[1][:23]:<25}{row[2][:18]:<20}{row[3]:<10}{row[4]}")
                
        elif item_type == "technologies":
            cursor.execute("""
            SELECT tech_id, tech_title, tech_category, research_institution
            FROM rnd_technologies
            WHERE tech_title ILIKE %s OR tech_category ILIKE %s OR tech_description ILIKE %s
            LIMIT %s
            """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit))
            
            results = cursor.fetchall()
            print(f"\nR&D 기술 검색 결과: '{keyword}' (상위 {limit}개)")
            print("-" * 80)
            print(f"{'기술 ID':<12}{'기술명':<30}{'분야':<15}{'연구기관'}")
            print("-" * 80)
            
            for row in results:
                print(f"{row[0]:<12}{row[1][:28]:<30}{row[2]:<15}{row[3]}")
        
    except Exception as e:
        print(f"검색 오류: {e}")
    finally:
        cursor.close()
        conn.close()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LangChain 기반 R&D 기술 추천 시스템")
    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령어")
    
    # 목록 조회 명령어
    list_parser = subparsers.add_parser("list", help="항목 목록 조회")
    list_parser.add_argument("--type", choices=["companies", "technologies"], required=True, help="조회할 항목 타입")
    list_parser.add_argument("--limit", type=int, default=10, help="조회할 항목 수")
    
    # 검색 명령어
    search_parser = subparsers.add_parser("search", help="항목 검색")
    search_parser.add_argument("--type", choices=["companies", "technologies"], required=True, help="검색할 항목 타입")
    search_parser.add_argument("--keyword", required=True, help="검색 키워드")
    search_parser.add_argument("--limit", type=int, default=10, help="검색 결과 수")
    
    # R&D 기술 추천 명령어
    recommend_tech_parser = subparsers.add_parser("recommend-tech", help="기업에 적합한 R&D 기술 추천")
    recommend_tech_parser.add_argument("company_id", help="기업 ID")
    recommend_tech_parser.add_argument("--top", type=int, default=10, help="추천할 기술 수")
    
    # 기업 추천 명령어
    recommend_company_parser = subparsers.add_parser("recommend-company", help="R&D 기술에 적합한 기업 추천")
    recommend_company_parser.add_argument("tech_id", help="기술 ID")
    recommend_company_parser.add_argument("--top", type=int, default=10, help="추천할 기업 수")
    
    # 가중치 설정 명령어
    weights_parser = subparsers.add_parser("weights", help="가중치 설정 관리")
    weights_parser.add_argument("--show", action="store_true", help="현재 가중치 설정 표시")
    weights_parser.add_argument("--update", action="store_true", help="가중치 설정 업데이트")
    weights_parser.add_argument("--name", help="설정 이름")
    weights_parser.add_argument("--text", type=float, help="텍스트 유사도 가중치")
    weights_parser.add_argument("--quant", type=float, help="정량적 데이터 가중치")
    weights_parser.add_argument("--category", type=float, help="카테고리 가중치")
    weights_parser.add_argument("--region", type=float, help="지역 가중치")
    weights_parser.add_argument("--size", type=float, help="기업 규모 가중치")
    
    # 평가 명령어
    evaluate_parser = subparsers.add_parser("evaluate", help="추천 시스템 평가")
    evaluate_parser.add_argument("--type", choices=["company_to_rnd", "rnd_to_company"], required=True, help="평가 타입")
    evaluate_parser.add_argument("--sample", type=int, default=5, help="평가 샘플 수")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 추천 엔진 초기화
    engine = LangChainRecommendationEngine()
    
    try:
        if args.command == "list":
            list_items(args.type, args.limit)
            
        elif args.command == "search":
            search_items(args.type, args.keyword, args.limit)
            
        elif args.command == "recommend-tech":
            print(f"\n기업 {args.company_id}에 적합한 R&D 기술 추천 (LangChain 기반)")
            recommendations = engine.recommend_rnd_tech_for_company(args.company_id, args.top)
            display_rnd_tech_recommendations(recommendations)
            
        elif args.command == "recommend-company":
            print(f"\nR&D 기술 {args.tech_id}에 적합한 기업 추천 (LangChain 기반)")
            recommendations = engine.recommend_companies_for_rnd_tech(args.tech_id, args.top)
            display_company_recommendations(recommendations)
            
        elif args.command == "weights":
            if args.show:
                weights = engine.get_active_weight_settings()
                display_weight_settings(weights)
            elif args.update:
                if not args.name:
                    print("오류: 설정 이름(--name)이 필요합니다.")
                    return
                
                weight_params = {}
                if args.text is not None:
                    weight_params['text_similarity_weight'] = args.text
                if args.quant is not None:
                    weight_params['quantitative_weight'] = args.quant
                if args.category is not None:
                    weight_params['category_weight'] = args.category
                if args.region is not None:
                    weight_params['region_weight'] = args.region
                if args.size is not None:
                    weight_params['size_weight'] = args.size
                
                success = engine.update_weight_settings(args.name, **weight_params)
                if success:
                    print(f"가중치 설정 '{args.name}'이 성공적으로 업데이트되었습니다.")
                    weights = engine.get_active_weight_settings()
                    display_weight_settings(weights)
                else:
                    print("가중치 설정 업데이트에 실패했습니다.")
            else:
                weights = engine.get_active_weight_settings()
                display_weight_settings(weights)
                
        elif args.command == "evaluate":
            print(f"\nLangChain 추천 시스템 평가 중... (타입: {args.type}, 샘플: {args.sample})")
            evaluator = LangChainEvaluationMetrics()
            results = evaluator.evaluate_recommendation_system(args.type, args.sample)
            display_evaluation_results(results)
            
    except Exception as e:
        print(f"명령 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()

