import os
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

# 환경 변수 로드
load_dotenv()

# 공공 R&D 기술 분야 정의
TECH_CATEGORIES = [
    "인공지능", "빅데이터", "클라우드", "사물인터넷", "블록체인", 
    "로봇공학", "자율주행", "스마트팩토리", "신재생에너지", "바이오헬스",
    "나노소재", "반도체", "디스플레이", "통신기술", "보안기술"
]

# 연구개발 단계 정의
RND_STAGES = ["기초연구", "응용연구", "개발연구", "상용화"]

# 지역 정의
REGIONS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]

# 산업 코드 정의 (10차 산업분류 기준 일부)
INDUSTRY_CODES = {
    "A": "농업, 임업 및 어업",
    "B": "광업",
    "C": "제조업",
    "D": "전기, 가스, 증기 및 공기 조절 공급업",
    "E": "수도, 하수 및 폐기물 처리, 원료 재생업",
    "F": "건설업",
    "G": "도매 및 소매업",
    "J": "정보통신업",
    "K": "금융 및 보험업",
    "M": "전문, 과학 및 기술 서비스업"
}

# 기업 규모 정의
COMPANY_SIZES = ["대기업", "중견기업", "중소기업", "스타트업"]

def generate_rnd_tech_data(num_samples=100):
    """공공 R&D 기술 샘플 데이터 생성"""
    rnd_data = []
    
    for i in range(1, num_samples + 1):
        # 기본 정보
        tech_id = f"RND-{i:05d}"
        tech_category = random.choice(TECH_CATEGORIES)
        tech_subcategory = f"{tech_category} {random.choice(['시스템', '플랫폼', '알고리즘', '솔루션', '기술'])}"
        
        # 기술 제목 생성
        tech_keywords = {
            "인공지능": ["딥러닝", "머신러닝", "자연어처리", "컴퓨터비전", "강화학습"],
            "빅데이터": ["데이터마이닝", "분석", "시각화", "예측모델", "클러스터링"],
            "클라우드": ["서버리스", "컨테이너", "가상화", "분산처리", "마이크로서비스"],
            "사물인터넷": ["센서", "네트워크", "모니터링", "제어", "연결"],
            "블록체인": ["스마트계약", "분산원장", "암호화", "합의알고리즘", "토큰"],
            "로봇공학": ["자동화", "제어", "인식", "모션", "협업"],
            "자율주행": ["인지", "판단", "제어", "맵핑", "센서퓨전"],
            "스마트팩토리": ["자동화", "모니터링", "최적화", "품질관리", "예지보전"],
            "신재생에너지": ["태양광", "풍력", "수소", "바이오매스", "지열"],
            "바이오헬스": ["진단", "치료", "모니터링", "약물전달", "재생의학"],
            "나노소재": ["합성", "코팅", "복합재", "필름", "촉매"],
            "반도체": ["설계", "공정", "패키징", "테스트", "집적"],
            "디스플레이": ["패널", "구동", "화질개선", "터치", "유연"],
            "통신기술": ["5G", "네트워크", "프로토콜", "무선", "대역폭"],
            "보안기술": ["암호화", "인증", "탐지", "방어", "복구"]
        }
        
        keywords = tech_keywords.get(tech_category, ["혁신", "첨단", "지능형", "융합", "고효율"])
        tech_title = f"{random.choice(keywords)} 기반 {tech_subcategory} {random.choice(['개발', '구축', '연구', '설계', '최적화'])}"
        
        # 기술 내용 생성
        tech_description = f"""
본 기술은 {tech_category} 분야의 {tech_subcategory}에 관한 것으로, {random.choice(['효율성', '정확성', '안정성', '확장성', '편의성'])}을 
{random.choice(['향상', '개선', '극대화', '최적화', '강화'])}하기 위한 {random.choice(['방법', '시스템', '알고리즘', '프레임워크', '플랫폼'])}을 제공합니다.

기존 {tech_category} 기술의 {random.choice(['한계', '문제점', '비효율성', '복잡성', '취약점'])}을 해결하기 위해, 
{random.choice(['새로운 접근법', '혁신적 방법론', '최신 알고리즘', '개선된 아키텍처', '최적화된 프로세스'])}를 적용하였습니다.

주요 특징:
1. {random.choice(['고성능', '고효율', '저비용', '사용자 친화적', '확장 가능한'])} 설계
2. {random.choice(['실시간', '자동화된', '지능형', '통합', '모듈화된'])} {random.choice(['처리', '분석', '제어', '관리', '최적화'])} 기능
3. {random.choice(['다양한', '유연한', '강력한', '직관적인', '안전한'])} {random.choice(['인터페이스', '알고리즘', '프로토콜', '메커니즘', '아키텍처'])}

본 기술은 {random.choice(['제조', '의료', '금융', '교통', '에너지', '환경', '농업', '국방'])} 분야에 적용 가능하며, 
{random.choice(['생산성 향상', '비용 절감', '품질 개선', '안전성 강화', '사용자 경험 향상'])}에 기여할 것으로 기대됩니다.
        """
        
        # 연구 정보
        rnd_stage = random.choice(RND_STAGES)
        region = random.choice(REGIONS)
        
        # 날짜 정보
        end_date = datetime.now() - timedelta(days=random.randint(30, 365*3))
        start_date = end_date - timedelta(days=random.randint(365, 365*5))
        
        # 연구 자원 및 실적
        research_budget = random.randint(5000, 50000) * 10000  # 5천만원 ~ 5억원
        research_team_size = random.randint(3, 20)
        patents = random.randint(0, 5)
        papers = random.randint(0, 10)
        
        # 데이터 구조화
        tech_data = {
            "tech_id": tech_id,
            "tech_title": tech_title,
            "tech_category": tech_category,
            "tech_subcategory": tech_subcategory,
            "tech_description": tech_description,
            "rnd_stage": rnd_stage,
            "region": region,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "research_budget": research_budget,
            "research_team_size": research_team_size,
            "patents": patents,
            "papers": papers
        }
        
        rnd_data.append(tech_data)
    
    return rnd_data

def generate_company_data(num_samples=100):
    """기업 프로파일 샘플 데이터 생성"""
    company_data = []
    
    for i in range(1, num_samples + 1):
        # 기본 정보
        company_id = f"COMP-{i:05d}"
        company_name = f"{random.choice(['테크', '솔루션', '시스템', '인포', '디지털', '스마트', '이노베이션', '퓨처', '넥스트', '글로벌'])}"\
                      f"{random.choice(['코리아', '테크', '소프트', '시스템즈', '랩스', '네트웍스', '인포', '솔루션즈', '파트너스', '그룹'])}"
        
        # 설립일
        founded_years_ago = random.randint(1, 30)
        founded_date = datetime.now() - timedelta(days=founded_years_ago*365)
        
        # 산업 코드
        industry_code = random.choice(list(INDUSTRY_CODES.keys()))
        industry_name = INDUSTRY_CODES[industry_code]
        
        # 기업 규모
        company_size = random.choice(COMPANY_SIZES)
        employees = 0
        if company_size == "대기업":
            employees = random.randint(1000, 10000)
        elif company_size == "중견기업":
            employees = random.randint(300, 999)
        elif company_size == "중소기업":
            employees = random.randint(50, 299)
        else:  # 스타트업
            employees = random.randint(5, 49)
        
        # 재무 정보
        revenue_per_employee = 0
        if company_size == "대기업":
            revenue_per_employee = random.randint(300, 1000) * 1000000
        elif company_size == "중견기업":
            revenue_per_employee = random.randint(200, 500) * 1000000
        elif company_size == "중소기업":
            revenue_per_employee = random.randint(100, 300) * 1000000
        else:  # 스타트업
            revenue_per_employee = random.randint(50, 200) * 1000000
        
        annual_revenue = employees * revenue_per_employee
        total_assets = annual_revenue * random.uniform(0.8, 2.5)
        total_capital = total_assets * random.uniform(0.3, 0.7)
        
        # 위치
        region = random.choice(REGIONS)
        
        # 주요 제품
        main_products = []
        if industry_code == "C":  # 제조업
            products = ["스마트 센서", "제어 시스템", "자동화 장비", "전자 부품", "산업용 로봇", "측정 장비"]
            main_products = random.sample(products, k=min(3, len(products)))
        elif industry_code == "J":  # 정보통신업
            products = ["클라우드 서비스", "모바일 앱", "보안 솔루션", "데이터 분석 플랫폼", "AI 솔루션", "IoT 플랫폼"]
            main_products = random.sample(products, k=min(3, len(products)))
        elif industry_code == "M":  # 전문, 과학 및 기술 서비스업
            products = ["R&D 컨설팅", "기술 특허 서비스", "엔지니어링 설계", "기술 검증 서비스", "연구 개발 지원"]
            main_products = random.sample(products, k=min(3, len(products)))
        else:
            products = ["솔루션", "서비스", "플랫폼", "시스템", "컨설팅", "관리 서비스"]
            main_products = random.sample(products, k=min(3, len(products)))
        
        # 사업 목적
        business_purposes = [
            f"{industry_name} 분야의 {random.choice(['혁신적인', '효율적인', '지속가능한', '고품질', '경쟁력 있는'])} "\
            f"{random.choice(['제품', '서비스', '솔루션', '시스템', '플랫폼'])} 제공",
            
            f"{random.choice(['국내', '글로벌', '산업', '시장'])} 내 {random.choice(['선도적', '혁신적', '차별화된', '특화된'])} "\
            f"{random.choice(['기술', '서비스', '제품', '솔루션'])} 개발",
            
            f"{random.choice(['고객', '파트너', '이해관계자'])}에게 {random.choice(['최고의', '최적화된', '맞춤형', '혁신적인'])} "\
            f"{random.choice(['가치', '경험', '성과', '솔루션'])} 제공"
        ]
        
        business_purpose = random.choice(business_purposes)
        
        # 데이터 구조화
        company_profile = {
            "company_id": company_id,
            "company_name": company_name,
            "founded_date": founded_date.strftime("%Y-%m-%d"),
            "industry_code": industry_code,
            "industry_name": industry_name,
            "company_size": company_size,
            "employees": employees,
            "annual_revenue": annual_revenue,
            "total_assets": total_assets,
            "total_capital": total_capital,
            "region": region,
            "main_products": main_products,
            "business_purpose": business_purpose
        }
        
        company_data.append(company_profile)
    
    return company_data

def save_sample_data(rnd_data, company_data):
    """샘플 데이터를 JSON 파일로 저장"""
    # 디렉토리 생성
    os.makedirs("data", exist_ok=True)
    
    # 공공 R&D 기술 데이터 저장
    with open("data/rnd_tech_data.json", "w", encoding="utf-8") as f:
        json.dump(rnd_data, f, ensure_ascii=False, indent=2)
    
    # 기업 프로파일 데이터 저장
    with open("data/company_data.json", "w", encoding="utf-8") as f:
        json.dump(company_data, f, ensure_ascii=False, indent=2)
    
    print(f"공공 R&D 기술 데이터 {len(rnd_data)}개와 기업 프로파일 데이터 {len(company_data)}개가 생성되어 저장되었습니다.")

def main():
    # 샘플 데이터 생성
    print("샘플 데이터 생성 중...")
    rnd_data = generate_rnd_tech_data(num_samples=200)
    company_data = generate_company_data(num_samples=150)
    
    # 데이터 저장
    save_sample_data(rnd_data, company_data)
    
    # 데이터 미리보기
    print("\n공공 R&D 기술 데이터 미리보기:")
    rnd_df = pd.DataFrame(rnd_data)
    print(rnd_df[['tech_id', 'tech_title', 'tech_category', 'rnd_stage']].head())
    
    print("\n기업 프로파일 데이터 미리보기:")
    company_df = pd.DataFrame(company_data)
    print(company_df[['company_id', 'company_name', 'industry_name', 'company_size']].head())

if __name__ == "__main__":
    main()
