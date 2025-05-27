import os
import json
import psycopg2
import pandas as pd
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

# OpenAI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

def load_sample_data():
    """샘플 데이터 로드 함수"""
    # 공공 R&D 기술 데이터 로드
    with open("data/rnd_tech_data.json", "r", encoding="utf-8") as f:
        rnd_data = json.load(f)
    
    # 기업 프로파일 데이터 로드
    with open("data/company_data.json", "r", encoding="utf-8") as f:
        company_data = json.load(f)
    
    return rnd_data, company_data

def insert_rnd_data(conn, rnd_data):
    """공공 R&D 기술 데이터 삽입 함수"""
    cursor = conn.cursor()
    
    for tech in rnd_data:
        cursor.execute("""
        INSERT INTO rnd_technologies (
            tech_id, tech_title, tech_category, tech_subcategory, tech_description,
            rnd_stage, region, start_date, end_date, research_budget,
            research_team_size, patents, papers
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (tech_id) DO UPDATE SET
            tech_title = EXCLUDED.tech_title,
            tech_category = EXCLUDED.tech_category,
            tech_subcategory = EXCLUDED.tech_subcategory,
            tech_description = EXCLUDED.tech_description,
            rnd_stage = EXCLUDED.rnd_stage,
            region = EXCLUDED.region,
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            research_budget = EXCLUDED.research_budget,
            research_team_size = EXCLUDED.research_team_size,
            patents = EXCLUDED.patents,
            papers = EXCLUDED.papers
        """, (
            tech["tech_id"], tech["tech_title"], tech["tech_category"], tech["tech_subcategory"],
            tech["tech_description"], tech["rnd_stage"], tech["region"], tech["start_date"],
            tech["end_date"], tech["research_budget"], tech["research_team_size"],
            tech["patents"], tech["papers"]
        ))
    
    conn.commit()
    print(f"{len(rnd_data)}개의 공공 R&D 기술 데이터가 데이터베이스에 저장되었습니다.")

def insert_company_data(conn, company_data):
    """기업 프로파일 데이터 삽입 함수"""
    cursor = conn.cursor()
    
    for company in company_data:
        cursor.execute("""
        INSERT INTO companies (
            company_id, company_name, founded_date, industry_code, industry_name,
            company_size, employees, annual_revenue, total_assets, total_capital,
            region, main_products, business_purpose
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (company_id) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            founded_date = EXCLUDED.founded_date,
            industry_code = EXCLUDED.industry_code,
            industry_name = EXCLUDED.industry_name,
            company_size = EXCLUDED.company_size,
            employees = EXCLUDED.employees,
            annual_revenue = EXCLUDED.annual_revenue,
            total_assets = EXCLUDED.total_assets,
            total_capital = EXCLUDED.total_capital,
            region = EXCLUDED.region,
            main_products = EXCLUDED.main_products,
            business_purpose = EXCLUDED.business_purpose
        """, (
            company["company_id"], company["company_name"], company["founded_date"],
            company["industry_code"], company["industry_name"], company["company_size"],
            company["employees"], company["annual_revenue"], company["total_assets"],
            company["total_capital"], company["region"], company["main_products"],
            company["business_purpose"]
        ))
    
    conn.commit()
    print(f"{len(company_data)}개의 기업 프로파일 데이터가 데이터베이스에 저장되었습니다.")

def main():
    # 샘플 데이터 로드
    print("샘플 데이터 로드 중...")
    rnd_data, company_data = load_sample_data()
    
    # 데이터베이스 연결
    print("데이터베이스에 연결 중...")
    conn = connect_to_db()
    
    # 데이터 삽입
    print("공공 R&D 기술 데이터 삽입 중...")
    insert_rnd_data(conn, rnd_data)
    
    print("기업 프로파일 데이터 삽입 중...")
    insert_company_data(conn, company_data)
    
    # 연결 종료
    conn.close()
    print("데이터베이스 연결이 종료되었습니다.")

if __name__ == "__main__":
    main()
