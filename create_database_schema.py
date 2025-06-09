import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

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
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn

def create_tables():
    """데이터베이스 테이블 생성 함수"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # 공공 R&D 기술 테이블 생성
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rnd_technologies (
        tech_id VARCHAR(10) PRIMARY KEY,
        tech_title TEXT NOT NULL,
        tech_category VARCHAR(50) NOT NULL,
        tech_subcategory VARCHAR(100),
        tech_description TEXT,
        rnd_stage VARCHAR(20) NOT NULL,
        region VARCHAR(20),
        start_date DATE,
        end_date DATE,
        research_budget BIGINT,
        research_team_size INTEGER,
        patents INTEGER,
        papers INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 기업 프로파일 테이블 생성
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS companies (
        company_id VARCHAR(10) PRIMARY KEY,
        company_name VARCHAR(100) NOT NULL,
        founded_date DATE,
        industry_code CHAR(1),
        industry_name VARCHAR(100),
        company_size VARCHAR(20),
        employees INTEGER,
        annual_revenue BIGINT,
        total_assets BIGINT,
        total_capital BIGINT,
        region VARCHAR(20),
        main_products TEXT[],
        business_purpose TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 임베딩 저장을 위한 테이블 생성 (PGVector 활용)
    # 공공 R&D 기술 임베딩 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rnd_embeddings (
        id SERIAL PRIMARY KEY,
        tech_id VARCHAR(20) NOT NULL,
        embedding_type VARCHAR(50) NOT NULL,
        embedding vector(1536),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_rnd_embedding UNIQUE (tech_id, embedding_type)
    )
    """)
    
    # 기업 프로파일 임베딩 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS company_embeddings (
        id SERIAL PRIMARY KEY,
        company_id VARCHAR(20) NOT NULL,
        embedding_type VARCHAR(50) NOT NULL,
        embedding vector(1536),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_company_embedding UNIQUE (company_id, embedding_type)
    )
    """)
    
    # 추천 결과 저장 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        id SERIAL PRIMARY KEY,
        source_id VARCHAR(20) NOT NULL,
        target_id VARCHAR(20) NOT NULL,
        recommendation_type VARCHAR(50) NOT NULL,
        similarity_score FLOAT,
        rank INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_recommendation UNIQUE (source_id, target_id, recommendation_type)
    )
    """)
    
    # 가중치 설정 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weight_settings (
        id SERIAL PRIMARY KEY,
        setting_name VARCHAR(50) UNIQUE NOT NULL,
        description TEXT,
        text_similarity_weight FLOAT DEFAULT 0.7,
        quantitative_weight FLOAT DEFAULT 0.3,
        category_weight FLOAT DEFAULT 0.5,
        region_weight FLOAT DEFAULT 0.2,
        size_weight FLOAT DEFAULT 0.3,
        is_active BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 기본 가중치 설정 추가
    cursor.execute("""
    INSERT INTO weight_settings 
    (setting_name, description, text_similarity_weight, quantitative_weight, category_weight, region_weight, size_weight, is_active)
    VALUES 
    ('default', '기본 가중치 설정', 0.7, 0.3, 0.5, 0.2, 0.3, TRUE)
    ON CONFLICT (setting_name) DO NOTHING;
    """)
    
    # 인덱스 생성
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_rnd_tech_category ON rnd_technologies(tech_category);
    CREATE INDEX IF NOT EXISTS idx_rnd_region ON rnd_technologies(region);
    CREATE INDEX IF NOT EXISTS idx_rnd_stage ON rnd_technologies(rnd_stage);
    
    CREATE INDEX IF NOT EXISTS idx_company_industry ON companies(industry_code);
    CREATE INDEX IF NOT EXISTS idx_company_region ON companies(region);
    CREATE INDEX IF NOT EXISTS idx_company_size ON companies(company_size);
    """)
    
    # PGVector 인덱스 생성
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS rnd_embeddings_idx ON rnd_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS company_embeddings_idx ON company_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """)
    
    cursor.close()
    conn.close()
    
    print("데이터베이스 테이블이 성공적으로 생성되었습니다.")

if __name__ == "__main__":
    create_tables()
