# LangChain 기반 R&D 기술 추천 시스템

이 프로젝트는 LangChain 프레임워크를 활용하여 기업 프로파일에 적합한 공공 R&D 기술을 추천하고, 보유 기술을 사업화할 수 있는 적합한 수요 기업을 예측하는 시스템입니다.

## 프로젝트 발전 과정

### 1단계: MVP 개발 (LangChain 기반)
빠른 프로토타이핑을 위해 LangChain의 다양한 모듈을 활용하여 핵심 기능을 구현했습니다.

### 2단계: 최적화 (직접 구현)
LLM 모델이 확정되고 성능 최적화가 필요해지면서 일부 컴포넌트를 직접 구현으로 전환할 예정입니다.
- 임베딩: `openai.Embedding.create()` 직접 호출
- 벡터 DB: `psycopg2`로 PostgreSQL 직접 관리
- 텍스트 처리: 정형 데이터 기반 문자열 구성

## 주요 기능

### LangChain 컴포넌트
- **OpenAIEmbeddings**: 텍스트 임베딩 생성
- **PGVector**: PostgreSQL 기반 벡터 데이터베이스
- **RecursiveCharacterTextSplitter**: 텍스트 청크 분할
- **LLMChain**: 추천 사유 생성을 위한 LLM 체인

### 핵심 기능
1. **기업 프로파일 맞춤형 공공 R&D 기술 추천**
   - 기업의 특성(규모, 매출액, 지역 등)에 최적화된 공공 R&D 기술을 적합도 순으로 추천
   - 상위 10개의 공공 R&D 기술 추천 목록 제공

2. **공공 R&D 기술의 유망 수요기업 예측**
   - 연구자가 보유한 공공 R&D 기술에 가장 적합한 수요 기업을 예측
   - 기술과 적합성이 높은 수요기업 리스트 제공 (기업 정보, 주요 제품, 재무 현황 등 포함)

3. **LLM 기반 추천 사유 생성**
   - LangChain의 LLMChain을 활용한 추천 사유 자동 생성
   - 200자 이내의 간결하고 명확한 매칭 사유 제공

4. **가중치 기반 스코어링 시스템**
   - 텍스트 유사도, 카테고리 매칭, 지역, 기업 규모 등 다양한 요소를 고려한 종합 점수 계산
   - 사용자가 손쉽게 가중치를 조절할 수 있는 시스템

## 기술 구현

### LangChain 기반 아키텍처
- **임베딩 생성**: LangChain OpenAIEmbeddings를 사용한 text-embedding-3-small 모델 활용
- **벡터 검색**: LangChain PGVector를 통한 PostgreSQL 기반 시맨틱 서치
- **텍스트 분할**: LangChain RecursiveCharacterTextSplitter를 활용한 청크 기반 처리 (선택적)
- **추천 사유 생성**: LangChain LLMChain과 GPT-4o 모델을 활용한 자연어 생성

### 데이터베이스 설계
- **PostgreSQL 14+** with **PGVector** 확장
- **벡터 인덱스**: ivfflat 인덱스를 활용한 고속 유사도 검색
- **정형 데이터**: 기업 정보, R&D 기술 정보, 가중치 설정 등

## 파일 구조

```
langchain/
├── langchain_main.py                   # 메인 실행 스크립트
├── langchain_recommendation_engine.py  # LangChain 기반 추천 엔진
├── langchain_cli.py                    # LangChain 기반 CLI
├── langchain_database_manager.py       # LangChain 기반 데이터베이스 관리
├── langchain_evaluation_metrics.py     # LangChain 기반 평가 지표
├── langchain_vector_db_manager.py      # LangChain PGVector 관리
├── langchain_embedding_utils.py        # LangChain 임베딩 유틸리티
├── langchain_text_splitter.py          # LangChain 텍스트 분할
└── README.md                           # 프로젝트 문서
```

### 최적화 구현 파일들
```
Rnd_recom/
├── cli.py                              # 최적화된 CLI
├── recommendation_engine.py            # 직접 구현 추천 엔진
├── create_database_schema.py           # 데이터베이스 스키마
├── generate_embeddings.py              # 직접 구현 임베딩 생성
├── evaluation_metrics.py               # 직접 구현 평가 지표
├── create_sample_data.py               # 샘플 데이터 생성
├── load_data_to_db.py                  # 데이터 로드
├── system_test.py                      # 시스템 테스트
└── requirements.txt                    # 의존성 목록
```

### 주요 차이점

| 구성 요소 | LangChain MVP | 최적화 버전 | 주요 차이점 |
|-----------|---------------|-------------|-------------|
| **추천 엔진** | `langchain_recommendation_engine.py` | `recommendation_engine.py` | LangChain 컴포넌트 vs 직접 구현 |
| **벡터 DB** | `langchain_vector_db_manager.py` | 직접 SQL 쿼리 | PGVector vs psycopg2 직접 사용 |
| **임베딩** | `langchain_embedding_utils.py` | `generate_embeddings.py` | OpenAIEmbeddings vs openai.Embedding.create() |
| **텍스트 처리** | `langchain_text_splitter.py` | 정형 데이터 결합 | RecursiveCharacterTextSplitter vs 단순 결합 |
| **CLI** | `langchain_cli.py` | `cli.py` | LangChain 통합 vs 독립적 구현 |
