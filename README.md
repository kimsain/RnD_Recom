# R&D 기술 추천 시스템 사용 설명서

## 개요
이 시스템은 최신 딥러닝 알고리즘을 활용하여 기업 프로파일에 적합한 공공 R&D 기술을 추천하고, 보유 기술을 사업화할 수 있는 적합한 수요 기업을 예측합니다.

## 주요 기능
1. **기업 프로파일 맞춤형 공공 R&D 기술 추천**
   - 기업의 특성(규모, 매출액, 지역 등)에 최적화된 공공 R&D 기술을 적합도 순으로 추천
   - 상위 100건의 공공 R&D 기술 추천 목록 제공

2. **공공 R&D 기술의 유망 수요기업 예측**
   - 연구자가 보유한 공공 R&D 기술에 가장 적합한 수요 기업을 예측
   - 기술과 적합성이 높은 수요기업 리스트 제공 (기업 정보, 주요 제품, 재무 현황 등 포함)

## 기술 구현
- PGVector와 PostgreSQL을 이용한 시맨틱 서치 기반 유사도 추천
- OpenAI text-embedding-3-small 모델을 활용한 텍스트 임베딩
- 정량적 데이터와 정성적 데이터에 대한 가중치 조절 가능한 추천 알고리즘

## 시스템 요구사항
- Python 3.10 이상
- PostgreSQL 14 이상 (PGVector 확장 설치 필요)
- OpenAI API 키

## 설치 방법
1. 저장소 클론
```bash
git clone https://github.com/your-username/rnd_recommendation.git
cd rnd_recommendation
```

2. 가상 환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정 (.env 파일 생성)
```
DB_NAME=rnd_recommendation
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
OPENAI_API_KEY=your_openai_api_key
```

5. 데이터베이스 스키마 생성
```bash
python create_database_schema.py
```

6. 샘플 데이터 생성 및 로드
```bash
python create_sample_data.py
python load_data_to_db.py
```

7. 임베딩 생성
```bash
python generate_embeddings.py
```

## 사용 방법
시스템은 명령줄 인터페이스(CLI)를 통해 사용할 수 있습니다.

### 기본 명령어
```bash
# 도움말 표시
python cli.py --help

# 기업 목록 조회
python cli.py list --type companies --limit 10

# 기술 목록 조회
python cli.py list --type technologies --limit 10

# 기업 검색
python cli.py search --type companies --keyword "제조" --limit 10

# 기술 검색
python cli.py search --type technologies --keyword "인공지능" --limit 10
```

### 추천 기능
```bash
# 기업에 적합한 R&D 기술 추천
python cli.py recommend-tech COMP-00001 --top 10

# R&D 기술에 적합한 기업 추천
python cli.py recommend-company RND-00001 --top 10
```

### 가중치 설정
```bash
# 현재 가중치 설정 확인
python cli.py weights --show

# 가중치 설정 업데이트
python cli.py weights --update --name custom --text 0.6 --quant 0.4 --category 0.4 --region 0.3 --size 0.3
```

### 시스템 평가
```bash
# 기업에 대한 R&D 기술 추천 평가
python cli.py evaluate --type company_to_rnd --sample 5

# R&D 기술에 대한 기업 추천 평가
python cli.py evaluate --type rnd_to_company --sample 5
```

## 평가 지표
시스템은 다음과 같은 평가 지표를 제공합니다:

1. **커버리지(Coverage)**: 추천을 받은 항목 수 / 전체 항목 수
2. **다양성(Diversity)**: 추천 결과의 카테고리 다양성
3. **관련성(Relevance)**: 추천 결과와 대상 항목 간의 관련성
4. **유사도(Similarity)**: 추천 결과 간의 평균 유사도

## 시스템 구조
- **create_sample_data.py**: 샘플 데이터 생성
- **create_database_schema.py**: 데이터베이스 스키마 생성
- **load_data_to_db.py**: 데이터베이스에 샘플 데이터 로드
- **generate_embeddings.py**: 텍스트 임베딩 생성
- **recommendation_engine.py**: 추천 알고리즘 구현
- **evaluation_metrics.py**: 평가 지표 계산
- **cli.py**: 명령줄 인터페이스

## 문제 해결
- **임베딩 생성 오류**: OpenAI API 키가 올바르게 설정되었는지 확인
- **데이터베이스 연결 오류**: PostgreSQL 서버가 실행 중인지, 연결 정보가 올바른지 확인
- **PGVector 관련 오류**: PostgreSQL에 PGVector 확장이 설치되었는지 확인

## 향후 개선 사항
- 웹 인터페이스 구현
- 더 많은 데이터 소스 통합
- 실시간 추천 기능 개선
- 사용자 피드백 기반 추천 알고리즘 개선
