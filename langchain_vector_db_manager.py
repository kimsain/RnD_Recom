"""
LangChain 기반 벡터 데이터베이스 관리 클래스

이 모듈은 LangChain의 PGVector를 활용하여 벡터 데이터베이스를 관리하는 기능을 제공합니다.
"""

from langchain.vectorstores import PGVector
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import APIConfig
import os
from typing import List, Dict, Any, Optional

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = APIConfig.OPENAI_API_KEY

class LangChainVectorDBManager:
    """
    LangChain PGVector를 활용한 벡터 데이터베이스 관리 클래스
    """
    
    def __init__(self, collection_name="documents"):
        """
        벡터 데이터베이스 매니저 초기화
        
        Args:
            collection_name (str): 컬렉션(테이블) 이름
        """
        self.collection_name = collection_name
        
        # DB 연결 정보 구성
        db_params = APIConfig.get_db_connection_params()
        self.connection_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model=APIConfig.OPENAI_EMBEDDING_MODEL,
            openai_api_key=APIConfig.OPENAI_API_KEY
        )
        
        # 텍스트 스플리터 초기화 (선택적 사용)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        
        # PGVector 인스턴스 (필요시 초기화)
        self.vectorstore = None
    
    def _get_vectorstore(self):
        """
        PGVector 인스턴스를 가져오거나 생성합니다.
        
        Returns:
            PGVector: 벡터스토어 인스턴스
        """
        if self.vectorstore is None:
            try:
                # 기존 컬렉션에 연결 시도
                self.vectorstore = PGVector(
                    collection_name=self.collection_name,
                    connection_string=self.connection_string,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"기존 벡터스토어 연결 실패: {e}")
                # 새로운 컬렉션 생성
                self.vectorstore = PGVector.from_texts(
                    texts=["초기화 문서"],  # 더미 문서로 초기화
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection_string=self.connection_string
                )
                # 더미 문서 삭제
                try:
                    self.vectorstore.delete(["0"])  # 첫 번째 문서 삭제 시도
                except:
                    pass
        
        return self.vectorstore
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None, use_splitter: bool = False):
        """
        문서를 벡터 데이터베이스에 추가
        
        Args:
            content (str): 문서 내용
            metadata (dict): 문서 메타데이터
            use_splitter (bool): 텍스트 분할 사용 여부
            
        Returns:
            list: 추가된 문서 ID 목록
        """
        try:
            vectorstore = self._get_vectorstore()
            
            if use_splitter:
                # 텍스트 분할 사용
                chunks = self.text_splitter.split_text(content)
                documents = [
                    Document(page_content=chunk, metadata=metadata or {})
                    for chunk in chunks
                ]
            else:
                # 텍스트 분할 없이 전체 문서 사용
                documents = [Document(page_content=content, metadata=metadata or {})]
            
            # 문서 추가
            doc_ids = vectorstore.add_documents(documents)
            return doc_ids
            
        except Exception as e:
            print(f"문서 추가 오류: {e}")
            return []
    
    def search_similar_documents(self, query_text: str, limit: int = 5, score_threshold: float = None):
        """
        쿼리 텍스트와 유사한 문서 검색
        
        Args:
            query_text (str): 검색 쿼리 텍스트
            limit (int): 반환할 최대 문서 수
            score_threshold (float): 유사도 임계값
            
        Returns:
            list: 유사한 문서 목록
        """
        try:
            vectorstore = self._get_vectorstore()
            
            if score_threshold is not None:
                # 유사도 임계값을 사용한 검색
                docs_with_scores = vectorstore.similarity_search_with_score(
                    query_text, k=limit
                )
                # 임계값 필터링
                filtered_docs = [
                    (doc, score) for doc, score in docs_with_scores 
                    if score >= score_threshold
                ]
            else:
                # 일반 유사도 검색
                docs_with_scores = vectorstore.similarity_search_with_score(
                    query_text, k=limit
                )
                filtered_docs = docs_with_scores
            
            # 결과 포맷팅
            results = []
            for i, (doc, score) in enumerate(filtered_docs):
                results.append({
                    "id": i,  # PGVector에서는 실제 ID 추출이 복잡할 수 있음
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": float(1 - score)  # 거리를 유사도로 변환
                })
            
            return results
            
        except Exception as e:
            print(f"유사 문서 검색 오류: {e}")
            return []
    
    def delete_collection(self):
        """
        전체 컬렉션 삭제
        
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            vectorstore = self._get_vectorstore()
            vectorstore.delete_collection()
            self.vectorstore = None  # 인스턴스 재설정
            return True
        except Exception as e:
            print(f"컬렉션 삭제 오류: {e}")
            return False
    
    def import_news_data(self, news_api, query=None, category=None, limit=10, use_splitter=False):
        """
        뉴스 API에서 데이터를 가져와 벡터 데이터베이스에 저장
        
        Args:
            news_api: 뉴스 API 인스턴스
            query (str): 검색 쿼리 (선택적)
            category (str): 뉴스 카테고리 (선택적)
            limit (int): 가져올 뉴스 수
            use_splitter (bool): 텍스트 분할 사용 여부
            
        Returns:
            list: 추가된 문서 ID 목록
        """
        try:
            news_data = []
            
            # 쿼리로 검색
            if query:
                news_data = news_api.search_news(query, limit)
            # 카테고리로 검색
            elif category:
                news_data = news_api.get_news_by_category(category, limit)
            # 최신 뉴스 가져오기
            else:
                news_data = news_api.get_recent_news(limit)
            
            # 뉴스 데이터를 벡터 데이터베이스에 저장
            all_doc_ids = []
            for news in news_data:
                # 뉴스 내용 구성
                content = f"{news['title']}\n\n{news['content']}"
                
                # 메타데이터 구성
                metadata = {
                    "source": news.get("source", ""),
                    "date": news.get("date", ""),
                    "category": news.get("category", ""),
                    "keywords": news.get("keywords", []),
                    "news_id": news.get("id", ""),
                    "title": news.get("title", "")
                }
                
                # 문서 추가
                doc_ids = self.add_document(content, metadata, use_splitter)
                all_doc_ids.extend(doc_ids)
            
            return all_doc_ids
            
        except Exception as e:
            print(f"뉴스 데이터 가져오기 오류: {e}")
            return []