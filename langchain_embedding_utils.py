"""
LangChain 기반 임베딩 유틸리티 모듈

이 모듈은 LangChain의 OpenAIEmbeddings를 사용하여 텍스트의 임베딩 벡터를 생성하는 기능을 제공합니다.
"""

from langchain.embeddings import OpenAIEmbeddings
from config import APIConfig
import os

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = APIConfig.OPENAI_API_KEY

class LangChainEmbeddingUtils:
    """
    LangChain을 사용한 임베딩 유틸리티 클래스
    """
    
    def __init__(self, model_name=None):
        """
        임베딩 모델 초기화
        
        Args:
            model_name (str): 사용할 OpenAI 임베딩 모델명
        """
        self.model_name = model_name or APIConfig.OPENAI_EMBEDDING_MODEL
        self.embeddings = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=APIConfig.OPENAI_API_KEY
        )
    
    def generate_embedding(self, text):
        """
        LangChain OpenAIEmbeddings를 사용하여 텍스트의 임베딩 벡터를 생성합니다.
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            list: 임베딩 벡터
        """
        try:
            # LangChain의 embed_query 메서드 사용
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"임베딩 생성 중 오류 발생: {e}")
            return None
    
    def generate_embeddings_batch(self, texts):
        """
        여러 텍스트에 대한 임베딩을 배치로 생성합니다.
        
        Args:
            texts (list): 임베딩할 텍스트 목록
            
        Returns:
            list: 임베딩 벡터 목록
        """
        try:
            # LangChain의 embed_documents 메서드 사용
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            print(f"배치 임베딩 생성 중 오류 발생: {e}")
            return None