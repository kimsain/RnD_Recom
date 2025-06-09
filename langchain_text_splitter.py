"""
LangChain 기반 텍스트 분할 유틸리티

이 모듈은 LangChain의 RecursiveCharacterTextSplitter를 사용하여 
텍스트를 청크로 분할하는 기능을 제공합니다.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

class LangChainTextSplitter:
    """
    LangChain을 사용한 텍스트 분할 유틸리티 클래스
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        텍스트 스플리터 초기화
        
        Args:
            chunk_size (int): 청크의 최대 크기
            chunk_overlap (int): 청크 간 중첩 크기
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할
        
        Args:
            text (str): 분할할 텍스트
            
        Returns:
            List[str]: 분할된 텍스트 청크 목록
        """
        return self.text_splitter.split_text(text)
    
    def split_news_article(self, title: str, content: str) -> List[str]:
        """
        뉴스 기사를 청크로 분할
        
        Args:
            title (str): 뉴스 제목
            content (str): 뉴스 본문
            
        Returns:
            List[str]: 분할된 텍스트 청크 목록
        """
        # 제목과 본문을 결합
        full_text = f"{title}\n\n{content}"
        return self.split_text(full_text)
    
    def create_chunks_with_metadata(self, text: str, base_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        텍스트를 분할하고 메타데이터와 함께 반환
        
        Args:
            text (str): 분할할 텍스트
            base_metadata (dict): 기본 메타데이터
            
        Returns:
            List[Dict]: 청크와 메타데이터 목록
        """
        chunks = self.split_text(text)
        base_metadata = base_metadata or {}
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
            
            result.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return result

# 편의 함수들
def split_text_simple(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    간단한 텍스트 분할 함수
    
    Args:
        text (str): 분할할 텍스트
        chunk_size (int): 청크 크기
        chunk_overlap (int): 중첩 크기
        
    Returns:
        List[str]: 분할된 텍스트 목록
    """
    splitter = LangChainTextSplitter(chunk_size, chunk_overlap)
    return splitter.split_text(text)

def split_news_article(title: str, content: str, chunk_size: int = 1000) -> List[str]:
    """
    뉴스 기사 분할 편의 함수
    
    Args:
        title (str): 뉴스 제목
        content (str): 뉴스 본문
        chunk_size (int): 청크 크기
        
    Returns:
        List[str]: 분할된 텍스트 목록
    """
    splitter = LangChainTextSplitter(chunk_size)
    return splitter.split_news_article(title, content)

