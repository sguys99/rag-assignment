import os
import re
import shutil
from typing import List, Optional

import pandas as pd
import yaml
from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """
    문서 목록에서 각 문서의 페이지 콘텐츠를 연결하여 하나의 문자열로 반환하는 함수.

    Args:
        docs (List[Document]): Langchain Document 인스턴스로 구성된 리스트.

    Returns:
        str: 각 문서의 페이지 콘텐츠가 두 줄 간격으로 연결된 하나의 문자열.
    """
    return "\n\n".join(document.page_content for document in docs)


def format_docs_with_meta(docs: List[Document]) -> str:
    """
    문서 목록에서 각 문서의 페이지 콘텐츠와 메타데이터를 포함하여 하나의 문자열로 반환하는 함수.

    Args:
        docs (List[Document]): Langchain Document 인스턴스로 구성된 리스트.

    Returns:
        str: 각 문서의 페이지 콘텐츠와 메타데이터가 포함되어 연결된 하나의 문자열.

    Examples:
        >>> docs = retriever.get_relevant_documents("fruity whisky")
        >>> formatted = format_docs_with_meta(docs)
        >>> print(formatted)
        [Whisky: Glenfiddich 12 Year Old]
        Link: https://...
        Tags: fruity, smooth
        Nose Score: 85, Taste Score: 87, Finish Score: 84

        Review Content:
        This whisky has a fruity nose...

        ---

        [Whisky: Macallan 18]
        ...
    """
    formatted_docs = []

    for doc in docs:
        parts = []

        # Whisky 이름 추가
        if 'whisky_name' in doc.metadata:
            parts.append(f"[Whisky: {doc.metadata['whisky_name']}]")

        # Link 추가
        if 'link' in doc.metadata:
            parts.append(f"Link: {doc.metadata['link']}")

        # Tags 추가
        if 'tags' in doc.metadata:
            parts.append(f"Tags: {doc.metadata['tags']}")

        # Scores 추가
        scores = []
        if 'nose_score' in doc.metadata:
            scores.append(f"Nose Score: {doc.metadata['nose_score']}")
        if 'taste_score' in doc.metadata:
            scores.append(f"Taste Score: {doc.metadata['taste_score']}")
        if 'finish_score' in doc.metadata:
            scores.append(f"Finish Score: {doc.metadata['finish_score']}")

        if scores:
            parts.append(", ".join(scores))

        # 메타데이터와 콘텐츠 구분
        if parts:
            parts.append("\nReview Content:")

        # 페이지 콘텐츠 추가
        parts.append(doc.page_content)

        # 하나의 문서로 합치기
        formatted_docs.append("\n".join(parts))

    # 모든 문서를 구분선으로 연결
    return "\n\n---\n\n".join(formatted_docs)


# def save_rag_configs(
#     save_path: str,
#     document_format: str,
#     documents: List[str],
#     text_splitter_type: str,
#     chunk_size: int,
#     chunk_overlap: int,
#     loader_type: str = "directory",
#     vectorstore_type: str = "FAISS",
#     embedding_type: str = "text-embedding-3-large",
# ) -> None:
#     """
#     RAG(Retrieval-Augmented Generation) 설정을 YAML 파일로 저장.

#     Args:
#         save_path (str): 설정 파일을 저장할 경로.
#         document_format (str): 문서 형식 (예: PDF, DOCX).
#         documents (List[str]): 처리할 문서 목록.
#         text_splitter_type (str): 사용할 텍스트 분할기 유형.
#         chunk_size (int): 분할할 텍스트 청크 크기.
#         chunk_overlap (int): 청크 간 중첩되는 문자 수.
#         loader_type (str, optional): 문서 로더 유형 (기본값: "directory").
#         vectorstore_type (str, optional): 벡터 스토어 유형 (기본값: "FAISS").
#         embedding_type (str, optional): 임베딩 모델 유형 (기본값: "text-embedding-3-large").

#     Returns:
#         None
#     """
#     config = {
#         "document_format": document_format,
#         "documents": documents,
#         "loader": {"type": loader_type},
#         "text_splitter": {
#             "type": text_splitter_type,
#             "chunk_size": chunk_size,
#             "chunk_overlap": chunk_overlap,
#         },
#         "vectorstore": {"type": vectorstore_type},
#         "embedding": embedding_type,
#     }

#     with open(save_path, "w") as file:
#         yaml.dump(config, file, default_flow_style=False)