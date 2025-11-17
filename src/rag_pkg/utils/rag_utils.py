import os
import re
import shutil
from typing import List, Optional

import pandas as pd
import yaml
from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(document.page_content for document in docs)


def format_docs_with_meta(docs: List[Document]) -> str:
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


def save_rag_configs(
    save_path: str,
    document_format: str,
    document: List[str],
    vectorstore_type: str = "FAISS",
    embedding_type: str = "gemini-embedding-001",
) -> None:
    
    config = {
        "document_format": document_format,
        "document": document,
        "vectorstore": {"type": vectorstore_type},
        "embedding": embedding_type,
    }

    with open(save_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def check_incomplete_logs(base_path: str, required_files: List[str]) -> List[str]:

    return [
        f
        for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
        and not all(os.path.exists(os.path.join(base_path, f, file)) for file in required_files)
    ]


def delete_incomplete_logs(base_path: str, required_files: List[str]) -> None:

    incomplete_dirs = check_incomplete_logs(base_path, required_files)

    for dir_name in incomplete_dirs:
        dir_path = os.path.join(base_path, dir_name)
        shutil.rmtree(dir_path)
        print(f"Deleted incomplete directory: {dir_path}")