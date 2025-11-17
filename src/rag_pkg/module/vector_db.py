import pandas as pd
from typing import List, Union
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone
from rag_pkg.module.preprocess import filter_valid_reviews, create_document_text, preprocess_for_rag
from dotenv import load_dotenv
load_dotenv()

def load_documents(df: pd.DataFrame, document_text_col: str = 'document_text') -> List[Document]:
    if document_text_col not in df.columns:
        raise ValueError(f"'{document_text_col}' 컬럼이 데이터프레임에 존재하지 않습니다.")

    documents = []

    for idx, row in df.iterrows():
        # page_content: document_text 내용
        page_content = row[document_text_col]

        # metadata: 기타 정보 (Whisky Name, Link, Tags, Scores 등)
        metadata = {
            'whisky_name': row['Whisky Name'],
            'link': row['Link'],
        }

        # Tags 추가 (있는 경우)
        if pd.notna(row['Tags']) and row['Tags'].strip():
            metadata['tags'] = row['Tags']

        # Scores 추가 (있는 경우)
        if pd.notna(row['Nose Score']) and str(row['Nose Score']).strip():
            metadata['nose_score'] = row['Nose Score']

        if pd.notna(row['Taste Score']) and str(row['Taste Score']).strip():
            metadata['taste_score'] = row['Taste Score']

        if pd.notna(row['Finish Score']) and str(row['Finish Score']).strip():
            metadata['finish_score'] = row['Finish Score']

        # Document 생성
        doc = Document(
            page_content=page_content,
            metadata=metadata
        )

        documents.append(doc)

    return documents


def get_vector_store(
    documents: List[Document],
    embedding: Embeddings,
    type: str = "faiss",
    index_name: str = "whisky-reviews",
    dimension:int = 1024
) -> Union[FAISS, Chroma, PineconeVectorStore]:
    if type.lower().startswith("faiss"):
        vector_store = FAISS.from_documents(documents=documents, embedding=embedding)

    elif type.lower().startswith("chroma"):
        vector_store = Chroma.from_documents(documents=documents, embedding=embedding)

    elif type.lower().startswith("pinecone"):
        if index_name is None:
            raise ValueError("Pinecone 벡터 스토어 사용 시 'index_name' 파라미터가 필수입니다.")

        pc = Pinecone()
        index_exists = pc.has_index(index_name)

        if not index_exists:
            # 인덱스가 없으면 새로 생성
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        index = pc.Index(index_name)

        vector_store = PineconeVectorStore(index=index, embedding=embedding)

        # 인덱스가 새로 생성된 경우에만 문서 추가
        if not index_exists:
            vector_store.add_documents(documents=documents)
            print(f"✓ Pinecone 인덱스 '{index_name}'에 {len(documents)}개 문서 추가 완료")
        else:
            print(f"✓ 기존 Pinecone 인덱스 '{index_name}' 사용 (문서 추가 생략)")

    else:
        raise ValueError(f"Unsupported vector store type: {type}")

    return vector_store

