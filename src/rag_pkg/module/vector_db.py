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
    """
    전처리된 데이터프레임을 LangChain Document 리스트로 변환합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        전처리된 위스키 리뷰 데이터프레임 (document_text 컬럼 포함)
    document_text_col : str, default='document_text'
        Document의 page_content로 사용할 컬럼명

    Returns:
    --------
    List[Document]
        LangChain Document 객체 리스트
        - page_content: document_text 내용
        - metadata: Whisky Name, Link 등 기타 정보

    Examples:
    ---------
    >>> df = preprocess_for_rag(reviews_df, min_comments=2)
    >>> documents = load_documents(df)
    >>> print(len(documents))  # df의 행 수와 동일
    >>> print(documents[0].page_content)  # document_text 내용
    >>> print(documents[0].metadata)  # {'whisky_name': '...', 'link': '...', ...}

    Notes:
    ------
    - 원본 DataFrame의 행 수와 반환된 Document 리스트의 길이가 동일합니다.
    - 각 Document의 metadata에는 Whisky Name, Link, Tags, Score 정보가 포함됩니다.
    """
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
) -> Union[FAISS, Chroma, PineconeVectorStore]:
    """
    주어진 Document 객체 목록과 임베딩을 사용하여 지정된 유형의 벡터 스토어를 반환하는 함수.

    Args:
        documents (List[Document]): 임베딩을 생성할 Document 객체 목록.
        embedding (Embeddings): 사용할 임베딩 모델.
        type (str, optional): 생성할 벡터 스토어의 유형. 기본값은 'faiss'.
            - "faiss": FAISS 벡터 스토어
            - "chroma": Chroma 벡터 스토어
            - "pinecone": Pinecone 벡터 스토어 (index_name 필수)
        index_name (str, optional): Pinecone 인덱스 이름. Pinecone 사용 시 필수.

    Returns:
        Union[FAISS, Chroma, PineconeVectorStore]: 지정된 유형의 벡터 스토어 객체를 반환.

    Raises:
        ValueError: 지원되지 않는 벡터 스토어 유형이 입력된 경우 발생.
        ValueError: Pinecone 선택 시 index_name이 제공되지 않은 경우 발생.
    """
    if type.lower().startswith("faiss"):
        vector_store = FAISS.from_documents(documents=documents, embedding=embedding)

    elif type.lower().startswith("chroma"):
        vector_store = Chroma.from_documents(documents=documents, embedding=embedding)

    elif type.lower().startswith("pinecone"):
        if index_name is None:
            raise ValueError("Pinecone 벡터 스토어 사용 시 'index_name' 파라미터가 필수입니다.")
        
        pc = Pinecone()
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension = 1536, #??
                metric="cosine", #??
                spec=ServerlessSpec(cloud="aws", region="us-east-1"), #??
            )
            
        index = pc.Index(index_name)

        vector_store = PineconeVectorStore(index=index, embedding=embedding)
        vector_store.add_documents(documents=documents)

    else:
        raise ValueError(f"Unsupported vector store type: {type}")

    return vector_store

