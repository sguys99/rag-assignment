from typing import List, Union

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeEmbeddings


def get_embedding(
    model: str = "gemini",
) -> Union[OpenAIEmbeddings, GoogleGenerativeAIEmbeddings]:
    """
    주어진 모델 이름에 따라 적절한 Embeddings 객체를 반환하는 함수.

    Args:
        model (str): 사용할 Embeddings 모델의 이름. 기본값은 'text-embedding-3-large'

    Returns:
        Union[OpenAIEmbeddings, OllamaEmbeddings]: 주어진 모델에 해당하는 Embeddings 객체.

    Raises:
        ValueError: 지원되지 않는 모델 이름이 주어졌을 때 발생.
    """
    if model.lower().startswith("gemini"):
        embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    elif model.lower().startswith("pinecone"):
        embedding = PineconeEmbeddings(model="multilingual-e5-large")
    else:
        raise ValueError(f"지원되지 않는 임베딩 모델: {model}")

    return embedding