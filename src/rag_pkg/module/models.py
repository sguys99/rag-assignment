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


def get_llm(
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.2,
    callbacks: List = [],
) -> Union[ChatOpenAI, GoogleGenerativeAI]:
    """
    주어진 모델 이름에 따라 적절한 LLM(언어 모델) 객체를 반환하는 함수.

    지원되는 모델:
    - 'gpt'로 시작하는 모델: ChatOpenAI 객체 반환
    - 'claude'로 시작하는 모델: ChatAnthropic 객체 반환
    - 'llama', 'gemma', 'mistral'로 시작하는 모델: ChatOllama 객체 반환

    Args:
        model (str): 사용할 LLM 모델 이름. 기본값은 'gpt-4o'
        temperature (float): 생성된 텍스트의 창의성 수준을 조절하는 값. 기본값은 0.2
        streaming (bool): 스트리밍 모드를 사용할지 여부. 기본값은 False
        callbacks (List): 생성 중 실행할 콜백 함수 목록. 기본값은 빈 리스트
        num_predict (int): Ollama 모델의 출력 토큰 최대 길이. 기본값은 300

    Returns:
        Union[ChatOpenAI, ChatAnthropic, ChatOllama]: 주어진 모델 이름에 해당하는 LLM 객체.

    Raises:
        ValueError: 지원되지 않는 모델 이름이 입력된 경우 발생합니다.
    """
    if model.startswith("gpt"):
        llm = ChatOpenAI(model=model, temperature=temperature, callbacks=callbacks)
    elif model.startswith("gemini"):
        llm = GoogleGenerativeAI(
            model=model,
            temperature=temperature,
            callbacks=callbacks,
        )
    else:
        raise ValueError(f"지원되지 않는 모델: {model}")

    return llm