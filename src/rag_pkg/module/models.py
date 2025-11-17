from typing import List, Union

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeEmbeddings


def get_embedding(
    model: str = "gemini",
) -> Union[OpenAIEmbeddings, GoogleGenerativeAIEmbeddings]:
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