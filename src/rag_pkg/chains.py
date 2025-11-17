from typing import Callable, List

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag_pkg.utils.rag_utils import format_docs, format_docs_with_meta


def build_simple_chain(
    retriever: Callable,
    prompt: Callable,
    llm: Callable,
    load_memory_func: Callable[[], str],
    format_docs_func: Callable[[List], str] = format_docs,
) -> Callable:
    """
    주어진 retriever, prompt, llm, 메모리 로딩 함수, 문서 포맷팅 함수를 사용하여 체인을 구성하는 함수.

    Args:
        retriever (Callable): 문서에서 정보를 검색하는 retriever 객체.
        prompt (Callable): LLM에게 전달할 프롬프트를 생성하는 템플릿.
        llm (Callable): LLM 모델을 실행하는 객체.
        load_memory_func (Callable[[], str]): 대화 히스토리를 불러오는 함수.
        format_docs_func (Callable[[List], str], optional): 검색된 문서를 포맷하는 함수. 기본값은 `format_docs` 함수.

    Returns:
        Callable: 구성된 체인을 반환.
    """
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs_func),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(load_memory_func),
        }
        | prompt
        | llm
    )
    return chain