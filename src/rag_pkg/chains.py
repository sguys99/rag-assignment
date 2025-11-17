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