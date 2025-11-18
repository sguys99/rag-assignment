import re
import shutil
from time import time
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic.storage import LocalFileStore
from langchain_community.vectorstores import FAISS

from rag_pkg.module.models import get_embedding
from rag_pkg.utils.path import DEMO_IMG_PATH

load_dotenv()

human_avatar = DEMO_IMG_PATH / "man-icon.png"
ai_avartar = DEMO_IMG_PATH / "vessel-icon.png"


def load_image(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        return image_file.read()


class ChatCallbackHandler(BaseCallbackHandler):
    """
    LLM 모델의 실행 중 상태를 처리하는 콜백 핸들러 클래스.

    이 핸들러는 메시지 박스를 업데이트하고, 새로운 토큰을 수신할 때마다 실시간으로
    사용자 인터페이스에 반영.

    Attributes:
        message (str): 현재까지 생성된 메시지를 저장하는 문자열.
        message_box: Streamlit에서 비어 있는 UI 요소로, 생성된 메시지를 실시간으로 업데이트하는 데 사용.
    """

    message: str = ""

    def on_llm_start(self, *args, **kwargs) -> None:
        """
        LLM 모델이 시작될 때 호출되는 메서드.
        빈 메시지 박스를 생성.

        Args:
            *args: 임의의 인자.
            **kwargs: 임의의 키워드 인자.
        """
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs) -> None:
        """
        LLM 모델이 종료될 때 호출되는 메서드.
        최종 생성된 메시지를 저장.

        Args:
            *args: 임의의 인자.
            **kwargs: 임의의 키워드 인자.
        """
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
        """
        새로운 토큰을 수신할 때 호출되는 메서드.
        메시지 박스를 실시간으로 업데이트.

        Args:
            token (str): 새로 생성된 토큰.
            *args: 임의의 인자.
            **kwargs: 임의의 키워드 인자.
        """
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_resource(show_spinner="Loading vectorstore...")
def load_retriver(
    db_path: str,
    embedding_model="gemini-embedding-001",
    retriever_k: int = 4,
):
    embeddings = get_embedding(model=embedding_model)
    vectorstore = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": retriever_k})
    return retriever


@st.dialog("로그 삭제")
def delete_log(selected_log_path: str) -> None:
    st.write(f"`{selected_log_path.as_posix().split('/')[-1]}`")
    st.write("위 경로의 설정을 삭제합니까?")
    if st.button("삭제"):
        try:
            shutil.rmtree(selected_log_path)
            st.success(f"설정 '{selected_log_path.as_posix().split('/')[-1]}'이 삭제되었습니다.")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"설정 삭제 중 오류가 발생했습니다: {str(e)}")
        st.rerun()


def save_message(message: str, role: str) -> None:
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message: str, role: str, save: bool = True, stream: bool = False) -> None:
    avatar_image = load_image(ai_avartar if role == "ai" else human_avatar)

    with st.chat_message(role, avatar=avatar_image):
        if stream and role == "ai":
            # 스트리밍 효과를 위한 write_stream 사용
            st.write_stream((char for char in message))
        else:
            st.markdown(message)

    if save:
        save_message(message, role)


def pain_history() -> None:
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


@st.cache_data(show_spinner=False)
def read_file_data(file: Any) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


def check_korean(text):
    p = re.compile("[ㄱ-힣]")
    r = p.search(text)
    if r is None:
        return False
    else:
        return True
