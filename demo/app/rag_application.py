import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 가벼운 utils만 먼저 import
from rag_pkg.utils.path import LOG_PATH, DEMO_IMG_PATH
from rag_pkg.utils.config_loader import dump_yaml, load_yaml
from rag_pkg.utils.rag_utils import delete_incomplete_logs

# 무거운 imports는 함수 내부에서 lazy loading

def load_image(image_path):
    """이미지 파일을 읽어서 바이트로 반환"""
    with open(image_path, "rb") as f:
        return f.read()

human_avatar = DEMO_IMG_PATH / "man-icon.png"
ai_avartar = DEMO_IMG_PATH / "vessel-icon.png"
ai_avatar_image = load_image(ai_avartar)


os.makedirs(LOG_PATH, exist_ok=True)
delete_incomplete_logs(base_path=LOG_PATH, required_files=["prompt.yaml", "rag_config.yaml"])


if "memory" not in st.session_state:
    from langchain_classic.memory import ConversationBufferWindowMemory
    st.session_state["memory"] = ConversationBufferWindowMemory(
        return_messages=True,
        k=3,
        memory_key="chat_history",
    )

memory = st.session_state["memory"]

def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


if "rag_qa_on" not in st.session_state:
    st.session_state["rag_qa_on"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "llm_temp" not in st.session_state:
    st.session_state["llm_temp"] = 0.1
if "retriever_k" not in st.session_state:
    st.session_state["retriever_k"] = 4
# if "selected_rag_llm" not in st.session_state:
#     st.session_state["selected_rag_llm"] = os.getenv("SELECTED_LLM")


with st.sidebar:
    st.write("")

    logs = [f for f in os.listdir(LOG_PATH) if os.path.isdir(os.path.join(LOG_PATH, f))]
    if not logs:
        st.error("저장된 항목이 없습니다.")
        st.session_state["build_bot"] = False
        selected_db = None
        selected_log = None
        selected_prompt = None
        selected_rag_config = None
    else:
        selected_log = st.selectbox("RAG 저장 목록", logs)
        selected_log_path = LOG_PATH / selected_log

        with st.expander(label="✔️ **Settings**", expanded=False):
            llm_temperature = st.slider(
                "LLM Temperature",
                min_value=0.1,
                max_value=1.0,
                value=0.2,
                step=0.1,
                disabled=st.session_state["rag_qa_on"],
            )
            retriever_k = st.slider(
                "Retriever K value",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
                disabled=st.session_state["rag_qa_on"],
            )
            llm_type = st.radio(
                "LLM",
                [
                    "gemini-2.5-flash-lite",
                ],
                horizontal=True,
                disabled=st.session_state["rag_qa_on"],
                help="사용할 LLM을 선택하세요.",
            )


        col1, col2, _ = st.columns((3, 3, 0.5))

        with col1:
            with st.popover("RAG 조회", use_container_width=False):
                selected_prompt = load_yaml(selected_log_path / "prompt.yaml")
                selected_rag_config = load_yaml(selected_log_path / "rag_config.yaml")

                tab1, tab2 = st.tabs(
                    ["prompt", "RAG 설정"],
                )

                with tab1:
                    st.code(dump_yaml(selected_prompt), language="yaml")

                with tab2:
                    st.code(dump_yaml(selected_rag_config), language="yaml")

        with col2:
            reset_btn = st.button("Reset history", disabled=not logs)
            build_bot_btn = st.toggle("Build bot", key="build_bot", disabled=not logs)

        if build_bot_btn:
            from langchain_core.prompts import load_prompt
            selected_data_path = selected_log_path / "data"
            selected_db_path = selected_log_path / "db"
            selected_prompt_path = selected_log_path / "prompt.yaml"
            prompt = load_prompt(selected_log_path / "prompt.yaml", encoding="utf-8")
            st.session_state["llm_temp"] = llm_temperature
            st.session_state["retriever_k"] = retriever_k
            st.session_state["selected_rag_llm"] = "gemini-2.5-flash-lite"
            st.session_state["rag_qa_on"] = True
            st.session_state["prompt"] = prompt
            st.session_state["selected_data_path"] = selected_data_path
            st.session_state["selected_db_path"] = selected_db_path
            st.success("Setting complete!")
        else:
            st.session_state["rag_qa_on"] = False

        if reset_btn and st.session_state["rag_qa_on"]:
            st.session_state["messages"] = []
            st.session_state["memory"].clear()

st.title("Whisky Recommendation Service")
st.caption("RAG 설정을 호출하여 선호하는 위스키를 추천천합니다.")
st.write("")

col11, col12 = st.columns([1, 1])
with col11:
    st.write("**:blue[1.Datasets]**")
    with st.container(height=700):
        if st.session_state["rag_qa_on"] and selected_rag_config:
            from function_utils import read_file_data
            data_path = st.session_state["selected_data_path"] / selected_rag_config["document"]

            csv_df = read_file_data(data_path)
            st.write(
                    f"**`{selected_rag_config["document"]}`**, `size({csv_df.shape[0]} x {csv_df.shape[1]})`",
            )
            st.dataframe(csv_df, height=540)
        else:
            st.write("`Build bot`을 toggle하면 문서가 표시됩니다.")
            
with col12:
    st.write(
        f"**:blue[RAG name :]** `{selected_log}` , **:blue[LLM Temp. :]** `{st.session_state['llm_temp']}`, **:blue[Retriever K :]** `{st.session_state['retriever_k']}`",
    )
    with st.container(height=700):
        if st.session_state["rag_qa_on"] and selected_rag_config:
            # Lazy import - 필요할 때만 로드
            from function_utils import send_message, pain_history, load_retriver
            from rag_pkg.module.models import get_llm
            from rag_pkg.chains import build_simple_chain
            from rag_pkg.utils.rag_utils import format_docs_with_meta

            # LLM과 Retriever를 session_state에 캐싱
            if "llm" not in st.session_state or "retriever" not in st.session_state:
                llm = get_llm(
                    model=st.session_state["selected_rag_llm"],
                    temperature=st.session_state["llm_temp"],
                )
                retriever = load_retriver(
                    db_path=st.session_state["selected_db_path"],
                    embedding_model=selected_rag_config["embedding"],
                    retriever_k=st.session_state["retriever_k"]
                )
                st.session_state["llm"] = llm
                st.session_state["retriever"] = retriever
            else:
                llm = st.session_state["llm"]
                retriever = st.session_state["retriever"]

            send_message("선호하는 위스키에 대해서 물어보세요!", "ai", save=False)
            pain_history()

            with st._bottom:
                _, col122 = st.columns([1, 1])
                with col122:
                    message = st.chat_input("여기에 질문을 입력하세요....")

            if message:
                send_message(message, "human")
                chain = build_simple_chain(
                    retriever=retriever,
                    prompt=st.session_state["prompt"],
                    llm=llm,
                    load_memory_func=load_memory,
                    format_docs_func=format_docs_with_meta,
                )

                # 스트리밍 응답 수집
                full_response = ""
                for chunk in chain.stream(message):
                    full_response += chunk

                print(full_response)
                memory.save_context({"input": message}, {"output": full_response})
                st.session_state["memory"] = memory
                send_message(full_response, "ai", stream=True)
        else:
            st.session_state["messages"] = []
            st.session_state["rag_qa_on"] = False
            memory.clear()