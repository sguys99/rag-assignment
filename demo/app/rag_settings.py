import gc
import os
import time
from datetime import datetime
import yaml
import streamlit as st
from dotenv import load_dotenv
from function_utils import delete_log, read_file_data, check_korean
from langchain_core.prompts import load_prompt
from rag_pkg.utils.path import LOG_PATH, PROMPT_CONFIG_PATH
from rag_pkg.utils.config_loader import dump_yaml, load_yaml
from rag_pkg.module.prompts import build_qa_prompt, save_fewshot_prompt, save_prompt
from rag_pkg.utils.rag_utils import delete_incomplete_logs, save_rag_configs
from rag_pkg.module.preprocess import preprocess_for_rag
from rag_pkg.module.vector_db import load_documents, get_vector_store
from rag_pkg.module.models import get_embedding

load_dotenv()

os.makedirs(LOG_PATH, exist_ok=True)
delete_incomplete_logs(base_path=LOG_PATH, required_files=["prompt.yaml", "rag_config.yaml"])

if "rag_name" not in st.session_state:
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    preposition = "QA_RAG_"
    st.session_state["rag_name"] = preposition + current_datetime

if "prompt" not in st.session_state:
    st.session_state["prompt"] = None
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
# if "doc_format" not in st.session_state:
#     st.session_state["doc_format"] = "pdf"
if "selected_system_prompt" not in st.session_state:
    st.session_state["selected_system_prompt"] = "qa_prompt"
if "selected_example" not in st.session_state:
    st.session_state["selected_example"] = "fewshot_template"

with st.sidebar:
    st.write("")
    st.session_state["uploaded_file"] = st.file_uploader(
        "CSV 파일을 업로드하세요.",
        type=["csv", "tsv"],
        accept_multiple_files=False,
    )

    logs = [
        f for f in os.listdir(LOG_PATH) if os.path.isdir(os.path.join(LOG_PATH, f))
    ]

    if not logs:
        st.error("저장된 항목이 없습니다.")
        selected_db = None
    else:
        selected_log = st.selectbox("RAG 저장 목록", logs)
        selected_log_path = LOG_PATH / selected_log

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
            delete_log_btn = st.button(":warning: :red[RAG 삭제]")

        if delete_log_btn:
            if selected_log:
                delete_log(selected_log_path)
            else:
                st.warning("삭제할 RAG 설정을 선택해주세요.")        

st.title(":gear: RAG Settings")
st.caption("RAG Chain을 위한 구성요소를 설정합니다.")
st.write("")

col11, col12, col13 = st.columns([1, 1, 1])

with col11:
    st.write("**:blue[1.Datasets]**")
    with st.container(height=700):
        if st.session_state["uploaded_file"]:
            csv_file = st.session_state["uploaded_file"]

            csv_df = read_file_data(csv_file)
            st.write(
                    f"**`{st.session_state['uploaded_file'].name}`**, `size({csv_df.shape[0]} x {csv_df.shape[1]})`",
            )
            st.dataframe(csv_df, height=540)
        else:
            st.write("데이터를 업로드하면 여기에 표시됩니다.")
            
with col12:
    st.write("**:blue[2.Prompt settings]**")
    with st.container(height=700):
        tab11, tab12 = st.tabs(["System Prompt", "Fewshot Prompt"])

        with tab11:
            col121, col122 = st.columns([1, 1])
            with col121:
                st.session_state["selected_system_prompt"] = st.selectbox(
                    label="Prompt template 선택",
                    options=[
                        "qa_prompt",
                        "qa_prompt_kor",
                    ],
                    index=[
                        "qa_prompt",
                        "qa_prompt_kor",
                    ].index(st.session_state["selected_system_prompt"]),
                    help="사용할 프롬프트를 선택하세요.",
                )

            prompt_template = load_prompt(
                PROMPT_CONFIG_PATH / f"{st.session_state['selected_system_prompt']}.yaml",
                encoding="utf-8",
            )
            system_prompt = prompt_template.template
            if "Context:" in system_prompt:
                system_prompt = system_prompt.split("Context:")[0].strip()

            st.write("")
            edited_message = st.text_area(
                "Prompt",
                system_prompt,
                height=380,
                label_visibility="collapsed",
            )

        with tab12:
            col121, col122 = st.columns([1, 1])
            with col121:
                st.write("")
                use_example_check = st.checkbox("fewshot 사용", value=False)
            with col122:
                st.session_state["selected_example"] = st.selectbox(
                    label="Fewshot template 선택",
                    options=["fewshot_template",],
                    index=["fewshot_template"].index(
                        st.session_state["selected_example"],
                    ),
                    disabled=not use_example_check,
                    help="사용할 프롬프트를 선택하세요.",
                )

            example_template = load_yaml(
                PROMPT_CONFIG_PATH / f"{st.session_state['selected_example']}.yaml",
            )
            example_content = example_template.get("answer_examples", "")

            st.write("")
            few_shot_msg = st.text_area(
                "Fewshot prompt",
                dump_yaml(example_content),
                height=420,
                disabled=not use_example_check,
                label_visibility="collapsed",
            )

        with st.form(key="prompt_setting", border=False):
            save_prompt_button = st.form_submit_button(label="Save prompt")
        if save_prompt_button:
            st.session_state["use_example"] = use_example_check
            st.session_state["prompt"] = build_qa_prompt(
                system_message=edited_message,
                examples=yaml.safe_load(few_shot_msg) if st.session_state["use_example"] else None,
            )
            st.success("프롬프트 생성완료!")
            
with col13:
    st.write("**:blue[3.Vector DB settings]**")
    with st.container(height=700):
        if st.session_state["uploaded_file"]:
            data_name = st.session_state["uploaded_file"].name
            st.text_area(
                "data_name",
                value=data_name,
                height=100,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.info("문서를 업로드 하세요.")
            
        st.write(" ")
        rag_name_input = st.text_input(
            "RAG 이름",
            value=st.session_state["rag_name"],
            help="저장할 RAG 이름을 지정하세요.",
        )
        
        st.write(" ")
        if st.button(label="Save settings"):
            if check_korean(rag_name_input):
                st.warning("영문 이름을 사용하세요.")

            elif "uploaded_file" not in st.session_state or not st.session_state["uploaded_file"]:
                st.warning("문서를 업로드해주세요")

            elif "prompt" not in st.session_state or not st.session_state["prompt"]:
                st.warning("prompt를 생성해주세요.")
                
            else:
                status = st.status("Saving RAG settings...", expanded=True)
                time.sleep(0.5)
                status.write("**`Preparing the log path`**")
                dir_path = LOG_PATH / rag_name_input
                prompt_path = dir_path / "prompt.yaml"
                data_path = dir_path / "data"
                db_path = dir_path / "db"
                os.makedirs(dir_path, exist_ok=True)
                os.makedirs(data_path, exist_ok=True)
                time.sleep(0.5)     
                
                status.write("**`Loading the csv file...`**")
                file_path = data_path / f"{st.session_state["uploaded_file"].name}"
                with open(file_path, "wb") as f:
                    f.write(st.session_state["uploaded_file"].getvalue())
                    
                csv_df_processed = preprocess_for_rag(csv_df, min_comments=2)
                documents = load_documents(
                    df=csv_df_processed,
                    document_text_col="document_text"
                    )
                    
                    
                status.write("**`Setting the Vector Store....`**")
                embedding = get_embedding(model = "gemini")
                vectorstore = get_vector_store(
                    documents = documents,
                    embedding=embedding,
                    type = "faiss",
                    dimension=3072
                    )
                vectorstore.save_local(folder_path=db_path)
                
                status.write("**`Saving the configs....`**")

                if st.session_state["use_example"]:
                    save_fewshot_prompt(prompt=st.session_state["prompt"], save_path=prompt_path)
                else:
                    save_prompt(prompt=st.session_state["prompt"], save_path=prompt_path)
                    
                save_rag_configs(
                    save_path=dir_path / "rag_config.yaml",
                    document_format="csv",
                    document=data_name,
                    vectorstore_type="FAISS",
                    embedding_type="gemini-embedding-001",
                )
                status.update(
                    label="**:blue[RAG 설정 저장 완료.]**",
                    state="complete",
                    expanded=False,
                )

                time.sleep(2)
                keys_to_clear = ["prompt", "uploaded_file", "rag_name"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                gc.collect()
                st.rerun()                