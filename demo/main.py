import streamlit as st
from dotenv import load_dotenv
from page_utils import login, logout
from PIL import Image

load_dotenv()

if "login" not in st.session_state:
    st.session_state["login"] = False


st.set_page_config(
    page_title="Whisky AI",
    page_icon=Image.open("../img/logo-circle.png"),
    layout="wide",
)

st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

home = st.Page(
    "home/home.py",
    title="Home page",
    icon=":material/home:",
    default=True,
)


rag_settings = st.Page(
    "app/rag_settings.py",
    title="Settings",
    icon=":material/settings:",
)

rag_application = st.Page(
    "app/rag_application.py",
    title="Application",
    icon=":material/smart_toy:",
)

if st.session_state["login"]:
    pg = st.navigation(
        {
            "⚙️ Logout": [logout_page],
            "1️⃣ Home": [home],
            "2️⃣ Whisky RAG Service": [rag_settings, rag_application],
        },
    )
else:
    pg = st.navigation([login_page])


with st.sidebar:
    st.image("../img/logo.png", width=200)

pg.run()
