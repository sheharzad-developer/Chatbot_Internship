"""
AI Chatbot: one Streamlit app with two pages.

    streamlit run app.py        (from streamlit_app/, or `streamlit run streamlit_app/app.py`)

💬 Chat        - multi-tenant chatbot: history, PDF uploads, web search (chat_page.py)
📚 Domain RAG  - answers only from the PDFs in documents/, built by `python ingest.py` (domain_page.py)
"""
import streamlit as st

from theme import CSS

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

page = st.navigation(
    [
        st.Page("chat_page.py", title="Chat", icon="💬", default=True),
        st.Page("domain_page.py", title="Domain RAG", icon="📚", url_path="domain-rag"),
    ]
)
page.run()
