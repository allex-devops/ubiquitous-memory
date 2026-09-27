"""Streamlit front end: uv run --env-file .env streamlit run src/research/app.py"""
import streamlit as st
from ragchat.rag import ConfigError

from research.factory import build_assistant
from research.memory import ConversationMemory
from research.ui import render_brief, render_reply

st.set_page_config(page_title="Ubiquitous Memory", layout="wide")


@st.cache_resource
def load():
    return build_assistant()


try:
    assistant, store = load()
except ConfigError as e:
    st.error(str(e))
    st.stop()
if store.count() == 0:
    st.error("The index is empty. Run `uv run python -m research fetch`, then `uv run --env-file .env python -m research index`.")
    st.stop()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(llm=assistant.rewriter)
    st.session_state.messages = []

with st.sidebar:
    st.metric("Indexed passages", store.count())
    assistant.k = st.slider("Passages per question", 3, 12, assistant.k)
    assistant.min_score = st.slider("Minimum similarity", 0.2, 0.95, assistant.min_score, 0.01)
    if st.button("New conversation"):
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.rerun()

ask_tab, brief_tab = st.tabs(["Ask", "Research brief"])

with ask_tab:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.text(m["content"]) if m["role"] == "assistant" else st.write(m["content"])
    if question := st.chat_input("Ask about the papers"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and reading"):
                reply = assistant.ask(question, st.session_state.memory)
            st.text(render_reply(reply))  # st.text, not markdown: answer text is untrusted
            if reply.standalone != question:
                st.caption(f"Searched for: {reply.standalone}")
        st.session_state.messages.append({"role": "assistant", "content": render_reply(reply)})

with brief_tab:
    topic = st.text_input("Topic", placeholder="e.g. reranking in retrieval-augmented generation")
    n = st.slider("Papers", 2, 8, 4)
    if topic and st.button("Write a brief"):
        with st.spinner("Reading papers"):
            brief = assistant.brief(topic, n_papers=n)
        st.text(render_brief(brief))
