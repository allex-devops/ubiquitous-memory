import os
from pathlib import Path

from ragchat.rag import default_min_score, embedder, llm_from_env
from ragchat.store import Store

from .assistant import Assistant
from .corpus import paper_meta


def data_dir() -> Path:
    return Path(os.environ.get("RESEARCH_DATA", "data"))


def build_assistant() -> tuple[Assistant, Store]:
    llm = llm_from_env(timeout=300)
    store = Store(data_dir() / "index", embed_model=llm.embed_model)
    # rewriting follow-ups is a small job; point REWRITE_MODEL at a cheaper model from the same provider if you like
    rewriter = llm_from_env(timeout=120)
    rewriter.model = os.environ.get("REWRITE_MODEL") or llm.model
    assistant = Assistant(store, llm, embedder(llm), paper_meta(), rewriter=rewriter, min_score=default_min_score())
    return assistant, store
