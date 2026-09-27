from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "research" / "app.py")


@pytest.fixture(autouse=True)
def fresh_cache():
    import streamlit as st

    st.cache_resource.clear()  # the app caches its assistant for the process, and tests must not share one


def fake_provider(monkeypatch):
    for name, value in {"LLM_BASE_URL": "http://provider.invalid/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m", "EMBED_MODEL": "e"}.items():
        monkeypatch.setenv(name, value)


def test_the_app_tells_you_what_to_do_when_the_index_is_empty(tmp_path, monkeypatch):
    fake_provider(monkeypatch)
    monkeypatch.setenv("RESEARCH_DATA", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "index is empty" in at.error[0].value and "research index" in at.error[0].value
    assert len(at.chat_input) == 0  # nothing to ask yet, so no question box


def test_the_app_asks_for_provider_settings_before_anything_else(tmp_path, monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "EMBED_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RESEARCH_DATA", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "LLM_API_KEY" in at.error[0].value
