"""Calls your provider (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, EMBED_MODEL) and needs the full collection indexed:

    uv run python -m research fetch
    uv run --env-file .env python -m research index
    uv run --env-file .env pytest -m live -s
"""
import json
import statistics
from pathlib import Path

import pytest

from research.factory import build_assistant
from research.memory import ConversationMemory, rewrite_question

pytestmark = pytest.mark.live
# questions written from single passages of the papers, each with the paper and page it came from
QUESTIONS = Path(__file__).parent / "data" / "questions.jsonl"
FOLLOW_UPS = ["What method does that paper propose?", "What are the main limitations of that work?", "How do they evaluate it?"]


@pytest.fixture(scope="module")
def stack():
    assistant, store = build_assistant()
    if store.count() == 0:
        pytest.skip("index is empty: run `uv run python -m research index` first")
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines()]
    return assistant, store, rows


def test_retrieval_over_the_whole_collection(stack):
    assistant, store, rows = stack
    ranks = []
    for r in rows[:40]:
        hits = store.query(assistant.embed([r["question"]], "query")[0], k=20)
        sources = []
        for h in hits:
            if h.source not in sources:
                sources.append(h.source)
        want = f"{r['paper_id']}.pdf"
        ranks.append(sources.index(want) + 1 if want in sources else None)
    found = [x for x in ranks if x]
    hit1 = sum(x == 1 for x in ranks) / len(ranks)
    hit5 = sum(bool(x) and x <= 5 for x in ranks) / len(ranks)
    mrr = statistics.mean(1 / x if x else 0 for x in ranks)
    print(f"\n{store.count()} chunks, {len(ranks)} questions: right paper first {hit1:.0%}, in top 5 {hit5:.0%}, MRR {mrr:.2f}")
    assert hit5 >= 0.6


def test_answers_are_cited_and_the_citations_check_out(stack):
    assistant, _, rows = stack
    replies = [assistant.ask(r["question"]) for r in rows[:10]]
    grounded = [r for r in replies if r.grounded]
    clean = [r for r in grounded if r.report.ok]
    phantom = sum(len(r.report.phantom) for r in grounded)
    share = statistics.mean(r.report.supported_share for r in grounded) if grounded else 0
    print(f"\nanswered {len(grounded)}/10, every citation valid and supported in {len(clean)}/{len(grounded)}, "
          f"phantom citations {phantom}, cited sentences supported {share:.0%}")
    print("example:\n" + __import__("research.ui", fromlist=["render_reply"]).render_reply(grounded[0])[:700])
    assert len(grounded) >= 8 and phantom == 0


def test_memory_lets_a_follow_up_find_the_same_paper(stack):
    assistant, store, rows = stack
    with_rewrite = without = total = 0
    for r, follow_up in zip(rows[:24], FOLLOW_UPS * 8):
        first = assistant.ask(r["question"], ConversationMemory(llm=assistant.rewriter))
        want = f"{r['paper_id']}.pdf"
        if not first.sources or first.sources[0].source != want:
            continue  # only test dialogs whose first turn really was about the right paper
        memory = ConversationMemory(llm=assistant.rewriter)
        memory.add(r["question"], first.text)
        standalone = rewrite_question(assistant.rewriter, memory, follow_up)
        def hit(q):
            return any(h.source == want for h in store.query(assistant.embed([q], "query")[0], k=6))
        with_rewrite += hit(standalone)
        without += hit(follow_up)
        total += 1
        if total == 1:
            print(f"\nfollow-up {follow_up!r} became {standalone!r}")
    print(f"{total} follow-ups: right paper found in top 6 with rewriting {with_rewrite}/{total}, without {without}/{total}")
    assert total >= 6 and with_rewrite > without


def test_literature_brief(stack):
    assistant, _, _ = stack
    from research.ui import render_brief

    b = assistant.brief("reranking retrieved passages in retrieval-augmented generation", n_papers=3)
    print("\n" + render_brief(b)[:1500])
    assert len(b.papers) >= 2 and b.overview
    assert sum(p.report.ok for p in b.papers) >= 1


def test_streamlit_app_answers_a_question(stack, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    rows = stack[2]
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src" / "research" / "app.py"), default_timeout=300).run()
    assert not at.exception
    at.chat_input[0].set_value(rows[0]["question"]).run()
    assert not at.exception
    texts = [t.value for t in at.text]
    print("\nfirst lines shown:", texts[0][:200] if texts else None)
    assert texts and "Sources" in texts[-1]
