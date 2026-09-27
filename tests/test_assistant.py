from conftest import DOCS, META, ScriptedLLM, fake_embed
from ragchat.rag import NO_ANSWER

from research.assistant import Assistant
from research.citations import cited_ids
from research.memory import ConversationMemory
from research.ui import render_brief, render_reply


def assistant(store, llm=None, **kw):
    llm = llm or ScriptedLLM()
    return Assistant(store, llm, fake_embed, META, min_score=0.2, **kw), llm


def test_a_first_question_is_answered_with_sources_and_a_clean_report(store):
    a, llm = assistant(store)
    r = a.ask("why do cats purr")
    assert r.grounded and r.report.ok
    assert [x["title"] for x in r.references] == ["On Purring"] and r.references[0]["url"].endswith("2401.00001")


def test_a_follow_up_only_finds_its_answer_because_it_was_rewritten(store):
    follow_up = "and what else do they get up to"  # none of these words appear in any document
    without, _ = assistant(store, ScriptedLLM(rewrite=follow_up))
    memory = ConversationMemory()
    memory.add("why do cats purr", "Because they are content.")
    r = without.ask(follow_up, memory)
    assert not r.grounded and r.text == NO_ANSWER  # nothing to search for

    with_rewrite, _ = assistant(store, ScriptedLLM(rewrite="what else do cats do when stressed"))
    memory2 = ConversationMemory()
    memory2.add("why do cats purr", "Because they are content.")
    r2 = with_rewrite.ask(follow_up, memory2)
    assert r2.grounded and r2.standalone == "what else do cats do when stressed"
    assert r2.references[0]["source"] == "cats.pdf"


def test_the_prompt_carries_earlier_turns_but_the_question_searched_is_the_standalone_one(store):
    llm = ScriptedLLM(rewrite="what else do cats do when stressed")
    a, _ = assistant(store, llm)
    m = ConversationMemory()
    m.add("why do cats purr", "Because they are content [1].")
    a.ask("and when stressed", m)
    final = next(c for c in llm.calls if "<context>" in c[-1]["content"])
    assert [x["role"] for x in final] == ["system", "user", "assistant", "user"]
    assert final[2]["content"] == "Because they are content."  # citation stripped on the way in
    assert final[-1]["content"].endswith("Question: what else do cats do when stressed")


def test_memory_records_what_the_user_actually_typed(store):
    a, _ = assistant(store, ScriptedLLM(rewrite="how do cats show affection"))
    m = ConversationMemory()
    m.add("why do cats purr", "Because they are content.")
    a.ask("and how do they show it", m)
    assert m.turns[-1].question == "and how do they show it"
    assert "[1]" not in m.turns[-1].answer


def test_a_made_up_citation_shows_up_in_the_report_and_the_rendered_answer(store):
    a, _ = assistant(store, ScriptedLLM(answer="Cats purr when they are content [7]."))
    r = a.ask("why do cats purr")
    assert r.report.phantom == [7] and r.references == []
    assert "cites passages that don't exist ([7])" in render_reply(r)


def test_an_unrelated_question_is_declined_and_still_remembered(store):
    a, llm = assistant(store)
    m = ConversationMemory()
    r = a.ask("quantum chromodynamics lattice gauge", m)
    assert r.text == NO_ANSWER and not r.grounded and r.report.ok
    assert llm.calls == [] and len(m.turns) == 1


def test_asking_without_a_memory_object_still_works(store):
    a, _ = assistant(store)
    assert a.ask("why do cats purr").grounded


# research brief

def test_a_brief_covers_each_matching_paper_from_its_own_passages_and_relates_them(store):
    a, llm = assistant(store)
    b = a.brief("purr cats stressed diesel engines fuel compression", n_papers=2)
    assert {p.source for p in b.papers} == {"cats.pdf", "engines.pdf"}
    assert all(p.report.ok for p in b.papers) and b.overview_report.ok
    assert sorted(set(cited_ids(b.overview))) == [1, 2]  # the overview cites both papers


def test_each_paper_note_only_sees_that_papers_passages(store):
    a, llm = assistant(store)
    a.brief("purr cats stressed diesel engines fuel compression", n_papers=2)
    notes = [c[-1]["content"] for c in llm.calls if "passages from one paper" in c[-1]["content"]]
    assert len(notes) == 2
    assert all(("Cats purr" in n) != ("Diesel engines" in n) for n in notes)  # one or the other, never both


def test_papers_below_the_similarity_cutoff_are_left_out(store):
    a, _ = assistant(store)
    a.min_score = 0.6
    b = a.brief("purr cats stressed", n_papers=4)
    assert [p.source for p in b.papers] == ["cats.pdf"] and b.overview == ""  # one paper: nothing to compare


def test_a_topic_nothing_matches_says_so_instead_of_inventing_a_brief(store):
    a, llm = assistant(store)
    b = a.brief("quantum chromodynamics lattice gauge")
    assert b.papers == [] and "No paper" in b.overview and llm.calls == []


def test_a_note_with_an_invented_citation_is_flagged_per_paper(store):
    a, _ = assistant(store, ScriptedLLM(note="It covers cats purring when content [4]."))
    b = a.brief("purr cats stressed", n_papers=1)
    assert b.papers[0].report.phantom == [4]
    assert "cites passages that don't exist" in render_brief(b)


# rendering

def test_a_clean_answer_has_sources_and_no_warning(store):
    a, _ = assistant(store)
    text = render_reply(a.ask("why do cats purr"))
    assert "**Sources**" in text and "[1] On Purring (2024), p.1 — https://arxiv.org/abs/2401.00001" in text
    assert "Check this answer" not in text


def test_an_uncited_claim_is_called_out(store):
    a, _ = assistant(store, ScriptedLLM(answer="Cats purr when they are content [1]. They also enjoy sleeping in warm sunny spots."))
    assert "1 claim(s) with no citation" in render_reply(a.ask("why do cats purr"))


def test_the_test_docs_have_metadata_for_every_file():
    assert set(DOCS) == set(META)
