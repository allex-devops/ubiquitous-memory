from conftest import ScriptedLLM

from research.memory import ConversationMemory, rewrite_question, strip_citations


def test_recent_turns_are_kept_in_order_and_replayed_as_messages():
    m = ConversationMemory()
    m.add("first question", "first answer")
    m.add("second question", "second answer")
    assert [x["content"] for x in m.messages()] == ["first question", "first answer", "second question", "second answer"]
    assert [x["role"] for x in m.messages()] == ["user", "assistant", "user", "assistant"]


def test_citation_numbers_are_stripped_because_the_next_turn_renumbers_from_one():
    m = ConversationMemory()
    m.add("why", "Because of X [1] and Y [2, 3].")
    assert m.messages()[1]["content"] == "Because of X and Y."
    assert strip_citations("a [1][2] b [10, 11].") == "a b."


def test_older_turns_are_folded_into_a_summary_when_the_window_is_full():
    llm = ScriptedLLM(summary="They asked about purring.")
    m = ConversationMemory(llm=llm, max_turns=2)
    for i in range(4):
        m.add(f"question {i}", f"answer {i}")
    assert [t.question for t in m.turns] == ["question 2", "question 3"]
    assert m.summary == "They asked about purring."
    first = m.messages()[0]
    assert first["role"] == "user" and "They asked about purring." in first["content"]
    assert "question 0" in llm.calls[0][0]["content"]  # what got folded in


def test_without_a_model_the_oldest_turns_are_just_dropped():
    m = ConversationMemory(llm=None, max_turns=2)
    for i in range(4):
        m.add(f"q{i}", f"a{i}")
    assert [t.question for t in m.turns] == ["q2", "q3"] and m.summary == ""


def test_the_character_budget_matters_as_much_as_the_turn_count():
    m = ConversationMemory(llm=None, max_turns=50, max_chars=300)
    for i in range(6):
        m.add(f"question {i}", "x" * 100)
    assert m._chars() <= 300 and len(m.turns) < 6


def test_a_single_huge_answer_is_truncated_and_never_evicts_itself():
    m = ConversationMemory(llm=None, max_chars=100)
    m.add("q", "y" * 10_000)
    assert len(m.turns) == 1 and len(m.turns[0].answer) <= 1200


def test_a_failing_summariser_never_breaks_the_conversation():
    class Broken:
        def chat(self, *a, **k):
            raise RuntimeError("model is down")

    m = ConversationMemory(llm=Broken(), max_turns=1)
    m.add("q1", "a1")
    m.add("q2", "a2")  # would fold q1, the model fails, the turn is lost but nothing raises
    assert [t.question for t in m.turns] == ["q2"]


def test_memory_holds_only_questions_and_answers_never_retrieved_text():
    m = ConversationMemory()
    m.add("what is it", "an answer")
    assert set(vars(m.turns[0])) == {"question", "answer"}


def test_clear_forgets_everything():
    m = ConversationMemory(llm=ScriptedLLM(), max_turns=1)
    m.add("a", "b")
    m.add("c", "d")
    m.clear()
    assert m.messages() == [] and m.summary == ""


# rewriting

def test_first_question_is_not_rewritten_and_costs_no_model_call():
    llm = ScriptedLLM()
    assert rewrite_question(llm, ConversationMemory(), "why do cats purr?") == "why do cats purr?"
    assert llm.calls == []


def test_a_follow_up_is_rewritten_using_the_conversation():
    llm = ScriptedLLM(rewrite="How do cats show affection?")
    m = ConversationMemory()
    m.add("Why do cats purr?", "Because they are content.")
    assert rewrite_question(llm, m, "and how do they show affection?") == "How do cats show affection?"
    prompt = llm.calls[0][0]["content"]
    assert "Why do cats purr?" in prompt and "Last question: and how do they show affection?" in prompt


def test_a_rewrite_that_looks_like_an_answer_or_a_ramble_is_ignored():
    m = ConversationMemory()
    m.add("q", "a")
    q = "and its limits?"
    for bad in ("", "Line one\nLine two", "x" * 500):
        assert rewrite_question(ScriptedLLM(rewrite=bad), m, q) == q


def test_a_rewrite_failure_falls_back_to_the_original_question():
    class Broken:
        def chat(self, *a, **k):
            raise RuntimeError("down")

    m = ConversationMemory()
    m.add("q", "a")
    assert rewrite_question(Broken(), m, "and then?") == "and then?"


def test_quotes_around_a_rewrite_are_removed():
    m = ConversationMemory()
    m.add("q", "a")
    assert rewrite_question(ScriptedLLM(rewrite='"What about tea?"'), m, "and that?") == "What about tea?"
