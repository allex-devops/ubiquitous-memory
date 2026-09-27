from ragchat.rag import NO_ANSWER
from ragchat.store import Hit

from research.citations import cited_ids, references, split_sentences, support_score, validate

PASSAGES = [
    "Cats purr when they are content and also when they are stressed.",
    "Diesel engines ignite fuel by compression alone with no spark plug.",
]


def test_citation_markers_are_found_in_all_their_forms():
    assert cited_ids("Cats purr [1]. Engines too [2, 3]. Both [1][2].") == [1, 2, 3, 1, 2]
    assert cited_ids("No citations here.") == []


def test_sentences_split_on_full_stops_bullets_and_keep_abbreviations_whole():
    text = "First claim [1]. Second claim [2].\n- A bullet point about something else [1]\nSmith et al. showed this."
    assert split_sentences(text) == ["First claim [1].", "Second claim [2].", "A bullet point about something else [1]", "Smith et al. showed this."]


def test_a_supported_cited_sentence_passes():
    r = validate("Cats purr when they are content [1].", PASSAGES)
    assert r.ok and r.sentences[0].support == 1.0


def test_a_citation_to_a_passage_that_does_not_exist_is_flagged():
    r = validate("Cats purr when they are content [3].", PASSAGES)
    assert not r.ok and r.phantom == [3]
    assert "doesn't exist" in r.sentences[0].problem


def test_a_sentence_that_the_cited_passage_does_not_back_up_is_flagged():
    r = validate("Diesel engines ignite fuel by compression alone [1].", PASSAGES)  # right sentence, wrong passage
    assert not r.ok and r.unsupported == ["Diesel engines ignite fuel by compression alone [1]."]


def test_citing_the_right_passage_for_the_same_sentence_passes():
    assert validate("Diesel engines ignite fuel by compression alone [2].", PASSAGES).ok


def test_a_claim_with_no_citation_is_reported_but_a_short_remark_is_not():
    r = validate("Kittens learn to purr within their first days of life. Sure thing.", PASSAGES)
    assert r.uncited == ["Kittens learn to purr within their first days of life."]
    assert not r.ok


def test_support_is_measured_against_every_passage_a_sentence_cites():
    both = validate("Cats purr when content and diesel engines ignite fuel by compression [1, 2].", PASSAGES)
    assert both.ok
    only_first = validate("Cats purr when content and diesel engines ignite fuel by compression [1].", PASSAGES)
    assert not only_first.ok


def test_declining_makes_no_claims_so_there_is_nothing_to_check():
    assert validate(NO_ANSWER, PASSAGES).ok


def test_supported_share_counts_only_cited_sentences():
    r = validate("Cats purr when content [1]. Diesel engines ignite fuel by compression [1]. Some uncited remark about weather patterns today.", PASSAGES)
    assert r.supported_share == 0.5


def test_support_score_ignores_filler_words():
    assert support_score("the of and it is", PASSAGES) == 1.0  # nothing to check counts as supported
    assert support_score("quantum chromodynamics lattice", PASSAGES) == 0.0


def test_references_list_only_what_was_cited_with_paper_details():
    hits = [Hit("t1", "cats.pdf", 3, 0.9), Hit("t2", "engines.pdf", 1, 0.8)]
    meta = {"cats.pdf": {"id": "2401.00001", "title": "On Purring", "published": "2024-01-05"}}
    refs = references([2, 1, 1, 9], hits, meta)
    assert [r["n"] for r in refs] == [1, 2]  # sorted, deduplicated, and the out-of-range 9 dropped
    assert refs[0] == {"n": 1, "title": "On Purring", "year": "2024", "page": 3, "source": "cats.pdf", "url": "https://arxiv.org/abs/2401.00001"}
    assert refs[1]["title"] == "engines.pdf" and refs[1]["url"].endswith("/engines")  # no metadata: fall back to the file name
