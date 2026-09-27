"""Checks that an answer's citations point at real passages and that the cited passages back up the claim."""
import re
from dataclasses import dataclass, field

from ragchat.rag import NO_ANSWER

CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
STOP = set(
    "a an and are as at be been being by can could did do does for from had has have how if in into is it its "
    "may might more most not of on or our than that the their them then there these they this those to was we "
    "were what when which who will with would you your also both each such very".split()
)
MIN_WORDS_TO_NEED_A_CITATION = 5


def cited_ids(text: str) -> list[int]:
    out: list[int] = []
    for group in CITE.findall(text):
        out += [int(n) for n in re.split(r"\s*,\s*", group)]
    return out


def content_words(text: str) -> set[str]:
    text = CITE.sub(" ", text.lower())
    return {w for w in re.findall(r"[a-z0-9]+", text) if w not in STOP and len(w) > 2}


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if line:
            # a full stop followed by a citation belongs to the sentence before it: "...shown. [2]"
            parts += [p.strip() for p in re.split(r"(?<=[.!?])(?<!\bet al\.)\s+(?=[A-Z0-9])", line) if p.strip()]
    return parts


@dataclass
class SentenceCheck:
    text: str
    cited: list[int]
    support: float | None  # share of the sentence's content words found in the cited passages
    ok: bool
    problem: str = ""


@dataclass
class CitationReport:
    sentences: list[SentenceCheck] = field(default_factory=list)
    phantom: list[int] = field(default_factory=list)  # cited numbers with no passage behind them

    @property
    def uncited(self) -> list[str]:
        return [s.text for s in self.sentences if s.problem == "no citation"]

    @property
    def unsupported(self) -> list[str]:
        return [s.text for s in self.sentences if s.problem == "not supported by the cited passage"]

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.sentences) and not self.phantom

    @property
    def supported_share(self) -> float:
        cited = [s for s in self.sentences if s.cited]
        return sum(s.ok for s in cited) / len(cited) if cited else 0.0


def support_score(sentence: str, passages: list[str]) -> float:
    words = content_words(sentence)
    if not words:
        return 1.0
    have = set().union(*(content_words(p) for p in passages)) if passages else set()
    return len(words & have) / len(words)


def validate(answer: str, passages: list[str], min_support: float = 0.5) -> CitationReport:
    """`passages[0]` is what the answer calls [1]. Lexical overlap is a cheap check, not proof of entailment."""
    report = CitationReport()
    if answer.strip() == NO_ANSWER:
        return report  # declining makes no claims

    for sentence in split_sentences(answer):
        ids = cited_ids(sentence)
        bad = sorted({i for i in ids if not 1 <= i <= len(passages)})
        report.phantom += [i for i in bad if i not in report.phantom]
        real = [i for i in ids if 1 <= i <= len(passages)]

        if bad:
            report.sentences.append(SentenceCheck(sentence, ids, None, False, f"cites [{bad[0]}], which doesn't exist"))
        elif real:
            score = support_score(sentence, [passages[i - 1] for i in real])
            ok = score >= min_support
            report.sentences.append(SentenceCheck(sentence, ids, round(score, 2), ok, "" if ok else "not supported by the cited passage"))
        elif len(content_words(sentence)) >= MIN_WORDS_TO_NEED_A_CITATION:
            report.sentences.append(SentenceCheck(sentence, [], None, False, "no citation"))
        else:
            report.sentences.append(SentenceCheck(sentence, [], None, True))  # too short to be a claim
    return report


def references(cited: list[int], hits: list, meta: dict[str, dict]) -> list[dict]:
    """One entry per cited passage number, with the paper it came from."""
    out = []
    for n in sorted(set(cited)):
        if not 1 <= n <= len(hits):
            continue
        h = hits[n - 1]
        m = meta.get(h.source, {})
        pid = m.get("id") or h.source.removesuffix(".pdf")
        out.append(
            {
                "n": n,
                "title": m.get("title", h.source),
                "year": (m.get("published") or "")[:4],
                "page": h.page,
                "source": h.source,
                "url": f"https://arxiv.org/abs/{pid}",
            }
        )
    return out
