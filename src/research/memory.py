import re
from dataclasses import dataclass, field

MAX_TURN_CHARS = 1200  # one long answer shouldn't crowd everything else out of the window

SUMMARY_PROMPT = """Update the running summary of a conversation between a user and a research assistant.
Keep names, papers, numbers and what the user is trying to find out. Be brief: at most 120 words.

Current summary:
{summary}

Turns to fold in:
{turns}

Reply with the new summary only."""

REWRITE_PROMPT = """Rewrite the user's last question so it makes sense on its own, using the conversation for context.
Replace pronouns and references like "it", "that paper", "the second one" with what they refer to.
Keep the question's meaning, don't answer it, and reply with the question only.

Conversation:
{conversation}

Last question: {question}"""


def strip_citations(text: str) -> str:
    # "[2]" pointed at one turn's passages, and the next turn's passages are numbered from 1 again
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text)


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    """What the assistant remembers of one conversation, kept small and free of document text.

    Only the user's questions and the assistant's own answers go in here, never the retrieved passages, so a
    poisoned document can't ride along from turn to turn. Older turns are folded into a short summary.
    """

    llm: object | None = None
    max_turns: int = 4
    max_chars: int = 4000
    summary: str = ""
    turns: list[Turn] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question.strip()[:MAX_TURN_CHARS], strip_citations(answer).strip()[:MAX_TURN_CHARS]))
        overflow: list[Turn] = []
        while len(self.turns) > self.max_turns or self._chars() > self.max_chars:
            if len(self.turns) == 1:
                break
            overflow.append(self.turns.pop(0))
        if overflow:
            self._fold(overflow)

    def _chars(self) -> int:
        return len(self.summary) + sum(len(t.question) + len(t.answer) for t in self.turns)

    def _fold(self, turns: list[Turn]) -> None:
        if self.llm is None:
            return  # no model to summarise with, so the oldest turns are simply dropped
        text = "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in turns)
        try:
            reply = self.llm.chat([{"role": "user", "content": SUMMARY_PROMPT.format(summary=self.summary or "(none)", turns=text)}], temperature=0)
        except Exception:
            return  # losing the summary is better than failing the user's question
        self.summary = (reply.get("content") or "").strip()[:800]

    def messages(self) -> list[dict]:
        out = []
        if self.summary:
            out.append({"role": "user", "content": f"Summary of the conversation so far: {self.summary}"})
            out.append({"role": "assistant", "content": "Understood."})
        for t in self.turns:
            out += [{"role": "user", "content": t.question}, {"role": "assistant", "content": t.answer}]
        return out

    def clear(self) -> None:
        self.summary, self.turns = "", []


def rewrite_question(llm, memory: ConversationMemory, question: str) -> str:
    """Turn a follow-up into a standalone question, so retrieval has something to search for."""
    if not memory.turns and not memory.summary:
        return question
    conversation = "\n".join(
        [f"Summary: {memory.summary}"] * bool(memory.summary)
        + [f"User: {t.question}\nAssistant: {t.answer[:400]}" for t in memory.turns]
    )
    try:
        reply = llm.chat([{"role": "user", "content": REWRITE_PROMPT.format(conversation=conversation, question=question)}], temperature=0)
    except Exception:
        return question
    text = (reply.get("content") or "").strip().strip('"')
    # a reply that's empty, several lines, or far longer than the question is the model answering or rambling
    if not text or "\n" in text or len(text) > 3 * len(question) + 200:
        return question
    return text
