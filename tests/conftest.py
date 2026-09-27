import re
import zlib

import numpy as np
import pytest
from ragchat.ingest import Chunk
from ragchat.store import Store

STOP = set("a an and are as at be by do does for how in is it of on or the to what when who why with that this from about their they".split())


def fake_embed(texts, kind="document"):
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in STOP:
                out[row, zlib.crc32(word.encode()) % 256] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


class ScriptedLLM:
    """Answers by looking at the prompt, so one object can play the answerer, the rewriter and the summariser."""

    def __init__(self, answer="Cats purr when happy [1].", rewrite=None, summary="They discussed cats.", note=None):
        self.answer, self.rewrite, self.summary, self.note = answer, rewrite, summary, note
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if "Rewrite the user's last question" in prompt:
            return {"content": self.rewrite if self.rewrite is not None else prompt.rsplit("Last question: ", 1)[1]}
        if "running summary" in prompt:
            return {"content": self.summary}
        if "Below are notes on" in prompt:
            # an overview built from the notes it was given, each cited by its number
            parts = []
            for line in prompt.splitlines():
                if re.match(r"\[\d+\] ", line):
                    n = re.match(r"\[(\d+)\]", line).group(1)
                    body = re.sub(r"\s*\[\d+\]\.?$", "", line.split(": ", 1)[1]).rstrip(".")
                    parts.append(f"{body} [{n}].")
            return {"content": " ".join(parts)}
        if "passages from one paper" in prompt:
            if self.note is not None:
                return {"content": self.note}
            # by default quote the first passage, which is what a well-behaved model would do
            first = prompt.split("[1] (p.", 1)[1].split(") ", 1)[1].splitlines()[0]
            return {"content": first.split(". ")[0].rstrip(".") + ". [1]"}
        return {"content": self.answer}


DOCS = {
    "cats.pdf": "Cats purr when they are content and also when they are stressed. Kittens learn to purr within days.",
    "engines.pdf": "Diesel engines ignite fuel by compression alone. Four stroke engines complete a cycle in two rotations.",
    "tea.pdf": "Green tea is steeped at eighty degrees to avoid bitterness. Black tea takes boiling water.",
}
META = {
    "cats.pdf": {"id": "2401.00001", "title": "On Purring", "published": "2024-01-05"},
    "engines.pdf": {"id": "2402.00002", "title": "Engine Cycles", "published": "2023-06-01"},
    "tea.pdf": {"id": "2403.00003", "title": "Steeping Temperatures", "published": "2022-03-09"},
}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "idx")
    for name, text in DOCS.items():
        s.add([Chunk(f"{name}:1:0", text, name, 1)], fake_embed([text]))
    return s
