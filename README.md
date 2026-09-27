# Ubiquitous Memory

A research assistant over 300 arXiv papers on retrieval-augmented generation. Ask it questions and every answer comes back with citations it has checked against the papers. It remembers the conversation, so follow-ups like "what about its limits?" work, and it can write a short literature brief that relates several papers on a topic.

## What it does

- **Cited answers.** Every answer cites the passages it used, with the paper title, year, page and arXiv link.
- **Citation checks.** A citation to a passage that doesn't exist, a claim with no citation, or a sentence the cited passage doesn't back up is flagged inline rather than hidden.
- **Conversation memory.** Follow-ups like "what about its limits?" are rewritten into a standalone question before searching. Older turns are folded into a short summary once the window fills.
- **Literature briefs.** Pick a topic and it finds the closest papers, summarises each from its own passages, then writes a short overview of how they relate.
- **Web app.** A Streamlit page with an Ask tab and a Research brief tab.

The citation check is **lexical overlap**, not an entailment model. It reliably catches invented citation numbers, but a correct paraphrase can still be flagged if it shares few exact words with the source (see Results).

Memory holds only the user's questions and the assistant's own answers, never the retrieved document text, so instructions hidden in a document can't carry over from one turn to the next.

## Requirements

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- An API key for OpenAI or Google Gemini (any OpenAI-compatible endpoint works), used for chat and embeddings. Your questions and the paper text are sent to the provider you choose.
- About 500 MB of disk for the papers

## Setup

```bash
uv sync
cp .env.example .env    # uncomment one provider block and paste your key

uv run python -m research fetch                      # download the 300 PDFs from arXiv (~25 min; --limit 20 for a quick try)
uv run --env-file .env python -m research index      # chunk and embed them
```

`fetch` pauses between downloads, as arXiv asks, and skips files already on disk, so an interrupted run can simply be started again. The papers and the index live in `./data`; set `RESEARCH_DATA` to put them somewhere else.

## Run

```bash
uv run --env-file .env python -m research ask "What does RAG change about hallucination?"
uv run --env-file .env python -m research chat           # follow-ups like "what about its limits?" work
uv run --env-file .env python -m research brief "reranking in retrieval-augmented generation" --papers 4
uv run --env-file .env streamlit run src/research/app.py # Ask and Research brief tabs

uv run pytest                                   # offline, no key needed
uv run --env-file .env pytest -m live -s        # against your provider and the indexed collection
```

## Configuration

Set these in `.env`. The first four are required and have no defaults.

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL` | the provider's OpenAI-compatible endpoint |
| `LLM_API_KEY` | your provider key |
| `LLM_MODEL` | chat model, for answers and briefs |
| `EMBED_MODEL` | embedding model name |
| `REWRITE_MODEL` | optional cheaper model for rewriting follow-up questions (defaults to `LLM_MODEL`) |
| `RAGCHAT_MIN_SCORE` | similarity below which it answers "I don't know" (default `0.7`); scores differ between embedding models, so tune it for yours |
| `RESEARCH_DATA` | where the papers and index are kept (default `./data`) |

Changing `EMBED_MODEL` after indexing is refused rather than silently mixing vectors: delete `data/index` and run `index` again.

## Results

Measured with open models run locally (a 20B-parameter chat model, a 4B model for rewriting and a small open embedding model). A hosted provider will give different numbers, so run the live tests against yours.

| Check | Result |
|---|---|
| Offline tests | 51 pass |
| Deliberately broken to confirm tests notice | phantom-citation check, support check, follow-up rewriting, citation stripping in memory: each made a test fail |
| Retrieval over 300 papers, 40 questions | right paper ranked first 72%, in top 5 78%, MRR 0.75 |
| Citations | 10/10 answered, 0 phantom citations, but only 3/10 answers had every sentence pass the lexical support check (30% of cited sentences). Checked by hand, the flagged sentences were correct paraphrases, not fabrications |
| Memory helps follow-ups | with question rewriting, 12/18 follow-ups found the right paper in the top 6; without rewriting, 0/18 did |
| Literature brief | 3 papers on reranking in RAG, each summarised from its own passages, related in one paragraph citing all three |
| Web app | answers a real question end to end |

The phantom-citation check (an invented `[7]` when there are 6 passages) is reliable. The support check is a floor, not a semantic judgement: don't read "unsupported" as "wrong" without checking.
