"""The paper collection: which papers it holds, downloading them from arXiv, and turning them into one index."""
import json
import os
import time
from pathlib import Path
from typing import Callable

import httpx

from ragchat.ingest import Chunk
from ragchat.rag import Embedder
from ragchat.store import Store
from shared.docs import chunk_text, read_pages

# 300 arXiv papers on retrieval-augmented generation: id, title and publication date
MANIFEST = Path(__file__).with_name("papers.jsonl")
PDF_DIR = Path(os.environ.get("RESEARCH_DATA", "data")) / "pdfs"

PDF_URL = "https://arxiv.org/pdf/{id}"
HEADERS = {"User-Agent": "ubiquitous-memory/0.1"}
# arXiv asks for a 3 second gap; 3-4s apart still drew bare 406s, 5s did not
DELAY = 5.0
# besides the usual 429/5xx, arXiv sometimes answers a busy client with a bare 406 that goes away on retry
BUSY = {406, 429, 500, 502, 503, 504}


def load_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


def pdf_path(paper_id: str) -> Path:
    # old-style ids look like cs/0601001
    return PDF_DIR / (paper_id.replace("/", "_") + ".pdf")


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def get_with_retry(client: httpx.Client, url: str, *, sleep=time.sleep, tries: int = 5) -> httpx.Response:
    for attempt in range(tries):
        resp = client.get(url)
        if resp.status_code not in BUSY or attempt == tries - 1:
            return resp
        sleep(DELAY * 2 ** (attempt + 1))  # 10s, 20s, 40s, 80s
    return resp


def download(
    limit: int | None = None,
    client: httpx.Client | None = None,
    sleep=time.sleep,
    progress: Callable[[str], None] = print,
) -> dict:
    """Fetch the collection's PDFs. Files already on disk are skipped, so an interrupted run can be restarted."""
    papers = load_manifest()[:limit]
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    client = client or httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True)
    got, skipped, failed = 0, 0, []
    for n, p in enumerate(papers, 1):
        dest = pdf_path(p["id"])
        if dest.exists() and is_pdf(dest.read_bytes()[:5]):
            skipped += 1
            continue
        try:
            resp = get_with_retry(client, PDF_URL.format(id=p["id"]), sleep=sleep)
            resp.raise_for_status()
            if not is_pdf(resp.content):
                raise ValueError("response was not a PDF")
        except (httpx.HTTPError, ValueError) as e:
            failed.append(p["id"])
            progress(f"failed {p['id']}: {str(e)[:80]}")
        else:
            # temp name + rename, so a run killed mid-write can't leave a truncated file that looks valid
            part = dest.with_suffix(".part")
            part.write_bytes(resp.content)
            part.replace(dest)
            got += 1
        if n % 25 == 0 or n == len(papers):
            progress(f"{n}/{len(papers)} papers checked")
        sleep(DELAY)
    return {"downloaded": got, "skipped": skipped, "failed": failed}


def paper_meta() -> dict[str, dict]:
    """Title, year and id for each paper, keyed by the PDF file name the index uses as a source."""
    return {pdf_path(p["id"]).name: p for p in load_manifest()}


def chunks_for_paper(pdf: Path) -> list[Chunk]:
    out = []
    for page, text in read_pages(pdf):
        for i, piece in enumerate(chunk_text(text)):
            out.append(Chunk(f"{pdf.name}:{page}:{i}", piece, pdf.name, page))
    return out


def build_index(store: Store, embed: Embedder, limit: int | None = None, progress: Callable[[str], None] = print) -> int:
    papers = [p for p in load_manifest() if pdf_path(p["id"]).exists()][:limit]
    if not papers:
        progress("no papers on disk yet: run `python -m research fetch` first")
        return 0
    total = 0
    for n, paper in enumerate(papers, 1):
        try:
            chunks = chunks_for_paper(pdf_path(paper["id"]))
        except Exception as e:  # one broken PDF shouldn't stop the rest
            progress(f"skip {paper['id']}: {str(e)[:80]}")
            continue
        if chunks:
            store.add(chunks, embed([c.text for c in chunks], "document"))
            total += len(chunks)
        if n % 25 == 0 or n == len(papers):
            progress(f"{n}/{len(papers)} papers, {total} chunks")
    return total
