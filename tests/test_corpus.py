import json

import fitz
import httpx
import pytest
from conftest import fake_embed
from ragchat.store import Store

from research import corpus
from research.corpus import build_index, chunks_for_paper, download, load_manifest, paper_meta


@pytest.fixture
def collection(tmp_path, monkeypatch):
    """A small fake collection: three papers, one of them a broken file."""
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    papers = []
    for pid, pages in {
        "2401.00001": ["Retrieval quality depends on chunking. " * 40, "A second page about embeddings and search."],
        "2402.00002": ["Reranking improves the ordering of retrieved passages."],
    }.items():
        doc = fitz.open()
        for text in pages:
            doc.new_page().insert_textbox(fitz.Rect(30, 30, 570, 800), text, fontsize=8)
        doc.save(pdfs / f"{pid}.pdf")
        doc.close()
        papers.append({"id": pid, "title": f"Paper {pid}", "published": "2024-01-01"})
    (pdfs / "2403.00003.pdf").write_bytes(b"%PDF-1.4 truncated nonsense")
    papers.append({"id": "2403.00003", "title": "Broken", "published": "2024-03-01"})
    papers.append({"id": "2404.00004", "title": "Never downloaded", "published": "2024-04-01"})  # in the manifest, no PDF

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(p) for p in papers))
    monkeypatch.setattr(corpus, "MANIFEST", manifest)
    monkeypatch.setattr(corpus, "PDF_DIR", pdfs)
    return pdfs


def test_metadata_is_keyed_by_the_file_name_the_index_uses_as_source(collection):
    meta = paper_meta()
    assert meta["2401.00001.pdf"]["title"] == "Paper 2401.00001"
    assert "2404.00004.pdf" in meta  # the manifest lists it even though the file is missing


def test_chunks_carry_page_numbers_and_stable_ids(collection):
    chunks = chunks_for_paper(collection / "2401.00001.pdf")
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[0].id == "2401.00001.pdf:1:0" and all(c.source == "2401.00001.pdf" for c in chunks)


def test_building_the_index_skips_broken_and_missing_files(collection, tmp_path):
    store = Store(tmp_path / "idx")
    messages = []
    total = build_index(store, fake_embed, progress=messages.append)
    assert total == store.count() > 0
    assert any("skip 2403.00003" in m for m in messages)
    sources = {h.source for h in store.query(fake_embed(["retrieval chunking embeddings reranking"], "query")[0], k=20)}
    assert sources == {"2401.00001.pdf", "2402.00002.pdf"}


def test_building_twice_does_not_duplicate(collection, tmp_path):
    store = Store(tmp_path / "idx")
    build_index(store, fake_embed, progress=lambda m: None)
    first = store.count()
    build_index(store, fake_embed, progress=lambda m: None)
    assert store.count() == first


def test_limit_restricts_how_many_papers_are_indexed(collection, tmp_path):
    store = Store(tmp_path / "idx")
    build_index(store, fake_embed, limit=1, progress=lambda m: None)
    assert {h.source for h in store.query(fake_embed(["retrieval reranking"], "query")[0], k=20)} == {"2401.00001.pdf"}


def test_the_bundled_collection_lists_300_papers_with_titles():
    papers = load_manifest()
    assert len(papers) == 300 and len({p["id"] for p in papers}) == 300
    assert all(p["title"] and p["published"] for p in papers)


def test_an_empty_folder_says_to_fetch_first(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "PDF_DIR", tmp_path / "none")
    messages = []
    assert build_index(Store(tmp_path / "idx"), fake_embed, progress=messages.append) == 0
    assert "research fetch" in messages[0]


def test_download_keeps_pdfs_skips_what_is_there_and_reports_failures(collection, tmp_path, monkeypatch):
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "2401.00001.pdf").write_bytes(b"%PDF-1.4 already here")
    monkeypatch.setattr(corpus, "PDF_DIR", fresh)
    busy = {"2402.00002": 1}

    def handler(request):
        pid = request.url.path.rsplit("/", 1)[-1]
        if busy.get(pid):
            busy[pid] -= 1
            return httpx.Response(406)  # arXiv's "slow down", which clears on retry
        if pid == "2403.00003":
            return httpx.Response(200, text="<html>not a pdf</html>")
        if pid == "2404.00004":
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF-1.4 " + pid.encode())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = download(client=client, sleep=lambda s: None, progress=lambda m: None)
    assert result == {"downloaded": 1, "skipped": 1, "failed": ["2403.00003", "2404.00004"]}
    assert (fresh / "2402.00002.pdf").read_bytes().startswith(b"%PDF-")
    assert not list(fresh.glob("*.part"))
