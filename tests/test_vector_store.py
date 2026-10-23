import hashlib
import re

import numpy as np
import pytest

from vector_store import VectorStore, chunk_text


class FakeEmbedder:
    """Deterministic bag-of-words embedder, so the tests need no model download."""

    def __init__(self, model_name="fake-model", dim=64):
        self.model_name = model_name
        self.dim = dim

    def _embed(self, text):
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return vector / (np.linalg.norm(vector) or 1)

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def store(tmp_path):
    return VectorStore(FakeEmbedder(), tmp_path / "store.db")


def test_search_returns_most_similar_chunk_first(store):
    store.add_text("Die Katze schläft auf dem Sofa.", "cats.md")
    store.add_text("The rocket launched into orbit.", "space.md")
    results = store.search("rocket orbit", k=2)
    assert [r["source"] for r in results] == ["space.md", "cats.md"]
    assert results[0]["content"] == "The rocket launched into orbit."
    assert results[0]["distance"] < results[1]["distance"]


def test_search_on_empty_store_returns_nothing(store):
    assert store.search("anything") == []


def test_adding_same_source_replaces_old_chunks(store):
    store.add_text("old content " * 300, "doc.md")
    store.add_text("new content", "doc.md")
    assert store.list_sources()[0]["source"] == "doc.md"
    assert store.list_sources()[0]["chunks"] == 1
    assert [r["content"] for r in store.search("content", k=10)] == ["new content"]


def test_delete_source_removes_chunks_and_vectors(store):
    store.add_text("keep me", "keep.md")
    store.add_text("delete me", "delete.md")
    assert store.delete_source("delete.md") == 1
    assert [s["source"] for s in store.list_sources()] == ["keep.md"]
    assert [r["source"] for r in store.search("delete me", k=10)] == ["keep.md"]


def test_empty_text_stores_nothing(store):
    assert store.add_text("   \n", "empty.md") == 0
    assert store.list_sources() == []


def test_store_persists_across_instances(tmp_path):
    VectorStore(FakeEmbedder(), tmp_path / "store.db").add_text("persistent fact", "fact.md")
    assert VectorStore(FakeEmbedder(), tmp_path / "store.db").search("persistent fact", k=1)[0]["source"] == "fact.md"


def test_reopening_with_different_model_fails(tmp_path):
    VectorStore(FakeEmbedder("model-a"), tmp_path / "store.db")
    with pytest.raises(ValueError, match="model-a"):
        VectorStore(FakeEmbedder("model-b"), tmp_path / "store.db")


def test_chunk_text_short_text_is_one_chunk():
    assert chunk_text("  Hallo Welt.  ") == ["Hallo Welt."]


def test_chunk_text_splits_long_text_with_overlap_at_paragraphs():
    paragraphs = [f"Paragraph {i}. " + "word " * 80 for i in range(10)]
    chunks = chunk_text("\n\n".join(paragraphs), chunk_size=1000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert all(c.startswith("Paragraph") or c.startswith("word") for c in chunks)
    # every paragraph survives, and neighbouring chunks overlap
    joined = " ".join(chunks)
    assert all(f"Paragraph {i}." in joined for i in range(10))
    assert any(chunks[i][-50:].strip() in chunks[i + 1] for i in range(len(chunks) - 1))


def test_chunk_text_without_breaks_still_terminates():
    chunks = chunk_text("x" * 2500, chunk_size=1000, overlap=200)
    assert [len(c) for c in chunks] == [1000, 1000, 900]


def test_openai_embedder_adds_task_prefixes_and_returns_float32(monkeypatch):
    import vector_store

    calls = []

    class StubEmbeddings:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def embed_documents(self, texts):
            calls.append(("documents", texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            calls.append(("query", text))
            return [0.0, 1.0, 0.0]

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", StubEmbeddings)
    for var in ("EMBEDDING_MODEL", "EMBEDDING_BASE_URL", "EMBEDDING_QUERY_PREFIX", "EMBEDDING_DOCUMENT_PREFIX"):
        monkeypatch.delenv(var, raising=False)

    embedder = vector_store.OpenAIEmbedder()
    assert embedder.model_name == "embed-mm" and embedder.dim == 3
    init_kwargs = calls[0][1]
    assert init_kwargs["base_url"] == "http://localhost:8080/v1"
    assert init_kwargs["check_embedding_ctx_length"] is False

    docs = embedder.embed_documents(["Hallo"])
    query = embedder.embed_query("Wo?")
    assert ("documents", ["title: none | text: Hallo"]) in calls
    assert ("query", "task: search result | query: Wo?") in calls
    assert docs[0].dtype == np.float32 and query.dtype == np.float32


def test_openai_embedder_prefixes_can_be_disabled(monkeypatch):
    import vector_store

    texts = []

    class StubEmbeddings:
        def __init__(self, **kwargs):
            pass

        def embed_documents(self, batch):
            texts.extend(batch)
            return [[1.0] for _ in batch]

        def embed_query(self, text):
            texts.append(text)
            return [1.0]

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", StubEmbeddings)
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", "")
    embedder = vector_store.OpenAIEmbedder()
    embedder.embed_documents(["doc"])
    embedder.embed_query("query")
    assert texts[-2:] == ["doc", "query"]


def test_cli_add_search_sources_delete(tmp_path, monkeypatch):
    import vector_store

    monkeypatch.setattr(vector_store, "open_default_store", lambda: VectorStore(FakeEmbedder(), tmp_path / "store.db"))
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    (docs / "rocket.md").write_text("The rocket launched into orbit.", encoding="utf-8")
    (docs / "sub" / "cat.txt").write_text("Die Katze schläft.", encoding="utf-8")
    (docs / "ignored.pdf").write_text("not a text file", encoding="utf-8")

    cli = vector_store.CLI()
    cli.add(str(docs))
    cli.add_text("A note", source=2024)  # fire passes numbers as int
    assert {line.split("\t")[0] for line in cli.sources()} == {
        "2024", (docs / "rocket.md").as_posix(), (docs / "sub" / "cat.txt").as_posix()}
    assert "rocket.md#0" in cli.search("rocket orbit", k=1)[0]
    assert cli.delete("2024") == "Deleted 1 chunk(s) of '2024'"
    with pytest.raises(ValueError, match="single file"):
        cli.add(str(docs), source="x")
