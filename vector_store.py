"""SQLite + sqlite-vec vector store used by the RAG tools in mcp_server.py.

Run as a script (a python-fire CLI) to add content to the store:
    python vector_store.py add notes.md docs/      # embed files (.txt/.md; directories are searched recursively)
    python vector_store.py add_text "Some fact" --source facts
    python vector_store.py search "some question" --k 3
    python vector_store.py sources
    python vector_store.py delete notes.md
    python vector_store.py --help
"""
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np
import sqlite_vec
from loguru import logger

ROOT_DIR = Path(__file__).resolve().parent

# Defaults, overridable via environment variables (e.g. in .env)
DEFAULT_DB_PATH = ROOT_DIR / "data" / "vector_store.db"

# OpenAI-compatible embeddings endpoint. Default: EmbeddingGemma-2 (768 dims), served by llama-swap/llama.cpp as "embed-mm"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8080/v1"
DEFAULT_OPENAI_EMBEDDING_MODEL = "embed-mm"
# EmbeddingGemma-2 expects these task prefixes. Other models need other prefixes (or none), e.g. Qwen3-Embedding:
# query "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", documents none
DEFAULT_QUERY_PREFIX = "task: search result | query: "
DEFAULT_DOCUMENT_PREFIX = "title: none | text: "

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200
FILE_SUFFIXES = (".txt", ".md")  # file types the command line embeds from directories


def _resolve(path: str | Path) -> Path:
    """Relative paths (e.g. from .env) are relative to the repo, not to the current working directory."""
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class OpenAIEmbedder:
    """Embeddings from an OpenAI-compatible /v1/embeddings endpoint (llama.cpp/llama-swap, LM Studio, Ollama, vLLM, OpenAI, ...)."""

    def __init__(self, model_name: str | None = None, base_url: str | None = None, api_key: str | None = None):
        from langchain_openai import OpenAIEmbeddings

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
        base_url = base_url or os.getenv("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
        self.query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", DEFAULT_QUERY_PREFIX)
        self.document_prefix = os.getenv("EMBEDDING_DOCUMENT_PREFIX", DEFAULT_DOCUMENT_PREFIX)
        logger.info(f"Using embedding model {self.model_name} at {base_url}")
        self._embeddings = OpenAIEmbeddings(
            model=self.model_name,
            base_url=base_url,
            api_key=api_key or os.getenv("EMBEDDING_API_KEY", "not-needed"),
            # By default OpenAIEmbeddings sends token ids instead of text, which non-OpenAI servers reject
            check_embedding_ctx_length=False,
        )
        self.dim = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        vectors = self._embeddings.embed_documents([self.document_prefix + t for t in texts])
        return [np.asarray(v, dtype=np.float32) for v in vectors]

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self._embeddings.embed_query(self.query_prefix + text), dtype=np.float32)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into overlapping chunks, preferring paragraph, line and sentence boundaries."""
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Cut at the last natural break in the second half of the window, if there is one
            for sep in ("\n\n", "\n", ". "):
                cut = text.rfind(sep, start + chunk_size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)  # start the overlap at a word boundary
        start = space + 1 if space != -1 else next_start
    return chunks


def _to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


class VectorStore:
    """Stores text chunks in a normal table and their embeddings in a sqlite-vec vec0 table (same rowid)."""

    def __init__(self, embedder: Embedder, db_path: str | Path | None = None):
        self.embedder = embedder
        self.db_path = _resolve(db_path or os.getenv("VECTOR_DB_PATH", DEFAULT_DB_PATH))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._init_schema(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A new connection per operation: sqlite3 connections are bound to the thread that created them,
        # and the MCP server may call tools from different threads.
        conn = sqlite3.connect(self.db_path)
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            with conn:  # one transaction: commit on success, rollback on error
                yield conn
        finally:
            conn.close()

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (source, chunk_index)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{self.embedder.dim}] distance_metric=cosine
            );
            """
        )
        # Vectors from different models are not comparable, so a store is tied to the model that created it
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('embedding_model', ?)", (self.embedder.model_name,))
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('embedding_dim', ?)", (str(self.embedder.dim),))
        stored = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        if stored["embedding_model"] != self.embedder.model_name or int(stored["embedding_dim"]) != self.embedder.dim:
            raise ValueError(
                f"{self.db_path} was created with {stored['embedding_model']} ({stored['embedding_dim']} dims), "
                f"but the configured model is {self.embedder.model_name} ({self.embedder.dim} dims). "
                "Use the original model or a new VECTOR_DB_PATH."
            )

    def add_text(self, text: str, source: str) -> int:
        """Chunks, embeds and stores a text. Existing chunks with the same source are replaced. Returns the chunk count."""
        chunks = chunk_text(text)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents(chunks)
        with self._connect() as conn:
            self._delete_source(conn, source)
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                cur = conn.execute(
                    "INSERT INTO chunks (source, chunk_index, content) VALUES (?, ?, ?)", (source, index, chunk)
                )
                conn.execute(
                    "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)", (cur.lastrowid, _to_blob(embedding))
                )
        logger.info(f"Stored {len(chunks)} chunks for source '{source}'")
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Returns the k chunks closest to the query, nearest first (cosine distance: 0 = identical)."""
        query_blob = _to_blob(self.embedder.embed_query(query))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.source, c.chunk_index, c.content, v.distance
                FROM vec_chunks v JOIN chunks c ON c.id = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (query_blob, k),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_sources(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) AS chunks, MAX(created_at) AS stored_at FROM chunks GROUP BY source ORDER BY source"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_source(self, source: str) -> int:
        """Removes all chunks of a source. Returns the number of deleted chunks."""
        with self._connect() as conn:
            return self._delete_source(conn, source)

    @staticmethod
    def _delete_source(conn: sqlite3.Connection, source: str) -> int:
        ids = [row["id"] for row in conn.execute("SELECT id FROM chunks WHERE source = ?", (source,))]
        conn.executemany("DELETE FROM vec_chunks WHERE rowid = ?", [(i,) for i in ids])
        conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        return len(ids)


def open_default_store() -> VectorStore:
    """Opens the store configured by VECTOR_DB_PATH / EMBEDDING_* (or the defaults above)."""
    return VectorStore(OpenAIEmbedder())


def iter_files(paths: Iterable[str]) -> Iterable[Path]:
    """Yields the given files, plus every .txt/.md file inside the given directories."""
    for path in map(Path, paths):
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in FILE_SUFFIXES)
        elif path.is_file():
            yield path
        else:
            logger.warning(f"Skipping {path}: not found")


class CLI:
    """Adds content to the RAG vector store and inspects it."""

    def __init__(self):
        self._opened: VectorStore | None = None

    def _store(self) -> VectorStore:
        # Opened on first use (a method, not a property, so that --help doesn't connect to the embedding endpoint)
        if self._opened is None:
            self._opened = open_default_store()
        return self._opened

    def add(self, *paths: str, source: str | None = None) -> None:
        """Embeds files (.txt/.md; directories are searched recursively). The source defaults to the file path."""
        files = list(iter_files(paths))
        if source is not None and len(files) != 1:
            raise ValueError("--source can only be used with a single file")
        for file in files:
            self._store().add_text(file.read_text(encoding="utf-8"), str(source) if source is not None else file.as_posix())

    def add_text(self, text: str, source: str) -> None:
        """Embeds a text and stores it under the given source."""
        self._store().add_text(str(text), str(source))

    def search(self, query: str, k: int = 4) -> list[str]:
        """Shows the k chunks closest to the query."""
        return [f"{r['distance']:.3f}  {r['source']}#{r['chunk_index']}  {r['content'][:200]!r}"
                for r in self._store().search(str(query), k)]

    def sources(self) -> list[str]:  # not named "list": that would shadow list[...] in this class's annotations
        """Lists the stored sources."""
        return [f"{s['source']}\t{s['chunks']} chunk(s)\t{s['stored_at']}" for s in self._store().list_sources()]

    def delete(self, source: str) -> str:
        """Removes a source from the store."""
        return f"Deleted {self._store().delete_source(str(source))} chunk(s) of '{source}'"


if __name__ == "__main__":
    # Only configured when run as a script; when imported, the importing program sets up .env and logging
    import fire
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(ROOT_DIR / "logs" / "vector_store.log", level="DEBUG")
    fire.Fire(CLI())
