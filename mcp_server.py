import sys
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from mcp.server.fastmcp import FastMCP

from vector_store import VectorStore, open_default_store

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# Logging Setup: stdout is the MCP stdio communication stream, so logs must only go to stderr and the log file
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(ROOT_DIR / "logs" / "mcp_server.log", level="DEBUG")

mcp = FastMCP("Agent Tools")


@cache
def get_store() -> VectorStore:
    # Loaded on the first RAG tool call, so starting the server (and listing tools) stays fast
    return open_default_store()


@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds 2 numbers together. Requires 2 inputs: an integer a and an integer b. """
    logger.info("RUNNING ADD TOOL")
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies 2 numbers. Requires 2 inputs: an integer a and an integer b. """
    logger.info("RUNNING MULTIPLY TOOL")
    return a * b

@mcp.tool()
def count_r(word: str) -> int:
    """Can count how many r's a given word has. Requires 1 input: a string word. """
    return word.lower().count('r')

@mcp.tool()
def search_knowledge_base(query: str, k: int = 4) -> str:
    """Searches the knowledge base (vector store) for the text passages most relevant to a query. Use it to answer questions about the stored documents and cite the sources. Requires 1 input: a string query. Optional: an integer k, the number of passages to return (default 4). """
    logger.info(f"RUNNING SEARCH TOOL: {query!r} (k={k})")
    results = get_store().search(query, k)
    if not results:
        return "The knowledge base is empty."
    return "\n\n".join(
        f"[{i}] source: {r['source']} (chunk {r['chunk_index']}, distance {r['distance']:.3f})\n{r['content']}"
        for i, r in enumerate(results, start=1)
    )

@mcp.tool()
def add_to_knowledge_base(text: str, source: str) -> str:
    """Stores a text in the knowledge base so it can be found later with search_knowledge_base. A text stored under an existing source replaces the old one. Requires 2 inputs: a string text and a string source (a short name for where the text comes from). """
    logger.info(f"RUNNING ADD TO KNOWLEDGE BASE TOOL: source={source!r}")
    chunks = get_store().add_text(text, source)
    return f"Stored {chunks} chunk(s) under source '{source}'."

@mcp.tool()
def list_knowledge_base_sources() -> list[str]:
    """Lists the sources (documents) stored in the knowledge base. Requires no inputs. """
    return [s["source"] for s in get_store().list_sources()]


if __name__ == "__main__":
    logger.info("Starting MCP Server via mcp.run(transport='stdio')...")
    mcp.run(transport="stdio")
    logger.info("MCP Server finished.")
