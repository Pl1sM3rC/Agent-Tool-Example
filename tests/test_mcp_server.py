import os
import sys
import urllib.request
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT_DIR = Path(__file__).resolve().parent.parent
SERVER = ROOT_DIR / "mcp_server.py"


async def test_server_starts_over_stdio_and_exposes_tools():
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            assert {"add", "multiply", "count_r", "search_knowledge_base", "add_to_knowledge_base",
                    "list_knowledge_base_sources"} <= names
            result = await session.call_tool("add", {"a": 2, "b": 3})
            assert result.content[0].text == "5"


def embedding_endpoint_reachable() -> bool:
    base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8080/v1")
    try:
        urllib.request.urlopen(f"{base_url}/models", timeout=2)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not embedding_endpoint_reachable(), reason="embedding endpoint (EMBEDDING_BASE_URL) not reachable")
async def test_rag_tools_store_and_find_text(tmp_path):
    env = {**os.environ, "VECTOR_DB_PATH": str(tmp_path / "store.db")}
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            empty = await session.call_tool("search_knowledge_base", {"query": "anything"})
            assert empty.content[0].text == "The knowledge base is empty."

            for text, source in [("Der Server Lindwurm steht im Keller von Gebäude 20.30.", "server.md"),
                                 ("Bananen sind reich an Kalium.", "obst.md")]:
                result = await session.call_tool("add_to_knowledge_base", {"text": text, "source": source})
                assert not result.isError, result.content

            sources = await session.call_tool("list_knowledge_base_sources", {})
            assert sources.structuredContent["result"] == ["obst.md", "server.md"]

            found = await session.call_tool("search_knowledge_base", {"query": "Wo steht Lindwurm?", "k": 1})
            assert found.content[0].text.startswith("[1] source: server.md")
            assert "Gebäude 20.30" in found.content[0].text
