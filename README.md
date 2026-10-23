# Agent Tool Example

A LangGraph/LangChain agent (`agent.py`) that loads its tools from a local MCP server (`mcp_server.py`).
Besides simple example tools (`add`, `multiply`, `count_r`), the server provides RAG tools that search and fill
a local vector store (SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec)).

| File | Purpose |
| --- | --- |
| `agent.py` | Agent (`langchain.agents.create_agent`) and LLM configuration (`create_model()`) |
| `mcp_server.py` | MCP server with all tools (stdio) |
| `vector_store.py` | Vector store: chunking, embeddings (any OpenAI-compatible endpoint), search; run as a script to add content |

## Setup
Requires Python 3.10 or newer.

**Option A: venv + pip**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: [uv](https://docs.astral.sh/uv/)**
```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Then create the configuration file and fill in the API key:
```bash
cp .env.example .env
```

## Usage
```bash
# Add content to the vector store (.txt/.md; directories are searched recursively)
python vector_store.py add notes.md documents/
python vector_store.py add_text "The Lindwurm server is in the basement." --source notes
python vector_store.py search "Where is the Lindwurm server?" --k 3   # test retrieval without the LLM
python vector_store.py sources                # list stored sources
python vector_store.py delete notes           # remove a source
python vector_store.py --help

# Ask the agent
python agent.py "Where is the Lindwurm server?"
```
Adding a source again replaces its previous version. The database is stored at `data/vector_store.db`
(configurable with `VECTOR_DB_PATH` in `.env`).

## Embedding model
Embeddings are created through LangChain's `OpenAIEmbeddings`, so any OpenAI-compatible `/v1/embeddings` endpoint
works. The default is EmbeddingGemma-2 (768 dimensions), served by a local llama.cpp server at
`http://localhost:8080/v1` under the model name `embed-mm`. All settings are in `.env` (see `.env.example`):

* `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY` select the endpoint and model.
* `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_DOCUMENT_PREFIX` hold the task prefixes the model expects. The defaults are
  EmbeddingGemma's; other models need different prefixes (`.env.example` has the values for Qwen3-Embedding) or none.

A vector store is tied to the model that created it. After switching models, either use a new `VECTOR_DB_PATH` or
delete the database and embed the content again.

### Hosting the embedding model
The embedding model runs as a separate local server. Any of these works:

**llama.cpp** (matches the defaults). EmbeddingGemma-2 needs a recent llama.cpp build.
```bash
hf download unsloth/embeddinggemma-2-GGUF embeddinggemma-2-UD-Q4_K_XL.gguf --local-dir models
llama-server --model models/embeddinggemma-2-UD-Q4_K_XL.gguf --alias embed-mm \
  --embeddings --pooling mean --ctx-size 2048 --port 8080
```
`--embeddings` enables the `/v1/embeddings` endpoint; `--alias` sets the model name that `EMBEDDING_MODEL` refers to.
[llama-swap](https://github.com/mostlygeek/llama-swap) can sit in front of it to serve several models on one port.

**Ollama** (first-generation EmbeddingGemma 300M, 768 dimensions, same prefixes)
```bash
ollama pull embeddinggemma
```
```
EMBEDDING_BASE_URL="http://localhost:11434/v1"
EMBEDDING_MODEL="embeddinggemma"
```

**LM Studio**: download an embedding model (e.g. EmbeddingGemma) in the app and start the local server
(Developer tab). LM Studio names models after their files, so the exact ID comes from
`curl http://localhost:1234/v1/models`:
```
EMBEDDING_BASE_URL="http://localhost:1234/v1"
EMBEDDING_MODEL="<id from /v1/models>"
```

For a model other than EmbeddingGemma, adjust (or empty) the two prefix variables.

## Changing the LLM / endpoint
The agent uses `ChatOpenAI`, which works with any OpenAI-compatible API (DeepSeek, OpenAI, OpenRouter,
LM Studio, Ollama, vLLM, ...). Switching only requires changing `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`
in `.env`. Examples are in `.env.example`; the docstring of `create_model()` in `agent.py` lists LangChain's
premade adapters for other providers.

Example for a local model served by LM Studio:
```
LLM_BASE_URL="http://localhost:1234/v1"
LLM_MODEL="<model loaded in LM Studio>"
```

## Adding tools
Tools are plain Python functions in `mcp_server.py`, registered with the `@mcp.tool()` decorator. The agent picks
them up automatically. Each tool needs a short, precise docstring that states what it does and which inputs it
expects, since the LLM decides from this description when and how to call it.
* https://docs.langchain.com/oss/python/langchain/mcp

## Tests
```bash
pip install pytest pytest-asyncio      # or: uv pip install pytest pytest-asyncio
python -m pytest
```
The RAG test over the MCP server only runs when the embedding endpoint (`EMBEDDING_BASE_URL`) is reachable.

## Notes
* Logs (loguru) are written to `logs/`. The MCP server logs to stderr only, because stdout is its communication channel with the agent.
* For DeepSeek, thinking mode is disabled, because together with tool calls through `ChatOpenAI` it causes HTTP 400 errors (see `create_model()`).
* DeepSeek stores data in China. Passages found by `search_knowledge_base` are sent to the LLM as context, so a local model is the better choice for private documents.
