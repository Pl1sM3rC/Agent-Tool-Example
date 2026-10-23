from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path
import sys, os, asyncio

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(ROOT_DIR / "logs" / "agent.log", level="DEBUG")

async def print_astream(stream):
    try:
        async for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
    except Exception as e:
        logger.exception(f"Couldn't print Stream: {e}")
        sys.exit(1)
    finally:
        print("--- Graph Async Stream End ---")

async def load_mcp_tools_from_session(session: ClientSession) -> list[BaseTool]:
    """Loads LangChain tools from an active MCP ClientSession."""
    try:
        tools = await load_mcp_tools(session)
        logger.debug(f"--- Tools available via session ({len(tools)} tools) ---")
        for tool in tools:
            logger.debug(f"  Tool Name: {getattr(tool, 'name', 'N/A')}")
            logger.debug(f"  Description: {getattr(tool, 'description', 'N/A')}")
        logger.info("-" * 50)
        return tools
    except Exception as e:
        logger.exception(f"Failed to load tools from session: {e}")
        return []

def create_model() -> ChatOpenAI:
    """Creates the chat model. ChatOpenAI works with every OpenAI-compatible endpoint, not only OpenAI itself.

    To change the endpoint, set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY in .env (see .env.example).
    Examples of OpenAI-compatible base URLs:
        DeepSeek    https://api.deepseek.com           (default; models: deepseek-flash, deepseek-v4-pro)
        OpenAI      https://api.openai.com/v1
        OpenRouter  https://openrouter.ai/api/v1
        LM Studio   http://localhost:1234/v1           (local; api key can be anything)
        Ollama      http://localhost:11434/v1          (local; api key can be anything)
        vLLM        http://localhost:8000/v1           (local)

    LangChain also has premade adapters for many providers, which support provider-specific
    features better (thinking, caching, ...). Install the package and swap ChatOpenAI for it:
        langchain-anthropic     ChatAnthropic
        langchain-google-genai  ChatGoogleGenerativeAI
        langchain-deepseek      ChatDeepSeek
        langchain-mistralai     ChatMistralAI
        langchain-ollama        ChatOllama
        langchain-groq          ChatGroq
        langchain-openrouter    ChatOpenRouter
        langchain-aws           ChatBedrockConverse
        langchain-huggingface   ChatHuggingFace
    Or let LangChain pick the adapter: from langchain.chat_models import init_chat_model
        model = init_chat_model("anthropic:claude-sonnet-5-5")
    Full list: https://docs.langchain.com/oss/python/integrations/chat
    """
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    extra_body = None
    if "deepseek.com" in base_url:
        # DeepSeek's thinking mode (on by default) requires reasoning_content to be sent back in tool loops,
        # which ChatOpenAI doesn't do -> HTTP 400 on the second call. So it is disabled for DeepSeek.
        extra_body = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(
        base_url=base_url,
        model=os.getenv("LLM_MODEL", "deepseek-flash"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        extra_body=extra_body,
    )

async def main(query: str):
    model = create_model()

    server_params = StdioServerParameters(
        command=sys.executable,  # same Python (and venv) as the agent
        args=[str(ROOT_DIR / "mcp_server.py")],
    )
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:

                await session.initialize()
                tools = await load_mcp_tools_from_session(session)

                prompt = (
                    "You are a helpful AI Assistant that can answer questions. Only reply in correct English or German sentences. "
                    "For questions about stored documents or facts, use the search_knowledge_base tool and cite the sources you used."
                )

                graph = create_agent(
                    model = model,
                    tools = tools,
                    system_prompt = prompt,
                )
                logger.debug("Agent created successfully")
                input = ({
                            "messages": [
                                {
                                    "role": "user",
                                    "content": query
                                }
                            ]
                        })
                await print_astream(graph.astream(input, stream_mode="values"))
    except Exception as e:
        logger.exception(f"An error occurred in the main workflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # query = "What is 2*(4+8)?"
    query = sys.argv[1] if len(sys.argv) > 1 else "How many r's does the word Strawberry have?"
    asyncio.run(main(query))
