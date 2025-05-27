from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_deepseek import ChatDeepSeek
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import BaseTool
import sys, os, asyncio

from logging_config import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

async def print_astream(stream): 
    try: 
        async for s in stream: 
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
    except Exception as e:
        logger.error(f"Coulden't print Stream: {e}", exc_info=True)
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
        logger.error(f"Failed to load tools from session: {e}", exc_info=True)
        return []

async def main(query: str): 
    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    
    server_params = StdioServerParameters(
        command="python",
        # Make sure to update to the full absolute path to your example_mcp_server.py file
        args=["./example_mcp_server.py"],
    )
    try: 
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:

                await session.initialize()
                tools = await load_mcp_tools(session)

                prompt = "You are a helpful AI Assistant that can answer questions. Only reply in correct English or German sentences. "

                subgraph = create_react_agent(
                    tools = tools, 
                    model = model, 
                    output_mode = "full_history", 
                    prompt = prompt, 
                )

                graph = subgraph.compile()
                logger.debug("Workflow Compiled Successfully")
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
        logger.error(f"An error occurred in the main workflow: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # query = "What is 2*(4+8)?"
    query = "How many r's does the word Strawberry have?"
    asyncio.run(main(query))