from mcp.server.fastmcp import FastMCP
import sys, os

mcp = FastMCP("Add")

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.chdir(root_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Logging Setup
from logging_config import setup_logging # Wenn man normale Print befehle nutzt, kann es mit dem STIO kommunikationsstream interferieren, daher ist logging hier besser
import logging
setup_logging()
logger = logging.getLogger(__name__)

@mcp.tool()
def add(a: int, b: int) -> int: 
    """Adds 2 numbers together. Requires 2 inputs: an integer a and an integer b. """
    print("RUNNING ADD TOOL", file=sys.stderr, flush=True)
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int: 
    """Multiplies 2 numbers. Requires 2 inputs: an integer a and an integer b. """
    print("RUNNING MULTIPLY TOOL", file=sys.stderr, flush=True)
    return a * b

@mcp.tool()
def count_r(word: str) -> int:
    """Can count how many r's a given word has. Requires 1 input: a string word. """
    return word.lower().count('r')


if __name__ == "__main__":
    logger.info("Starting Test MCP Server via mcp.run(transport='stdio')...", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")
    logger.info("MCP Server finished.", file=sys.stderr, flush=True) 