"""A tiny MCP server for learning RoboClaw Next tool adapters.

Run through the adapter demo instead of starting this file manually:

    PYTHONPATH=. uv run --no-project --with "mcp[cli]<2" \
        python roboclaw_next/examples/mcp_adapter_demo.py
"""

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("RoboClaw Next Demo", json_response=True)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""

    return a + b


@mcp.tool()
def echo(text: str) -> str:
    """Return the input text."""

    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")
