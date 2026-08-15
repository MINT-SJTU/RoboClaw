"""Example MCP server used by the MCP tool-call demo."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("RoboClaw Next Example MCP Server", json_response=True)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers through the MCP server."""

    return a + b


@mcp.tool()
def echo(text: str) -> str:
    """Return the input text through the MCP server."""

    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")
