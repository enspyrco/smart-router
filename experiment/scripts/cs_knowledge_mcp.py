#!/usr/bin/env python3
"""Local stdio MCP server for the open-book computer-science experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from mcp.server import MCPServer
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from knowledge_retrieval import format_context, search_wikipedia

mcp = MCPServer("smart-router-cs-knowledge")


@mcp.tool()
def search_cs_knowledge(question: str, limit: int = 3) -> str:
    """Search Wikipedia for short reference passages relevant to a CS question."""
    return format_context(search_wikipedia(question, limit))


if __name__ == "__main__":
    mcp.run()
