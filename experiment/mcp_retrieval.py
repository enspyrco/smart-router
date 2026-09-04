"""Small synchronous adapter around the local stdio MCP knowledge server."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


async def _retrieve(question: str) -> str:
    server = Path(__file__).resolve().parent / "scripts" / "cs_knowledge_mcp.py"
    try:
        from mcp import Client, StdioServerParameters

        params = StdioServerParameters(command=sys.executable, args=[str(server)])
        async with Client(params) as client:
            result = await client.call_tool(
                "search_cs_knowledge", {"question": question, "limit": 3}
            )
    except ImportError:  # MCP Python SDK 1.x
        from mcp import StdioServerParameters
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=sys.executable, args=[str(server)])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_cs_knowledge", {"question": question, "limit": 3}
                )
    texts = [block.text for block in result.content if getattr(block, "text", None)]
    return "\n".join(texts)


def retrieve_cs_context(question: str) -> str:
    return asyncio.run(_retrieve(question))
