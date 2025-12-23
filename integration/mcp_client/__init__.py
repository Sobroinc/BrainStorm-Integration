# -*- coding: utf-8 -*-
"""
MCP Client - SSE/HTTP и stdio клиенты для BraineMemory и LegalOps.

Два способа подключения:

1. SSE/HTTP (к уже запущенным серверам - рекомендуемый):
    from integration.mcp_client import BraineMemoryClient, LegalOpsClient

    async with BraineMemoryClient() as braine:
        result = await braine.memory_recall("test query")

    async with LegalOpsClient() as legal:
        result = await legal.search_documents("contract")

2. Stdio subprocess (для Claude Desktop):
    from integration.mcp_client import MCPClient, MCPClientPool

    client = MCPClient(command="python", args=["-m", "src.mcp.server"])
    await client.start()
    result = await client.call_tool("memory_recall", {"query": "test"})
"""

# SSE/HTTP typed clients (primary)
from integration.mcp_client.braine_client import BraineMemoryClient
from integration.mcp_client.legal_client import LegalOpsClient
from integration.mcp_client.sse_session import MCPSession, connect_sse, connect_http

# Stdio subprocess (for Claude Desktop / legacy)
from integration.mcp_client.base import MCPClient, MCPProtocolError, Tool
from integration.mcp_client.pool import MCPClientPool

__all__ = [
    # SSE/HTTP clients
    "BraineMemoryClient",
    "LegalOpsClient",
    "MCPSession",
    "connect_sse",
    "connect_http",
    # Stdio subprocess
    "MCPClient",
    "MCPClientPool",
    "MCPProtocolError",
    "Tool",
]
