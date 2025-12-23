# -*- coding: utf-8 -*-
"""
Agent MCP Helper - Connect Agent to BraineMemory and LegalOps.

Использует существующий MCP клиент Agent для подключения к серверам.

Пример использования:
    from integration.agent_mcp_helper import (
        get_all_mcp_tools,
        call_braine_tool,
        call_legal_tool,
        BrainStormMCPClient,
    )

    # Получить все tools от обоих серверов
    tools = await get_all_mcp_tools()

    # Вызвать tool BraineMemory
    result = await call_braine_tool("memory.recall", {"query": "test"})

    # Вызвать tool LegalOps
    result = await call_legal_tool("search_documents", {"query": "контракт"})

    # Или через unified client
    async with BrainStormMCPClient() as client:
        tools = await client.list_all_tools()
        result = await client.call_tool("braine", "memory.recall", {"query": "test"})
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default server URLs
BRAINE_MCP_URL = os.environ.get("BRAINE_MCP_URL", "http://localhost:8001/mcp/")
LEGAL_MCP_URL = os.environ.get("LEGAL_MCP_URL", "http://localhost:8002/mcp/")


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    url: str
    transport: str = "streamable_http"
    timeout: int = 30


# Default server configurations
BRAINE_CONFIG = MCPServerConfig(
    name="braine",
    url=BRAINE_MCP_URL,
)

LEGAL_CONFIG = MCPServerConfig(
    name="legal",
    url=LEGAL_MCP_URL,
)


async def get_mcp_tools(url: str, format: str = "openai") -> List[Dict[str, Any]]:
    """
    Get tools from an MCP server.

    Args:
        url: MCP server URL (e.g., http://localhost:8001/mcp/)
        format: Tool format ('openai' or 'mcp')

    Returns:
        List of tools in specified format
    """
    try:
        # Import Agent's MCP client
        from swarms.swarms.tools.mcp_client_tools import aget_mcp_tools

        tools = await aget_mcp_tools(
            server_path=url,
            format=format,
            transport="streamable_http",
        )
        return tools
    except ImportError:
        logger.error("Agent MCP client not available. Install swarms package.")
        raise
    except Exception as e:
        logger.error(f"Error getting tools from {url}: {e}")
        raise


async def get_braine_tools(format: str = "openai") -> List[Dict[str, Any]]:
    """Get all tools from BraineMemory MCP server."""
    return await get_mcp_tools(BRAINE_MCP_URL, format)


async def get_legal_tools(format: str = "openai") -> List[Dict[str, Any]]:
    """Get all tools from LegalOps MCP server."""
    return await get_mcp_tools(LEGAL_MCP_URL, format)


async def get_all_mcp_tools(format: str = "openai") -> Dict[str, List[Dict[str, Any]]]:
    """
    Get tools from both MCP servers.

    Returns:
        Dict with 'braine' and 'legal' keys containing respective tools
    """
    braine_task = get_braine_tools(format)
    legal_task = get_legal_tools(format)

    braine_tools, legal_tools = await asyncio.gather(
        braine_task,
        legal_task,
        return_exceptions=True,
    )

    result = {}

    if isinstance(braine_tools, Exception):
        logger.warning(f"Failed to get BraineMemory tools: {braine_tools}")
        result["braine"] = []
    else:
        result["braine"] = braine_tools

    if isinstance(legal_tools, Exception):
        logger.warning(f"Failed to get LegalOps tools: {legal_tools}")
        result["legal"] = []
    else:
        result["legal"] = legal_tools

    return result


async def call_mcp_tool(
    url: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Call a tool on an MCP server.

    Args:
        url: MCP server URL
        tool_name: Name of the tool to call
        arguments: Tool arguments

    Returns:
        Tool result
    """
    try:
        from swarms.swarms.tools.mcp_client_tools import execute_tool_call_simple
        from swarms.schemas.mcp_schemas import MCPConnection

        # Create fake tool call in OpenAI format
        tool_call = {
            "function": {
                "name": tool_name,
                "arguments": arguments,
            }
        }

        connection = MCPConnection(url=url, transport="streamable_http")

        result = await execute_tool_call_simple(
            response=tool_call,
            server_path=url,
            connection=connection,
            output_type="dict",
            transport="streamable_http",
        )
        return result
    except ImportError:
        logger.error("Agent MCP client not available")
        raise
    except Exception as e:
        logger.error(f"Error calling tool {tool_name} on {url}: {e}")
        raise


async def call_braine_tool(
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Call a tool on BraineMemory MCP server."""
    return await call_mcp_tool(BRAINE_MCP_URL, tool_name, arguments)


async def call_legal_tool(
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Call a tool on LegalOps MCP server."""
    return await call_mcp_tool(LEGAL_MCP_URL, tool_name, arguments)


class BrainStormMCPClient:
    """
    Unified MCP client for BrainStorm modules.

    Provides access to both BraineMemory and LegalOps through a single interface.

    Usage:
        async with BrainStormMCPClient() as client:
            # List all tools
            tools = await client.list_all_tools()

            # Call BraineMemory tool
            result = await client.call_tool("braine", "memory.recall", {"query": "test"})

            # Call LegalOps tool
            result = await client.call_tool("legal", "search_documents", {"query": "contract"})
    """

    def __init__(
        self,
        braine_url: str = BRAINE_MCP_URL,
        legal_url: str = LEGAL_MCP_URL,
    ):
        self.braine_url = braine_url
        self.legal_url = legal_url
        self._tools_cache: Dict[str, List[Dict]] = {}

    async def __aenter__(self) -> "BrainStormMCPClient":
        # Pre-fetch tools on entry
        await self.refresh_tools()
        return self

    async def __aexit__(self, *args) -> None:
        pass

    async def refresh_tools(self) -> None:
        """Refresh tools cache from both servers."""
        self._tools_cache = await get_all_mcp_tools()

    async def list_all_tools(self) -> Dict[str, List[Dict]]:
        """List all tools from both servers."""
        if not self._tools_cache:
            await self.refresh_tools()
        return self._tools_cache

    async def list_braine_tools(self) -> List[Dict]:
        """List BraineMemory tools."""
        if "braine" not in self._tools_cache:
            self._tools_cache["braine"] = await get_braine_tools()
        return self._tools_cache["braine"]

    async def list_legal_tools(self) -> List[Dict]:
        """List LegalOps tools."""
        if "legal" not in self._tools_cache:
            self._tools_cache["legal"] = await get_legal_tools()
        return self._tools_cache["legal"]

    async def call_tool(
        self,
        server: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call a tool on specified server.

        Args:
            server: Server name ('braine' or 'legal')
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if server == "braine":
            return await call_braine_tool(tool_name, arguments)
        elif server == "legal":
            return await call_legal_tool(tool_name, arguments)
        else:
            raise ValueError(f"Unknown server: {server}")

    # ─────────────────────────────────────────────────────────────────────────
    # BraineMemory convenience methods
    # ─────────────────────────────────────────────────────────────────────────

    async def memory_ingest(
        self,
        content: str,
        source_url: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Ingest content into BraineMemory."""
        args = {"content": content, **kwargs}
        if source_url:
            args["source_url"] = source_url
        return await self.call_tool("braine", "memory.ingest", args)

    async def memory_recall(
        self,
        query: str,
        mode: str = "auto",
        limit: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Recall from BraineMemory."""
        return await self.call_tool("braine", "memory.recall", {
            "query": query,
            "mode": mode,
            "limit": limit,
            **kwargs,
        })

    async def memory_context_pack(
        self,
        goal: str,
        token_budget: int = 4000,
        **kwargs,
    ) -> Dict[str, Any]:
        """Pack context from BraineMemory."""
        return await self.call_tool("braine", "memory.context_pack", {
            "goal": goal,
            "token_budget": token_budget,
            **kwargs,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # LegalOps convenience methods
    # ─────────────────────────────────────────────────────────────────────────

    async def ingest_pdf(
        self,
        pdf_path: str,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Ingest PDF into LegalOps."""
        args = {"pdf_path": pdf_path, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.call_tool("legal", "ingest_pdf", args)

    async def search_documents(
        self,
        query: str,
        project_id: Optional[str] = None,
        limit: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Search documents in LegalOps."""
        args = {"query": query, "limit": limit, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.call_tool("legal", "search_documents", args)

    async def detect_contradictions(
        self,
        doc_ids: List[str],
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Detect contradictions between documents."""
        args = {"doc_ids": doc_ids, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.call_tool("legal", "detect_contradictions", args)


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

async def demo():
    """Demo of BrainStorm MCP integration."""
    print("BrainStorm MCP Integration Demo")
    print("=" * 50)

    async with BrainStormMCPClient() as client:
        # List tools
        tools = await client.list_all_tools()
        print(f"\nBraineMemory tools: {len(tools.get('braine', []))}")
        print(f"LegalOps tools: {len(tools.get('legal', []))}")

        # Example: recall from memory
        try:
            result = await client.memory_recall(
                query="тестовый запрос",
                limit=5,
            )
            print(f"\nMemory recall result: {result}")
        except Exception as e:
            print(f"\nMemory recall error: {e}")


if __name__ == "__main__":
    asyncio.run(demo())
