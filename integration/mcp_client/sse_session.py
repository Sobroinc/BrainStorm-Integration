# -*- coding: utf-8 -*-
"""
MCP Session - Unified wrapper for SSE/HTTP MCP connections.

Provides a consistent interface for calling MCP tools regardless of transport.

Production features:
- trace_id injection for end-to-end correlation
- Semaphore for parallelism control
- Retry with exponential backoff (transport errors only)
- Tools caching with capability assertions
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_INFLIGHT = 6
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAYS = [0.3, 0.8, 1.6]  # Exponential backoff

# Errors that should be retried
RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


@dataclass
class MCPSession:
    """
    Unified MCP session wrapper.

    Provides call_tool() and list_tools() with:
    - Automatic JSON parsing
    - trace_id injection for correlation
    - Semaphore for parallelism control
    - Retry with backoff (transport errors only)
    - Capability assertions
    """

    session: ClientSession
    url: str
    transport: str
    tools_cache: List[Dict[str, Any]] = field(default_factory=list)
    _tools_set: Set[str] = field(default_factory=set)
    _semaphore: asyncio.Semaphore | None = field(default=None)
    max_inflight: int = field(default=DEFAULT_MAX_INFLIGHT)
    max_retries: int = field(default=DEFAULT_MAX_RETRIES)

    def __post_init__(self):
        """Initialize semaphore after dataclass creation."""
        if self._semaphore is None:
            object.__setattr__(self, '_semaphore', asyncio.Semaphore(self.max_inflight))

    async def list_tools(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.

        Args:
            refresh: Force refresh the tools cache

        Returns:
            List of tool definitions
        """
        if self.tools_cache and not refresh:
            return self.tools_cache

        result = await self.session.list_tools()
        self.tools_cache = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
        ]
        # Update tools set for capability checking
        object.__setattr__(self, '_tools_set', {t["name"] for t in self.tools_cache})
        return self.tools_cache

    async def assert_capabilities(self, required: Set[str]) -> None:
        """
        Assert that required tools are available.

        Args:
            required: Set of tool names that must be present

        Raises:
            RuntimeError: If any required tools are missing
        """
        if not self._tools_set:
            await self.list_tools()

        missing = required - self._tools_set
        if missing:
            raise RuntimeError(
                f"Missing required tools on {self.url}: {sorted(missing)}"
            )

    def has_tool(self, name: str) -> bool:
        """Check if a tool is available (requires list_tools() to be called first)."""
        return name in self._tools_set

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        trace_id: Optional[str] = None,
        retry: bool = True,
    ) -> Dict[str, Any]:
        """
        Call an MCP tool and return parsed result.

        Args:
            name: Tool name
            arguments: Tool arguments
            trace_id: Optional trace ID for correlation (injected into args)
            retry: Whether to retry on transport errors

        Returns:
            Parsed JSON result from tool

        Raises:
            ValueError: If tool returns non-JSON content
            Exception: If tool call fails after retries
        """
        trace_id = trace_id or str(uuid.uuid4())[:8]

        # Inject trace_id into arguments for server-side correlation
        args_with_trace = {**arguments, "trace_id": trace_id}

        logger.debug(f"[{trace_id}] Calling {name} on {self.url}")

        last_error: Exception | None = None
        attempts = self.max_retries if retry else 1

        for attempt in range(attempts):
            try:
                # Use semaphore to limit parallelism
                async with self._semaphore:
                    result = await self.session.call_tool(name, args_with_trace)

                # Parse result content
                if not result.content:
                    return {"error": "Empty response", "_trace_id": trace_id}

                text = result.content[0].text
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        data["_trace_id"] = trace_id
                    return data
                except json.JSONDecodeError:
                    return {"raw": text, "_trace_id": trace_id}

            except RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < attempts - 1:
                    delay = DEFAULT_RETRY_DELAYS[min(attempt, len(DEFAULT_RETRY_DELAYS) - 1)]
                    logger.warning(
                        f"[{trace_id}] Retry {attempt + 1}/{attempts} for {name}: {e}"
                    )
                    await asyncio.sleep(delay)
            except Exception as e:
                # Non-retryable error (validation, tool not found, etc.)
                logger.error(f"[{trace_id}] Error calling {name}: {e}")
                raise

        # All retries exhausted
        logger.error(f"[{trace_id}] All {attempts} attempts failed for {name}")
        raise last_error or RuntimeError(f"call_tool failed: {name}")

    async def health_check(self) -> Dict[str, Any]:
        """Quick health check - list tools and return count."""
        try:
            tools = await self.list_tools(refresh=True)
            return {
                "status": "healthy",
                "url": self.url,
                "transport": self.transport,
                "tools_count": len(tools),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "url": self.url,
                "transport": self.transport,
                "error": str(e),
            }


@asynccontextmanager
async def connect_sse(
    url: str,
    timeout: int = 30,
    max_inflight: int = DEFAULT_MAX_INFLIGHT,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    """
    Connect to MCP server via SSE transport.

    Args:
        url: SSE endpoint URL (e.g., http://localhost:8001/sse)
        timeout: Connection timeout in seconds
        max_inflight: Maximum parallel tool calls (semaphore)
        max_retries: Maximum retries for transport errors

    Yields:
        MCPSession instance

    Example:
        async with connect_sse("http://localhost:8001/sse") as session:
            result = await session.call_tool("memory_recall", {"query": "test"})
    """
    async with sse_client(url, timeout=timeout) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            mcp_session = MCPSession(
                session=session,
                url=url,
                transport="sse",
                max_inflight=max_inflight,
                max_retries=max_retries,
            )
            # Pre-cache tools list for capability checking
            await mcp_session.list_tools()
            yield mcp_session


@asynccontextmanager
async def connect_http(
    url: str,
    timeout: int = 30,
    max_inflight: int = DEFAULT_MAX_INFLIGHT,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    """
    Connect to MCP server via HTTP streamable transport.

    Args:
        url: HTTP endpoint URL (e.g., http://localhost:8001/mcp)
        timeout: Connection timeout in seconds
        max_inflight: Maximum parallel tool calls (semaphore)
        max_retries: Maximum retries for transport errors

    Yields:
        MCPSession instance

    Example:
        async with connect_http("http://localhost:8001/mcp") as session:
            result = await session.call_tool("memory_recall", {"query": "test"})
    """
    async with streamablehttp_client(url, timeout=timeout) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            mcp_session = MCPSession(
                session=session,
                url=url,
                transport="http",
                max_inflight=max_inflight,
                max_retries=max_retries,
            )
            # Pre-cache tools list for capability checking
            await mcp_session.list_tools()
            yield mcp_session


async def connect(
    url: str,
    transport: str = "sse",
    timeout: int = 30,
    max_inflight: int = DEFAULT_MAX_INFLIGHT,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    """
    Connect to MCP server with specified transport.

    Args:
        url: MCP endpoint URL
        transport: Transport type ("sse" or "http")
        timeout: Connection timeout
        max_inflight: Maximum parallel tool calls
        max_retries: Maximum retries for transport errors

    Returns:
        Async context manager yielding MCPSession
    """
    if transport == "sse":
        return connect_sse(url, timeout, max_inflight, max_retries)
    elif transport == "http":
        return connect_http(url, timeout, max_inflight, max_retries)
    else:
        raise ValueError(f"Unknown transport: {transport}")
