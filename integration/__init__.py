# -*- coding: utf-8 -*-
"""
BrainStorm Integration Layer v1.0.0

Объединяет три модуля:
- Agent: Оркестрация AI агентов (40+ swarm-архитектур)
- BraineMemory: Долгосрочная память (MCP, GraphRAG)
- LegalOps: Юридический анализ документов

Архитектура:
┌─────────────────────────────────────────────────────────────┐
│                      CentralHub (Agent)                     │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ Services │  │  Providers   │  │       Memories        │ │
│  │          │  │              │  │                       │ │
│  │ - swarms │  │ - llm        │  │ - braine (knowledge)  │ │
│  │ - chat   │  │ - embeddings │  │ - legal (documents)   │ │
│  │ - legal  │  │              │  │                       │ │
│  └──────────┘  └──────────────┘  └───────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol (HTTP/SSE)
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ BraineMemory│  │  LegalOps   │  │   SurrealDB │
│ MCP :8001   │  │ MCP :8002   │  │  (per-mod)  │
│ 17 tools    │  │ 68 tools    │  │             │
└─────────────┘  └─────────────┘  └─────────────┘

Два способа интеграции:

1. Через stdio subprocess (для Claude Desktop):
    from integration import BrainStormOrchestrator

    async with BrainStormOrchestrator() as orch:
        await orch.braine.store("важная информация")
        doc = await orch.legal.ingest_pdf("contract.pdf")

2. Через HTTP/SSE (для Agent MCP client):
    # Сначала запустите серверы:
    # python -m integration.start_all

    from integration.agent_mcp_helper import BrainStormMCPClient

    async with BrainStormMCPClient() as client:
        tools = await client.list_all_tools()
        result = await client.memory_recall("query")
        docs = await client.search_documents("contract")
"""

from .config import config, IntegrationConfig
from .mcp_client import (
    MCPSession,
    connect_sse,
    connect_http,
    BraineMemoryClient,
    LegalOpsClient,
)
from .providers import BraineMemoryProvider, LegalOpsProvider
from .orchestrator import BrainStormOrchestrator
from .agent_mcp_helper import (
    BrainStormMCPClient,
    get_braine_tools,
    get_legal_tools,
    get_all_mcp_tools,
    call_braine_tool,
    call_legal_tool,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "config",
    "IntegrationConfig",
    # MCP Clients (SSE/HTTP)
    "MCPSession",
    "connect_sse",
    "connect_http",
    "BraineMemoryClient",
    "LegalOpsClient",
    # Providers
    "BraineMemoryProvider",
    "LegalOpsProvider",
    # Orchestrator (stdio)
    "BrainStormOrchestrator",
    # Agent MCP Helper (HTTP/SSE)
    "BrainStormMCPClient",
    "get_braine_tools",
    "get_legal_tools",
    "get_all_mcp_tools",
    "call_braine_tool",
    "call_legal_tool",
]
