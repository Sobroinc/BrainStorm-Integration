# -*- coding: utf-8 -*-
"""
BraineMemory MCP Client - Typed wrapper for 17 memory tools.

Provides type-safe methods matching exact tool signatures from BraineMemory server.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from integration.config import config
from integration.mcp_client.sse_session import MCPSession, connect_sse, connect_http

logger = logging.getLogger(__name__)

# Required tools for BraineMemory (minimum viable set)
BRAINE_REQUIRED_TOOLS: Set[str] = {
    "memory_ingest",
    "memory_recall",
    "memory_recall_graph",
    "memory_context_pack",
}

# Optional but recommended tools
BRAINE_RECOMMENDED_TOOLS: Set[str] = {
    "memory_recall_explain",
    "memory_recall_claims",
    "memory_link",
    "memory_compare",
    "memory_explain",
    "memory_forget",
    "memory_detect_conflicts",
    "memory_list_conflicts",
    "memory_resolve_conflict",
    "memory_build_communities",
    "user_memory_remember",
    "user_memory_recall",
    "user_memory_forget",
}


class BraineMemoryClient:
    """
    Typed MCP client for BraineMemory.

    All methods match exact tool signatures from BraineMemory/src/mcp/server.py.

    Usage:
        async with BraineMemoryClient() as braine:
            # Ingest content
            result = await braine.memory_ingest("Document text", type="document")

            # Recall with auto-routing
            result = await braine.memory_recall("search query", mode="auto")

            # User personalization
            await braine.user_memory_remember("user123", "Prefers bullet points")
    """

    def __init__(
        self,
        url: Optional[str] = None,
        transport: str = "sse",
        timeout: Optional[int] = None,
        max_inflight: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize BraineMemory client.

        Args:
            url: MCP endpoint URL (default from config)
            transport: Transport type ("sse" or "http")
            timeout: Connection timeout in seconds (default from config)
            max_inflight: Max parallel calls (default from config)
            max_retries: Max retries on transport errors (default from config)
        """
        if url is None:
            url = config.braine_sse_url if transport == "sse" else config.braine_http_url
        self.url = url
        self.transport = transport
        self.timeout = timeout or int(config.mcp_timeout)
        self.max_inflight = max_inflight or config.braine_max_inflight
        self.max_retries = max_retries or config.mcp_max_retries
        self._session: Optional[MCPSession] = None
        self._context_manager = None

    async def __aenter__(self) -> "BraineMemoryClient":
        """Connect to BraineMemory MCP server."""
        if self.transport == "sse":
            self._context_manager = connect_sse(
                self.url, self.timeout, self.max_inflight, self.max_retries
            )
        else:
            self._context_manager = connect_http(
                self.url, self.timeout, self.max_inflight, self.max_retries
            )
        self._session = await self._context_manager.__aenter__()

        # Check required capabilities
        await self._session.assert_capabilities(BRAINE_REQUIRED_TOOLS)

        # Log available tools
        tools_count = len(self._session.tools_cache)
        logger.info(f"Connected to BraineMemory at {self.url} ({tools_count} tools)")
        return self

    async def __aexit__(self, *args) -> None:
        """Disconnect from BraineMemory MCP server."""
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(*args)
            except RuntimeError as e:
                # Handle anyio cancel scope task boundary issue
                if "cancel scope" in str(e):
                    logger.debug(f"Ignoring cancel scope warning on disconnect: {e}")
                else:
                    raise
        self._session = None
        self._context_manager = None
        logger.debug("Disconnected from BraineMemory")

    @property
    def session(self) -> MCPSession:
        """Get active session (raises if not connected)."""
        if self._session is None:
            raise RuntimeError("Not connected. Use 'async with BraineMemoryClient()' context.")
        return self._session

    # ─────────────────────────────────────────────────────────────────────────
    # Ingestion
    # ─────────────────────────────────────────────────────────────────────────

    async def memory_ingest(
        self,
        content: str,
        type: str = "document",
        source_url: Optional[str] = None,
        lang: Optional[str] = None,
        version_of: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extract_entities: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Ingest content into BraineMemory.

        Args:
            content: Text content to ingest
            type: Asset type (document, image, cad, audio, video)
            source_url: Original source URL
            lang: Language (ru, en, fr, multi) - auto-detected if not provided
            version_of: Asset ID if this is a new version
            metadata: Additional metadata
            extract_entities: Extract entities, relations, claims using LLM

        Returns:
            IngestResult with asset_id, chunks_created, entities_extracted
        """
        args = {
            "content": content,
            "type": type,
            "extract_entities": extract_entities,
            **kwargs,
        }
        if source_url:
            args["source_url"] = source_url
        if lang:
            args["lang"] = lang
        if version_of:
            args["version_of"] = version_of
        if metadata:
            args["metadata"] = metadata
        return await self.session.call_tool("memory_ingest", args)

    # ─────────────────────────────────────────────────────────────────────────
    # Recall
    # ─────────────────────────────────────────────────────────────────────────

    async def memory_recall(
        self,
        query: str,
        limit: int = 10,
        mode: str = "auto",
        user_id: Optional[str] = None,
        asset_ids: Optional[List[str]] = None,
        lang: Optional[str] = None,
        include_evidence: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Recall relevant information using hybrid search with auto-routing.

        Args:
            query: Search query
            limit: Max results
            mode: Retrieval mode (auto, fts, vector, hybrid, local, global, research)
            user_id: User ID for personalization (Mem0-style)
            asset_ids: Limit to specific assets
            lang: Filter by language
            include_evidence: Include source quotes

        Returns:
            RecallResult with items, total_items, mode used
        """
        args = {
            "query": query,
            "limit": limit,
            "mode": mode,
            "include_evidence": include_evidence,
            **kwargs,
        }
        if user_id:
            args["user_id"] = user_id
        if asset_ids:
            args["asset_ids"] = asset_ids
        if lang:
            args["lang"] = lang
        return await self.session.call_tool("memory_recall", args)

    async def memory_recall_explain(
        self,
        query: str,
        mode: str = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Explain how the router would handle a query (debugging/transparency).

        Args:
            query: The search query to analyze
            mode: Retrieval mode override

        Returns:
            Routing explanation with intent, pipeline, weights
        """
        return await self.session.call_tool("memory_recall_explain", {
            "query": query,
            "mode": mode,
            **kwargs,
        })

    async def memory_context_pack(
        self,
        goal: str,
        token_budget: int = 4000,
        audience: str = "general",
        style: str = "structured",
        include_sources: bool = True,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Pack relevant context for a goal within a token budget.

        Args:
            goal: What the context is for
            token_budget: Max tokens for context
            audience: Target audience (lawyer, engineer, executive, general)
            style: Output format (bullets, structured, narrative, table)
            include_sources: Include source references
            user_id: User ID for personalization

        Returns:
            Packed context with content, sources, token_count
        """
        args = {
            "goal": goal,
            "token_budget": token_budget,
            "audience": audience,
            "style": style,
            "include_sources": include_sources,
            **kwargs,
        }
        if user_id:
            args["user_id"] = user_id
        return await self.session.call_tool("memory_context_pack", args)

    async def memory_recall_claims(
        self,
        query: str,
        limit: int = 10,
        include_conflicting: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Recall claims with conflict annotations.

        Args:
            query: Search query to find relevant claims
            limit: Maximum number of claims
            include_conflicting: Include claims that have conflicts

        Returns:
            Claims with conflict annotations
        """
        return await self.session.call_tool("memory_recall_claims", {
            "query": query,
            "limit": limit,
            "include_conflicting": include_conflicting,
            **kwargs,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Graph (GraphRAG)
    # ─────────────────────────────────────────────────────────────────────────

    async def memory_recall_graph(
        self,
        query: str,
        mode: str = "local",
        limit: int = 10,
        include_entities: bool = True,
        include_relations: bool = True,
        include_communities: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Recall through entity graph (GraphRAG).

        Args:
            query: Search query
            mode: local=entity expansion, global=community summaries, both=combined
            limit: Max results per category
            include_entities: Include matched entities
            include_relations: Include relations between entities
            include_communities: Include community summaries (global mode)

        Returns:
            Graph search results with entities, relations, communities
        """
        return await self.session.call_tool("memory_recall_graph", {
            "query": query,
            "mode": mode,
            "limit": limit,
            "include_entities": include_entities,
            "include_relations": include_relations,
            "include_communities": include_communities,
            **kwargs,
        })

    async def memory_build_communities(
        self,
        max_levels: int = 2,
        min_community_size: int = 2,
        resolution: float = 1.0,
        regenerate: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build community index for GraphRAG global search.

        Args:
            max_levels: Maximum hierarchy levels
            min_community_size: Minimum entities per community
            resolution: Louvain resolution (higher = more communities)
            regenerate: Delete existing communities first

        Returns:
            Build result with communities_created, entities_processed
        """
        return await self.session.call_tool("memory_build_communities", {
            "max_levels": max_levels,
            "min_community_size": min_community_size,
            "resolution": resolution,
            "regenerate": regenerate,
            **kwargs,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Links & Provenance
    # ─────────────────────────────────────────────────────────────────────────

    async def memory_link(
        self,
        source: str,
        target: str,
        relation: str = "relates_to",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a link/relation between two records.

        Args:
            source: Source record ID (entity:xxx, claim:xxx, asset:xxx)
            target: Target record ID
            relation: Relation type
            metadata: Additional edge data

        Returns:
            Created link details
        """
        args = {
            "source": source,
            "target": target,
            "relation": relation,
            **kwargs,
        }
        if metadata:
            args["metadata"] = metadata
        return await self.session.call_tool("memory_link", args)

    async def memory_compare(
        self,
        items: List[str],
        compare_type: str = "diff",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compare multiple items and find differences or conflicts.

        Args:
            items: Record IDs to compare (2+)
            compare_type: Comparison type (diff, conflict, timeline, full)

        Returns:
            Comparison results with differences, conflicts
        """
        return await self.session.call_tool("memory_compare", {
            "items": items,
            "compare_type": compare_type,
            **kwargs,
        })

    async def memory_explain(
        self,
        target: str,
        depth: str = "shallow",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Explain the provenance and evidence chain for a fact/claim.

        Args:
            target: Record ID to explain (claim:xxx, entity:xxx)
            depth: Explanation depth (shallow, deep)

        Returns:
            Provenance chain with sources, evidence
        """
        return await self.session.call_tool("memory_explain", {
            "target": target,
            "depth": depth,
            **kwargs,
        })

    async def memory_forget(
        self,
        target: str,
        reason: str,
        scope: str = "hide",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Soft-delete a record with audit trail.

        Args:
            target: Record ID to forget
            reason: Why it's being forgotten
            scope: Scope of forgetting (hide, anonymize)

        Returns:
            Forget result with policy_decision_id
        """
        return await self.session.call_tool("memory_forget", {
            "target": target,
            "reason": reason,
            "scope": scope,
            **kwargs,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Conflicts
    # ─────────────────────────────────────────────────────────────────────────

    async def memory_detect_conflicts(
        self,
        claim_id: Optional[str] = None,
        scan_all: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Detect contradictions between claims.

        Args:
            claim_id: Specific claim ID to check for conflicts
            scan_all: Scan all claims for conflicts

        Returns:
            Detected conflicts with severity, type
        """
        args = {"scan_all": scan_all, **kwargs}
        if claim_id:
            args["claim_id"] = claim_id
        return await self.session.call_tool("memory_detect_conflicts", args)

    async def memory_resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        winning_claim_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Resolve a detected conflict between claims.

        Args:
            conflict_id: ID of the conflict to resolve
            resolution: Description of how the conflict was resolved
            winning_claim_id: If one claim is correct, its ID

        Returns:
            Resolution result
        """
        args = {
            "conflict_id": conflict_id,
            "resolution": resolution,
            **kwargs,
        }
        if winning_claim_id:
            args["winning_claim_id"] = winning_claim_id
        return await self.session.call_tool("memory_resolve_conflict", args)

    async def memory_list_conflicts(
        self,
        status: str = "open",
        severity: Optional[str] = None,
        limit: int = 20,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        List conflicts in the database.

        Args:
            status: Filter by status (open, resolved)
            severity: Filter by severity (critical, high, medium, low)
            limit: Max results

        Returns:
            List of conflicts with details
        """
        args = {
            "status": status,
            "limit": limit,
            **kwargs,
        }
        if severity:
            args["severity"] = severity
        return await self.session.call_tool("memory_list_conflicts", args)

    # ─────────────────────────────────────────────────────────────────────────
    # User Memory (Mem0-style)
    # ─────────────────────────────────────────────────────────────────────────

    async def user_memory_remember(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Store user-specific memory for personalization.

        Args:
            user_id: External user identifier
            content: What to remember about this user
            category: Memory category (preference, fact, instruction, correction)
            importance: Importance score 0-1 (auto-estimated if not provided)
            metadata: Additional context

        Returns:
            Created memory details
        """
        args = {
            "user_id": user_id,
            "content": content,
            "category": category,
            **kwargs,
        }
        if importance is not None:
            args["importance"] = importance
        if metadata:
            args["metadata"] = metadata
        return await self.session.call_tool("user_memory_remember", args)

    async def user_memory_recall(
        self,
        user_id: str,
        query: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrieve user-specific memories for personalization.

        Args:
            user_id: External user identifier
            query: Optional semantic search query
            categories: Filter by categories (preference, fact, instruction, correction)
            limit: Max results
            min_importance: Minimum importance threshold

        Returns:
            User memories matching criteria
        """
        args = {
            "user_id": user_id,
            "limit": limit,
            "min_importance": min_importance,
            **kwargs,
        }
        if query:
            args["query"] = query
        if categories:
            args["categories"] = categories
        return await self.session.call_tool("user_memory_recall", args)

    async def user_memory_forget(
        self,
        user_id: str,
        memory_id: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Soft-delete user memories.

        Args:
            user_id: External user identifier
            memory_id: Specific memory to forget
            category: Forget all in category

        Returns:
            Forget result
        """
        args = {"user_id": user_id, **kwargs}
        if memory_id:
            args["memory_id"] = memory_id
        if category:
            args["category"] = category
        return await self.session.call_tool("user_memory_forget", args)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Check BraineMemory MCP server health."""
        return await self.session.health_check()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available BraineMemory tools."""
        return await self.session.list_tools()
