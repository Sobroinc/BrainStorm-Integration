# -*- coding: utf-8 -*-
"""
BraineMemory Provider для Agent.CentralHub.

Предоставляет интерфейс долгосрочной памяти для агентов,
абстрагируя детали MCP протокола.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from integration.mcp_client.braine_client import BraineMemoryClient
from integration.mcp_client.pool import MCPClientPool

logger = logging.getLogger(__name__)


class BraineMemoryProvider:
    """
    Провайдер памяти для регистрации в Agent.CentralHub.

    Реализует интерфейс MemoryInterface для CentralHub:
    - store() - сохранить в память
    - retrieve() - извлечь из памяти
    - search_graph() - поиск по графу знаний
    - forget() - удалить из памяти

    Может работать как с одним клиентом, так и с пулом.

    Использование:
        # С одним клиентом
        provider = BraineMemoryProvider(client)

        # С пулом (для параллельных операций)
        provider = BraineMemoryProvider.from_pool(pool)

        # Регистрация в CentralHub
        hub.register_memory("braine", provider)

        # Использование через hub
        await hub.memory.store("важная информация")
        results = await hub.memory.retrieve("поиск")
    """

    name = "braine"
    provider_type = "memory"

    def __init__(
        self,
        client: BraineMemoryClient | None = None,
        pool: MCPClientPool[BraineMemoryClient] | None = None,
    ):
        if not client and not pool:
            raise ValueError("Either client or pool must be provided")

        self._client = client
        self._pool = pool
        self._initialized = False

    @classmethod
    def from_pool(cls, pool: MCPClientPool[BraineMemoryClient]) -> "BraineMemoryProvider":
        """Создать провайдер из пула клиентов."""
        return cls(pool=pool)

    @classmethod
    def from_client(cls, client: BraineMemoryClient) -> "BraineMemoryProvider":
        """Создать провайдер из одного клиента."""
        return cls(client=client)

    async def _call(self, method: str, *args, **kwargs) -> Any:
        """Вызвать метод на клиенте (через пул если есть)."""
        if self._pool:
            return await self._pool.with_client(
                lambda c: getattr(c, method)(*args, **kwargs)
            )
        elif self._client:
            return await getattr(self._client, method)(*args, **kwargs)
        else:
            raise RuntimeError("No client available")

    # ─────────────────────────────────────────────────────────────────────────
    # Core Memory Interface
    # ─────────────────────────────────────────────────────────────────────────

    async def store(
        self,
        content: str,
        metadata: Dict[str, Any] | None = None,
        extract_entities: bool = True,
    ) -> str:
        """
        Сохранить контент в долгосрочную память.

        Args:
            content: Текст для сохранения
            metadata: Дополнительные метаданные (source_url, type, lang, etc.)
            extract_entities: Извлекать сущности и claims

        Returns:
            ID созданного asset
        """
        kwargs = {"content": content, "extract_entities": extract_entities}
        if metadata:
            kwargs.update(metadata)

        result = await self._call("memory_ingest", **kwargs)
        asset_id = result.get("asset_id", "")
        logger.debug(f"Stored content, asset_id={asset_id}")
        return asset_id

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        mode: str = "auto",
        user_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Извлечь релевантный контент из памяти.

        Args:
            query: Поисковый запрос
            limit: Максимум результатов
            mode: Режим поиска (auto, fts, vector, hybrid, local, global)
            user_id: ID пользователя для персонализации

        Returns:
            Список релевантных элементов
        """
        result = await self._call(
            "memory_recall",
            query=query,
            limit=limit,
            mode=mode,
            user_id=user_id,
        )

        # SSE клиент возвращает Dict, извлекаем items
        items = result.get("items", [])
        return items

    async def search_graph(
        self,
        query: str,
        mode: str = "local",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Поиск по графу знаний (GraphRAG).

        Args:
            query: Поисковый запрос
            mode: local (entity expansion), global (communities), both
            limit: Максимум результатов

        Returns:
            Граф с entities, relations, communities
        """
        return await self._call(
            "memory_recall_graph",
            query=query,
            mode=mode,
            limit=limit,
        )

    async def forget(
        self,
        target: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Soft-delete записи из памяти.

        Args:
            target: ID записи для удаления
            reason: Причина удаления

        Returns:
            Результат операции
        """
        return await self._call("memory_forget", target=target, reason=reason)

    # ─────────────────────────────────────────────────────────────────────────
    # Extended Memory Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def context_pack(
        self,
        goal: str,
        token_budget: int = 4000,
        user_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Собрать контекст для конкретной цели в пределах токен-бюджета.

        Полезно для подготовки контекста перед вызовом LLM.
        """
        return await self._call(
            "memory_context_pack",
            goal=goal,
            token_budget=token_budget,
            user_id=user_id,
        )

    async def compare(
        self,
        items: List[str],
        compare_type: str = "diff",
    ) -> Dict[str, Any]:
        """Сравнить несколько записей."""
        return await self._call("memory_compare", items=items, compare_type=compare_type)

    async def explain(
        self,
        target: str,
        depth: str = "shallow",
    ) -> Dict[str, Any]:
        """Объяснить провенанс факта/claim."""
        return await self._call("memory_explain", target=target, depth=depth)

    # ─────────────────────────────────────────────────────────────────────────
    # User Memory (Mem0-style)
    # ─────────────────────────────────────────────────────────────────────────

    async def user_remember(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
        importance: float | None = None,
    ) -> Dict[str, Any]:
        """Сохранить персонализированную память пользователя."""
        return await self._call(
            "user_memory_remember",
            user_id=user_id,
            content=content,
            category=category,
            importance=importance,
        )

    async def user_recall(
        self,
        user_id: str,
        query: str | None = None,
        categories: List[str] | None = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Извлечь персонализированную память пользователя."""
        return await self._call(
            "user_memory_recall",
            user_id=user_id,
            query=query,
            categories=categories,
            limit=limit,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Conflict Detection
    # ─────────────────────────────────────────────────────────────────────────

    async def detect_conflicts(
        self,
        claim_id: str | None = None,
        scan_all: bool = False,
    ) -> Dict[str, Any]:
        """Обнаружить противоречия между claims."""
        return await self._call(
            "memory_detect_conflicts",
            claim_id=claim_id,
            scan_all=scan_all,
        )

    async def list_conflicts(
        self,
        status: str = "open",
        severity: str | None = None,
    ) -> Dict[str, Any]:
        """Список конфликтов."""
        return await self._call(
            "memory_list_conflicts",
            status=status,
            severity=severity,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GraphRAG
    # ─────────────────────────────────────────────────────────────────────────

    async def build_communities(
        self,
        max_levels: int = 2,
        regenerate: bool = False,
    ) -> Dict[str, Any]:
        """Построить community index для GraphRAG."""
        return await self._call(
            "memory_build_communities",
            max_levels=max_levels,
            regenerate=regenerate,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Health & Status
    # ─────────────────────────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья провайдера."""
        if self._pool:
            return await self._pool.health_check()
        elif self._client:
            return await self._client.health_check()
        return {"status": "unhealthy", "error": "No client"}

    @property
    def is_ready(self) -> bool:
        """Готов ли провайдер к работе."""
        if self._pool:
            return self._pool.available > 0
        elif self._client:
            return self._client.is_running
        return False

    def __repr__(self) -> str:
        if self._pool:
            return f"<BraineMemoryProvider pool={self._pool}>"
        return f"<BraineMemoryProvider client={self._client}>"
