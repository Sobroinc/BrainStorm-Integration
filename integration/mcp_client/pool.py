# -*- coding: utf-8 -*-
"""
MCP Client Pool для параллельной обработки запросов.

Позволяет swarm-агентам работать параллельно без блокировок,
держа пул из нескольких subprocess для каждого MCP сервера.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Generic, List, TypeVar

from .base import MCPClient

T = TypeVar("T", bound=MCPClient)
R = TypeVar("R")

logger = logging.getLogger(__name__)


class MCPClientPool(Generic[T]):
    """
    Пул MCP клиентов для параллельной обработки.

    Использование:
        clients = [BraineMemoryClient(...) for _ in range(3)]
        pool = MCPClientPool(clients)
        await pool.start()

        # Параллельные запросы автоматически распределяются
        result = await pool.with_client(lambda c: c.recall("query"))

        await pool.stop()
    """

    def __init__(self, clients: List[T]) -> None:
        if not clients:
            raise ValueError("Pool requires at least one client")

        self._clients = clients
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._started = False

    @property
    def size(self) -> int:
        """Размер пула."""
        return len(self._clients)

    @property
    def available(self) -> int:
        """Количество доступных клиентов."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Запустить все клиенты в пуле."""
        if self._started:
            return

        logger.info(f"Starting pool with {len(self._clients)} clients...")

        # Запускаем всех клиентов параллельно
        results = await asyncio.gather(
            *(c.start() for c in self._clients),
            return_exceptions=True,
        )

        # Проверяем ошибки
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Client #{i} failed to start: {result}")
            else:
                # Добавляем успешно запущенных в очередь
                self._queue.put_nowait(self._clients[i])

        if self._queue.empty():
            raise RuntimeError("All clients failed to start")

        self._started = True
        logger.info(f"Pool started: {self._queue.qsize()}/{len(self._clients)} clients ready")

    async def stop(self) -> None:
        """Остановить все клиенты в пуле."""
        if not self._started:
            return

        logger.info("Stopping pool...")

        # Очищаем очередь
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Останавливаем всех клиентов
        await asyncio.gather(
            *(c.stop() for c in self._clients),
            return_exceptions=True,
        )

        self._started = False
        logger.info("Pool stopped")

    async def with_client(self, fn: Callable[[T], Awaitable[R]]) -> R:
        """
        Выполнить операцию с клиентом из пула.

        Автоматически берёт клиент из очереди, выполняет операцию,
        и возвращает клиент обратно.

        Args:
            fn: Асинхронная функция, принимающая клиент

        Returns:
            Результат выполнения функции
        """
        if not self._started:
            raise RuntimeError("Pool not started")

        # Берём клиент из очереди (блокирующе)
        client = await self._queue.get()
        try:
            return await fn(client)
        finally:
            # Всегда возвращаем клиент в очередь
            self._queue.put_nowait(client)

    async def broadcast(self, fn: Callable[[T], Awaitable[R]]) -> List[R]:
        """
        Выполнить операцию на ВСЕХ клиентах параллельно.

        Полезно для warmup или health check.

        Args:
            fn: Асинхронная функция, принимающая клиент

        Returns:
            Список результатов от всех клиентов
        """
        if not self._started:
            raise RuntimeError("Pool not started")

        # Собираем всех клиентов из очереди
        clients_to_use = []
        while not self._queue.empty():
            try:
                clients_to_use.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        try:
            # Выполняем на всех параллельно
            results = await asyncio.gather(
                *(fn(c) for c in clients_to_use),
                return_exceptions=True,
            )
            return results
        finally:
            # Возвращаем всех обратно
            for c in clients_to_use:
                self._queue.put_nowait(c)

    async def health_check(self) -> dict:
        """Проверка здоровья пула."""
        checks = await self.broadcast(lambda c: c.health_check())
        healthy = sum(1 for c in checks if isinstance(c, dict) and c.get("running"))
        return {
            "pool_size": len(self._clients),
            "available": self.available,
            "healthy": healthy,
            "clients": checks,
        }

    def __repr__(self) -> str:
        status = "started" if self._started else "stopped"
        return f"<MCPClientPool size={len(self._clients)} available={self.available} [{status}]>"
