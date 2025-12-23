# -*- coding: utf-8 -*-
"""
Базовый MCP клиент для stdio subprocess.

Особенности:
- JSON-RPC 2.0 по stdin/stdout
- Асинхронный reader loop
- Таймауты и retry
- Graceful shutdown

ВАЖНО: MCP сервер должен писать JSON только в stdout, логи - в stderr!
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Описание MCP инструмента."""
    name: str
    description: str | None = None
    input_schema: dict | None = None


class MCPProtocolError(RuntimeError):
    """Ошибка протокола MCP."""
    pass


class MCPClient:
    """
    Базовый MCP клиент для MCP серверов через stdio subprocess.

    Предполагает JSON-RPC 2.0 протокол (каждое сообщение = одна JSON строка).

    Использование:
        client = MCPClient(command="python", args=["-m", "src.mcp.server"], cwd="../BraineMemory")
        await client.start()
        tools = await client.list_tools()
        result = await client.call_tool("memory.recall", {"query": "test"})
        await client.stop()
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        cwd: str | None = None,
        timeout_s: float = 30.0,
        env: dict[str, str] | None = None,
        name: str = "mcp",
        max_retries: int = 2,
    ) -> None:
        self.command = command
        self.args = args
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.env = env or {}
        self.name = name
        self.max_retries = max_retries

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._id_seq = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._write_lock = asyncio.Lock()
        self._stopping = False
        self._tools_cache: list[Tool] | None = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Запуск subprocess с MCP сервером."""
        if self._proc:
            return

        merged_env = os.environ.copy()
        merged_env.update(self.env)

        logger.info(f"[{self.name}] Starting: {self.command} {' '.join(self.args)}")

        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            cwd=self.cwd,
            env=merged_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Запускаем loop чтения stdout (JSON-RPC ответы)
        self._reader_task = asyncio.create_task(
            self._read_stdout_loop(),
            name=f"{self.name}_stdout_reader"
        )

        # Запускаем loop чтения stderr (логи)
        self._stderr_task = asyncio.create_task(
            self._read_stderr_loop(),
            name=f"{self.name}_stderr_reader"
        )

        # Даём серверу время на инициализацию
        await asyncio.sleep(0.5)

        # Опционально: проверяем что сервер отвечает
        # await self._handshake()

        logger.info(f"[{self.name}] Started, PID={self._proc.pid}")

    async def stop(self, kill_after_s: float = 3.0) -> None:
        """Graceful остановка subprocess."""
        if not self._proc:
            return

        self._stopping = True
        logger.info(f"[{self.name}] Stopping...")

        # Отменяем все pending futures
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(asyncio.CancelledError(f"{self.name} stopping"))
        self._pending.clear()

        # Пытаемся graceful terminate
        try:
            self._proc.terminate()
        except ProcessLookupError:
            pass

        try:
            await asyncio.wait_for(self._proc.wait(), timeout=kill_after_s)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.name}] Force killing after {kill_after_s}s")
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

        # Отменяем reader tasks
        for task in [self._reader_task, self._stderr_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._proc = None
        self._reader_task = None
        self._stderr_task = None
        self._stopping = False
        self._tools_cache = None

        logger.info(f"[{self.name}] Stopped")

    async def list_tools(self, use_cache: bool = True) -> List[Tool]:
        """
        Получить список доступных инструментов.

        MCP Protocol: метод tools/list
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        result = await self.call_raw("tools/list", {})
        tools_raw = result.get("tools", [])

        tools: list[Tool] = []
        for t in tools_raw:
            tools.append(
                Tool(
                    name=t.get("name", ""),
                    description=t.get("description"),
                    input_schema=t.get("inputSchema") or t.get("input_schema"),
                )
            )

        self._tools_cache = tools
        logger.debug(f"[{self.name}] Found {len(tools)} tools")
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        retry: bool = True,
    ) -> Dict[str, Any]:
        """
        Вызов инструмента через MCP (tools/call).

        Args:
            name: Имя инструмента (например "memory.recall")
            arguments: Аргументы вызова
            retry: Повторять ли при транспортных ошибках

        Returns:
            Результат вызова инструмента
        """
        payload = {"name": name, "arguments": arguments}

        last_error = None
        attempts = self.max_retries if retry else 1

        for attempt in range(attempts):
            try:
                result = await self.call_raw("tools/call", payload)

                # MCP tools/call возвращает content[]
                content = result.get("content", [])
                if content and isinstance(content, list):
                    # Обычно первый элемент - TextContent
                    first = content[0]
                    if isinstance(first, dict) and first.get("type") == "text":
                        text = first.get("text", "")
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return {"text": text}
                    return first
                return result

            except (MCPProtocolError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < attempts - 1:
                    wait = 0.5 * (attempt + 1)
                    logger.warning(f"[{self.name}] Retry {attempt + 1}/{attempts} for {name}: {e}")
                    await asyncio.sleep(wait)

        raise last_error or MCPProtocolError(f"call_tool failed: {name}")

    async def call_raw(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Низкоуровневый JSON-RPC вызов.

        Args:
            method: JSON-RPC метод (например "tools/list", "tools/call")
            params: Параметры метода
        """
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            raise RuntimeError(f"[{self.name}] Not started")

        self._id_seq += 1
        req_id = self._id_seq

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")

        async with self._write_lock:
            try:
                self._proc.stdin.write(data)
                await self._proc.stdin.drain()
            except Exception as e:
                self._pending.pop(req_id, None)
                raise MCPProtocolError(f"[{self.name}] Write failed: {e}") from e

        try:
            resp = await asyncio.wait_for(fut, timeout=self.timeout_s)
        except asyncio.TimeoutError as e:
            self._pending.pop(req_id, None)
            raise asyncio.TimeoutError(f"[{self.name}] Timeout calling {method}") from e

        if not isinstance(resp, dict):
            raise MCPProtocolError(f"[{self.name}] Invalid response type: {type(resp)}")

        if "error" in resp and resp["error"]:
            err = resp["error"]
            code = err.get("code", -1) if isinstance(err, dict) else -1
            message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise MCPProtocolError(f"[{self.name}] Error {code}: {message}")

        return resp.get("result", resp)

    async def _read_stdout_loop(self) -> None:
        """Читаем stdout (JSON-RPC ответы)."""
        assert self._proc and self._proc.stdout
        reader = self._proc.stdout

        while True:
            try:
                line = await reader.readline()
            except Exception as e:
                if self._stopping:
                    return
                logger.error(f"[{self.name}] stdout read error: {e}")
                break

            if not line:
                # Процесс закрыл stdout
                if self._stopping:
                    return
                # Поднимаем ошибку всем pending
                for fut in list(self._pending.values()):
                    if not fut.done():
                        fut.set_exception(MCPProtocolError(f"[{self.name}] stdout closed"))
                self._pending.clear()
                return

            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                msg = json.loads(line_str)
            except json.JSONDecodeError:
                # Пропускаем мусор (лучше серверу писать логи в stderr!)
                logger.debug(f"[{self.name}] Non-JSON stdout: {line_str[:100]}")
                continue

            # JSON-RPC ответ: есть id
            if isinstance(msg, dict) and "id" in msg:
                req_id = msg["id"]
                fut = self._pending.pop(req_id, None)
                if fut and not fut.done():
                    fut.set_result(msg)
                continue

            # Notifications/events (если MCP сервер шлёт)
            if isinstance(msg, dict) and "method" in msg:
                method = msg.get("method")
                params = msg.get("params", {})
                await self._handle_notification(method, params)
                continue

    async def _read_stderr_loop(self) -> None:
        """Читаем stderr (логи сервера) и пробрасываем в наш logger."""
        assert self._proc and self._proc.stderr
        reader = self._proc.stderr

        while True:
            try:
                line = await reader.readline()
            except Exception:
                if self._stopping:
                    return
                break

            if not line:
                return

            line_str = line.decode("utf-8", errors="replace").rstrip()
            if line_str:
                # Можно парсить уровень логирования из строки
                logger.debug(f"[{self.name}:stderr] {line_str}")

    async def _handle_notification(self, method: str, params: dict) -> None:
        """Обработка уведомлений от сервера (можно переопределить)."""
        logger.debug(f"[{self.name}] Notification: {method} {params}")

    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья клиента."""
        return {
            "name": self.name,
            "running": self.is_running,
            "pid": self._proc.pid if self._proc else None,
            "pending_requests": len(self._pending),
        }

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return f"<MCPClient {self.name} [{status}]>"
