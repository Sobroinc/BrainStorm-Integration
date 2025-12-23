# -*- coding: utf-8 -*-
"""
Конфигурация интеграции BrainStorm.

Управляет настройками для:
- BraineMemory MCP Server
- LegalOps MCP Server
- Shared SurrealDB
- Pool sizes и timeouts
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_base_path() -> Path:
    """Получить базовый путь к BrainStorm Moduls."""
    # Предполагаем что integration/ находится в BrainStorm Moduls/
    return Path(__file__).parent.parent


class IntegrationConfig(BaseSettings):
    """
    Конфигурация интеграции BrainStorm.

    Все параметры могут быть переопределены через переменные окружения
    с префиксом BRAINSTORM_.

    Пример:
        BRAINSTORM_BRAINE_POOL_SIZE=3
        BRAINSTORM_LEGAL_POOL_SIZE=2
        BRAINSTORM_MCP_TIMEOUT=60.0
    """

    model_config = SettingsConfigDict(
        env_prefix="BRAINSTORM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SSE Endpoints (PRIMARY - для подключения к уже запущенным серверам)
    # ─────────────────────────────────────────────────────────────────────────

    braine_sse_url: str = Field(
        default="http://localhost:8001/sse",
        description="BraineMemory SSE endpoint",
    )
    legal_sse_url: str = Field(
        default="http://localhost:8002/sse",
        description="LegalOps SSE endpoint",
    )
    braine_http_url: str = Field(
        default="http://localhost:8001/mcp",
        description="BraineMemory HTTP endpoint (alternative)",
    )
    legal_http_url: str = Field(
        default="http://localhost:8002/mcp",
        description="LegalOps HTTP endpoint (alternative)",
    )
    default_transport: str = Field(
        default="sse",
        description="Default transport: sse or http",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # BraineMemory MCP Server (for subprocess launch - fallback)
    # ─────────────────────────────────────────────────────────────────────────

    braine_command: str = Field(
        default="python",
        description="Команда для запуска BraineMemory",
    )
    braine_args: list[str] = Field(
        default=["-m", "src.mcp.server"],
        description="Аргументы командной строки",
    )
    braine_cwd: Optional[str] = Field(
        default=None,
        description="Рабочая директория (авто-определение если не указана)",
    )
    braine_pool_size: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Размер пула клиентов BraineMemory",
    )
    braine_env: dict[str, str] = Field(
        default_factory=dict,
        description="Дополнительные переменные окружения",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # LegalOps MCP Server
    # ─────────────────────────────────────────────────────────────────────────

    legal_command: str = Field(
        default="python",
        description="Команда для запуска LegalOps",
    )
    legal_args: list[str] = Field(
        default=["-m", "legalops.mcp.server"],
        description="Аргументы командной строки",
    )
    legal_cwd: Optional[str] = Field(
        default=None,
        description="Рабочая директория (авто-определение если не указана)",
    )
    legal_pool_size: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Размер пула клиентов LegalOps (больше для параллельных операций)",
    )
    legal_env: dict[str, str] = Field(
        default_factory=dict,
        description="Дополнительные переменные окружения",
    )
    legal_default_project: Optional[str] = Field(
        default=None,
        description="ID проекта по умолчанию для LegalOps",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SurrealDB (опционально, каждый модуль имеет свои настройки в .env)
    # ─────────────────────────────────────────────────────────────────────────
    # BraineMemory: ws://localhost:8000, ns=brainememory (сервер)
    # LegalOps: file://./data/legalops.db, ns=legalops (embedded)
    #
    # Эти настройки передаются ТОЛЬКО если нужно переопределить .env модулей

    surreal_override: bool = Field(
        default=False,
        description="Переопределить SurrealDB настройки модулей",
    )
    surreal_url: str | None = Field(
        default=None,
        description="URL SurrealDB (если surreal_override=True)",
    )
    surreal_ns: str | None = Field(
        default=None,
        description="Namespace (если surreal_override=True)",
    )
    surreal_user: str = Field(
        default="root",
        description="Пользователь SurrealDB",
    )
    surreal_pass: str = Field(
        default="root",
        description="Пароль SurrealDB",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Timeouts, Retries и Parallelism
    # ─────────────────────────────────────────────────────────────────────────

    mcp_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Таймаут для MCP вызовов (секунды)",
    )
    mcp_max_retries: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Максимум повторов при транспортных ошибках",
    )
    startup_timeout: float = Field(
        default=60.0,
        description="Таймаут запуска серверов",
    )

    # Parallelism limits (semaphore)
    braine_max_inflight: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Макс. параллельных вызовов к BraineMemory",
    )
    legal_max_inflight: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Макс. параллельных вызовов к LegalOps (тяжёлый анализ)",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Computed paths
    # ─────────────────────────────────────────────────────────────────────────

    def get_braine_cwd(self) -> str:
        """Получить рабочую директорию для BraineMemory."""
        if self.braine_cwd:
            return self.braine_cwd
        return str(get_base_path() / "BraineMemory")

    def get_legal_cwd(self) -> str:
        """Получить рабочую директорию для LegalOps."""
        if self.legal_cwd:
            return self.legal_cwd
        return str(get_base_path() / "LegalOps")

    def get_braine_env(self) -> dict[str, str]:
        """
        Получить переменные окружения для BraineMemory.

        По умолчанию BraineMemory использует свой .env:
        - SURREAL_URL=ws://localhost:8000
        - SURREAL_NS=brainememory

        Переопределение происходит только если surreal_override=True.
        """
        env = {}

        # Переопределяем SurrealDB только если явно указано
        if self.surreal_override:
            if self.surreal_url:
                env["SURREAL_URL"] = self.surreal_url
            if self.surreal_ns:
                env["SURREAL_NS"] = self.surreal_ns
            env["SURREAL_USER"] = self.surreal_user
            env["SURREAL_PASS"] = self.surreal_pass

        # Дополнительные переменные из конфига
        env.update(self.braine_env)
        return env

    def get_legal_env(self) -> dict[str, str]:
        """
        Получить переменные окружения для LegalOps.

        По умолчанию LegalOps использует свой .env:
        - SURREALDB_URL=file://./data/legalops.db (embedded режим!)
        - SURREALDB_NAMESPACE=legalops

        Переопределение происходит только если surreal_override=True.
        """
        env = {}

        # Переопределяем SurrealDB только если явно указано
        if self.surreal_override:
            if self.surreal_url:
                env["SURREALDB_URL"] = self.surreal_url
            if self.surreal_ns:
                env["SURREALDB_NAMESPACE"] = self.surreal_ns
            env["SURREALDB_USERNAME"] = self.surreal_user
            env["SURREALDB_PASSWORD"] = self.surreal_pass

        # Default project
        if self.legal_default_project:
            env["LEGALOPS_DEFAULT_PROJECT"] = self.legal_default_project

        # Дополнительные переменные из конфига
        env.update(self.legal_env)
        return env


# Singleton instance
config = IntegrationConfig()
