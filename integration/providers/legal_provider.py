# -*- coding: utf-8 -*-
"""
LegalOps Provider для Agent.CentralHub.

Предоставляет интерфейс юридического анализа для агентов,
абстрагируя детали MCP протокола.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from integration.mcp_client.legal_client import LegalOpsClient
from integration.mcp_client.pool import MCPClientPool

# Type aliases for backward compatibility (SSE client returns Dict)
Document = Dict[str, Any]
SearchResult = Dict[str, Any]
Entity = Dict[str, Any]
Contradiction = Dict[str, Any]

logger = logging.getLogger(__name__)


class LegalOpsProvider:
    """
    Провайдер юридического анализа для регистрации в Agent.CentralHub.

    Предоставляет высокоуровневый интерфейс для:
    - Загрузки и поиска документов
    - Извлечения сущностей
    - Обнаружения противоречий
    - Поиска доказательств
    - Управления кейсами
    - Интеграции с Legifrance/JudiLibre

    Использование:
        # С одним клиентом
        provider = LegalOpsProvider(client)

        # С пулом (рекомендуется для параллельных операций)
        provider = LegalOpsProvider.from_pool(pool)

        # Регистрация в CentralHub
        hub.register_service("legal", provider)

        # Использование
        doc = await hub.legal.ingest_pdf("contract.pdf")
        contradictions = await hub.legal.detect_contradictions([doc.id])
    """

    name = "legal"
    provider_type = "service"

    def __init__(
        self,
        client: LegalOpsClient | None = None,
        pool: MCPClientPool[LegalOpsClient] | None = None,
        default_project_id: str | None = None,
    ):
        if not client and not pool:
            raise ValueError("Either client or pool must be provided")

        self._client = client
        self._pool = pool
        self._default_project_id = default_project_id

    @classmethod
    def from_pool(
        cls,
        pool: MCPClientPool[LegalOpsClient],
        default_project_id: str | None = None,
    ) -> "LegalOpsProvider":
        """Создать провайдер из пула клиентов."""
        return cls(pool=pool, default_project_id=default_project_id)

    @classmethod
    def from_client(
        cls,
        client: LegalOpsClient,
        default_project_id: str | None = None,
    ) -> "LegalOpsProvider":
        """Создать провайдер из одного клиента."""
        return cls(client=client, default_project_id=default_project_id)

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

    def _resolve_project(self, project_id: str | None) -> str | None:
        """Получить project_id с fallback на default."""
        return project_id or self._default_project_id

    # ─────────────────────────────────────────────────────────────────────────
    # L2 - Document Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def ingest_pdf(
        self,
        pdf_path: str | Path,
        project_id: str | None = None,
        ner_mode: str = "quick",
    ) -> Document:
        """
        Загрузить PDF документ.

        Args:
            pdf_path: Путь к файлу
            project_id: ID проекта
            ner_mode: Режим NER (quick, deep, none)

        Returns:
            Document с информацией о загруженном документе
        """
        return await self._call(
            "ingest_pdf",
            pdf_path=pdf_path,
            project_id=self._resolve_project(project_id),
            ner_mode=ner_mode,
        )

    async def ingest_text(
        self,
        content: str,
        title: str,
        project_id: str | None = None,
        doc_type: str = "text",
    ) -> Document:
        """Загрузить текстовый контент."""
        return await self._call(
            "ingest_text",
            content=content,
            title=title,
            project_id=self._resolve_project(project_id),
            doc_type=doc_type,
        )

    async def search(
        self,
        query: str,
        project_id: str | None = None,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> List[SearchResult]:
        """
        Поиск по документам (гибридный: vector + FTS).

        Args:
            query: Поисковый запрос
            project_id: ID проекта
            limit: Максимум результатов
            mode: hybrid, vector, fts

        Returns:
            Список результатов поиска
        """
        return await self._call(
            "search_documents",
            query=query,
            project_id=self._resolve_project(project_id),
            limit=limit,
            mode=mode,
        )

    async def smart_search(
        self,
        query: str,
        project_id: str | None = None,
        include_entities: bool = True,
        include_timeline: bool = False,
    ) -> Dict[str, Any]:
        """Умный поиск с автоопределением intent."""
        return await self._call(
            "smart_search",
            query=query,
            project_id=self._resolve_project(project_id),
            include_entities=include_entities,
            include_timeline=include_timeline,
        )

    async def ask(
        self,
        question: str,
        doc_ids: List[str] | None = None,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """RAG Q&A с цитатами."""
        return await self._call(
            "ask_question",
            question=question,
            doc_ids=doc_ids,
            project_id=self._resolve_project(project_id),
        )

    async def get_document(self, document_id: str) -> Document:
        """Получить информацию о документе."""
        return await self._call("get_document", document_id=document_id)

    async def list_documents(
        self,
        project_id: str | None = None,
        limit: int = 50,
    ) -> List[Document]:
        """Список документов в проекте."""
        return await self._call(
            "list_documents",
            project_id=self._resolve_project(project_id),
            limit=limit,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # L3 - Entity Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def extract_entities(
        self,
        document_id: str,
        mode: str = "quick",
    ) -> List[Entity]:
        """Извлечь сущности из документа."""
        return await self._call(
            "extract_entities",
            document_id=document_id,
            mode=mode,
        )

    async def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        project_id: str | None = None,
    ) -> List[Entity]:
        """Поиск сущностей."""
        return await self._call(
            "search_entities",
            query=query,
            entity_type=entity_type,
            project_id=self._resolve_project(project_id),
        )

    async def get_entity_graph(
        self,
        project_id: str | None = None,
        entity_ids: List[str] | None = None,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """Получить граф сущностей."""
        return await self._call(
            "get_entity_graph",
            project_id=self._resolve_project(project_id),
            entity_ids=entity_ids,
            depth=depth,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # L4 - Analysis Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def detect_contradictions(
        self,
        doc_ids: List[str] | None = None,
        project_id: str | None = None,
    ) -> List[Contradiction]:
        """
        Обнаружить противоречия между документами.

        Args:
            doc_ids: Конкретные документы для сравнения
            project_id: ID проекта (сканирует все документы)

        Returns:
            Список обнаруженных противоречий
        """
        return await self._call(
            "detect_contradictions",
            doc_ids=doc_ids,
            project_id=self._resolve_project(project_id),
        )

    async def get_contradictions(
        self,
        project_id: str | None = None,
        status: str = "open",
    ) -> List[Contradiction]:
        """Получить список противоречий по статусу."""
        return await self._call(
            "get_contradictions",
            project_id=self._resolve_project(project_id),
            status=status,
        )

    async def compare_documents(
        self,
        doc_a: str,
        doc_b: str,
        comparison_type: str = "semantic",
    ) -> Dict[str, Any]:
        """Сравнить два документа."""
        return await self._call(
            "compare_documents",
            doc_a=doc_a,
            doc_b=doc_b,
            comparison_type=comparison_type,
        )

    async def get_contradiction_report(
        self,
        project_id: str | None = None,
        format: str = "markdown",
    ) -> Dict[str, Any]:
        """Сгенерировать отчёт о противоречиях."""
        return await self._call(
            "get_contradiction_report",
            project_id=self._resolve_project(project_id),
            format=format,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # L5 - Evidence Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def find_evidence(
        self,
        claim: str,
        project_id: str | None = None,
        min_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Найти доказательства для утверждения."""
        return await self._call(
            "find_evidence",
            claim=claim,
            project_id=self._resolve_project(project_id),
            min_confidence=min_confidence,
        )

    async def analyze_coverage(
        self,
        draft_text: str,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """Анализ покрытия черновика доказательствами."""
        return await self._call(
            "analyze_coverage",
            draft_text=draft_text,
            project_id=self._resolve_project(project_id),
        )

    async def audit_draft(
        self,
        draft_text: str,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """Аудит черновика на предмет неподтверждённых утверждений."""
        return await self._call(
            "audit_draft",
            draft_text=draft_text,
            project_id=self._resolve_project(project_id),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # L6 - Drafting Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def annotate_text(
        self,
        text: str,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """Аннотировать текст ссылками на доказательства."""
        return await self._call(
            "annotate_text",
            text=text,
            project_id=self._resolve_project(project_id),
        )

    async def generate_skeleton(
        self,
        case_type: str,
        project_id: str | None = None,
        template: str | None = None,
    ) -> Dict[str, Any]:
        """Сгенерировать скелет документа."""
        return await self._call(
            "generate_skeleton",
            case_type=case_type,
            project_id=self._resolve_project(project_id),
            template=template,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # L7 - Bundle Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def create_case(
        self,
        name: str,
        case_type: str = "civil",
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """Создать новый кейс."""
        return await self._call(
            "create_case",
            name=name,
            case_type=case_type,
            project_id=self._resolve_project(project_id),
        )

    async def add_piece(
        self,
        case_id: str,
        document_id: str,
        piece_number: int | None = None,
        label: str | None = None,
    ) -> Dict[str, Any]:
        """Добавить документ в кейс."""
        return await self._call(
            "add_piece",
            case_id=case_id,
            document_id=document_id,
            piece_number=piece_number,
            label=label,
        )

    async def get_case(self, case_id: str) -> Dict[str, Any]:
        """Получить информацию о кейсе."""
        return await self._call("get_case", case_id=case_id)

    async def list_cases(
        self,
        project_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Список кейсов."""
        return await self._call(
            "list_cases",
            project_id=self._resolve_project(project_id),
        )

    async def export_bundle(
        self,
        case_id: str,
        format: str = "pdf",
    ) -> Dict[str, Any]:
        """Экспортировать bundle для подачи в суд."""
        return await self._call(
            "export_bundle",
            case_id=case_id,
            format=format,
        )

    async def get_timeline(
        self,
        case_id: str | None = None,
        project_id: str | None = None,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Получить timeline событий."""
        return await self._call(
            "get_timeline",
            case_id=case_id,
            project_id=self._resolve_project(project_id),
            format=format,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # L8 - Legal References
    # ─────────────────────────────────────────────────────────────────────────

    async def search_legifrance(
        self,
        query: str,
        code: str | None = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Поиск в Legifrance."""
        return await self._call(
            "search_legifrance",
            query=query,
            code=code,
            limit=limit,
        )

    async def search_judilibre(
        self,
        query: str,
        jurisdiction: str | None = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Поиск в JudiLibre (судебная практика)."""
        return await self._call(
            "search_judilibre",
            query=query,
            jurisdiction=jurisdiction,
            limit=limit,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Health & Status
    # ─────────────────────────────────────────────────────────────────────────

    async def get_server_info(self) -> Dict[str, Any]:
        """Получить информацию о сервере."""
        return await self._call("get_server_info")

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
            return f"<LegalOpsProvider pool={self._pool}>"
        return f"<LegalOpsProvider client={self._client}>"
