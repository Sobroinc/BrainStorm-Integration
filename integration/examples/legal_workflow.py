#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример: Полный workflow анализа юридического дела.

Демонстрирует интеграцию Agent + BraineMemory + LegalOps:
1. Загрузка PDF документов через LegalOps
2. Сохранение в долгосрочную память BraineMemory
3. Обнаружение противоречий
4. Построение графа знаний
5. Поиск доказательств
6. Генерация отчёта

Запуск:
    # Сначала запустите SurrealDB
    surreal start --user root --pass root file:data/brainstorm.db

    # Затем запустите пример
    python -m integration.examples.legal_workflow
"""

import asyncio
import logging
import sys
from pathlib import Path

# Добавляем parent в path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration import BrainStormOrchestrator

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main():
    """Основной workflow."""

    print("=" * 60)
    print("BrainStorm Legal Analysis Workflow")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Запуск оркестратора
    # ─────────────────────────────────────────────────────────────────────────

    print("\n[1] Starting orchestrator...")

    async with BrainStormOrchestrator() as orch:
        # Health check
        health = await orch.health_check()
        print(f"    Orchestrator: {health['orchestrator']}")
        print(f"    BraineMemory: {health['braine']}")
        print(f"    LegalOps: {health['legal']}")

        # ─────────────────────────────────────────────────────────────────────
        # 2. Проверка доступных инструментов
        # ─────────────────────────────────────────────────────────────────────

        print("\n[2] Checking available tools...")

        # Получаем список инструментов BraineMemory
        braine_tools = await orch._braine_pool.with_client(
            lambda c: c.list_tools()
        )
        print(f"    BraineMemory tools: {len(braine_tools)}")
        for tool in braine_tools[:5]:
            print(f"      - {tool.name}")
        if len(braine_tools) > 5:
            print(f"      ... and {len(braine_tools) - 5} more")

        # Получаем список инструментов LegalOps
        legal_tools = await orch._legal_pool.with_client(
            lambda c: c.list_tools()
        )
        print(f"    LegalOps tools: {len(legal_tools)}")
        for tool in legal_tools[:5]:
            print(f"      - {tool.name}")
        if len(legal_tools) > 5:
            print(f"      ... and {len(legal_tools) - 5} more")

        # ─────────────────────────────────────────────────────────────────────
        # 3. Тестовый запрос к памяти
        # ─────────────────────────────────────────────────────────────────────

        print("\n[3] Testing BraineMemory recall...")

        try:
            recall_result = await orch.braine.retrieve(
                query="договор аренды",
                limit=3,
                mode="auto",
            )
            print(f"    Found {len(recall_result)} items")
            for item in recall_result[:3]:
                content_preview = item.get("content", "")[:100]
                print(f"      - [{item.get('score', 0):.2f}] {content_preview}...")
        except Exception as e:
            print(f"    Error: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # 4. Тестовый запрос к LegalOps
        # ─────────────────────────────────────────────────────────────────────

        print("\n[4] Testing LegalOps search...")

        try:
            docs = await orch.legal.list_documents(limit=5)
            print(f"    Found {len(docs)} documents")
            for doc in docs[:5]:
                print(f"      - {doc.filename} ({doc.pages} pages)")
        except Exception as e:
            print(f"    Error: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # 5. Объединённый поиск
        # ─────────────────────────────────────────────────────────────────────

        print("\n[5] Combined search (BraineMemory + LegalOps)...")

        try:
            combined = await orch.recall_with_legal_context(
                query="обязательства сторон",
            )
            print(f"    BraineMemory results: {type(combined.get('braine_memory'))}")
            print(f"    LegalOps results: {type(combined.get('legal_documents'))}")
        except Exception as e:
            print(f"    Error: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # 6. Информация о сервере LegalOps
        # ─────────────────────────────────────────────────────────────────────

        print("\n[6] LegalOps server info...")

        try:
            server_info = await orch.legal.get_server_info()
            if isinstance(server_info, dict):
                server = server_info.get("server", {})
                print(f"    Name: {server.get('name', 'unknown')}")
                print(f"    Version: {server.get('version', 'unknown')}")
                caps = server_info.get("capabilities", {})
                print(f"    Tools: {caps.get('tools', 'unknown')}")
        except Exception as e:
            print(f"    Error: {e}")

    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)


async def demo_full_analysis():
    """
    Демо полного анализа дела (требует PDF файлы).

    Раскомментируйте и укажите пути к вашим PDF файлам.
    """
    async with BrainStormOrchestrator() as orch:
        # Пример с реальными PDF
        result = await orch.analyze_legal_case(
            pdf_paths=[
                "path/to/contract.pdf",
                "path/to/annex.pdf",
            ],
            project_id="projects:demo_case",
            store_in_memory=True,
        )

        print("Analysis Result:")
        print(f"  Documents: {len(result['documents'])}")
        print(f"  Contradictions: {len(result['contradictions'])}")
        print(f"  Entity Graph: {result['entity_graph']}")


if __name__ == "__main__":
    asyncio.run(main())
