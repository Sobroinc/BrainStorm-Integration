# -*- coding: utf-8 -*-
"""
Pytest fixtures for load testing.

Uses real legal documents from OlgaFinal dataset.

Note: Path setup is handled by Integration/conftest.py
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import pytest
import pytest_asyncio

# Data source paths
OLGA_BASE = Path(r"C:\Users\s7\OneDrive\Backup-Doc\OlgaFinal")
EMAIL_ATTACHMENTS = OLGA_BASE / "Olga" / "Email_Attachments"
EMAIL_BODIES = OLGA_BASE / "Olga" / "Email_Bodies"
OLGA_ATTACH = OLGA_BASE / "OlgaAttach"


@dataclass
class LoadTestMetrics:
    """Metrics collected during load tests."""
    start_time: float = 0.0
    end_time: float = 0.0
    docs_ingested: int = 0
    docs_failed: int = 0
    entities_extracted: int = 0
    chunks_created: int = 0
    peak_memory_mb: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def docs_per_second(self) -> float:
        if self.duration_s > 0:
            return self.docs_ingested / self.duration_s
        return 0.0

    def to_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 2),
            "docs_ingested": self.docs_ingested,
            "docs_failed": self.docs_failed,
            "docs_per_second": round(self.docs_per_second, 2),
            "entities_extracted": self.entities_extracted,
            "chunks_created": self.chunks_created,
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "error_count": len(self.errors),
        }


def collect_txt_files(limit: Optional[int] = None, shuffle: bool = True) -> List[Path]:
    """
    Collect .txt files from OlgaFinal dataset.

    Args:
        limit: Max files to return (None = all)
        shuffle: Randomize selection

    Returns:
        List of Path objects
    """
    files = []

    # Collect from Email_Bodies (primary source - extracted email text)
    if EMAIL_BODIES.exists():
        files.extend(EMAIL_BODIES.glob("*.txt"))

    # Collect from OlgaAttach (extracted PDF text)
    if OLGA_ATTACH.exists():
        files.extend(OLGA_ATTACH.glob("*.txt"))

    files = list(files)

    if shuffle:
        random.shuffle(files)

    if limit:
        files = files[:limit]

    return files


def collect_pdf_files(limit: Optional[int] = None, shuffle: bool = True) -> List[Path]:
    """
    Collect .pdf files from OlgaFinal dataset.

    Args:
        limit: Max files to return (None = all)
        shuffle: Randomize selection

    Returns:
        List of Path objects
    """
    files = []

    # Collect from Email_Attachments
    if EMAIL_ATTACHMENTS.exists():
        files.extend(EMAIL_ATTACHMENTS.glob("*.pdf"))

    # Collect from OlgaAttach
    if OLGA_ATTACH.exists():
        files.extend(OLGA_ATTACH.glob("*.pdf"))

    files = list(files)

    if shuffle:
        random.shuffle(files)

    if limit:
        files = files[:limit]

    return files


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def olga_txt_files_100() -> List[Path]:
    """100 random .txt files for basic load test."""
    return collect_txt_files(limit=100)


@pytest.fixture(scope="session")
def olga_txt_files_500() -> List[Path]:
    """500 random .txt files for standard load test."""
    return collect_txt_files(limit=500)


@pytest.fixture(scope="session")
def olga_txt_files_1000() -> List[Path]:
    """1000 random .txt files for stress test."""
    return collect_txt_files(limit=1000)


@pytest.fixture(scope="session")
def olga_all_txt_files() -> List[Path]:
    """All available .txt files for extreme stress test."""
    return collect_txt_files(limit=None)


@pytest_asyncio.fixture
async def orchestrator():
    """
    Create and connect orchestrator for tests.

    Assumes MCP servers are already running:
    - BraineMemory on port 8001
    - LegalOps on port 8002
    """
    from integration.orchestrator import BrainStormOrchestrator

    async with BrainStormOrchestrator() as orch:
        yield orch


@pytest_asyncio.fixture
async def test_project_id():
    """Generate unique project ID for test isolation."""
    import uuid
    return f"proj:load-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def metrics() -> LoadTestMetrics:
    """Fresh metrics instance for each test."""
    return LoadTestMetrics()


# Markers for test categorization
def pytest_configure(config):
    config.addinivalue_line("markers", "load: Load tests (100 docs)")
    config.addinivalue_line("markers", "stress: Stress tests (500+ docs)")
    config.addinivalue_line("markers", "extreme: Extreme tests (1000+ docs)")
    config.addinivalue_line("markers", "slow: Tests that take > 60s")
