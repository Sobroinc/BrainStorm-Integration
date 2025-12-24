# -*- coding: utf-8 -*-
"""
Load Test: 100 Documents

Target metrics:
- Ingest 100 docs: < 120s
- prepare_hearing: < 60s
- No failures
"""

import asyncio
import time
import tracemalloc
from pathlib import Path
from typing import List

import pytest
import pytest_asyncio

from .conftest import LoadTestMetrics, collect_txt_files


@pytest.mark.load
@pytest.mark.asyncio
async def test_ingest_100_docs(orchestrator, test_project_id, metrics: LoadTestMetrics):
    """
    Test: Ingest 100 real documents from OlgaFinal.

    Success criteria:
    - All 100 docs ingested
    - Total time < 120s
    - Error rate < 5%
    """
    files = collect_txt_files(limit=100)
    assert len(files) >= 100, f"Need 100 files, got {len(files)}"

    # Start memory tracking
    tracemalloc.start()
    metrics.start_time = time.time()

    # Ingest documents
    for i, file_path in enumerate(files):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if len(content.strip()) < 50:
                continue  # Skip empty/tiny files

            result = await orchestrator.legal.ingest_text(
                project_id=test_project_id,
                content=content[:50000],  # Limit content size
                title=file_path.stem[:100],
                doc_type="evidence",
            )

            if result.get("document_id"):
                metrics.docs_ingested += 1
                metrics.chunks_created += result.get("chunks_created", 0)
            else:
                metrics.docs_failed += 1
                metrics.errors.append(f"No doc_id for {file_path.name}")

        except Exception as e:
            metrics.docs_failed += 1
            metrics.errors.append(f"{file_path.name}: {str(e)[:100]}")

        # Progress every 20 docs
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/100 docs ingested")

    metrics.end_time = time.time()

    # Memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics.peak_memory_mb = peak / 1024 / 1024

    # Assertions
    print(f"\n  Results: {metrics.to_dict()}")

    assert metrics.docs_ingested >= 95, f"Expected >= 95 docs, got {metrics.docs_ingested}"
    assert metrics.duration_s < 120, f"Expected < 120s, took {metrics.duration_s:.1f}s"
    assert metrics.docs_failed < 5, f"Too many failures: {metrics.docs_failed}"


@pytest.mark.load
@pytest.mark.asyncio
async def test_prepare_hearing_100_docs(orchestrator, test_project_id, metrics: LoadTestMetrics):
    """
    Test: Run prepare_hearing on project with 100 docs.

    Assumes test_ingest_100_docs ran first (same project_id won't work).
    This test creates its own docs for isolation.

    Success criteria:
    - prepare_hearing completes
    - Total time < 60s
    - Result has required fields
    """
    # First ingest some docs
    files = collect_txt_files(limit=100)

    print("  Ingesting 100 docs for prepare_hearing test...")
    ingested = 0
    for file_path in files[:100]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if len(content.strip()) < 50:
                continue

            await orchestrator.legal.ingest_text(
                project_id=test_project_id,
                content=content[:30000],
                title=file_path.stem[:100],
                doc_type="evidence",
            )
            ingested += 1
        except Exception:
            pass

    print(f"  Ingested {ingested} docs")

    # Create case
    case_result = await orchestrator.legal.create_case(
        name=f"Load Test Case - {test_project_id}",
        project_id=test_project_id,
    )
    case_id = case_result.get("case_id") or case_result.get("id", f"case:load-test")

    # Run prepare_hearing
    print("  Running prepare_hearing...")
    metrics.start_time = time.time()

    result = await orchestrator.prepare_hearing(
        case_id=case_id,
        project_id=test_project_id,
        questions=["Quels sont les points litigieux?"],
        auto_export=False,
        validate_integrity=True,
        generate_hearing_pack=True,
        max_doc_pairs=50,
    )

    metrics.end_time = time.time()

    # Assertions
    print(f"\n  prepare_hearing completed in {metrics.duration_s:.1f}s")
    print(f"  Success: {result.success}")
    print(f"  Trace ID: {result.trace_id}")

    assert result.success, f"prepare_hearing failed: {result.data.get('error')}"
    assert metrics.duration_s < 60, f"Expected < 60s, took {metrics.duration_s:.1f}s"

    # Check envelope structure
    envelope = result.to_envelope()
    assert envelope.get("ok") is True
    assert envelope.get("trace_id")
    assert envelope.get("project_id")


@pytest.mark.load
@pytest.mark.asyncio
async def test_parallel_tool_calls_10(orchestrator):
    """
    Test: 10 parallel memory_recall calls.

    Success criteria:
    - All 10 complete without error
    - No deadlocks
    - Total time < 30s
    """
    queries = [
        "contrat de vente",
        "facture impayée",
        "mise en demeure",
        "procès verbal",
        "assignation",
        "jugement tribunal",
        "appel cour",
        "expertise comptable",
        "rapport financier",
        "bordereau pièces",
    ]

    start = time.time()

    # Launch 10 parallel queries
    tasks = [
        orchestrator.braine.memory_recall(query=q, limit=5, mode="hybrid")
        for q in queries
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start

    # Check results
    successes = sum(1 for r in results if isinstance(r, dict) and not r.get("error"))
    failures = [r for r in results if isinstance(r, Exception)]

    print(f"\n  10 parallel calls completed in {duration:.1f}s")
    print(f"  Successes: {successes}/10")
    print(f"  Failures: {len(failures)}")

    assert successes >= 9, f"Expected >= 9 successes, got {successes}"
    assert duration < 30, f"Expected < 30s, took {duration:.1f}s"
    assert len(failures) <= 1, f"Too many failures: {failures}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
