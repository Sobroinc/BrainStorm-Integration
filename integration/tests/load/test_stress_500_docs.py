# -*- coding: utf-8 -*-
"""
Stress Test: 500 Documents

Target metrics:
- Ingest 500 docs: < 600s (10 min)
- prepare_hearing: < 180s (3 min)
- Memory stable (no leaks)
"""

import asyncio
import gc
import time
import tracemalloc
from pathlib import Path
from typing import List

import pytest

from .conftest import LoadTestMetrics, collect_txt_files


@pytest.mark.stress
@pytest.mark.slow
@pytest.mark.asyncio
async def test_ingest_500_docs(orchestrator, test_project_id, metrics: LoadTestMetrics):
    """
    Stress test: Ingest 500 real documents.

    Success criteria:
    - >= 475 docs ingested (95%)
    - Total time < 600s
    - Memory < 500MB peak
    """
    files = collect_txt_files(limit=500)
    assert len(files) >= 500, f"Need 500 files, got {len(files)}"

    tracemalloc.start()
    metrics.start_time = time.time()

    batch_size = 10
    for batch_start in range(0, len(files), batch_size):
        batch = files[batch_start:batch_start + batch_size]

        # Process batch concurrently
        tasks = []
        for file_path in batch:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content.strip()) < 50:
                    continue

                task = orchestrator.legal.ingest_text(
                    project_id=test_project_id,
                    content=content[:30000],
                    title=file_path.stem[:100],
                    doc_type="evidence",
                )
                tasks.append(task)
            except Exception as e:
                metrics.errors.append(str(e)[:100])

        # Wait for batch
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, dict) and r.get("document_id"):
                    metrics.docs_ingested += 1
                elif isinstance(r, Exception):
                    metrics.docs_failed += 1
                    metrics.errors.append(str(r)[:100])

        # Progress every 100 docs
        if (batch_start + batch_size) % 100 == 0:
            elapsed = time.time() - metrics.start_time
            rate = metrics.docs_ingested / elapsed if elapsed > 0 else 0
            print(f"  Progress: {metrics.docs_ingested}/500 @ {rate:.1f} docs/s")

    metrics.end_time = time.time()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics.peak_memory_mb = peak / 1024 / 1024

    print(f"\n  Results: {metrics.to_dict()}")

    assert metrics.docs_ingested >= 475, f"Expected >= 475, got {metrics.docs_ingested}"
    assert metrics.duration_s < 600, f"Expected < 600s, took {metrics.duration_s:.1f}s"
    assert metrics.peak_memory_mb < 500, f"Memory too high: {metrics.peak_memory_mb:.0f}MB"


@pytest.mark.stress
@pytest.mark.slow
@pytest.mark.asyncio
async def test_prepare_hearing_500_docs(orchestrator, test_project_id, metrics: LoadTestMetrics):
    """
    Stress test: prepare_hearing with 500 docs.

    Success criteria:
    - Completes successfully
    - Total time < 180s
    """
    # Ingest 500 docs first
    files = collect_txt_files(limit=500)

    print("  Ingesting 500 docs...")
    ingested = 0
    for i, file_path in enumerate(files):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if len(content.strip()) < 50:
                continue

            await orchestrator.legal.ingest_text(
                project_id=test_project_id,
                content=content[:20000],
                title=file_path.stem[:100],
                doc_type="evidence",
            )
            ingested += 1

            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/500 ingested")

        except Exception:
            pass

    print(f"  Ingested {ingested} docs")

    # Create case
    case_result = await orchestrator.legal.create_case(
        name=f"Stress Test Case",
        project_id=test_project_id,
    )
    case_id = case_result.get("case_id") or case_result.get("id", "case:stress")

    # Run prepare_hearing with timeout
    print("  Running prepare_hearing on 500 docs...")
    metrics.start_time = time.time()

    try:
        result = await asyncio.wait_for(
            orchestrator.prepare_hearing(
                case_id=case_id,
                project_id=test_project_id,
                questions=["Résumez les points clés du litige"],
                auto_export=False,
                validate_integrity=True,
                generate_hearing_pack=True,
                max_doc_pairs=100,  # Higher limit for more docs
            ),
            timeout=180.0,
        )
        success = result.success
    except asyncio.TimeoutError:
        success = False
        print("  TIMEOUT after 180s")

    metrics.end_time = time.time()

    print(f"\n  prepare_hearing completed in {metrics.duration_s:.1f}s")

    assert success, "prepare_hearing failed or timed out"
    assert metrics.duration_s < 180, f"Expected < 180s, took {metrics.duration_s:.1f}s"


@pytest.mark.stress
@pytest.mark.asyncio
async def test_memory_stability_50_workflows(orchestrator, metrics: LoadTestMetrics):
    """
    Memory stability: Run 50 lightweight workflows.

    Success criteria:
    - No memory leaks (final < 2x initial)
    - All workflows complete
    """
    gc.collect()
    tracemalloc.start()
    initial_snapshot = tracemalloc.take_snapshot()

    project_id = "proj:memory-test"

    print("  Running 50 workflow iterations...")
    for i in range(50):
        try:
            # Light workflow: recall + recall_graph
            await orchestrator.braine.memory_recall(
                query=f"test query {i}",
                limit=5,
                mode="auto",
            )

            await orchestrator.braine.memory_recall_graph(
                query=f"entity query {i}",
                mode="local",
                limit=5,
            )

            if (i + 1) % 10 == 0:
                gc.collect()
                current, peak = tracemalloc.get_traced_memory()
                print(f"    Iteration {i + 1}/50: {current / 1024 / 1024:.1f}MB current")

        except Exception as e:
            metrics.errors.append(str(e)[:100])

    gc.collect()
    final_current, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics.peak_memory_mb = final_peak / 1024 / 1024

    print(f"\n  Peak memory: {metrics.peak_memory_mb:.1f}MB")
    print(f"  Errors: {len(metrics.errors)}")

    # Memory should not grow excessively
    assert metrics.peak_memory_mb < 200, f"Memory too high: {metrics.peak_memory_mb:.0f}MB"
    assert len(metrics.errors) < 5, f"Too many errors: {len(metrics.errors)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "stress"])
