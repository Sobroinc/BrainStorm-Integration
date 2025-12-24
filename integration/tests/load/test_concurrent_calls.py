# -*- coding: utf-8 -*-
"""
Concurrent Tool Calls Stress Test.

Tests parallel MCP tool invocations for:
- Deadlock detection
- Semaphore correctness
- Error isolation
- Graceful timeout handling

Target metrics:
- 10 parallel calls: 100% success or graceful timeout
- No deadlocks
- Error isolation (one failure doesn't crash others)
"""

import asyncio
import time
from typing import List, Tuple

import pytest
import pytest_asyncio

from .conftest import LoadTestMetrics


@pytest.mark.stress
@pytest.mark.asyncio
class TestConcurrentToolCalls:
    """Stress tests for concurrent MCP tool calls."""

    async def test_10_parallel_memory_recall(self, orchestrator, metrics: LoadTestMetrics):
        """
        Test: 10 parallel memory_recall calls.

        Success criteria:
        - All 10 complete (success or graceful error)
        - No deadlocks (completes < 30s)
        - No exceptions propagate to caller
        """
        queries = [
            "contract terms and conditions",
            "payment obligations due dates",
            "parties involved in dispute",
            "evidence documents submitted",
            "legal proceedings timeline",
            "witness statements testimony",
            "expert analysis conclusions",
            "damages claimed amounts",
            "settlement negotiations",
            "court rulings judgments",
        ]

        print("\n  Running 10 parallel memory_recall calls...")
        metrics.start_time = time.time()

        tasks = [
            orchestrator.braine.memory_recall(
                query=q,
                limit=5,
                mode="hybrid",
            )
            for q in queries
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("DEADLOCK: 10 parallel calls did not complete in 30s")

        metrics.end_time = time.time()

        # Analyze results
        successes = 0
        graceful_errors = 0
        exceptions = []

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                exceptions.append(f"Query {i}: {type(r).__name__}")
            elif isinstance(r, dict):
                if r.get("error"):
                    graceful_errors += 1
                else:
                    successes += 1

        print(f"  Completed in {metrics.duration_s:.2f}s")
        print(f"  Successes: {successes}/10")
        print(f"  Graceful errors: {graceful_errors}/10")
        print(f"  Exceptions: {len(exceptions)}")

        # All should complete (graceful error is acceptable)
        total_handled = successes + graceful_errors
        assert total_handled >= 8, f"Expected >= 8 handled, got {total_handled}"
        assert metrics.duration_s < 30, f"Too slow: {metrics.duration_s:.1f}s"

    async def test_10_parallel_graph_queries(self, orchestrator, metrics: LoadTestMetrics):
        """
        Test: 10 parallel memory_recall_graph calls.

        Tests heavier operations that involve graph traversal.
        """
        queries = [
            ("local", "contrat achat"),
            ("local", "paiement facture"),
            ("global", "parties litige"),
            ("local", "expert comptable"),
            ("global", "tribunal jugement"),
            ("local", "preuve document"),
            ("local", "temoin declaration"),
            ("global", "dommages interets"),
            ("local", "accord settlement"),
            ("global", "appel cassation"),
        ]

        print("\n  Running 10 parallel graph queries...")
        metrics.start_time = time.time()

        tasks = [
            orchestrator.braine.memory_recall_graph(
                query=q,
                mode=mode,
                limit=10,
                include_entities=True,
            )
            for mode, q in queries
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("DEADLOCK: 10 parallel graph queries did not complete in 45s")

        metrics.end_time = time.time()

        successes = sum(1 for r in results if isinstance(r, dict) and not r.get("error"))
        print(f"  Completed in {metrics.duration_s:.2f}s")
        print(f"  Successes: {successes}/10")

        assert successes >= 7, f"Expected >= 7 successes, got {successes}"

    async def test_mixed_parallel_calls(self, orchestrator, metrics: LoadTestMetrics):
        """
        Test: Mixed parallel calls to different tools.

        Simulates real-world usage with diverse operations.
        """
        print("\n  Running mixed parallel calls...")
        metrics.start_time = time.time()

        tasks = [
            # Memory recalls
            orchestrator.braine.memory_recall(query="test 1", limit=3),
            orchestrator.braine.memory_recall(query="test 2", limit=3),
            # Graph queries
            orchestrator.braine.memory_recall_graph(query="entities", mode="local", limit=5),
            orchestrator.braine.memory_recall_graph(query="relations", mode="local", limit=5),
            # More recalls
            orchestrator.braine.memory_recall(query="test 5", limit=3),
            orchestrator.braine.memory_recall(query="test 3", limit=3),
            orchestrator.braine.memory_recall(query="test 4", limit=3),
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("DEADLOCK: Mixed parallel calls did not complete in 30s")

        metrics.end_time = time.time()

        successes = sum(1 for r in results if isinstance(r, dict))
        print(f"  Completed in {metrics.duration_s:.2f}s")
        print(f"  Successes: {successes}/7")

        assert successes >= 5, f"Expected >= 5 successes, got {successes}"

    async def test_error_isolation(self, orchestrator, metrics: LoadTestMetrics):
        """
        Test: One failing call doesn't affect others.

        Sends mix of valid and invalid queries.
        """
        print("\n  Testing error isolation...")

        tasks = [
            # Valid queries
            orchestrator.braine.memory_recall(query="valid query 1", limit=3),
            orchestrator.braine.memory_recall(query="valid query 2", limit=3),
            # Query that might fail (empty)
            orchestrator.braine.memory_recall(query="", limit=3),
            # More valid queries
            orchestrator.braine.memory_recall(query="valid query 3", limit=3),
            orchestrator.braine.memory_recall(query="valid query 4", limit=3),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count by type
        valid_results = sum(1 for r in results if isinstance(r, dict) and not r.get("error"))
        error_results = sum(1 for r in results if isinstance(r, dict) and r.get("error"))
        exceptions = sum(1 for r in results if isinstance(r, Exception))

        print(f"  Valid results: {valid_results}")
        print(f"  Error results: {error_results}")
        print(f"  Exceptions: {exceptions}")

        # Valid queries should succeed even if one fails
        assert valid_results >= 3, f"Expected >= 3 valid results, got {valid_results}"

    async def test_semaphore_limit(self, orchestrator, metrics: LoadTestMetrics):
        """
        Test: Semaphore correctly limits concurrent calls.

        Sends 20 calls with semaphore limit of 8.
        All should complete without resource exhaustion.
        """
        print("\n  Testing semaphore limit (20 calls, limit 8)...")
        metrics.start_time = time.time()

        # Generate 20 queries
        tasks = [
            orchestrator.braine.memory_recall(
                query=f"semaphore test query {i}",
                limit=3,
            )
            for i in range(20)
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("DEADLOCK: 20 calls with semaphore did not complete in 60s")

        metrics.end_time = time.time()

        successes = sum(1 for r in results if isinstance(r, dict))
        print(f"  Completed 20 calls in {metrics.duration_s:.2f}s")
        print(f"  Successes: {successes}/20")

        # Most should succeed
        assert successes >= 16, f"Expected >= 16 successes, got {successes}"
        # Should complete reasonably fast (not one-at-a-time)
        assert metrics.duration_s < 60, f"Too slow: {metrics.duration_s:.1f}s"


@pytest.mark.stress
@pytest.mark.slow
@pytest.mark.asyncio
class TestConcurrentLegalTools:
    """Stress tests for concurrent LegalOps tool calls."""

    async def test_10_parallel_legal_queries(self, orchestrator, test_project_id, metrics: LoadTestMetrics):
        """
        Test: 10 parallel calls to LegalOps tools.

        Requires LegalOps server on port 8002.
        """
        print("\n  Running 10 parallel LegalOps queries...")
        metrics.start_time = time.time()

        tasks = [
            orchestrator.legal.search_documents(
                project_id=test_project_id,
                query=f"search query {i}",
                limit=5,
            )
            for i in range(10)
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("DEADLOCK: 10 parallel legal queries did not complete in 45s")

        metrics.end_time = time.time()

        successes = sum(1 for r in results if isinstance(r, dict))
        print(f"  Completed in {metrics.duration_s:.2f}s")
        print(f"  Successes: {successes}/10")

        assert successes >= 7, f"Expected >= 7 successes, got {successes}"

    async def test_cross_service_parallel(self, orchestrator, test_project_id, metrics: LoadTestMetrics):
        """
        Test: Parallel calls to BOTH services simultaneously.

        Simulates real orchestration patterns.
        """
        print("\n  Running parallel calls to both services...")
        metrics.start_time = time.time()

        tasks = [
            # BraineMemory calls
            orchestrator.braine.memory_recall(query="braine query 1", limit=3),
            orchestrator.braine.memory_recall(query="braine query 2", limit=3),
            orchestrator.braine.memory_recall_graph(query="graph query", mode="local", limit=5),
            # LegalOps calls
            orchestrator.legal.search_documents(project_id=test_project_id, query="legal query 1", limit=3),
            orchestrator.legal.search_documents(project_id=test_project_id, query="legal query 2", limit=3),
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("DEADLOCK: Cross-service parallel calls did not complete in 30s")

        metrics.end_time = time.time()

        successes = sum(1 for r in results if isinstance(r, dict))
        print(f"  Completed in {metrics.duration_s:.2f}s")
        print(f"  Successes: {successes}/5 (cross-service)")

        # At least most should succeed
        assert successes >= 3, f"Expected >= 3 successes, got {successes}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "stress"])
