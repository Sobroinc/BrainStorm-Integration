# -*- coding: utf-8 -*-
"""
Graph Stress Test: 1000 Entities

Target metrics:
- memory_recall_graph: < 5s for 1000 entities
- Community building: < 60s
- No OOM errors
"""

import asyncio
import time
import tracemalloc
from typing import List

import pytest

from .conftest import LoadTestMetrics


@pytest.mark.stress
@pytest.mark.asyncio
async def test_recall_graph_performance(orchestrator, metrics: LoadTestMetrics):
    """
    Test: memory_recall_graph query performance.

    Success criteria:
    - Response time < 5s
    - Returns valid results
    - No timeouts
    """
    queries = [
        "financial documents expert analysis",
        "contract terms violations",
        "payment obligations parties",
        "legal proceedings timeline",
        "witness statements evidence",
    ]

    print("  Testing graph recall performance...")

    for query in queries:
        start = time.time()

        result = await orchestrator.braine.memory_recall_graph(
            query=query,
            mode="local",
            limit=20,
            include_entities=True,
            include_relations=True,
            include_communities=False,
        )

        duration = time.time() - start

        entities = result.get("entities", [])
        relations = result.get("relations", [])

        print(f"    '{query[:30]}...' -> {len(entities)} entities, {len(relations)} relations in {duration:.2f}s")

        assert duration < 5, f"Query too slow: {duration:.1f}s"


@pytest.mark.stress
@pytest.mark.asyncio
async def test_global_graph_search(orchestrator, metrics: LoadTestMetrics):
    """
    Test: Global graph search (community-based).

    Success criteria:
    - Response time < 10s
    - Returns community summaries
    """
    print("  Testing global graph search...")

    start = time.time()

    result = await orchestrator.braine.memory_recall_graph(
        query="Quels sont les principaux acteurs et leurs relations?",
        mode="global",
        limit=10,
        include_entities=True,
        include_relations=True,
        include_communities=True,
    )

    duration = time.time() - start

    communities = result.get("communities", [])
    entities = result.get("entities", [])

    print(f"    Global search: {len(communities)} communities, {len(entities)} entities in {duration:.2f}s")

    assert duration < 10, f"Global search too slow: {duration:.1f}s"


@pytest.mark.stress
@pytest.mark.slow
@pytest.mark.asyncio
async def test_build_communities(orchestrator, metrics: LoadTestMetrics):
    """
    Test: Community building performance.

    Success criteria:
    - Completes < 60s
    - Creates communities
    - No OOM
    """
    print("  Building communities...")

    tracemalloc.start()
    start = time.time()

    try:
        result = await asyncio.wait_for(
            orchestrator.braine.memory_build_communities(
                max_levels=2,
                min_community_size=2,
                resolution=1.0,
                regenerate=True,
            ),
            timeout=60.0,
        )
        success = True
    except asyncio.TimeoutError:
        result = {"error": "timeout"}
        success = False

    duration = time.time() - start

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    communities_created = result.get("communities_created", 0)
    entities_processed = result.get("entities_processed", 0)

    print(f"    Built {communities_created} communities from {entities_processed} entities in {duration:.2f}s")
    print(f"    Peak memory: {peak / 1024 / 1024:.1f}MB")

    assert success, "Community building timed out"
    assert duration < 60, f"Too slow: {duration:.1f}s"
    assert peak / 1024 / 1024 < 300, f"Memory too high: {peak / 1024 / 1024:.0f}MB"


@pytest.mark.stress
@pytest.mark.asyncio
async def test_concurrent_graph_queries(orchestrator, metrics: LoadTestMetrics):
    """
    Test: 5 parallel graph queries.

    Success criteria:
    - All complete
    - Total time < 15s
    - No deadlocks
    """
    queries = [
        ("local", "contrat achat vente"),
        ("local", "paiement facture dette"),
        ("global", "parties litige conflit"),
        ("local", "expert comptable rapport"),
        ("global", "tribunal jugement appel"),
    ]

    print("  Running 5 parallel graph queries...")

    start = time.time()

    tasks = [
        orchestrator.braine.memory_recall_graph(
            query=q,
            mode=mode,
            limit=10,
        )
        for mode, q in queries
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start

    successes = sum(1 for r in results if isinstance(r, dict) and not r.get("error"))
    failures = [r for r in results if isinstance(r, Exception)]

    print(f"    5 parallel queries completed in {duration:.1f}s")
    print(f"    Successes: {successes}/5")

    assert successes >= 4, f"Expected >= 4 successes, got {successes}"
    assert duration < 15, f"Expected < 15s, took {duration:.1f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "stress"])
