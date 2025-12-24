# -*- coding: utf-8 -*-
"""
Memory Profiling for BrainStorm Integration.

Provides:
- tracemalloc-based memory tracking
- Memory leak detection
- Workflow memory profiling
- 50-workflow stress test for leak detection

Usage:
    python -m pytest Integration/tests/load/memory_profile.py -v
"""

import asyncio
import gc
import logging
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pytest
import pytest_asyncio

from .conftest import collect_txt_files, LoadTestMetrics

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    current_mb: float
    peak_mb: float
    traced_blocks: int
    timestamp: float = 0.0
    label: str = ""

    def __str__(self) -> str:
        return f"{self.label}: current={self.current_mb:.2f}MB, peak={self.peak_mb:.2f}MB"


@dataclass
class MemoryProfile:
    """Full memory profile for a workflow."""
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    baseline_mb: float = 0.0
    final_mb: float = 0.0
    peak_mb: float = 0.0
    leaked_mb: float = 0.0
    top_allocators: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def growth_ratio(self) -> float:
        """Memory growth ratio (final / baseline)."""
        if self.baseline_mb > 0:
            return self.final_mb / self.baseline_mb
        return 1.0

    @property
    def is_leak_detected(self) -> bool:
        """True if memory grew > 2x baseline after gc."""
        return self.growth_ratio > 2.0

    def to_dict(self) -> dict:
        return {
            "baseline_mb": round(self.baseline_mb, 2),
            "final_mb": round(self.final_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "leaked_mb": round(self.leaked_mb, 2),
            "growth_ratio": round(self.growth_ratio, 2),
            "leak_detected": self.is_leak_detected,
            "snapshots_count": len(self.snapshots),
        }


class MemoryProfiler:
    """
    Memory profiler using tracemalloc.

    Example:
        profiler = MemoryProfiler()
        profiler.start()

        # Run your workflow
        await do_something()

        profile = profiler.stop()
        print(f"Peak memory: {profile.peak_mb:.2f}MB")
    """

    def __init__(self, nframes: int = 10):
        """
        Initialize profiler.

        Args:
            nframes: Stack frames to capture per allocation
        """
        self.nframes = nframes
        self._started = False
        self._snapshots: List[MemorySnapshot] = []
        self._baseline_snapshot: Optional[tracemalloc.Snapshot] = None

    def start(self) -> None:
        """Start memory tracing."""
        if self._started:
            return

        # Force GC before baseline
        gc.collect()
        gc.collect()

        tracemalloc.start(self.nframes)
        self._started = True
        self._snapshots = []
        self._baseline_snapshot = tracemalloc.take_snapshot()

        # Record baseline
        current, peak = tracemalloc.get_traced_memory()
        self._snapshots.append(MemorySnapshot(
            current_mb=current / 1024 / 1024,
            peak_mb=peak / 1024 / 1024,
            traced_blocks=len(self._baseline_snapshot.statistics("lineno")),
            label="baseline",
        ))

    def snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        if not self._started:
            raise RuntimeError("Profiler not started")

        current, peak = tracemalloc.get_traced_memory()
        snap = MemorySnapshot(
            current_mb=current / 1024 / 1024,
            peak_mb=peak / 1024 / 1024,
            traced_blocks=len(tracemalloc.take_snapshot().statistics("lineno")),
            label=label,
        )
        self._snapshots.append(snap)
        return snap

    def stop(self, top_n: int = 10) -> MemoryProfile:
        """
        Stop tracing and return profile.

        Args:
            top_n: Number of top allocators to include

        Returns:
            Complete memory profile
        """
        if not self._started:
            return MemoryProfile()

        # Force GC before final measurement
        gc.collect()
        gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        final_snapshot = tracemalloc.take_snapshot()

        # Record final
        self._snapshots.append(MemorySnapshot(
            current_mb=current / 1024 / 1024,
            peak_mb=peak / 1024 / 1024,
            traced_blocks=len(final_snapshot.statistics("lineno")),
            label="final",
        ))

        # Get top allocators
        top_allocators = []
        if self._baseline_snapshot:
            try:
                stats = final_snapshot.compare_to(self._baseline_snapshot, "lineno")
                for stat in stats[:top_n]:
                    size_mb = stat.size / 1024 / 1024
                    location = str(stat.traceback)[:100]
                    top_allocators.append((location, size_mb))
            except Exception as e:
                logger.warning(f"Failed to compare snapshots: {e}")

        tracemalloc.stop()
        self._started = False

        baseline_mb = self._snapshots[0].current_mb if self._snapshots else 0
        final_mb = current / 1024 / 1024

        return MemoryProfile(
            snapshots=self._snapshots.copy(),
            baseline_mb=baseline_mb,
            final_mb=final_mb,
            peak_mb=peak / 1024 / 1024,
            leaked_mb=max(0, final_mb - baseline_mb),
            top_allocators=top_allocators,
        )


async def profile_workflow(
    workflow_fn: Callable,
    iterations: int = 1,
    gc_between: bool = True,
) -> MemoryProfile:
    """
    Profile a workflow function.

    Args:
        workflow_fn: Async function to profile
        iterations: Number of times to run
        gc_between: Force GC between iterations

    Returns:
        Memory profile
    """
    profiler = MemoryProfiler()
    profiler.start()

    for i in range(iterations):
        await workflow_fn()

        if gc_between:
            gc.collect()

        profiler.snapshot(f"iter_{i+1}")

    return profiler.stop()


# =============================================================================
# Tests
# =============================================================================


@pytest.fixture
def profiler() -> MemoryProfiler:
    """Fresh profiler for each test."""
    return MemoryProfiler()


class TestMemoryProfiler:
    """Unit tests for MemoryProfiler."""

    def test_start_stop(self, profiler: MemoryProfiler):
        """Profiler starts and stops cleanly."""
        profiler.start()
        profile = profiler.stop()

        assert profile.baseline_mb >= 0
        assert profile.peak_mb >= profile.baseline_mb
        assert len(profile.snapshots) >= 2  # baseline + final

    def test_snapshot(self, profiler: MemoryProfiler):
        """Can take intermediate snapshots."""
        profiler.start()

        snap1 = profiler.snapshot("step_1")
        assert snap1.label == "step_1"

        snap2 = profiler.snapshot("step_2")
        assert snap2.label == "step_2"

        profile = profiler.stop()
        assert len(profile.snapshots) >= 4  # baseline + 2 snaps + final

    def test_growth_ratio(self):
        """Growth ratio calculated correctly."""
        profile = MemoryProfile(baseline_mb=10.0, final_mb=15.0)
        assert profile.growth_ratio == 1.5

        profile2 = MemoryProfile(baseline_mb=10.0, final_mb=25.0)
        assert profile2.growth_ratio == 2.5
        assert profile2.is_leak_detected


@pytest.mark.asyncio
class TestMemoryLeakDetection:
    """Memory leak detection tests."""

    async def test_no_leak_simple_operation(self, profiler: MemoryProfiler):
        """Simple operations should not leak."""
        profiler.start()

        # Simple CPU-bound operation
        for _ in range(100):
            data = [i * 2 for i in range(1000)]
            del data

        gc.collect()
        profile = profiler.stop()

        assert profile.growth_ratio < 1.5, f"Unexpected growth: {profile.growth_ratio:.2f}x"

    async def test_no_leak_async_operations(self, profiler: MemoryProfiler):
        """Async operations should not leak."""
        profiler.start()

        async def dummy_task(n: int) -> int:
            await asyncio.sleep(0.001)
            return n * 2

        # Run many async tasks
        for _ in range(10):
            tasks = [dummy_task(i) for i in range(100)]
            await asyncio.gather(*tasks)

        gc.collect()
        profile = profiler.stop()

        # Use absolute threshold when baseline is very small (< 1MB)
        # asyncio initialization allocates ~0.1MB which looks like huge ratio
        if profile.baseline_mb < 1.0:
            assert profile.final_mb < 5.0, f"Memory too high: {profile.final_mb:.2f}MB"
        else:
            assert profile.growth_ratio < 1.5, f"Unexpected growth: {profile.growth_ratio:.2f}x"


@pytest.mark.slow
@pytest.mark.asyncio
class TestWorkflowMemoryProfile:
    """Profile actual workflows for memory usage."""

    async def test_recall_memory_profile(self, profiler: MemoryProfiler):
        """
        Profile memory_recall operations.

        Requires BraineMemory server running on port 8001.
        """
        pytest.importorskip("mcp")

        from integration.mcp_client.sse_session import connect_sse

        profiler.start()

        try:
            async with connect_sse("http://localhost:8001/sse", timeout=30) as session:
                profiler.snapshot("connected")

                # Run 10 recall operations
                for i in range(10):
                    result = await session.call_tool(
                        "memory_recall",
                        {"query": f"test query {i}", "limit": 5},
                    )
                    profiler.snapshot(f"recall_{i}")

                profiler.snapshot("done_recalls")

        except Exception as e:
            pytest.skip(f"BraineMemory server not available: {e}")

        profile = profiler.stop()

        logger.info(f"Recall profile: {profile.to_dict()}")
        assert profile.peak_mb < 500, f"Peak memory too high: {profile.peak_mb:.2f}MB"


@pytest.mark.stress
@pytest.mark.slow
@pytest.mark.asyncio
class TestMemoryLeakStress:
    """
    Stress tests for memory leak detection.

    Run 50 workflows and check memory doesn't grow > 2x.
    """

    async def test_50_workflow_no_leak(self):
        """
        Run 50 sequential workflows, assert no memory leak.

        Target: peak memory < 2x baseline after 50 runs.
        """
        pytest.importorskip("mcp")

        from integration.mcp_client.sse_session import connect_sse

        profiler = MemoryProfiler()
        profiler.start()

        try:
            async with connect_sse("http://localhost:8001/sse", timeout=30) as session:
                profiler.snapshot("connected")

                for i in range(50):
                    # Simulate a mini-workflow
                    await session.call_tool(
                        "memory_recall",
                        {"query": f"stress test query {i}", "limit": 3},
                    )

                    # Force GC every 10 iterations
                    if (i + 1) % 10 == 0:
                        gc.collect()
                        profiler.snapshot(f"after_{i+1}_workflows")

                profiler.snapshot("done_50")

        except Exception as e:
            pytest.skip(f"BraineMemory server not available: {e}")

        profile = profiler.stop()

        logger.info(f"50-workflow profile: {profile.to_dict()}")

        # Check memory stability after initial connection
        # Compare "connected" to "done_50" - should not grow more than 2x
        connected_snap = next((s for s in profile.snapshots if s.label == "connected"), None)
        done_snap = next((s for s in profile.snapshots if s.label == "done_50"), None)

        if connected_snap and done_snap:
            workflow_growth = done_snap.current_mb / max(connected_snap.current_mb, 0.1)
            assert workflow_growth < 2.0, (
                f"Memory leak during workflows: {connected_snap.current_mb:.2f}MB → "
                f"{done_snap.current_mb:.2f}MB ({workflow_growth:.2f}x)"
            )
        else:
            # Fallback: absolute threshold
            assert profile.peak_mb < 50.0, f"Peak memory too high: {profile.peak_mb:.2f}MB"

    async def test_entity_cache_no_leak(self):
        """
        Test entity cache doesn't grow unbounded.

        Seed data, query repeatedly, check cache size stable.
        """
        pytest.importorskip("mcp")

        from integration.mcp_client.sse_session import connect_sse

        profiler = MemoryProfiler()
        profiler.start()

        try:
            async with connect_sse("http://localhost:8001/sse", timeout=30) as session:
                profiler.snapshot("connected")

                # Query graph 50 times
                for i in range(50):
                    await session.call_tool(
                        "memory_recall_graph",
                        {"query": f"entity cache test {i}", "max_hops": 1},
                    )

                    if (i + 1) % 10 == 0:
                        gc.collect()
                        profiler.snapshot(f"after_{i+1}_graph_queries")

        except Exception as e:
            pytest.skip(f"BraineMemory server not available: {e}")

        profile = profiler.stop()

        logger.info(f"Entity cache profile: {profile.to_dict()}")

        # Check memory stability after initial connection
        connected_snap = next((s for s in profile.snapshots if s.label == "connected"), None)
        last_query_snap = next((s for s in profile.snapshots if "50" in s.label), None)

        if connected_snap and last_query_snap:
            cache_growth = last_query_snap.current_mb / max(connected_snap.current_mb, 0.1)
            assert cache_growth < 2.0, (
                f"Entity cache leak: {connected_snap.current_mb:.2f}MB → "
                f"{last_query_snap.current_mb:.2f}MB ({cache_growth:.2f}x)"
            )
        else:
            # Fallback: absolute threshold
            assert profile.peak_mb < 20.0, f"Peak memory too high: {profile.peak_mb:.2f}MB"


def run_profile(doc_limit: int = 10) -> MemoryProfile:
    """
    CLI entry point for quick profiling.

    Usage:
        python -c "from integration.tests.load.memory_profile import run_profile; run_profile(10)"
    """
    import asyncio

    async def main():
        from integration.mcp_client.sse_session import connect_sse

        profiler = MemoryProfiler()
        profiler.start()

        async with connect_sse("http://localhost:8001/sse", timeout=30) as session:
            for i in range(doc_limit):
                await session.call_tool(
                    "memory_recall",
                    {"query": f"test {i}", "limit": 5},
                )
                if (i + 1) % 5 == 0:
                    gc.collect()
                    profiler.snapshot(f"after_{i+1}")

        return profiler.stop()

    profile = asyncio.run(main())
    print(f"\n=== Memory Profile ({doc_limit} operations) ===")
    print(f"Baseline:     {profile.baseline_mb:.2f} MB")
    print(f"Peak:         {profile.peak_mb:.2f} MB")
    print(f"Final:        {profile.final_mb:.2f} MB")
    print(f"Growth ratio: {profile.growth_ratio:.2f}x")
    print(f"Leak detected: {profile.is_leak_detected}")

    if profile.top_allocators:
        print("\nTop allocators:")
        for loc, size in profile.top_allocators[:5]:
            print(f"  {size:.2f}MB: {loc}")

    return profile
