# -*- coding: utf-8 -*-
"""
BrainStorm Orchestrator - High-level API for combined BraineMemory + LegalOps workflows.

Provides 6 main scenarios:
1. ingest_to_memory - Ingest documents to both LegalOps and BraineMemory
2. analyze_docset - Analyze documents for contradictions and entities
3. legal_answer - RAG Q&A with context packing and evidence
4. build_case_brief - Build comprehensive case brief with timeline, parties, claims
5. evidence_matrix - Build evidence matrix for trial preparation
6. prepare_hearing - One-button hearing preparation with integrity chain

Uses SSE transport for both MCP servers (production-ready).
Each scenario maintains a single trace_id for end-to-end correlation.

Usage:
    async with BrainStormOrchestrator() as orch:
        # Scenario 1: Ingest PDF
        result = await orch.ingest_to_memory("contract.pdf", project_id="proj:1")

        # Scenario 2: Analyze documents
        result = await orch.analyze_docset(project_id="proj:1", goal="Find inconsistencies")

        # Scenario 3: Legal Q&A
        result = await orch.legal_answer("What are the key terms?", project_id="proj:1")

        # Scenario 4: Case Brief
        result = await orch.build_case_brief(project_id="proj:1")

        # Scenario 5: Evidence Matrix
        result = await orch.evidence_matrix(project_id="proj:1", claims=["Defendant breached contract"])

        # Scenario 6: One-Button Hearing Prep
        result = await orch.prepare_hearing(
            case_id="case:1",
            project_id="proj:1",
            questions=["What is the main breach?"],
        )
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from integration.config import config
from integration.mcp_client import BraineMemoryClient, LegalOpsClient

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """
    Result of an orchestrator workflow.

    Provides unified envelope format for all scenarios:
    - ok: boolean success indicator
    - workflow: scenario name
    - trace_id: correlation ID
    - project_id/case_id: context IDs
    - inputs: original parameters
    - outputs: main result data
    - artifacts: generated files/exports
    - action_plan: P0/P1/P2 task list
    - warnings: non-fatal issues
    - timings_ms: step-level timing
    """

    success: bool
    trace_id: str
    scenario: str = ""
    project_id: str = ""
    case_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    _start_time: float = field(default=0.0, repr=False)
    _step_timings: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Initialize start time for timing."""
        import time
        if self._start_time == 0.0:
            object.__setattr__(self, '_start_time', time.time())

    def add_step(
        self,
        name: str,
        result: Dict[str, Any],
        success: bool = True,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Add a step to the workflow log."""
        import time

        step = {
            "name": name,
            "success": success,
            "trace_id": result.get("_trace_id", self.trace_id),
        }
        if duration_ms:
            step["duration_ms"] = duration_ms
            self._step_timings[name] = duration_ms
        self.steps.append(step)

        if not success:
            error = result.get("error", str(result.get("message", "Unknown error")))
            self.errors.append(f"{name}: {error}")

    def start_timer(self, step_name: str) -> float:
        """Start timer for a step. Returns start time."""
        import time
        return time.time()

    def end_timer(self, step_name: str, start_time: float) -> float:
        """End timer for a step. Returns elapsed ms."""
        import time
        elapsed_ms = (time.time() - start_time) * 1000
        self._step_timings[step_name] = elapsed_ms
        return elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dictionary format (backwards compatible)."""
        return {
            "success": self.success,
            "trace_id": self.trace_id,
            "scenario": self.scenario,
            "data": self.data,
            "errors": self.errors,
            "steps_count": len(self.steps),
            "steps": self.steps,
        }

    def to_envelope(self) -> Dict[str, Any]:
        """
        Convert to unified envelope format.

        Standard output format for all scenarios:
        - ok: boolean success
        - workflow: scenario name
        - trace_id: correlation ID
        - project_id/case_id: context
        - inputs: original parameters
        - outputs: main result data (filtered from internal fields)
        - artifacts: generated files
        - action_plan: P0/P1/P2 tasks
        - warnings: non-fatal issues
        - timings_ms: step timing
        """
        import time

        # Calculate total elapsed time
        total_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0

        # Extract outputs (main data without internal fields)
        internal_keys = {"_trace_id", "_export_profile", "_note", "steps"}
        outputs = {k: v for k, v in self.data.items()
                   if not k.startswith("_") and k not in internal_keys}

        # Extract artifacts if present
        artifacts = {}
        for key in ["artifacts", "integrity", "export_bundle"]:
            if key in self.data:
                artifacts[key] = self.data[key]
                outputs.pop(key, None)

        # Extract action_plan and warnings
        action_plan = self.data.get("action_plan", [])
        warnings = self.data.get("warnings", [])

        # Remove from outputs to avoid duplication
        outputs.pop("action_plan", None)
        outputs.pop("warnings", None)

        return {
            "ok": self.success and len(self.errors) == 0,
            "workflow": self.scenario,
            "trace_id": self.trace_id,
            "project_id": self.project_id or self.data.get("project_id", ""),
            "case_id": self.case_id or self.data.get("case_id", ""),
            "inputs": self.inputs,
            "outputs": outputs,
            "artifacts": artifacts,
            "action_plan": action_plan,
            "warnings": warnings,
            "errors": self.errors,
            "timings_ms": {
                "total": round(total_ms, 1),
                "steps": {k: round(v, 1) for k, v in self._step_timings.items()},
            },
            "meta": {
                "steps_count": len(self.steps),
                "steps_succeeded": len([s for s in self.steps if s.get("success", True)]),
            },
        }


class BrainStormOrchestrator:
    """
    High-level orchestrator for BraineMemory + LegalOps workflows.

    Provides unified API for complex multi-step operations that span
    both memory and legal document analysis. Uses SSE transport.

    Attributes:
        braine: BraineMemoryClient for memory operations
        legal: LegalOpsClient for legal document operations
    """

    def __init__(
        self,
        braine_url: Optional[str] = None,
        legal_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            braine_url: BraineMemory SSE URL (default from config)
            legal_url: LegalOps SSE URL (default from config)
            timeout: Connection timeout in seconds
        """
        self.braine_url = braine_url or config.braine_sse_url
        self.legal_url = legal_url or config.legal_sse_url
        self.timeout = timeout or int(config.mcp_timeout)

        self._braine: Optional[BraineMemoryClient] = None
        self._legal: Optional[LegalOpsClient] = None

    async def __aenter__(self) -> "BrainStormOrchestrator":
        """Connect to both MCP servers in parallel."""
        self._braine = BraineMemoryClient(url=self.braine_url, timeout=self.timeout)
        self._legal = LegalOpsClient(url=self.legal_url, timeout=self.timeout)

        # Connect in parallel
        await asyncio.gather(
            self._braine.__aenter__(),
            self._legal.__aenter__(),
        )

        logger.info(
            f"Orchestrator connected: "
            f"BraineMemory ({len(self._braine.session.tools_cache)} tools), "
            f"LegalOps ({len(self._legal.session.tools_cache)} tools)"
        )
        return self

    async def __aexit__(self, *args) -> None:
        """Disconnect from both MCP servers."""
        errors = []

        if self._braine:
            try:
                await self._braine.__aexit__(*args)
            except Exception as e:
                errors.append(f"BraineMemory: {e}")

        if self._legal:
            try:
                await self._legal.__aexit__(*args)
            except Exception as e:
                errors.append(f"LegalOps: {e}")

        if errors:
            logger.warning(f"Disconnect errors: {errors}")

        self._braine = None
        self._legal = None
        logger.debug("Orchestrator disconnected")

    @property
    def braine(self) -> BraineMemoryClient:
        """Get BraineMemory client."""
        if not self._braine:
            raise RuntimeError("Orchestrator not connected. Use 'async with' context.")
        return self._braine

    @property
    def legal(self) -> LegalOpsClient:
        """Get LegalOps client."""
        if not self._legal:
            raise RuntimeError("Orchestrator not connected. Use 'async with' context.")
        return self._legal

    def _new_trace_id(self) -> str:
        """Generate a new trace ID for a workflow."""
        return str(uuid.uuid4())[:8]

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 1: Ingest to Memory
    # ─────────────────────────────────────────────────────────────────────────

    async def ingest_to_memory(
        self,
        pdf_path: str | Path,
        project_id: Optional[str] = None,
        extract_entities: bool = True,
        store_in_braine: bool = True,
        trace_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Ingest a PDF document to LegalOps and optionally to BraineMemory.

        Workflow:
        1. Ingest PDF to LegalOps (document processing, chunking)
        2. Extract entities from document
        3. Store document content in BraineMemory (for cross-project recall)

        Args:
            pdf_path: Path to PDF file
            project_id: LegalOps project ID
            extract_entities: Whether to extract entities
            store_in_braine: Also store in BraineMemory
            trace_id: Optional trace ID (generated if not provided)

        Returns:
            WorkflowResult with document_id, entities, memory_asset_id
        """
        trace_id = trace_id or self._new_trace_id()
        result = WorkflowResult(success=True, trace_id=trace_id, scenario="ingest_to_memory")

        logger.info(f"[{trace_id}] ingest_to_memory: {pdf_path}")

        try:
            # Step 1: Ingest PDF to LegalOps
            ingest_result = await self.legal.ingest_pdf(
                pdf_path=str(pdf_path),
                project_id=project_id,
            )
            result.add_step("legal.ingest_pdf", ingest_result)

            document_id = ingest_result.get("document_id") or ingest_result.get("id")
            if not document_id:
                result.success = False
                result.errors.append("No document_id returned")
                return result

            result.data["document_id"] = document_id
            result.data["title"] = ingest_result.get("title", "")
            result.data["chunks_count"] = ingest_result.get("chunks_count", 0)

            logger.info(f"[{trace_id}] Ingested: {document_id}")

            # Step 2: Extract entities (parallel with memory storage possible)
            entities_task = None
            memory_task = None

            if extract_entities:
                entities_task = self._extract_entities(document_id, result)

            if store_in_braine:
                memory_task = self._store_in_memory(
                    document_id, pdf_path, project_id, result
                )

            # Run in parallel
            await asyncio.gather(
                entities_task if entities_task else asyncio.sleep(0),
                memory_task if memory_task else asyncio.sleep(0),
                return_exceptions=True,
            )

        except Exception as e:
            logger.error(f"[{trace_id}] ingest_to_memory failed: {e}")
            result.success = False
            result.errors.append(str(e))

        logger.info(f"[{trace_id}] ingest_to_memory: success={result.success}")
        return result

    async def _extract_entities(
        self,
        document_id: str,
        result: WorkflowResult,
    ) -> None:
        """Extract entities from document (helper)."""
        try:
            entities_result = await self.legal.extract_entities(document_id=document_id)
            result.add_step("legal.extract_entities", entities_result)

            entities = entities_result.get("entities", [])
            result.data["entities_count"] = len(entities)
            result.data["entities"] = entities[:10]  # Summary

            logger.info(f"[{result.trace_id}] Extracted {len(entities)} entities")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Entity extraction failed: {e}")
            result.add_step("legal.extract_entities", {"error": str(e)}, success=False)

    async def _store_in_memory(
        self,
        document_id: str,
        pdf_path: str | Path,
        project_id: Optional[str],
        result: WorkflowResult,
    ) -> None:
        """Store document in BraineMemory (helper)."""
        try:
            # Get document content
            doc_result = await self.legal.get_document(document_id=document_id)
            content = doc_result.get("content", "") or doc_result.get("text", "")

            if not content:
                logger.warning(f"[{result.trace_id}] No content to store in memory")
                return

            memory_result = await self.braine.memory_ingest(
                content=content,
                type="document",
                source_url=str(pdf_path),
                metadata={
                    "legal_doc_id": document_id,
                    "project_id": project_id,
                    "title": result.data.get("title", ""),
                },
                extract_entities=True,
            )
            result.add_step("braine.memory_ingest", memory_result)

            result.data["memory_asset_id"] = memory_result.get("asset_id")
            logger.info(f"[{result.trace_id}] Stored in memory: {result.data['memory_asset_id']}")

        except Exception as e:
            logger.warning(f"[{result.trace_id}] Memory storage failed: {e}")
            result.add_step("braine.memory_ingest", {"error": str(e)}, success=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 2: Analyze Document Set
    # ─────────────────────────────────────────────────────────────────────────

    async def analyze_docset(
        self,
        project_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        goal: Optional[str] = None,
        include_contradictions: bool = True,
        include_timeline: bool = True,
        include_entity_graph: bool = True,
        context_budget: int = 4000,
        trace_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Analyze a set of documents for contradictions, timeline, and entities.

        Workflow:
        1. Detect contradictions between documents
        2. Build timeline from documents
        3. Get entity graph (via BraineMemory GraphRAG)
        4. Pack context for analysis goal

        Args:
            project_id: LegalOps project ID
            doc_ids: Specific document IDs to analyze
            goal: Analysis goal for context packing
            include_contradictions: Detect contradictions
            include_timeline: Build timeline
            include_entity_graph: Get entity relationships
            context_budget: Token budget for context
            trace_id: Optional trace ID

        Returns:
            WorkflowResult with contradictions, timeline, entities, context
        """
        trace_id = trace_id or self._new_trace_id()
        result = WorkflowResult(success=True, trace_id=trace_id, scenario="analyze_docset")

        logger.info(f"[{trace_id}] analyze_docset: project={project_id}, docs={doc_ids}")

        try:
            # Parallel analysis
            tasks = []

            if include_contradictions:
                tasks.append(("contradictions", self._detect_contradictions(doc_ids, project_id, result)))

            if include_timeline and project_id:
                tasks.append(("timeline", self._get_timeline(project_id, result)))

            if include_entity_graph:
                tasks.append(("graph", self._get_entity_graph(goal or "all entities", result)))

            # Run in parallel
            if tasks:
                await asyncio.gather(
                    *(t[1] for t in tasks),
                    return_exceptions=True,
                )

            # Context packing (after analysis to include results)
            if goal:
                await self._pack_context(goal, context_budget, result)

        except Exception as e:
            logger.error(f"[{trace_id}] analyze_docset failed: {e}")
            result.success = False
            result.errors.append(str(e))

        # Partial success is OK
        if result.errors and len(result.errors) < len(result.steps):
            result.success = True  # At least some steps succeeded

        logger.info(f"[{trace_id}] analyze_docset: success={result.success}")
        return result

    async def _detect_contradictions(
        self,
        doc_ids: Optional[List[str]],
        project_id: Optional[str],
        result: WorkflowResult,
    ) -> None:
        """Detect contradictions (helper)."""
        try:
            contr_result = await self.legal.detect_contradictions(
                document_ids=doc_ids,
                project_id=project_id,
            )
            result.add_step("legal.detect_contradictions", contr_result)

            contradictions = contr_result.get("contradictions", [])
            result.data["contradictions"] = contradictions
            result.data["contradictions_count"] = len(contradictions)

            # By severity
            by_severity = {}
            for c in contradictions:
                sev = c.get("severity", "unknown")
                by_severity[sev] = by_severity.get(sev, 0) + 1
            result.data["contradictions_by_severity"] = by_severity

            logger.info(f"[{result.trace_id}] Contradictions: {len(contradictions)}")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Contradiction detection failed: {e}")
            result.add_step("legal.detect_contradictions", {"error": str(e)}, success=False)

    async def _get_timeline(
        self,
        project_id: str,
        result: WorkflowResult,
    ) -> None:
        """Get timeline (helper)."""
        try:
            timeline_result = await self.legal.get_timeline(case_id=project_id)
            result.add_step("legal.get_timeline", timeline_result)

            events = timeline_result.get("events", [])
            result.data["timeline"] = events
            result.data["timeline_count"] = len(events)

            logger.info(f"[{result.trace_id}] Timeline: {len(events)} events")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Timeline failed: {e}")
            result.add_step("legal.get_timeline", {"error": str(e)}, success=False)

    async def _get_entity_graph(
        self,
        query: str,
        result: WorkflowResult,
    ) -> None:
        """Get entity graph via BraineMemory GraphRAG (helper)."""
        try:
            graph_result = await self.braine.memory_recall_graph(
                query=query,
                mode="local",
                limit=50,
                include_entities=True,
                include_relations=True,
                include_communities=False,
            )
            result.add_step("braine.memory_recall_graph", graph_result)

            entities = graph_result.get("entities", [])
            relations = graph_result.get("relations", [])

            result.data["entities"] = entities[:20]
            result.data["relations"] = relations[:20]
            result.data["entities_count"] = len(entities)
            result.data["relations_count"] = len(relations)

            logger.info(f"[{result.trace_id}] Graph: {len(entities)} entities, {len(relations)} relations")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Entity graph failed: {e}")
            result.add_step("braine.memory_recall_graph", {"error": str(e)}, success=False)

    async def _pack_context(
        self,
        goal: str,
        token_budget: int,
        result: WorkflowResult,
        user_id: Optional[str] = None,
    ) -> None:
        """Pack context from BraineMemory (helper)."""
        try:
            context_result = await self.braine.memory_context_pack(
                goal=goal,
                token_budget=token_budget,
                audience="lawyer",
                style="structured",
                include_sources=True,
                user_id=user_id,
            )
            result.add_step("braine.memory_context_pack", context_result)

            result.data["context"] = context_result.get("content", "")
            result.data["context_sources"] = context_result.get("sources", [])
            result.data["context_tokens"] = context_result.get("token_count", 0)

            logger.info(f"[{result.trace_id}] Context: {result.data['context_tokens']} tokens")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Context packing failed: {e}")
            result.add_step("braine.memory_context_pack", {"error": str(e)}, success=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 3: Legal Answer with Evidence
    # ─────────────────────────────────────────────────────────────────────────

    async def legal_answer(
        self,
        question: str,
        project_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        include_evidence: bool = True,
        check_contradictions: bool = True,
        context_budget: int = 4000,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Answer a legal question with evidence and contradiction checking.

        Workflow:
        1. Smart search for relevant documents
        2. Pack context from BraineMemory (with personalization)
        3. Find evidence for the question
        4. Check contradictions in sources
        5. Generate answer with citations

        Args:
            question: Legal question to answer
            project_id: LegalOps project ID
            doc_ids: Limit to specific documents
            include_evidence: Find supporting evidence
            check_contradictions: Check for contradictions in sources
            context_budget: Token budget for context
            user_id: User ID for personalization
            trace_id: Optional trace ID

        Returns:
            WorkflowResult with answer, sources, evidence, contradictions
        """
        trace_id = trace_id or self._new_trace_id()
        result = WorkflowResult(success=True, trace_id=trace_id, scenario="legal_answer")

        logger.info(f"[{trace_id}] legal_answer: {question[:50]}...")

        try:
            # Step 1: Smart search
            search_result = await self.legal.smart_search(
                query=question,
                project_id=project_id,
            )
            result.add_step("legal.smart_search", search_result)

            search_docs = search_result.get("documents", search_result.get("results", []))
            result.data["search_results_count"] = len(search_docs)

            logger.info(f"[{trace_id}] Found {len(search_docs)} relevant docs")

            # Parallel: context + evidence
            tasks = [
                self._pack_context(
                    f"Answer legal question: {question}",
                    context_budget,
                    result,
                    user_id,
                ),
            ]

            if include_evidence:
                tasks.append(self._find_evidence(question, project_id, result))

            await asyncio.gather(*tasks, return_exceptions=True)

            # Step 4: Check contradictions in sources
            if check_contradictions and search_docs:
                source_ids = [d.get("id") or d.get("document_id") for d in search_docs[:5]]
                source_ids = [sid for sid in source_ids if sid]

                if source_ids:
                    try:
                        contr_result = await self.legal.detect_contradictions(
                            document_ids=source_ids,
                        )
                        result.add_step("legal.detect_contradictions", contr_result)

                        contradictions = contr_result.get("contradictions", [])
                        result.data["source_contradictions"] = contradictions

                        if contradictions:
                            result.data["warning"] = (
                                f"Found {len(contradictions)} contradictions in sources - "
                                "verify answer carefully"
                            )
                            logger.warning(f"[{trace_id}] Sources have contradictions!")
                    except Exception as e:
                        logger.warning(f"[{trace_id}] Contradiction check failed: {e}")

            # Step 5: Generate answer
            answer_result = await self.legal.ask_question(
                question=question,
                project_id=project_id,
            )
            result.add_step("legal.ask_question", answer_result)

            result.data["answer"] = answer_result.get("answer", "")
            result.data["sources"] = answer_result.get("sources", [])
            result.data["confidence"] = answer_result.get("confidence", 0)

            # Mark if context was enriched
            if result.data.get("context"):
                result.data["context_enriched"] = True

            logger.info(f"[{trace_id}] Answer generated: {len(result.data['sources'])} sources")

        except Exception as e:
            logger.error(f"[{trace_id}] legal_answer failed: {e}")
            result.success = False
            result.errors.append(str(e))

        logger.info(f"[{trace_id}] legal_answer: success={result.success}")
        return result

    async def _find_evidence(
        self,
        claim: str,
        project_id: Optional[str],
        result: WorkflowResult,
    ) -> None:
        """Find evidence for claim (helper)."""
        try:
            evidence_result = await self.legal.find_evidence(
                claim=claim,
                project_id=project_id,
            )
            result.add_step("legal.find_evidence", evidence_result)

            supporting = evidence_result.get("supporting", [])
            contradicting = evidence_result.get("contradicting", [])

            result.data["evidence_supporting"] = supporting
            result.data["evidence_contradicting"] = contradicting
            result.data["evidence_count"] = len(supporting) + len(contradicting)

            logger.info(
                f"[{result.trace_id}] Evidence: "
                f"{len(supporting)} supporting, {len(contradicting)} contradicting"
            )
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Evidence search failed: {e}")
            result.add_step("legal.find_evidence", {"error": str(e)}, success=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 4: Build Case Brief
    # ─────────────────────────────────────────────────────────────────────────

    async def build_case_brief(
        self,
        project_id: str,
        case_id: Optional[str] = None,
        include_timeline: bool = True,
        include_contradictions: bool = True,
        include_claims: bool = True,
        include_parties: bool = True,
        include_evidence_matrix: bool = True,
        include_action_plan: bool = True,
        evidence_max_claims: int = 10,
        export_profile: str = "work",
        auto_add_pieces: bool = True,
        deterministic: bool = True,
        summary_budget: int = 2000,
        max_doc_pairs: int = 0,
        trace_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Build a comprehensive case brief from all available documents.

        This is the "main button" for hearing preparation - combines all analyses
        into a court-ready brief with evidence matrix, red-flags, and action plan.

        Workflow:
        1. List all documents in project
        2. Get timeline of events
        3. Detect contradictions across documents
        4. Extract key claims from memory (GraphRAG)
        5. Search for parties/entities involved
        6. Build evidence matrix with red-flags
        7. Compute unified action_plan (P0/P1/P2)
        8. Track top_sources and missing_in_pieces
        9. Pack structured summary context
        10. (Optional) Export to case bundle

        Args:
            project_id: LegalOps project ID (required)
            case_id: Optional case ID for timeline and pieces
            include_timeline: Include chronological events
            include_contradictions: Detect and include contradictions
            include_claims: Extract key claims via BraineMemory
            include_parties: Identify parties and key entities
            include_evidence_matrix: Build evidence matrix with red-flags
            include_action_plan: Build unified action plan with priorities
            evidence_max_claims: Max claims for evidence matrix
            export_profile: "court" (concise) or "work" (full details)
            auto_add_pieces: Auto-add evidence docs to case if missing
            deterministic: Use deterministic sorting for reproducibility
            summary_budget: Token budget for summary context
            trace_id: Optional trace ID

        Returns:
            WorkflowResult with complete case brief:
            - documents: list of all documents
            - parties: identified parties and roles
            - timeline: chronological events
            - key_claims: important factual claims
            - contradictions: detected inconsistencies
            - evidence_matrix: claim evidence with red-flags
            - points_litigieux: contested claims needing attention
            - action_plan: unified task list with P0/P1/P2 priorities
            - top_sources: most cited documents
            - missing_in_pieces: evidence docs not yet in case
            - summary: structured case summary
            - brief_stats: comprehensive statistics
        """
        trace_id = trace_id or self._new_trace_id()
        result = WorkflowResult(success=True, trace_id=trace_id, scenario="build_case_brief")

        logger.info(f"[{trace_id}] build_case_brief: project={project_id}")

        try:
            # ─────────────────────────────────────────────────────────────────
            # Step 1: List all documents in project
            # ─────────────────────────────────────────────────────────────────
            docs_result = await self.legal.list_documents(
                project_id=project_id,
                limit=100,
            )
            result.add_step("legal.list_documents", docs_result)

            documents = docs_result.get("documents", docs_result.get("items", []))
            doc_ids = [d.get("id") or d.get("document_id") for d in documents]
            doc_ids = [did for did in doc_ids if did]

            result.data["documents"] = documents
            result.data["document_count"] = len(documents)
            result.data["export_profile"] = export_profile

            logger.info(f"[{trace_id}] Found {len(documents)} documents")

            if not documents:
                # Return skeleton brief with starter tasks instead of empty
                result.data["summary"] = "Projet vide - ajoutez des documents pour commencer l'analyse."
                result.data["warnings"] = [
                    "Projet vide: ajoutez des documents (ingest_pdf/ingest_text)",
                    "Evidence matrix non construite: aucune donnée disponible",
                ]
                result.data["action_plan"] = [
                    {
                        "priority": "P0",
                        "action": "ingest_contract",
                        "description": "Charger le contrat principal ou l'acte fondateur",
                        "hint": "Utilisez ingest_pdf pour le document clé du litige",
                    },
                    {
                        "priority": "P0",
                        "action": "ingest_correspondence",
                        "description": "Charger la correspondance pertinente (emails, lettres)",
                        "hint": "Les échanges avant/après le litige sont cruciaux",
                    },
                    {
                        "priority": "P1",
                        "action": "ingest_invoices",
                        "description": "Charger les factures et preuves de paiement",
                        "hint": "Nécessaire pour tout préjudice financier",
                    },
                    {
                        "priority": "P1",
                        "action": "ingest_expert_reports",
                        "description": "Charger les expertises et rapports techniques",
                        "hint": "Force probante élevée devant le tribunal",
                    },
                    {
                        "priority": "P2",
                        "action": "ingest_legal_refs",
                        "description": "Importer les références juridiques (jurisprudence, lois)",
                        "hint": "Utilisez import_legal_reference pour Legifrance/JudiLibre",
                    },
                ]
                result.data["evidence_matrix"] = []
                result.data["points_litigieux"] = []
                result.data["evidence_todo"] = []
                result.data["top_sources"] = []
                result.data["missing_in_pieces"] = []
                result.data["brief_stats"] = {
                    "document_count": 0,
                    "status": "empty_project",
                    "action_plan_count": 5,
                }
                logger.warning(f"[{trace_id}] Empty project - returning skeleton brief")
                return result

            # ─────────────────────────────────────────────────────────────────
            # Steps 2-5: Parallel analysis tasks
            # ─────────────────────────────────────────────────────────────────
            tasks = []

            # Timeline
            if include_timeline:
                tasks.append(self._brief_timeline(case_id or project_id, result))

            # Contradictions (with optional pair limit for large projects)
            if include_contradictions and len(doc_ids) >= 2:
                tasks.append(self._brief_contradictions(
                    doc_ids, project_id, result,
                    max_doc_pairs=max_doc_pairs,
                ))

            # Key claims (via BraineMemory)
            if include_claims:
                tasks.append(self._brief_claims(project_id, result))

            # Parties/entities
            if include_parties:
                tasks.append(self._brief_parties(project_id, result))

            # Run analysis in parallel
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # ─────────────────────────────────────────────────────────────────
            # Step 6: Build evidence matrix with red-flags
            # ─────────────────────────────────────────────────────────────────
            if include_evidence_matrix:
                await self._brief_evidence_matrix(
                    project_id=project_id,
                    case_id=case_id,
                    max_claims=evidence_max_claims,
                    result=result,
                )

            # ─────────────────────────────────────────────────────────────────
            # Step 7: Build unified action_plan with priorities (P0/P1/P2)
            # ─────────────────────────────────────────────────────────────────
            if include_action_plan:
                self._build_action_plan(result)

            # ─────────────────────────────────────────────────────────────────
            # Step 8: Track top_sources and missing_in_pieces
            # ─────────────────────────────────────────────────────────────────
            await self._track_sources_and_pieces(
                case_id=case_id,
                auto_add=auto_add_pieces,
                result=result,
            )

            # ─────────────────────────────────────────────────────────────────
            # Step 9: Build structured summary
            # ─────────────────────────────────────────────────────────────────
            await self._brief_summary(project_id, summary_budget, result)

            # ─────────────────────────────────────────────────────────────────
            # Step 10: Apply export profile filtering
            # ─────────────────────────────────────────────────────────────────
            if export_profile == "court":
                self._apply_court_profile(result)

            # ─────────────────────────────────────────────────────────────────
            # Compile brief metadata
            # ─────────────────────────────────────────────────────────────────
            em_summary = result.data.get("evidence_matrix_summary", {})
            rf_summary = result.data.get("red_flag_summary", {})

            result.data["brief_stats"] = {
                "document_count": result.data.get("document_count", 0),
                "timeline_events": result.data.get("timeline_count", 0),
                "contradictions_count": result.data.get("contradictions_count", 0),
                "claims_count": len(result.data.get("key_claims", [])),
                "parties_count": len(result.data.get("parties", [])),
                "has_critical_contradictions": any(
                    c.get("severity") in ("critical", "high")
                    for c in result.data.get("contradictions", [])
                ),
                # Evidence matrix stats
                "evidence_claims_analyzed": em_summary.get("total_claims", 0),
                "evidence_strong": len(result.data.get("strong_evidence", [])),
                "evidence_weak": len(result.data.get("weak_points", [])),
                "evidence_contested": len(result.data.get("contested_claims", [])),
                # Red-flag stats
                "red_flags_total": rf_summary.get("total", 0),
                "red_flags_high": rf_summary.get("high", 0),
                "red_flags_medium": rf_summary.get("medium", 0),
                # Action plan stats
                "action_plan_count": len(result.data.get("action_plan", [])),
                "action_plan_p0": len([a for a in result.data.get("action_plan", []) if a.get("priority") == "P0"]),
                # Source tracking
                "top_sources_count": len(result.data.get("top_sources", [])),
                "missing_pieces_count": len(result.data.get("missing_in_pieces", [])),
                "pieces_auto_added": result.data.get("pieces_auto_added", 0),
                # Export profile
                "export_profile": export_profile,
            }

            # Add warnings if needed
            if result.data["brief_stats"]["has_critical_contradictions"]:
                result.data["warnings"] = result.data.get("warnings", [])
                result.data["warnings"].append(
                    "Case has critical contradictions - review carefully"
                )

            # Warning for high-severity red-flags
            if rf_summary.get("high", 0) > 0:
                result.data["warnings"] = result.data.get("warnings", [])
                result.data["warnings"].append(
                    f"{rf_summary['high']} claims have high-severity evidence issues"
                )

            # Warning for P0 action items
            p0_count = result.data["brief_stats"]["action_plan_p0"]
            if p0_count > 0:
                result.data["warnings"] = result.data.get("warnings", [])
                result.data["warnings"].append(
                    f"{p0_count} critical actions required (P0 priority)"
                )

        except Exception as e:
            logger.error(f"[{trace_id}] build_case_brief failed: {e}")
            result.success = False
            result.errors.append(str(e))

        # Partial success is OK for briefs
        if result.errors and len(result.steps) > len(result.errors):
            result.success = True

        logger.info(
            f"[{trace_id}] build_case_brief: success={result.success}, "
            f"docs={result.data.get('document_count', 0)}"
        )
        return result

    async def _brief_timeline(
        self,
        case_id: str,
        result: WorkflowResult,
    ) -> None:
        """Get timeline for brief (helper)."""
        try:
            timeline_result = await self.legal.get_timeline(case_id=case_id)
            result.add_step("legal.get_timeline", timeline_result)

            events = timeline_result.get("events", [])

            # Sort by date if present
            events_sorted = sorted(
                events,
                key=lambda e: e.get("date", e.get("timestamp", "")),
            )

            result.data["timeline"] = events_sorted
            result.data["timeline_count"] = len(events_sorted)

            # Extract date range
            if events_sorted:
                result.data["timeline_range"] = {
                    "start": events_sorted[0].get("date", "unknown"),
                    "end": events_sorted[-1].get("date", "unknown"),
                }

            logger.info(f"[{result.trace_id}] Timeline: {len(events)} events")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Timeline failed: {e}")
            result.add_step("legal.get_timeline", {"error": str(e)}, success=False)
            result.data["timeline"] = []
            result.data["timeline_count"] = 0

    async def _brief_contradictions(
        self,
        doc_ids: List[str],
        project_id: str,
        result: WorkflowResult,
        max_doc_pairs: int = 0,
        top_sources: Optional[List[str]] = None,
    ) -> None:
        """
        Detect contradictions for brief (helper).

        Args:
            doc_ids: All document IDs in scope
            project_id: Project ID
            result: WorkflowResult to update
            max_doc_pairs: Max pairs to compare (0 = no limit)
            top_sources: Priority document IDs from evidence matrix
        """
        import math

        try:
            # Smart pair selection if max_doc_pairs > 0
            selected_doc_ids = doc_ids
            if max_doc_pairs > 0 and len(doc_ids) > 2:
                # Prioritize: top_sources first, then others
                priority_ids = set(top_sources or [])
                priority_list = [d for d in doc_ids if d in priority_ids]
                others = [d for d in doc_ids if d not in priority_ids]

                # Calculate max docs from max_pairs: n docs = n*(n-1)/2 pairs
                # Solve for n: n ~= sqrt(2*pairs)
                max_docs = max(3, int(math.sqrt(2 * max_doc_pairs) + 1))

                # Take priority first, fill with others
                selected_doc_ids = priority_list[:max_docs]
                remaining = max_docs - len(selected_doc_ids)
                if remaining > 0:
                    selected_doc_ids.extend(others[:remaining])

                actual_pairs = len(selected_doc_ids) * (len(selected_doc_ids) - 1) // 2
                logger.info(
                    f"[{result.trace_id}] Contradiction throttling: "
                    f"{len(doc_ids)} docs -> {len(selected_doc_ids)} docs "
                    f"({actual_pairs} pairs, limit={max_doc_pairs})"
                )

            contr_result = await self.legal.detect_contradictions(
                document_ids=selected_doc_ids,
                project_id=project_id,
            )
            result.add_step("legal.detect_contradictions", contr_result)

            contradictions = contr_result.get("contradictions", [])

            # Sort by severity
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            contradictions_sorted = sorted(
                contradictions,
                key=lambda c: severity_order.get(c.get("severity", "low"), 4),
            )

            result.data["contradictions"] = contradictions_sorted
            result.data["contradictions_count"] = len(contradictions_sorted)

            # Group by severity
            by_severity = {}
            for c in contradictions_sorted:
                sev = c.get("severity", "unknown")
                by_severity[sev] = by_severity.get(sev, 0) + 1
            result.data["contradictions_by_severity"] = by_severity

            logger.info(f"[{result.trace_id}] Contradictions: {len(contradictions)}")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Contradictions failed: {e}")
            result.add_step("legal.detect_contradictions", {"error": str(e)}, success=False)
            result.data["contradictions"] = []
            result.data["contradictions_count"] = 0

    async def _brief_claims(
        self,
        project_id: str,
        result: WorkflowResult,
    ) -> None:
        """Extract key claims via BraineMemory (helper)."""
        try:
            # Use memory_recall_claims for structured claim extraction
            claims_result = await self.braine.memory_recall_claims(
                query=f"key facts and claims for project {project_id}",
                limit=20,
                include_conflicting=True,
            )
            result.add_step("braine.memory_recall_claims", claims_result)

            claims = claims_result.get("claims", [])

            # Also try graph recall for entity-linked claims
            graph_result = await self.braine.memory_recall_graph(
                query="important claims and facts",
                mode="local",
                limit=15,
                include_entities=True,
                include_relations=True,
            )
            result.add_step("braine.memory_recall_graph", graph_result)

            # Merge claims from both sources
            graph_entities = graph_result.get("entities", [])

            result.data["key_claims"] = claims
            result.data["claim_entities"] = graph_entities[:10]

            logger.info(f"[{result.trace_id}] Claims: {len(claims)}, entities: {len(graph_entities)}")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Claims extraction failed: {e}")
            result.add_step("braine.memory_recall_claims", {"error": str(e)}, success=False)
            result.data["key_claims"] = []

    async def _brief_parties(
        self,
        project_id: str,
        result: WorkflowResult,
    ) -> None:
        """Identify parties and key entities (helper)."""
        try:
            # Search for person/organization entities
            parties_result = await self.legal.search_entities(
                query="parties plaintiff defendant claimant respondent",
                entity_type="person",
                project_id=project_id,
            )
            result.add_step("legal.search_entities[person]", parties_result)

            persons = parties_result.get("entities", [])

            # Also get organizations
            orgs_result = await self.legal.search_entities(
                query="company organization corporation",
                entity_type="organization",
                project_id=project_id,
            )
            result.add_step("legal.search_entities[org]", orgs_result)

            organizations = orgs_result.get("entities", [])

            # Combine and deduplicate
            all_parties = []
            seen_names = set()

            for entity in persons + organizations:
                name = entity.get("name", entity.get("text", ""))
                if name and name.lower() not in seen_names:
                    seen_names.add(name.lower())
                    all_parties.append({
                        "name": name,
                        "type": entity.get("type", "unknown"),
                        "role": entity.get("role", ""),
                        "mentions": entity.get("mention_count", entity.get("count", 1)),
                    })

            # Sort by mentions (most mentioned first)
            all_parties.sort(key=lambda p: p.get("mentions", 0), reverse=True)

            result.data["parties"] = all_parties[:15]  # Top 15

            logger.info(f"[{result.trace_id}] Parties: {len(all_parties)}")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Parties search failed: {e}")
            result.add_step("legal.search_entities", {"error": str(e)}, success=False)
            result.data["parties"] = []

    async def _brief_evidence_matrix(
        self,
        project_id: str,
        case_id: Optional[str],
        max_claims: int,
        result: WorkflowResult,
    ) -> None:
        """Build evidence matrix with red-flags for brief (helper)."""
        try:
            # Get claims from already-extracted key_claims if available
            claims = None
            key_claims = result.data.get("key_claims", [])
            if key_claims:
                claims = [
                    c.get("text", c.get("claim", str(c)))[:300]
                    for c in key_claims[:max_claims]
                    if c
                ]

            # Call evidence_matrix (internal - no nested context)
            em_result = await self._run_evidence_matrix_internal(
                project_id=project_id,
                claims=claims,
                max_claims=max_claims,
                export_case_id=case_id,
            )

            # Extract key data for brief
            if em_result:
                result.data["evidence_matrix"] = em_result.get("matrix", [])
                result.data["evidence_matrix_summary"] = em_result.get("evidence_summary", {})
                result.data["red_flag_summary"] = em_result.get("red_flag_summary", {})
                result.data["strong_evidence"] = em_result.get("strong_evidence", [])
                result.data["weak_points"] = em_result.get("weak_points", [])
                result.data["contested_claims"] = em_result.get("contested_claims", [])

                # Build "Points litigieux" section
                points_litigieux = []
                for claim in em_result.get("contested_claims", []):
                    points_litigieux.append({
                        "claim": claim.get("claim", "")[:150],
                        "index": claim.get("index"),
                        "issue": "contested_evidence",
                        "supporting": claim.get("supporting_count", 0),
                        "contradicting": claim.get("contradicting_count", 0),
                    })

                # Add high-severity red-flag claims
                for entry in em_result.get("matrix", []):
                    flags = entry.get("red_flags", [])
                    high_flags = [f for f in flags if f.get("severity") == "high"]
                    if high_flags and entry.get("index") not in [p.get("index") for p in points_litigieux]:
                        points_litigieux.append({
                            "claim": entry.get("claim", "")[:150],
                            "index": entry.get("index"),
                            "issue": high_flags[0].get("code", "RF-XX"),
                            "action": high_flags[0].get("action", ""),
                        })

                result.data["points_litigieux"] = points_litigieux[:10]

                # Build "Evidence To-Do" section from red-flags
                evidence_todo = []
                for entry in em_result.get("matrix", []):
                    for flag in entry.get("red_flags", []):
                        if flag.get("severity") in ("high", "medium"):
                            evidence_todo.append({
                                "claim_index": entry.get("index"),
                                "claim_excerpt": entry.get("claim", "")[:80],
                                "code": flag.get("code"),
                                "action": flag.get("action"),
                                "hint": flag.get("hint"),
                                "priority": "high" if flag.get("severity") == "high" else "medium",
                            })

                # Sort by priority and deduplicate
                evidence_todo.sort(key=lambda x: (0 if x["priority"] == "high" else 1, x.get("claim_index", 0)))
                result.data["evidence_todo"] = evidence_todo[:15]

                result.add_step("evidence_matrix", {
                    "claims_analyzed": len(em_result.get("matrix", [])),
                    "red_flags": em_result.get("red_flag_summary", {}).get("total", 0),
                })

                logger.info(
                    f"[{result.trace_id}] Evidence matrix: "
                    f"{len(em_result.get('matrix', []))} claims, "
                    f"{em_result.get('red_flag_summary', {}).get('total', 0)} red-flags"
                )
            else:
                result.data["evidence_matrix"] = []
                result.add_step("evidence_matrix", {"error": "No result"}, success=False)

        except Exception as e:
            logger.warning(f"[{result.trace_id}] Evidence matrix failed: {e}")
            result.add_step("evidence_matrix", {"error": str(e)}, success=False)
            result.data["evidence_matrix"] = []
            result.data["points_litigieux"] = []
            result.data["evidence_todo"] = []

    async def _run_evidence_matrix_internal(
        self,
        project_id: str,
        claims: Optional[List[str]],
        max_claims: int,
        export_case_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Run evidence matrix analysis without nested orchestrator context."""
        try:
            # Coverage analysis
            coverage_data = None
            if claims:
                claims_text = "\n\n".join(claims)
                try:
                    coverage_result = await self.legal.analyze_coverage(
                        text=claims_text,
                        project_id=project_id,
                        split_mode="paragraph",
                    )
                    coverage_data = {
                        "claims": coverage_result.get("claims", []),
                        "map": {},
                    }
                except Exception:
                    pass

            # Get claims if not provided
            if not claims:
                claims = await self._extract_claims_dual_channel(
                    project_id, max_claims,
                    WorkflowResult(success=True, trace_id="internal", scenario="em_internal")
                )

            if not claims:
                return {"matrix": [], "evidence_summary": {"total_claims": 0}, "red_flag_summary": {}}

            # Build matrix
            matrix = []
            for i, claim in enumerate(claims[:max_claims]):
                entry = await self._analyze_claim_evidence_v2(
                    claim=claim,
                    claim_index=i,
                    project_id=project_id,
                    include_contradictions=True,
                    include_citations=True,
                    citation_style="french_court",
                    coverage_data=coverage_data,
                    result=WorkflowResult(success=True, trace_id="internal", scenario="em_internal"),
                )
                matrix.append(entry)

            # Compute stats
            stats_result = WorkflowResult(success=True, trace_id="internal", scenario="em_internal")
            await self._compute_evidence_stats(matrix, stats_result)

            # Compute red-flags
            piece_mapping: Dict[str, int] = {}
            if export_case_id:
                try:
                    pieces = await self.legal.list_case_pieces(case_id=export_case_id)
                    for p in pieces.get("pieces", []):
                        doc_id = p.get("document_id")
                        num = p.get("piece_number", p.get("number"))
                        if doc_id and num:
                            piece_mapping[str(doc_id)] = num
                except Exception:
                    pass

            await self._compute_red_flags(matrix, piece_mapping, export_case_id, stats_result)

            return {
                "matrix": matrix,
                "evidence_summary": stats_result.data.get("evidence_summary", {}),
                "red_flag_summary": stats_result.data.get("red_flag_summary", {}),
                "strong_evidence": stats_result.data.get("strong_evidence", []),
                "weak_points": stats_result.data.get("weak_points", []),
                "contested_claims": stats_result.data.get("contested_claims", []),
            }

        except Exception as e:
            logger.warning(f"Evidence matrix internal failed: {e}")
            return None

    async def _brief_summary(
        self,
        project_id: str,
        token_budget: int,
        result: WorkflowResult,
    ) -> None:
        """Build structured summary via context_pack (helper)."""
        try:
            # Build goal from collected data
            parts_info = ""
            if result.data.get("parties"):
                party_names = [p["name"] for p in result.data["parties"][:5]]
                parts_info = f"Parties: {', '.join(party_names)}. "

            contr_info = ""
            if result.data.get("contradictions_count", 0) > 0:
                contr_info = f"Found {result.data['contradictions_count']} contradictions. "

            goal = (
                f"Create a structured legal case brief for project {project_id}. "
                f"{parts_info}{contr_info}"
                f"Include: case overview, key facts, legal issues, timeline summary."
            )

            context_result = await self.braine.memory_context_pack(
                goal=goal,
                token_budget=token_budget,
                audience="lawyer",
                style="structured",
                include_sources=True,
            )
            result.add_step("braine.memory_context_pack", context_result)

            result.data["summary"] = context_result.get("content", "")
            result.data["summary_sources"] = context_result.get("sources", [])
            result.data["summary_tokens"] = context_result.get("token_count", 0)

            logger.info(f"[{result.trace_id}] Summary: {result.data['summary_tokens']} tokens")
        except Exception as e:
            logger.warning(f"[{result.trace_id}] Summary generation failed: {e}")
            result.add_step("braine.memory_context_pack", {"error": str(e)}, success=False)

            # Fallback: build simple summary from collected data
            result.data["summary"] = self._build_fallback_summary(result)

    def _build_fallback_summary(self, result: WorkflowResult) -> str:
        """Build a simple summary when context_pack fails."""
        lines = [f"# Case Brief (Project: {result.scenario})", ""]

        # Documents
        doc_count = result.data.get("document_count", 0)
        lines.append(f"## Documents: {doc_count}")
        lines.append("")

        # Parties
        parties = result.data.get("parties", [])
        if parties:
            lines.append("## Key Parties")
            for p in parties[:5]:
                lines.append(f"- {p['name']} ({p.get('type', 'unknown')})")
            lines.append("")

        # Timeline
        timeline = result.data.get("timeline", [])
        if timeline:
            lines.append(f"## Timeline: {len(timeline)} events")
            tr = result.data.get("timeline_range", {})
            if tr:
                lines.append(f"Period: {tr.get('start', '?')} to {tr.get('end', '?')}")
            lines.append("")

        # Contradictions
        contr = result.data.get("contradictions", [])
        if contr:
            lines.append(f"## Contradictions: {len(contr)}")
            by_sev = result.data.get("contradictions_by_severity", {})
            for sev, count in by_sev.items():
                lines.append(f"- {sev}: {count}")
            lines.append("")

        # Claims
        claims = result.data.get("key_claims", [])
        if claims:
            lines.append(f"## Key Claims: {len(claims)}")
            for c in claims[:5]:
                text = c.get("text", c.get("claim", str(c)))[:100]
                lines.append(f"- {text}")
            lines.append("")

        return "\n".join(lines)

    def _build_action_plan(self, result: WorkflowResult) -> None:
        """
        Build unified action_plan with P0/P1/P2 priorities.

        Collects actions from:
        - evidence_todo (red-flags)
        - contradictions (requiring resolution)
        - missing_in_pieces (evidence not in case)

        Priority mapping:
        - P0: Critical (high severity red-flags, missing key evidence)
        - P1: Important (medium red-flags, contradictions to address)
        - P2: Recommended (improvements, additional evidence)
        """
        action_plan: List[Dict[str, Any]] = []
        seen_actions: set = set()

        # ── Source 1: Evidence red-flags ───────────────────────────────────────
        evidence_todo = result.data.get("evidence_todo", [])
        for item in evidence_todo:
            action_key = (item.get("code", ""), item.get("claim_index", 0))
            if action_key in seen_actions:
                continue
            seen_actions.add(action_key)

            # Map priority
            pri = item.get("priority", "medium")
            priority = "P0" if pri == "high" else "P1"

            action_plan.append({
                "priority": priority,
                "source": "evidence",
                "code": item.get("code"),
                "action": item.get("action", "Résoudre le problème de preuve"),
                "description": item.get("hint", item.get("action", "")),
                "claim_index": item.get("claim_index"),
                "claim_excerpt": item.get("claim_excerpt", ""),
            })

        # ── Source 2: Contradictions ───────────────────────────────────────────
        contradictions = result.data.get("contradictions", [])
        for i, contr in enumerate(contradictions[:5]):  # Top 5 contradictions
            severity = contr.get("severity", "low")
            if severity in ("critical", "high"):
                priority = "P0"
            elif severity == "medium":
                priority = "P1"
            else:
                priority = "P2"

            action_plan.append({
                "priority": priority,
                "source": "contradiction",
                "action": "resolve_contradiction",
                "description": f"Résoudre la contradiction: {contr.get('description', '')}",
                "severity": severity,
                "documents": contr.get("documents", [])[:2],
            })

        # ── Source 3: Contested claims (supporting vs contradicting) ───────────
        contested = result.data.get("contested_claims", [])
        for claim in contested[:3]:  # Top 3 contested claims
            action_plan.append({
                "priority": "P1",
                "source": "contested",
                "action": "clarify_claim",
                "description": f"Clarifier le point litigieux (s{claim.get('supporting_count', 0)} vs c{claim.get('contradicting_count', 0)})",
                "claim_index": claim.get("index"),
                "claim_excerpt": claim.get("claim", "")[:80],
            })

        # ── Source 4: Missing pieces (will be added by _track_sources_and_pieces) ──
        # Placeholder - gets populated later
        missing_pieces = result.data.get("missing_in_pieces", [])
        for piece in missing_pieces[:5]:
            action_plan.append({
                "priority": "P1",
                "source": "missing_piece",
                "action": "add_to_case",
                "description": f"Ajouter au dossier: {piece.get('document_name', piece.get('document_id', ''))}",
                "document_id": piece.get("document_id"),
            })

        # ── Sort by priority ───────────────────────────────────────────────────
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        action_plan.sort(key=lambda a: (priority_order.get(a["priority"], 3), a.get("claim_index", 999)))

        result.data["action_plan"] = action_plan

        logger.info(
            f"[{result.trace_id}] Action plan: {len(action_plan)} items "
            f"(P0:{len([a for a in action_plan if a['priority']=='P0'])}, "
            f"P1:{len([a for a in action_plan if a['priority']=='P1'])}, "
            f"P2:{len([a for a in action_plan if a['priority']=='P2'])})"
        )

    async def _track_sources_and_pieces(
        self,
        case_id: Optional[str],
        auto_add: bool,
        result: WorkflowResult,
    ) -> None:
        """
        Track top_sources and missing_in_pieces.

        - top_sources: Documents most cited as evidence (sorted by citation count)
        - missing_in_pieces: Evidence documents not yet added to the case

        If auto_add=True and case_id provided, auto-adds missing evidence docs.
        """
        # Count document citations from evidence matrix
        doc_citations: Dict[str, Dict[str, Any]] = {}

        matrix = result.data.get("evidence_matrix", [])
        for entry in matrix:
            # Count supporting evidence docs
            for ev in entry.get("best_supporting", entry.get("supporting", [])):
                doc_id = ev.get("document_id")
                if doc_id:
                    if doc_id not in doc_citations:
                        doc_citations[doc_id] = {
                            "document_id": doc_id,
                            "document_name": ev.get("document_name", ""),
                            "citation": ev.get("citation", ""),
                            "count": 0,
                            "as_supporting": 0,
                            "as_contradicting": 0,
                        }
                    doc_citations[doc_id]["count"] += 1
                    doc_citations[doc_id]["as_supporting"] += 1

            # Count contradicting evidence docs
            for ev in entry.get("best_contradicting", entry.get("contradicting", [])):
                doc_id = ev.get("document_id")
                if doc_id:
                    if doc_id not in doc_citations:
                        doc_citations[doc_id] = {
                            "document_id": doc_id,
                            "document_name": ev.get("document_name", ""),
                            "citation": ev.get("citation", ""),
                            "count": 0,
                            "as_supporting": 0,
                            "as_contradicting": 0,
                        }
                    doc_citations[doc_id]["count"] += 1
                    doc_citations[doc_id]["as_contradicting"] += 1

        # Sort by citation count
        top_sources = sorted(
            doc_citations.values(),
            key=lambda d: d["count"],
            reverse=True,
        )[:10]
        result.data["top_sources"] = top_sources

        # Check which docs are missing from case pieces
        missing_in_pieces: List[Dict[str, Any]] = []
        pieces_auto_added = 0

        if case_id:
            try:
                # Get current case pieces
                pieces_result = await self.legal.list_case_pieces(case_id=case_id)
                piece_doc_ids = {
                    str(p.get("document_id")) for p in pieces_result.get("pieces", [])
                }

                # Find cited docs not in case
                for source in top_sources:
                    doc_id = str(source.get("document_id", ""))
                    if doc_id and doc_id not in piece_doc_ids:
                        missing_in_pieces.append({
                            "document_id": doc_id,
                            "document_name": source.get("document_name", ""),
                            "citation_count": source.get("count", 0),
                            "as_supporting": source.get("as_supporting", 0),
                            "as_contradicting": source.get("as_contradicting", 0),
                        })

                        # Auto-add if enabled
                        if auto_add:
                            try:
                                await self.legal.add_piece_to_case(
                                    case_id=case_id,
                                    document_id=doc_id,
                                )
                                pieces_auto_added += 1
                                logger.info(
                                    f"[{result.trace_id}] Auto-added piece: {doc_id}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[{result.trace_id}] Failed to auto-add piece {doc_id}: {e}"
                                )

            except Exception as e:
                logger.warning(f"[{result.trace_id}] Failed to check case pieces: {e}")

        result.data["missing_in_pieces"] = missing_in_pieces
        result.data["pieces_auto_added"] = pieces_auto_added

        logger.info(
            f"[{result.trace_id}] Sources: top={len(top_sources)}, "
            f"missing={len(missing_in_pieces)}, auto_added={pieces_auto_added}"
        )

    def _apply_court_profile(self, result: WorkflowResult) -> None:
        """
        Apply court export profile - filter for concise tribunal output.

        Court profile produces:
        - Only top evidence per claim (best_supporting[0], best_contradicting[0])
        - Only high-severity red-flags
        - Condensed action_plan (P0 and P1 only)
        - No internal metadata (_trace_id, etc.)
        - Formatted for printing

        Work profile (default) keeps everything for internal analysis.
        """
        # ── Filter evidence matrix ─────────────────────────────────────────────
        matrix = result.data.get("evidence_matrix", [])
        court_matrix = []

        for entry in matrix:
            court_entry = {
                "index": entry.get("index"),
                "claim": entry.get("claim"),
                "status": entry.get("evidence_status", entry.get("status")),
            }

            # Keep only best supporting evidence
            supporting = entry.get("best_supporting", entry.get("supporting", []))
            if supporting:
                best = supporting[0]
                court_entry["pièce"] = best.get("citation", f"Pièce n°{best.get('piece_number', '?')}")
                court_entry["quote"] = best.get("quote", "")[:150]

            # Keep only high-severity red-flags
            flags = entry.get("red_flags", [])
            high_flags = [f for f in flags if f.get("severity") == "high"]
            if high_flags:
                court_entry["alerte"] = high_flags[0].get("code")
                court_entry["action"] = high_flags[0].get("action")

            court_matrix.append(court_entry)

        result.data["evidence_matrix"] = court_matrix

        # ── Filter action_plan ─────────────────────────────────────────────────
        action_plan = result.data.get("action_plan", [])
        court_actions = [
            {
                "priority": a["priority"],
                "action": a["action"],
                "description": a.get("description", "")[:100],
            }
            for a in action_plan
            if a.get("priority") in ("P0", "P1")
        ]
        result.data["action_plan"] = court_actions[:10]  # Max 10 actions for court

        # ── Simplify points_litigieux ──────────────────────────────────────────
        points = result.data.get("points_litigieux", [])
        court_points = [
            {
                "claim": p.get("claim", "")[:100],
                "issue": p.get("issue"),
            }
            for p in points
        ]
        result.data["points_litigieux"] = court_points[:5]

        # ── Remove internal fields ─────────────────────────────────────────────
        internal_fields = [
            "evidence_todo",
            "claim_entities",
            "summary_sources",
            "summary_tokens",
            "steps",
        ]
        for field in internal_fields:
            result.data.pop(field, None)

        # ── Add court profile marker ───────────────────────────────────────────
        result.data["_export_profile"] = "court"
        result.data["_note"] = "Profil tribunal - version condensée pour audience"

        logger.info(f"[{result.trace_id}] Applied court export profile")

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 5: Evidence Matrix (Court-Ready)
    # ─────────────────────────────────────────────────────────────────────────

    async def evidence_matrix(
        self,
        project_id: str,
        claims: Optional[List[str]] = None,
        auto_extract_claims: bool = True,
        max_claims: int = 20,
        include_contradictions: bool = True,
        include_citations: bool = True,
        include_coverage_analysis: bool = True,
        deep_contradiction_check: bool = False,
        compute_red_flags: bool = True,
        auto_add_pieces: bool = True,
        deterministic: bool = True,
        supporting_top: int = 5,
        contradicting_top: int = 3,
        export_case_id: Optional[str] = None,
        citation_style: str = "french_court",
        trace_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Build a court-ready evidence matrix for trial preparation.

        Creates a structured matrix with legal citations showing:
        - Claims/assertions to prove or disprove
        - Evidence supporting each claim (with Pièce n°, page, citation)
        - Evidence contradicting each claim
        - Coverage analysis validation
        - Deep contradiction notes for contested claims
        - Red-flag warnings with actionable hints

        This is essential for trial preparation, showing what evidence
        supports or weakens each key point in the case.

        Workflow:
        1. Load piece mapping (if export_case_id)
        2. Extract or use provided claims
        3. Run coverage analysis for strength validation
        4. For each claim, find supporting/contradicting evidence
        5. Enrich evidence with legal citations (get_citation_format)
        6. Auto-add missing pieces to case (if auto_add_pieces)
        7. Deep contradiction check for contested claims
        8. Compute red-flags and action hints
        9. Optionally export to case bundle

        Args:
            project_id: LegalOps project ID (required)
            claims: List of claims to evaluate (if not provided, auto-extract)
            auto_extract_claims: Auto-extract claims from documents if not provided
            max_claims: Maximum number of claims to analyze
            include_contradictions: Include contradicting evidence
            include_citations: Add legal citations via get_citation_format
            include_coverage_analysis: Run analyze_coverage for strength validation
            deep_contradiction_check: Run compare_documents_for_contradictions
            compute_red_flags: Compute red-flag warnings (RF-01 to RF-09)
            auto_add_pieces: Auto-add evidence docs to case if not in bordereau
            deterministic: Use deterministic sorting for reproducibility
            supporting_top: Max supporting evidence items per claim
            contradicting_top: Max contradicting evidence items per claim
            export_case_id: If provided, add matrix to case as piece
            citation_style: Citation format (french_court, academic, short)
            trace_id: Optional trace ID

        Returns:
            WorkflowResult with court-ready evidence matrix:
            - matrix: structured evidence with citations per claim
            - coverage: analyze_coverage results
            - evidence_summary: counts and statistics
            - strong_evidence: well-supported claims
            - weak_points: claims with actions needed
            - contested_claims: claims with contradictions
            - contradiction_notes: detailed contradiction analysis
            - red_flag_summary: count and breakdown of warnings
            - piece_mapping: document_id → piece_number map
            - export: case piece info if exported
        """
        trace_id = trace_id or self._new_trace_id()
        result = WorkflowResult(success=True, trace_id=trace_id, scenario="evidence_matrix")

        logger.info(f"[{trace_id}] evidence_matrix: project={project_id}, claims={len(claims) if claims else 'auto'}")

        try:
            # ─────────────────────────────────────────────────────────────────
            # Step 0: Load piece mapping (if case export requested)
            # ─────────────────────────────────────────────────────────────────
            piece_mapping: Dict[str, int] = {}
            if export_case_id:
                piece_mapping = await self._load_piece_mapping(export_case_id, result)

            # ─────────────────────────────────────────────────────────────────
            # Step 1: Get or extract claims (dual-channel)
            # ─────────────────────────────────────────────────────────────────
            if claims:
                claims_list = claims[:max_claims]
                result.data["claims_source"] = "provided"
            elif auto_extract_claims:
                claims_list = await self._extract_claims_dual_channel(
                    project_id, max_claims, result
                )
                result.data["claims_source"] = "auto_extracted"
            else:
                result.success = False
                result.errors.append("No claims provided and auto_extract_claims=False")
                return result

            result.data["claims_count"] = len(claims_list)
            logger.info(f"[{trace_id}] Analyzing {len(claims_list)} claims")

            if not claims_list:
                result.data["matrix"] = []
                result.data["evidence_summary"] = {"total_claims": 0}
                logger.warning(f"[{trace_id}] No claims to analyze")
                return result

            # ─────────────────────────────────────────────────────────────────
            # Step 2: Coverage analysis for strength validation
            # ─────────────────────────────────────────────────────────────────
            coverage_data = None
            if include_coverage_analysis:
                coverage_data = await self._run_coverage_analysis(
                    claims_list, project_id, result
                )

            # ─────────────────────────────────────────────────────────────────
            # Step 3-4: Build evidence matrix with citations (parallel)
            # ─────────────────────────────────────────────────────────────────
            matrix_tasks = [
                self._analyze_claim_evidence_v2(
                    claim=claim,
                    claim_index=i,
                    project_id=project_id,
                    include_contradictions=include_contradictions,
                    include_citations=include_citations,
                    citation_style=citation_style,
                    coverage_data=coverage_data,
                    result=result,
                )
                for i, claim in enumerate(claims_list)
            ]

            matrix_results = await asyncio.gather(*matrix_tasks, return_exceptions=True)

            # Collect results
            matrix = []
            for i, mr in enumerate(matrix_results):
                if isinstance(mr, Exception):
                    logger.warning(f"[{trace_id}] Claim {i} analysis failed: {mr}")
                    matrix.append({
                        "index": i,
                        "claim": claims_list[i] if i < len(claims_list) else f"claim_{i}",
                        "error": str(mr),
                        "supporting": [],
                        "contradicting": [],
                        "strength": "unknown",
                    })
                elif mr:
                    matrix.append(mr)

            # ─────────────────────────────────────────────────────────────────
            # Step 4b: Apply deterministic sorting and top-k limits
            # ─────────────────────────────────────────────────────────────────
            for entry in matrix:
                # Sort by score (deterministic)
                if deterministic:
                    entry["supporting"] = sorted(
                        entry.get("supporting", []),
                        key=lambda e: (-e.get("relevance", 0), e.get("document_id", "")),
                    )[:supporting_top]
                    entry["contradicting"] = sorted(
                        entry.get("contradicting", []),
                        key=lambda e: (-e.get("relevance", 0), e.get("document_id", "")),
                    )[:contradicting_top]

                # Select best evidence
                if entry.get("supporting"):
                    entry["best_supporting"] = entry["supporting"][0]
                if entry.get("contradicting"):
                    entry["best_contradicting"] = entry["contradicting"][0]

            result.data["matrix"] = matrix

            # ─────────────────────────────────────────────────────────────────
            # Step 5: Auto-add missing pieces to case
            # ─────────────────────────────────────────────────────────────────
            if export_case_id and auto_add_pieces:
                piece_mapping = await self._auto_add_missing_pieces(
                    matrix, export_case_id, piece_mapping, result
                )

            result.data["piece_mapping"] = piece_mapping

            # ─────────────────────────────────────────────────────────────────
            # Step 6: Compute statistics and categorize
            # ─────────────────────────────────────────────────────────────────
            await self._compute_evidence_stats(matrix, result)

            # ─────────────────────────────────────────────────────────────────
            # Step 7: Deep contradiction check for contested claims
            # ─────────────────────────────────────────────────────────────────
            if deep_contradiction_check and result.data.get("contested_claims"):
                await self._deep_contradiction_analysis(matrix, project_id, result)

            # ─────────────────────────────────────────────────────────────────
            # Step 8: Compute red-flags and action hints
            # ─────────────────────────────────────────────────────────────────
            if compute_red_flags:
                await self._compute_red_flags(
                    matrix=matrix,
                    piece_mapping=piece_mapping,
                    export_case_id=export_case_id,
                    result=result,
                )

            # ─────────────────────────────────────────────────────────────────
            # Step 9: Export to case if requested
            # ─────────────────────────────────────────────────────────────────
            if export_case_id:
                await self._export_matrix_to_case(
                    matrix, export_case_id, project_id, result
                )

            logger.info(
                f"[{trace_id}] Matrix built: {len(matrix)} claims, "
                f"strong={len(result.data.get('strong_evidence', []))}, "
                f"weak={len(result.data.get('weak_points', []))}, "
                f"red_flags={result.data.get('red_flag_summary', {}).get('total', 0)}"
            )

        except Exception as e:
            logger.error(f"[{trace_id}] evidence_matrix failed: {e}")
            result.success = False
            result.errors.append(str(e))

        logger.info(f"[{trace_id}] evidence_matrix: success={result.success}")
        return result

    async def _extract_claims_dual_channel(
        self,
        project_id: str,
        max_claims: int,
        result: WorkflowResult,
    ) -> List[str]:
        """
        Extract claims using dual-channel approach (helper).

        Channel 1: LegalOps - contradiction reports, coverage gaps
        Channel 2: BraineMemory - memory_recall_claims, conflicts
        """
        claims_list = []
        seen_claims = set()

        def add_claim(text: str) -> bool:
            """Add claim if not duplicate."""
            if not text or not isinstance(text, str):
                return False
            normalized = text.strip().lower()[:100]
            if normalized in seen_claims:
                return False
            seen_claims.add(normalized)
            claims_list.append(text.strip())
            return True

        # ─────────────────────────────────────────────────────────────────────
        # Channel 1: LegalOps
        # ─────────────────────────────────────────────────────────────────────
        try:
            # Check project readiness first
            readiness = await self.legal.get_project_readiness(project_id=project_id)
            result.add_step("legal.get_project_readiness", readiness)

            if readiness.get("status") == "cold":
                logger.warning(f"[{result.trace_id}] Project is cold - limited claims extraction")

            # Get contradictions (often contain claim-like statements)
            contr_result = await self.legal.get_contradictions(project_id=project_id)
            result.add_step("legal.get_contradictions", contr_result)

            for contr in contr_result.get("contradictions", [])[:10]:
                # Extract claims from contradiction statements
                stmt_a = contr.get("statement_a", contr.get("text_a", ""))
                stmt_b = contr.get("statement_b", contr.get("text_b", ""))
                if stmt_a:
                    add_claim(stmt_a[:300])
                if stmt_b:
                    add_claim(stmt_b[:300])

            logger.info(f"[{result.trace_id}] Channel 1 (LegalOps): {len(claims_list)} claims")

        except Exception as e:
            logger.warning(f"[{result.trace_id}] LegalOps claims channel failed: {e}")
            result.add_step("legal.get_contradictions", {"error": str(e)}, success=False)

        # ─────────────────────────────────────────────────────────────────────
        # Channel 2: BraineMemory
        # ─────────────────────────────────────────────────────────────────────
        try:
            claims_result = await self.braine.memory_recall_claims(
                query=f"key factual claims assertions for project {project_id}",
                limit=max_claims,
                include_conflicting=True,
            )
            result.add_step("braine.memory_recall_claims", claims_result)

            for c in claims_result.get("claims", []):
                text = c.get("text", c.get("claim", str(c) if not isinstance(c, str) else c))
                add_claim(text[:300] if text else "")

            # Also try conflict detection for claim-like formulations
            conflicts_result = await self.braine.memory_detect_conflicts(
                query=f"disputed facts for {project_id}",
                limit=10,
            )
            result.add_step("braine.memory_detect_conflicts", conflicts_result)

            for conflict in conflicts_result.get("conflicts", []):
                claim_text = conflict.get("claim", conflict.get("assertion", ""))
                if claim_text:
                    add_claim(claim_text[:300])

            logger.info(f"[{result.trace_id}] Channel 2 (BraineMemory): total {len(claims_list)} claims")

        except Exception as e:
            logger.warning(f"[{result.trace_id}] BraineMemory claims channel failed: {e}")
            result.add_step("braine.memory_recall_claims", {"error": str(e)}, success=False)

        # ─────────────────────────────────────────────────────────────────────
        # Fallback: Smart search if still not enough
        # ─────────────────────────────────────────────────────────────────────
        if len(claims_list) < max_claims // 3:
            try:
                search_result = await self.legal.smart_search(
                    query="alleged claimed stated asserted contended affirmed",
                    project_id=project_id,
                )
                result.add_step("legal.smart_search[claims_fallback]", search_result)

                for doc in search_result.get("documents", search_result.get("results", []))[:10]:
                    snippet = doc.get("snippet", doc.get("text", ""))
                    if snippet:
                        add_claim(snippet[:200])

            except Exception as e:
                logger.warning(f"[{result.trace_id}] Fallback search failed: {e}")

        return claims_list[:max_claims]

    async def _analyze_claim_evidence(
        self,
        claim: str,
        claim_index: int,
        project_id: str,
        include_contradictions: bool,
        include_sources: bool,
        result: WorkflowResult,
    ) -> Dict[str, Any]:
        """Analyze evidence for a single claim (helper)."""
        claim_entry = {
            "index": claim_index,
            "claim": claim[:500],  # Truncate long claims
            "supporting": [],
            "contradicting": [],
            "sources": [],
            "strength": "unknown",
        }

        try:
            # Find evidence for this claim
            evidence_result = await self.legal.find_evidence(
                claim=claim,
                project_id=project_id,
            )

            supporting = evidence_result.get("supporting", [])
            contradicting = evidence_result.get("contradicting", [])

            # Process supporting evidence
            for ev in supporting[:5]:
                entry = {
                    "text": ev.get("text", ev.get("content", ""))[:300],
                    "relevance": ev.get("relevance", ev.get("score", 0.5)),
                    "document_id": ev.get("document_id", ev.get("source", "")),
                }
                if include_sources:
                    entry["source_title"] = ev.get("title", ev.get("document_title", ""))
                claim_entry["supporting"].append(entry)

            # Process contradicting evidence
            if include_contradictions:
                for ev in contradicting[:5]:
                    entry = {
                        "text": ev.get("text", ev.get("content", ""))[:300],
                        "relevance": ev.get("relevance", ev.get("score", 0.5)),
                        "document_id": ev.get("document_id", ev.get("source", "")),
                    }
                    if include_sources:
                        entry["source_title"] = ev.get("title", ev.get("document_title", ""))
                    claim_entry["contradicting"].append(entry)

            # Collect unique sources
            if include_sources:
                source_ids = set()
                for ev in supporting + contradicting:
                    doc_id = ev.get("document_id", ev.get("source", ""))
                    if doc_id:
                        source_ids.add(doc_id)
                claim_entry["sources"] = list(source_ids)

            # Assess strength
            claim_entry["strength"] = self._assess_evidence_strength(
                len(supporting), len(contradicting)
            )

        except Exception as e:
            logger.warning(f"[{result.trace_id}] Evidence for claim {claim_index} failed: {e}")
            claim_entry["error"] = str(e)
            claim_entry["strength"] = "unknown"

        return claim_entry

    def _assess_evidence_strength(
        self,
        supporting_count: int,
        contradicting_count: int,
        coverage_status: Optional[str] = None,
    ) -> str:
        """
        Assess overall evidence strength for a claim.

        Uses both evidence counts and optional coverage analysis status.
        """
        # Coverage analysis can override pure count-based assessment
        if coverage_status == "missing":
            if supporting_count == 0:
                return "no_evidence"
            return "weak"  # Has some evidence but coverage says missing

        if supporting_count == 0 and contradicting_count == 0:
            return "no_evidence"
        elif supporting_count >= 3 and contradicting_count == 0:
            return "strong"
        elif supporting_count >= 2 and contradicting_count <= 1:
            return "moderate"
        elif contradicting_count > supporting_count:
            return "weak"
        elif contradicting_count > 0 and supporting_count > 0:
            return "contested"
        elif supporting_count > 0:
            return "some_support"
        else:
            return "contested"

    async def _run_coverage_analysis(
        self,
        claims_list: List[str],
        project_id: str,
        result: WorkflowResult,
    ) -> Optional[Dict[str, Any]]:
        """Run analyze_coverage for strength validation (helper)."""
        try:
            # Combine claims into text block
            claims_text = "\n\n".join(claims_list)

            coverage_result = await self.legal.analyze_coverage(
                text=claims_text,
                project_id=project_id,
                split_mode="paragraph",
                min_score=0.7,
            )
            result.add_step("legal.analyze_coverage", coverage_result)

            # Store coverage data
            result.data["coverage"] = {
                "coverage_pct": coverage_result.get("coverage_pct", 0),
                "claims_total": coverage_result.get("claims_total", 0),
                "claims_covered": coverage_result.get("claims_covered", 0),
                "claims_partial": coverage_result.get("claims_partial", 0),
                "claims_missing": coverage_result.get("claims_missing", 0),
            }

            # Build claim-status map for quick lookup
            coverage_map = {}
            for claim_data in coverage_result.get("claims", []):
                claim_id = claim_data.get("claim_id")
                if claim_id:
                    coverage_map[claim_id] = {
                        "status": claim_data.get("status", "unknown"),
                        "best_score": claim_data.get("best_score", 0),
                        "evidence_count": claim_data.get("evidence_count", 0),
                    }

            logger.info(
                f"[{result.trace_id}] Coverage: {result.data['coverage']['coverage_pct']}% "
                f"({result.data['coverage']['claims_covered']}/{result.data['coverage']['claims_total']} covered)"
            )

            return {
                "map": coverage_map,
                "claims": coverage_result.get("claims", []),
            }

        except Exception as e:
            logger.warning(f"[{result.trace_id}] Coverage analysis failed: {e}")
            result.add_step("legal.analyze_coverage", {"error": str(e)}, success=False)
            return None

    async def _analyze_claim_evidence_v2(
        self,
        claim: str,
        claim_index: int,
        project_id: str,
        include_contradictions: bool,
        include_citations: bool,
        citation_style: str,
        coverage_data: Optional[Dict[str, Any]],
        result: WorkflowResult,
    ) -> Dict[str, Any]:
        """
        Analyze evidence for a claim with court-ready citations (helper).

        Enhanced version that includes:
        - Legal citations via get_citation_format
        - Locator info (page, chunk)
        - Coverage validation
        """
        claim_entry = {
            "index": claim_index,
            "claim": claim[:500],
            "supporting": [],
            "contradicting": [],
            "sources": [],
            "strength": "unknown",
            "coverage_status": None,
        }

        try:
            # Get evidence from LegalOps
            evidence_result = await self.legal.find_evidence(
                claim=claim,
                project_id=project_id,
                top_k=5,
            )

            # Parse find_evidence response (it returns evidence array directly)
            evidence_items = evidence_result.get("evidence", [])
            claim_status = evidence_result.get("status", "unknown")
            best_score = evidence_result.get("best_score", 0)

            claim_entry["find_evidence_status"] = claim_status
            claim_entry["best_score"] = best_score

            # Process each evidence item
            for ev in evidence_items[:5]:
                doc_id = ev.get("document_id")
                page_num = ev.get("page_number")
                quote = ev.get("quote", "")
                score = ev.get("score_total", ev.get("score_semantic", 0))

                entry = {
                    "document_id": doc_id,
                    "quote": quote[:400] if quote else "",
                    "relevance": round(score, 3) if score else 0,
                    "locator": {
                        "page": page_num,
                        "chunk_id": ev.get("chunk_id"),
                    },
                    "scores": {
                        "total": round(ev.get("score_total", 0), 3),
                        "semantic": round(ev.get("score_semantic", 0), 3),
                        "entity_bonus": round(ev.get("score_entity_bonus", 0), 3),
                    },
                    "entity_matches": ev.get("entity_matches", []),
                }

                # Get legal citation if requested
                if include_citations and doc_id:
                    try:
                        citation_result = await self.legal.get_citation_format(
                            document_id=doc_id,
                            page_number=page_num,
                            style=citation_style,
                        )
                        entry["citation"] = citation_result.get(
                            "citation",
                            citation_result.get("formatted", f"Document {doc_id}")
                        )
                    except Exception as cite_err:
                        logger.debug(f"Citation failed for {doc_id}: {cite_err}")
                        entry["citation"] = f"Document {doc_id}" + (f", p.{page_num}" if page_num else "")

                # Determine if supporting or contradicting based on score
                # High score = supporting, find_evidence returns supporting evidence
                claim_entry["supporting"].append(entry)

            # For contradictions, we need to search for contradicting evidence
            if include_contradictions:
                try:
                    # Use detect_contradictions or find evidence for negated claim
                    contra_result = await self.legal.find_evidence(
                        claim=f"NOT: {claim}" if len(claim) < 200 else claim[:100],
                        project_id=project_id,
                        top_k=3,
                    )
                    contra_evidence = contra_result.get("evidence", [])

                    for ev in contra_evidence[:3]:
                        doc_id = ev.get("document_id")
                        page_num = ev.get("page_number")

                        entry = {
                            "document_id": doc_id,
                            "quote": (ev.get("quote", "") or "")[:400],
                            "relevance": round(ev.get("score_total", 0), 3),
                            "locator": {
                                "page": page_num,
                                "chunk_id": ev.get("chunk_id"),
                            },
                        }

                        if include_citations and doc_id:
                            try:
                                citation_result = await self.legal.get_citation_format(
                                    document_id=doc_id,
                                    page_number=page_num,
                                    style=citation_style,
                                )
                                entry["citation"] = citation_result.get("citation", f"Document {doc_id}")
                            except:
                                entry["citation"] = f"Document {doc_id}"

                        claim_entry["contradicting"].append(entry)

                except Exception as contra_err:
                    logger.debug(f"Contradiction search failed: {contra_err}")

            # Collect unique sources
            source_ids = set()
            for ev in claim_entry["supporting"] + claim_entry["contradicting"]:
                doc_id = ev.get("document_id")
                if doc_id:
                    source_ids.add(str(doc_id))
            claim_entry["sources"] = list(source_ids)

            # Get coverage status for this claim if available
            coverage_status = None
            if coverage_data and coverage_data.get("claims"):
                # Try to match by index
                if claim_index < len(coverage_data["claims"]):
                    cov_claim = coverage_data["claims"][claim_index]
                    coverage_status = cov_claim.get("status")
                    claim_entry["coverage_status"] = coverage_status
                    claim_entry["coverage_score"] = cov_claim.get("best_score", 0)

            # Assess strength with coverage validation
            claim_entry["strength"] = self._assess_evidence_strength(
                len(claim_entry["supporting"]),
                len(claim_entry["contradicting"]),
                coverage_status,
            )

        except Exception as e:
            logger.warning(f"[{result.trace_id}] Evidence for claim {claim_index} failed: {e}")
            claim_entry["error"] = str(e)
            claim_entry["strength"] = "unknown"

        return claim_entry

    async def _deep_contradiction_analysis(
        self,
        matrix: List[Dict[str, Any]],
        project_id: str,
        result: WorkflowResult,
    ) -> None:
        """
        Run deep contradiction analysis for contested claims (helper).

        Uses compare_documents_for_contradictions on doc pairs.
        """
        contradiction_notes = []

        contested = result.data.get("contested_claims", [])
        if not contested:
            return

        logger.info(f"[{result.trace_id}] Deep analysis for {len(contested)} contested claims")

        for claim_info in contested[:5]:  # Limit to 5 for performance
            claim_idx = claim_info.get("index", 0)
            if claim_idx >= len(matrix):
                continue

            claim_entry = matrix[claim_idx]
            supporting_docs = [e.get("document_id") for e in claim_entry.get("supporting", []) if e.get("document_id")]
            contradicting_docs = [e.get("document_id") for e in claim_entry.get("contradicting", []) if e.get("document_id")]

            # Compare first supporting doc with first contradicting doc
            if supporting_docs and contradicting_docs:
                doc_a = supporting_docs[0]
                doc_b = contradicting_docs[0]

                try:
                    compare_result = await self.legal.compare_documents_for_contradictions(
                        doc_a_id=str(doc_a),
                        doc_b_id=str(doc_b),
                    )
                    result.add_step(f"legal.compare_docs[{claim_idx}]", compare_result)

                    contradictions = compare_result.get("contradictions", [])
                    if contradictions:
                        note = {
                            "claim_index": claim_idx,
                            "claim": claim_entry.get("claim", "")[:100],
                            "doc_a": str(doc_a),
                            "doc_b": str(doc_b),
                            "contradictions": [
                                {
                                    "text_a": c.get("text_a", c.get("statement_a", ""))[:200],
                                    "text_b": c.get("text_b", c.get("statement_b", ""))[:200],
                                    "severity": c.get("severity", "medium"),
                                }
                                for c in contradictions[:3]
                            ],
                        }
                        contradiction_notes.append(note)

                except Exception as e:
                    logger.warning(f"[{result.trace_id}] Deep compare failed for claim {claim_idx}: {e}")

        result.data["contradiction_notes"] = contradiction_notes

        if contradiction_notes:
            logger.info(f"[{result.trace_id}] Found {len(contradiction_notes)} detailed contradiction notes")

    async def _export_matrix_to_case(
        self,
        matrix: List[Dict[str, Any]],
        case_id: str,
        project_id: str,
        result: WorkflowResult,
    ) -> None:
        """
        Export evidence matrix to a case as a piece (helper).

        Creates a formatted text document and adds it to the case.
        """
        try:
            # Build markdown document
            lines = [
                "# Evidence Matrix",
                f"Project: {project_id}",
                f"Generated: {result.trace_id}",
                "",
                f"## Summary",
                f"- Total claims: {result.data.get('claims_count', 0)}",
                f"- Strong evidence: {len(result.data.get('strong_evidence', []))}",
                f"- Weak points: {len(result.data.get('weak_points', []))}",
                f"- Contested: {len(result.data.get('contested_claims', []))}",
                "",
                "---",
                "",
            ]

            for entry in matrix:
                idx = entry.get("index", 0)
                claim = entry.get("claim", "")[:200]
                strength = entry.get("strength", "unknown")

                lines.append(f"## Claim {idx + 1}: {strength.upper()}")
                lines.append(f"> {claim}")
                lines.append("")

                # Supporting evidence
                if entry.get("supporting"):
                    lines.append("### Supporting Evidence")
                    for i, ev in enumerate(entry["supporting"], 1):
                        citation = ev.get("citation", ev.get("document_id", "Unknown"))
                        quote = ev.get("quote", "")[:150]
                        lines.append(f"{i}. **{citation}**")
                        if quote:
                            lines.append(f'   > "{quote}..."')
                        lines.append("")

                # Contradicting evidence
                if entry.get("contradicting"):
                    lines.append("### Contradicting Evidence")
                    for i, ev in enumerate(entry["contradicting"], 1):
                        citation = ev.get("citation", ev.get("document_id", "Unknown"))
                        quote = ev.get("quote", "")[:150]
                        lines.append(f"{i}. **{citation}**")
                        if quote:
                            lines.append(f'   > "{quote}..."')
                        lines.append("")

                lines.append("---")
                lines.append("")

            matrix_text = "\n".join(lines)

            # Ingest as document
            ingest_result = await self.legal.ingest_text(
                content=matrix_text,
                title=f"Evidence Matrix - {result.trace_id}",
                project_id=project_id,
                document_type="evidence_matrix",
            )
            result.add_step("legal.ingest_text[matrix]", ingest_result)

            doc_id = ingest_result.get("document_id", ingest_result.get("id"))

            if doc_id:
                # Add to case as piece
                piece_result = await self.legal.add_piece(
                    case_id=case_id,
                    document_id=doc_id,
                    label="Evidence Matrix",
                )
                result.add_step("legal.add_piece[matrix]", piece_result)

                result.data["export"] = {
                    "case_id": case_id,
                    "document_id": doc_id,
                    "piece_number": piece_result.get("piece_number"),
                    "status": "exported",
                }

                logger.info(f"[{result.trace_id}] Matrix exported to case {case_id}")
            else:
                result.data["export"] = {"status": "failed", "error": "No document_id returned"}

        except Exception as e:
            logger.warning(f"[{result.trace_id}] Matrix export failed: {e}")
            result.add_step("legal.export_matrix", {"error": str(e)}, success=False)
            result.data["export"] = {"status": "failed", "error": str(e)}

    async def _compute_evidence_stats(
        self,
        matrix: List[Dict[str, Any]],
        result: WorkflowResult,
    ) -> None:
        """Compute statistics and categorize claims (helper)."""
        stats = {
            "total_claims": len(matrix),
            "total_supporting": 0,
            "total_contradicting": 0,
            "by_strength": {},
        }

        strong_evidence = []
        weak_points = []
        contested_claims = []

        for entry in matrix:
            sup_count = len(entry.get("supporting", []))
            con_count = len(entry.get("contradicting", []))
            strength = entry.get("strength", "unknown")

            stats["total_supporting"] += sup_count
            stats["total_contradicting"] += con_count
            stats["by_strength"][strength] = stats["by_strength"].get(strength, 0) + 1

            # Categorize
            claim_summary = {
                "claim": entry.get("claim", "")[:100],
                "index": entry.get("index", 0),
                "supporting_count": sup_count,
                "contradicting_count": con_count,
            }

            if strength == "strong":
                strong_evidence.append(claim_summary)
            elif strength in ("weak", "no_evidence"):
                weak_points.append(claim_summary)
            elif strength == "contested":
                contested_claims.append(claim_summary)

        result.data["evidence_summary"] = stats
        result.data["strong_evidence"] = strong_evidence
        result.data["weak_points"] = weak_points
        result.data["contested_claims"] = contested_claims

        # Add warnings for weak points
        if weak_points:
            result.data["warnings"] = result.data.get("warnings", [])
            result.data["warnings"].append(
                f"{len(weak_points)} claims have weak or no supporting evidence"
            )

        if contested_claims:
            result.data["warnings"] = result.data.get("warnings", [])
            result.data["warnings"].append(
                f"{len(contested_claims)} claims have contested evidence"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Piece Mapping & Auto-Add
    # ─────────────────────────────────────────────────────────────────────────

    async def _load_piece_mapping(
        self,
        case_id: str,
        result: WorkflowResult,
    ) -> Dict[str, int]:
        """Load document_id → piece_number mapping from case (helper)."""
        mapping: Dict[str, int] = {}

        try:
            pieces_result = await self.legal.list_case_pieces(case_id=case_id)
            result.add_step("legal.list_case_pieces", pieces_result)

            for piece in pieces_result.get("pieces", []):
                doc_id = piece.get("document_id")
                piece_num = piece.get("piece_number", piece.get("number"))
                if doc_id and piece_num:
                    mapping[str(doc_id)] = piece_num

            logger.info(f"[{result.trace_id}] Loaded {len(mapping)} piece mappings")

        except Exception as e:
            logger.warning(f"[{result.trace_id}] Piece mapping load failed: {e}")
            result.add_step("legal.list_case_pieces", {"error": str(e)}, success=False)

        return mapping

    async def _auto_add_missing_pieces(
        self,
        matrix: List[Dict[str, Any]],
        case_id: str,
        piece_mapping: Dict[str, int],
        result: WorkflowResult,
    ) -> Dict[str, int]:
        """Auto-add evidence documents to case if not in bordereau (helper)."""
        # Collect all document IDs from matrix
        all_doc_ids = set()
        for entry in matrix:
            for ev in entry.get("supporting", []) + entry.get("contradicting", []):
                doc_id = ev.get("document_id")
                if doc_id:
                    all_doc_ids.add(str(doc_id))

        # Find missing
        missing = all_doc_ids - set(piece_mapping.keys())

        if not missing:
            return piece_mapping

        logger.info(f"[{result.trace_id}] Auto-adding {len(missing)} missing pieces")

        # Add missing pieces
        for doc_id in missing:
            try:
                add_result = await self.legal.add_piece(
                    case_id=case_id,
                    document_id=doc_id,
                )
                piece_num = add_result.get("piece_number", add_result.get("number"))
                if piece_num:
                    piece_mapping[doc_id] = piece_num
                    logger.debug(f"[{result.trace_id}] Added piece {piece_num} for {doc_id}")

            except Exception as e:
                logger.warning(f"[{result.trace_id}] Failed to add piece for {doc_id}: {e}")

        result.add_step("legal.auto_add_pieces", {
            "added": len(missing),
            "mapping_size": len(piece_mapping),
        })

        return piece_mapping

    # ─────────────────────────────────────────────────────────────────────────
    # Red-Flag Rules (RF-01 to RF-09)
    # ─────────────────────────────────────────────────────────────────────────

    async def _compute_red_flags(
        self,
        matrix: List[Dict[str, Any]],
        piece_mapping: Dict[str, int],
        export_case_id: Optional[str],
        result: WorkflowResult,
    ) -> None:
        """
        Compute red-flag warnings for each claim (helper).

        Implements RF-01 through RF-09:
        - RF-01: no_evidence (aucune preuve)
        - RF-02: single_source_risk (preuve unique)
        - RF-03: date_without_evidence (date sans preuve datée)
        - RF-04: amount_without_evidence (montant sans preuve chiffrée)
        - RF-05: contradiction_official (contradiction avec doc officiel)
        - RF-06: direct_contradiction (contradiction directe)
        - RF-07: evidence_out_of_chronology (preuve hors chronologie)
        - RF-08: citation_no_locator (citation sans localisation)
        - RF-09: evidence_not_in_case (preuve non versée au dossier)
        """
        red_flag_summary = {
            "total": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "by_code": {},
        }

        for entry in matrix:
            flags = []
            claim_text = entry.get("claim", "").lower()
            supporting = entry.get("supporting", [])
            contradicting = entry.get("contradicting", [])
            strength = entry.get("strength", "unknown")

            # ─────────────────────────────────────────────────────────────
            # RF-01: Aucune preuve (no evidence)
            # ─────────────────────────────────────────────────────────────
            if strength == "no_evidence" or not supporting:
                flags.append({
                    "code": "RF-01",
                    "severity": "high",
                    "message": "No supporting evidence found for this claim",
                    "message_fr": "Aucune preuve à l'appui de cette allégation",
                    "action": "ingest_missing_doc",
                    "hint": "Search for invoice, letter, contract, or minutes",
                })

            # ─────────────────────────────────────────────────────────────
            # RF-02: Preuve unique (single source risk)
            # ─────────────────────────────────────────────────────────────
            if supporting and not flags:  # Skip if RF-01 already triggered
                unique_docs = set(e.get("document_id") for e in supporting if e.get("document_id"))
                if len(unique_docs) == 1:
                    flags.append({
                        "code": "RF-02",
                        "severity": "medium",
                        "message": "Evidence from single source only",
                        "message_fr": "Preuve provenant d'une seule source",
                        "action": "find_independent_source",
                        "hint": "Look for email, annex, or testimony to corroborate",
                    })

            # ─────────────────────────────────────────────────────────────
            # RF-03: Date claim without dated evidence
            # ─────────────────────────────────────────────────────────────
            date_patterns = ["janvier", "février", "mars", "avril", "mai", "juin",
                           "juillet", "août", "septembre", "octobre", "novembre", "décembre",
                           "january", "february", "march", "april", "may", "june",
                           "july", "august", "september", "october", "november", "december",
                           "/20", "/19", "-20", "-19"]
            has_date_in_claim = any(p in claim_text for p in date_patterns)

            if has_date_in_claim and supporting:
                # Check if any evidence has date entity matches
                has_dated_evidence = False
                for ev in supporting:
                    entity_matches = ev.get("entity_matches", [])
                    if any("date" in str(em).lower() for em in entity_matches):
                        has_dated_evidence = True
                        break

                if not has_dated_evidence:
                    flags.append({
                        "code": "RF-03",
                        "severity": "high",
                        "message": "Claim contains date but no dated evidence found",
                        "message_fr": "Allégation datée sans preuve datée correspondante",
                        "action": "verify_date",
                        "hint": "Search for signed/dated document",
                    })

            # ─────────────────────────────────────────────────────────────
            # RF-04: Amount claim without amount evidence
            # ─────────────────────────────────────────────────────────────
            amount_patterns = ["€", "$", "eur", "euro", "dollars", "francs",
                             "000", "préjudice", "dommage", "montant", "somme"]
            has_amount_in_claim = any(p in claim_text for p in amount_patterns)

            if has_amount_in_claim and supporting:
                has_amount_evidence = False
                for ev in supporting:
                    entity_matches = ev.get("entity_matches", [])
                    quote = (ev.get("quote", "") or "").lower()
                    if any("amount" in str(em).lower() or "money" in str(em).lower() for em in entity_matches):
                        has_amount_evidence = True
                        break
                    if any(p in quote for p in ["€", "$", "euro"]):
                        has_amount_evidence = True
                        break

                if not has_amount_evidence:
                    flags.append({
                        "code": "RF-04",
                        "severity": "high",
                        "message": "Claim contains amount but no financial evidence found",
                        "message_fr": "Allégation chiffrée sans preuve financière",
                        "action": "request_financial_proof",
                        "hint": "Invoice, quote, expert accounting report",
                    })

            # ─────────────────────────────────────────────────────────────
            # RF-05: Contradiction with official document
            # (checked via contradiction_notes from deep analysis)
            # ─────────────────────────────────────────────────────────────
            contradiction_notes = result.data.get("contradiction_notes", [])
            entry_contras = [cn for cn in contradiction_notes if cn.get("claim_index") == entry.get("index")]

            if entry_contras:
                flags.append({
                    "code": "RF-05",
                    "severity": "high",
                    "message": "Contradiction detected between documents",
                    "message_fr": "Contradiction détectée entre documents",
                    "action": "prioritize_contradiction_analysis",
                    "hint": "Analyze probative value hierarchy",
                })

            # ─────────────────────────────────────────────────────────────
            # RF-06: Direct contradiction (supporting vs contradicting)
            # ─────────────────────────────────────────────────────────────
            if supporting and contradicting:
                # Check if contradiction is direct (high relevance on both sides)
                sup_best = max((e.get("relevance", 0) for e in supporting), default=0)
                con_best = max((e.get("relevance", 0) for e in contradicting), default=0)

                if sup_best > 0.7 and con_best > 0.7:
                    flags.append({
                        "code": "RF-06",
                        "severity": "high",
                        "message": "Direct contradiction between key documents",
                        "message_fr": "Contradiction directe entre pièces clés",
                        "action": "prepare_response_to_contradiction",
                        "hint": "Evidence hierarchy, chronology analysis",
                    })

            # ─────────────────────────────────────────────────────────────
            # RF-08: Citation without precise locator
            # ─────────────────────────────────────────────────────────────
            for ev in supporting:
                locator = ev.get("locator", {})
                if not locator.get("page") and locator.get("chunk_id"):
                    flags.append({
                        "code": "RF-08",
                        "severity": "low",
                        "message": "Citation without page number",
                        "message_fr": "Citation sans numéro de page",
                        "action": "enrich_locator",
                        "hint": "Attach to specific page/piece",
                    })
                    break  # Only flag once per claim

            # ─────────────────────────────────────────────────────────────
            # RF-09: Evidence not in case (piece mapping check)
            # ─────────────────────────────────────────────────────────────
            if export_case_id and piece_mapping:
                for ev in supporting + contradicting:
                    doc_id = ev.get("document_id")
                    if doc_id and str(doc_id) not in piece_mapping:
                        flags.append({
                            "code": "RF-09",
                            "severity": "medium",
                            "message": f"Evidence document {doc_id} not in case pieces",
                            "message_fr": f"Pièce {doc_id} non versée au dossier",
                            "action": "add_piece",
                            "hint": "Add document to bordereau",
                        })
                        break  # Only flag once per claim

            # Store flags in entry
            entry["red_flags"] = flags

            # Update summary
            for flag in flags:
                red_flag_summary["total"] += 1
                severity = flag.get("severity", "low")
                red_flag_summary[severity] = red_flag_summary.get(severity, 0) + 1

                code = flag.get("code", "unknown")
                red_flag_summary["by_code"][code] = red_flag_summary["by_code"].get(code, 0) + 1

        result.data["red_flag_summary"] = red_flag_summary

        logger.info(
            f"[{result.trace_id}] Red-flags: {red_flag_summary['total']} total "
            f"(H:{red_flag_summary['high']}, M:{red_flag_summary['medium']}, L:{red_flag_summary['low']})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 6: Prepare Hearing (One-Button)
    # ─────────────────────────────────────────────────────────────────────────

    async def prepare_hearing(
        self,
        case_id: str,
        project_id: str,
        questions: Optional[List[str]] = None,
        export_format: str = "html",
        auto_export: bool = True,
        validate_integrity: bool = True,
        generate_hearing_pack: bool = True,
        max_doc_pairs: int = 50,
        trace_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        One-button hearing preparation.

        Complete workflow for preparing a case for court hearing:
        1. Build case brief with evidence matrix (court profile)
        2. Get case state and blocking issues
        3. Validate export completeness
        4. Create integrity hash chain
        5. Generate hearing pack note
        6. Optionally export bundle

        Args:
            case_id: Case ID for export/pieces
            project_id: Project ID for analysis
            questions: Optional list of key questions to research
            export_format: Bundle format (html, pdf, zip)
            auto_export: Automatically export bundle if valid
            validate_integrity: Create and verify integrity hash
            generate_hearing_pack: Create HEARING_PACK note as artifact
            max_doc_pairs: Max document pairs for deep contradiction check (perf limit)
            trace_id: Optional trace ID for correlation

        Returns:
            WorkflowResult with unified envelope format (use .to_envelope()):
            - outputs.brief: Full case brief data
            - outputs.case_state: Current state and blocking issues
            - outputs.validation: Export validation results
            - artifacts.integrity: Hash chain for court verification
            - artifacts.hearing_pack: Generated note (if enabled)
            - action_plan: Unified P0/P1/P2 task list
            - warnings: Non-fatal issues
            - timings_ms: Step-level timing
        """
        trace_id = trace_id or self._new_trace_id()
        result = WorkflowResult(
            success=True,
            trace_id=trace_id,
            scenario="prepare_hearing",
            project_id=project_id,
            case_id=case_id,
            inputs={
                "case_id": case_id,
                "project_id": project_id,
                "questions": questions,
                "export_format": export_format,
                "auto_export": auto_export,
                "validate_integrity": validate_integrity,
                "generate_hearing_pack": generate_hearing_pack,
                "max_doc_pairs": max_doc_pairs,
            },
        )

        logger.info(f"[{trace_id}] prepare_hearing: case={case_id}, project={project_id}")

        try:
            # ─────────────────────────────────────────────────────────────────
            # Step 1: Get initial case state (for delta comparison)
            # ─────────────────────────────────────────────────────────────────
            initial_state = None
            try:
                initial_state = await self.legal.get_case_state(case_id=case_id)
                result.add_step("legal.get_case_state[initial]", initial_state)
            except Exception as e:
                logger.warning(f"[{trace_id}] Could not get initial case state: {e}")

            # ─────────────────────────────────────────────────────────────────
            # Step 2: Build comprehensive case brief
            # ─────────────────────────────────────────────────────────────────
            brief_result = await self.build_case_brief(
                project_id=project_id,
                case_id=case_id,
                include_evidence_matrix=True,
                include_action_plan=True,
                export_profile="court",
                auto_add_pieces=True,
                deterministic=True,
                max_doc_pairs=max_doc_pairs,
                trace_id=trace_id,
            )
            result.add_step("build_case_brief", {"success": brief_result.success})

            # Copy brief data to result
            result.data["brief"] = {
                "documents": brief_result.data.get("documents", []),
                "document_count": brief_result.data.get("document_count", 0),
                "timeline": brief_result.data.get("timeline", []),
                "contradictions": brief_result.data.get("contradictions", []),
                "evidence_matrix": brief_result.data.get("evidence_matrix", []),
                "points_litigieux": brief_result.data.get("points_litigieux", []),
                "summary": brief_result.data.get("summary", ""),
            }
            result.data["brief_stats"] = brief_result.data.get("brief_stats", {})
            result.data["action_plan"] = brief_result.data.get("action_plan", [])
            result.data["top_sources"] = brief_result.data.get("top_sources", [])
            result.data["warnings"] = brief_result.data.get("warnings", [])

            if not brief_result.success:
                result.errors.extend(brief_result.errors)

            # ─────────────────────────────────────────────────────────────────
            # Step 3: Answer key questions (if provided)
            # ─────────────────────────────────────────────────────────────────
            if questions:
                answers = []
                for q in questions[:5]:  # Max 5 questions
                    try:
                        qa_result = await self.legal_answer(
                            question=q,
                            project_id=project_id,
                            context_budget=1500,
                            trace_id=trace_id,
                        )
                        answers.append({
                            "question": q,
                            "answer": qa_result.data.get("answer", ""),
                            "sources": qa_result.data.get("sources", [])[:3],
                        })
                    except Exception as e:
                        answers.append({
                            "question": q,
                            "error": str(e),
                        })
                result.data["questions_answered"] = answers
                result.add_step("answer_questions", {"count": len(answers)})

            # ─────────────────────────────────────────────────────────────────
            # Step 4: Get final case state
            # ─────────────────────────────────────────────────────────────────
            try:
                case_state = await self.legal.get_case_state(case_id=case_id)
                result.add_step("legal.get_case_state[final]", case_state)

                result.data["case_state"] = {
                    "current_state": case_state.get("current_state", "unknown"),
                    "possible_transitions": case_state.get("possible_transitions", []),
                    "is_terminal": case_state.get("is_terminal", False),
                }

                # Extract blocking issues as P0 actions
                blocking_issues = case_state.get("blocking_issues", [])
                if blocking_issues:
                    result.data["blocking_issues"] = blocking_issues
                    # Add to action_plan as P0
                    for issue in blocking_issues:
                        result.data["action_plan"].insert(0, {
                            "priority": "P0",
                            "source": "blocking_issue",
                            "code": issue.get("code", "BLOCK"),
                            "action": "resolve_blocking_issue",
                            "description": issue.get("message", ""),
                            "severity": issue.get("severity", "high"),
                        })

                # Compute delta if we have initial state
                if initial_state:
                    result.data["delta"] = {
                        "initial_state": initial_state.get("current_state"),
                        "final_state": case_state.get("current_state"),
                        "state_changed": initial_state.get("current_state") != case_state.get("current_state"),
                        "blocking_resolved": len(initial_state.get("blocking_issues", [])) - len(blocking_issues),
                    }

            except Exception as e:
                logger.warning(f"[{trace_id}] Case state check failed: {e}")
                result.data["case_state"] = {"error": str(e)}

            # ─────────────────────────────────────────────────────────────────
            # Step 5: Validate export completeness
            # ─────────────────────────────────────────────────────────────────
            try:
                validation = await self.legal.validate_export(
                    case_id=case_id,
                    project_id=project_id,
                )
                result.add_step("legal.validate_export", validation)

                result.data["validation"] = {
                    "is_valid": validation.get("is_valid", False),
                    "is_ready": validation.get("is_ready", False),
                    "errors": validation.get("errors", []),
                    "warnings": validation.get("warnings", []),
                    "recommendations": validation.get("recommendations", []),
                }

                # Add validation warnings to action plan
                for warning in validation.get("warnings", []):
                    result.data["action_plan"].append({
                        "priority": "P1",
                        "source": "validation",
                        "action": "fix_validation_warning",
                        "description": warning,
                    })

            except Exception as e:
                logger.warning(f"[{trace_id}] Validation failed: {e}")
                result.data["validation"] = {"error": str(e)}

            # ─────────────────────────────────────────────────────────────────
            # Step 6: Create integrity hash (if enabled)
            # ─────────────────────────────────────────────────────────────────
            if validate_integrity:
                try:
                    hash_result = await self.legal.create_export_hash(case_id=case_id)
                    result.add_step("legal.create_export_hash", hash_result)

                    if not hash_result.get("error"):
                        # Verify immediately
                        verify_result = await self.legal.verify_export_integrity(
                            content_hash=hash_result.get("content_hash", ""),
                            pieces=hash_result.get("pieces", []),
                            tool_version="1.0.0",
                            timestamp=hash_result.get("timestamp", ""),
                        )
                        result.add_step("legal.verify_export_integrity", verify_result)

                        result.data["integrity"] = {
                            "export_id": hash_result.get("export_id"),
                            "content_hash": hash_result.get("content_hash"),
                            "pieces_count": hash_result.get("pieces_count", 0),
                            "timestamp": hash_result.get("timestamp"),
                            "verified": verify_result.get("is_valid", False),
                            "verdict": verify_result.get("verdict", "UNKNOWN"),
                        }
                    else:
                        result.data["integrity"] = {"error": hash_result.get("error")}

                except Exception as e:
                    logger.warning(f"[{trace_id}] Integrity hash failed: {e}")
                    result.data["integrity"] = {"error": str(e)}

            # ─────────────────────────────────────────────────────────────────
            # Step 7: Export bundle (if auto_export and valid)
            # ─────────────────────────────────────────────────────────────────
            is_ready = result.data.get("validation", {}).get("is_ready", False)

            if auto_export and is_ready:
                try:
                    export_result = await self.legal.export_bundle(
                        case_id=case_id,
                        format=export_format,
                    )
                    result.add_step("legal.export_bundle", export_result)

                    result.data["artifacts"] = {
                        "exported": True,
                        "format": export_format,
                        "title": export_result.get("title", ""),
                        "size_bytes": export_result.get("size_bytes"),
                        # Don't include base64 content in result
                        "has_content": bool(export_result.get("content_base64") or export_result.get("html_content")),
                    }

                except Exception as e:
                    logger.warning(f"[{trace_id}] Export failed: {e}")
                    result.data["artifacts"] = {"exported": False, "error": str(e)}
            else:
                result.data["artifacts"] = {
                    "exported": False,
                    "reason": "validation_failed" if not is_ready else "auto_export_disabled",
                }

            # ─────────────────────────────────────────────────────────────────
            # Step 8: Generate Hearing Pack Note (if enabled)
            # ─────────────────────────────────────────────────────────────────
            if generate_hearing_pack:
                try:
                    hearing_pack = await self._generate_hearing_pack(
                        result=result,
                        case_id=case_id,
                        project_id=project_id,
                    )
                    result.add_step("generate_hearing_pack", {"success": bool(hearing_pack)})

                    if hearing_pack:
                        result.data["hearing_pack"] = hearing_pack

                        # Add as Pièce if auto_export is enabled
                        if auto_export and hearing_pack.get("document_id"):
                            try:
                                await self.legal.add_piece_to_case(
                                    case_id=case_id,
                                    document_id=hearing_pack["document_id"],
                                    label="Note de synthèse (généré)",
                                )
                                result.data["hearing_pack"]["added_as_piece"] = True
                                logger.info(f"[{trace_id}] Hearing pack added as piece")
                            except Exception as e:
                                logger.warning(f"[{trace_id}] Failed to add hearing pack as piece: {e}")

                except Exception as e:
                    logger.warning(f"[{trace_id}] Hearing pack generation failed: {e}")
                    result.data["hearing_pack"] = {"error": str(e)}

            # ─────────────────────────────────────────────────────────────────
            # Final summary
            # ─────────────────────────────────────────────────────────────────
            # Sort action plan by priority
            priority_order = {"P0": 0, "P1": 1, "P2": 2}
            result.data["action_plan"].sort(
                key=lambda a: priority_order.get(a.get("priority", "P2"), 3)
            )

            # Compute readiness summary
            p0_count = len([a for a in result.data["action_plan"] if a.get("priority") == "P0"])
            is_hearing_ready = (
                is_ready
                and p0_count == 0
                and result.data.get("integrity", {}).get("verified", False)
            )

            result.data["hearing_readiness"] = {
                "ready": is_hearing_ready,
                "blocking_p0": p0_count,
                "validation_passed": is_ready,
                "integrity_verified": result.data.get("integrity", {}).get("verified", False),
                "artifacts_exported": result.data.get("artifacts", {}).get("exported", False),
            }

            # Add final warning if not ready
            if not is_hearing_ready:
                reasons = []
                if p0_count > 0:
                    reasons.append(f"{p0_count} blocking issues (P0)")
                if not is_ready:
                    reasons.append("validation failed")
                if not result.data.get("integrity", {}).get("verified", False):
                    reasons.append("integrity not verified")

                result.data["warnings"].append(
                    f"NOT READY for hearing: {', '.join(reasons)}"
                )

            logger.info(
                f"[{trace_id}] prepare_hearing: ready={is_hearing_ready}, "
                f"P0={p0_count}, validated={is_ready}"
            )

        except Exception as e:
            logger.error(f"[{trace_id}] prepare_hearing failed: {e}")
            result.success = False
            result.errors.append(str(e))

        return result

    async def _generate_hearing_pack(
        self,
        result: WorkflowResult,
        case_id: str,
        project_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate HEARING_PACK note from prepare_hearing results.

        Creates a markdown document with:
        - Résumé (1 paragraph summary)
        - Points litigieux (top 10 contested claims)
        - Best evidence (1-2 quotes per point)
        - Contradictions (high severity only)
        - Action plan P0/P1 (max 10)
        - Integrity hash

        Ingests as document and returns metadata.
        """
        from datetime import datetime

        lines = []

        # ═══════════════════════════════════════════════════════════════════
        # Header
        # ═══════════════════════════════════════════════════════════════════
        lines.append("# NOTE DE SYNTHÈSE - PRÉPARATION AUDIENCE")
        lines.append("")
        lines.append(f"**Dossier:** {case_id}")
        lines.append(f"**Projet:** {project_id}")
        lines.append(f"**Date:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        lines.append(f"**Trace ID:** {result.trace_id}")
        lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # Résumé (from brief summary)
        # ═══════════════════════════════════════════════════════════════════
        lines.append("## 1. RÉSUMÉ")
        lines.append("")
        summary = result.data.get("brief", {}).get("summary", "")
        if summary:
            # Take first paragraph or first 500 chars
            first_para = summary.split("\n\n")[0][:500]
            lines.append(first_para)
        else:
            doc_count = result.data.get("brief_stats", {}).get("document_count", 0)
            lines.append(f"Dossier comportant {doc_count} documents analysés.")
        lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # Points litigieux (top 10)
        # ═══════════════════════════════════════════════════════════════════
        lines.append("## 2. POINTS LITIGIEUX")
        lines.append("")

        points = result.data.get("brief", {}).get("points_litigieux", [])
        if points:
            for i, pt in enumerate(points[:10], 1):
                claim = pt.get("claim", "")[:100]
                issue = pt.get("issue", "")
                lines.append(f"{i}. **{claim}**")
                if issue:
                    lines.append(f"   - Issue: {issue}")
                lines.append("")
        else:
            lines.append("_Aucun point litigieux identifié._")
            lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # Best Evidence (per point)
        # ═══════════════════════════════════════════════════════════════════
        lines.append("## 3. PREUVES PRINCIPALES")
        lines.append("")

        matrix = result.data.get("brief", {}).get("evidence_matrix", [])
        if matrix:
            for entry in matrix[:10]:
                claim = entry.get("claim", "")[:80]
                piece = entry.get("pièce", entry.get("citation", ""))
                quote = entry.get("quote", "")[:150]

                lines.append(f"### {claim}")
                if piece:
                    lines.append(f"- **{piece}**")
                if quote:
                    lines.append(f"  > \"{quote}...\"")
                lines.append("")
        else:
            lines.append("_Aucune preuve analysée._")
            lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # Contradictions (high severity only)
        # ═══════════════════════════════════════════════════════════════════
        lines.append("## 4. CONTRADICTIONS CRITIQUES")
        lines.append("")

        contradictions = result.data.get("brief", {}).get("contradictions", [])
        high_contradictions = [
            c for c in contradictions
            if c.get("severity") in ("critical", "high")
        ]

        if high_contradictions:
            for i, c in enumerate(high_contradictions[:5], 1):
                desc = c.get("description", "")[:150]
                severity = c.get("severity", "")
                lines.append(f"{i}. [{severity.upper()}] {desc}")
            lines.append("")
        else:
            lines.append("_Aucune contradiction critique détectée._")
            lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # Action Plan P0/P1
        # ═══════════════════════════════════════════════════════════════════
        lines.append("## 5. ACTIONS REQUISES")
        lines.append("")

        action_plan = result.data.get("action_plan", [])
        p0_p1 = [a for a in action_plan if a.get("priority") in ("P0", "P1")]

        if p0_p1:
            for i, a in enumerate(p0_p1[:10], 1):
                pri = a.get("priority", "P1")
                action = a.get("action", "")
                desc = a.get("description", "")[:80]
                lines.append(f"{i}. **[{pri}]** {action}")
                if desc:
                    lines.append(f"   - {desc}")
            lines.append("")
        else:
            lines.append("_Aucune action requise._")
            lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # Integrity Hash
        # ═══════════════════════════════════════════════════════════════════
        lines.append("## 6. INTÉGRITÉ")
        lines.append("")

        integrity = result.data.get("integrity", {})
        if integrity.get("content_hash"):
            lines.append(f"- **Hash:** `{integrity.get('content_hash')}`")
            lines.append(f"- **Vérifié:** {'✓ Oui' if integrity.get('verified') else '✗ Non'}")
            lines.append(f"- **Verdict:** {integrity.get('verdict', 'N/A')}")
            lines.append(f"- **Timestamp:** {integrity.get('timestamp', 'N/A')}")
        else:
            lines.append("_Hash d'intégrité non disponible._")
        lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # Hearing Readiness
        # ═══════════════════════════════════════════════════════════════════
        lines.append("---")
        lines.append("")

        readiness = result.data.get("hearing_readiness", {})
        if readiness.get("ready"):
            lines.append("✅ **DOSSIER PRÊT POUR L'AUDIENCE**")
        else:
            lines.append("⚠️ **DOSSIER NON PRÊT POUR L'AUDIENCE**")
            if readiness.get("blocking_p0", 0) > 0:
                lines.append(f"   - {readiness['blocking_p0']} actions critiques (P0) en attente")
            if not readiness.get("validation_passed"):
                lines.append("   - Validation export échouée")
            if not readiness.get("integrity_verified"):
                lines.append("   - Intégrité non vérifiée")

        lines.append("")
        lines.append("---")
        lines.append(f"_Généré automatiquement par BrainStorm Orchestrator v1.0_")

        # Join all lines
        content = "\n".join(lines)

        # Ingest as document
        try:
            ingest_result = await self.legal.ingest_text(
                content=content,
                title=f"HEARING_PACK_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                project_id=project_id,
            )

            doc_id = ingest_result.get("document_id", ingest_result.get("id"))

            logger.info(f"[{result.trace_id}] Hearing pack generated: {doc_id}")

            return {
                "document_id": doc_id,
                "title": f"Note de synthèse - {case_id}",
                "format": "markdown",
                "char_count": len(content),
                "sections": ["résumé", "points_litigieux", "preuves", "contradictions", "actions", "intégrité"],
            }

        except Exception as e:
            logger.warning(f"[{result.trace_id}] Failed to ingest hearing pack: {e}")
            return {
                "error": str(e),
                "content_preview": content[:500],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Check health of both MCP servers."""
        braine_health = await self.braine.health_check()
        legal_health = await self.legal.health_check()

        all_healthy = (
            braine_health.get("status") == "healthy"
            and legal_health.get("status") == "healthy"
        )

        return {
            "orchestrator": "healthy" if all_healthy else "degraded",
            "braine": braine_health,
            "legal": legal_health,
        }

    def __repr__(self) -> str:
        connected = self._braine is not None and self._legal is not None
        return f"<BrainStormOrchestrator [{'connected' if connected else 'disconnected'}]>"


# ─────────────────────────────────────────────────────────────────────────────
# CLI Demo
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Demo usage of orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    async with BrainStormOrchestrator() as orch:
        print("=== BrainStorm Orchestrator Demo ===\n")

        # Health check
        health = await orch.health_check()
        print(f"Health: {health['orchestrator']}")
        print(f"  BraineMemory: {health['braine']['status']} ({health['braine'].get('tools_count', 0)} tools)")
        print(f"  LegalOps: {health['legal']['status']} ({health['legal'].get('tools_count', 0)} tools)")
        print()

        # Demo Scenario 3: Legal Answer
        print("=== Scenario: legal_answer ===")
        result = await orch.legal_answer(
            question="What are the main obligations of the parties?",
            context_budget=2000,
        )
        print(f"Success: {result.success}")
        print(f"Trace ID: {result.trace_id}")
        print(f"Steps: {len(result.steps)}")
        if result.errors:
            print(f"Errors: {result.errors}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
