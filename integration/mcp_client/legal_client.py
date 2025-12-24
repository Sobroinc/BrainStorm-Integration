# -*- coding: utf-8 -*-
"""
LegalOps MCP Client - Typed wrapper for 68 legal document tools.

Provides type-safe methods for document ingestion, entity extraction,
contradiction detection, and case bundle management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from integration.config import config
from integration.mcp_client.sse_session import MCPSession, connect_sse, connect_http

logger = logging.getLogger(__name__)

# Required tools for LegalOps (minimum viable set)
LEGAL_REQUIRED_TOOLS: Set[str] = {
    "ingest_pdf",
    "search_documents",
    "ask_question",
    "detect_contradictions",
    "extract_entities",
}


class LegalOpsClient:
    """
    Typed MCP client for LegalOps.

    Provides access to 68 tools organized by layer:
    - L2: Document Operations
    - L3: Entity Operations
    - L4: Analysis (Contradictions)
    - L5: Evidence
    - L6: Drafting
    - L7: Bundle/Case Management
    - L8: Legal References

    Usage:
        async with LegalOpsClient() as legal:
            # Search documents
            result = await legal.search_documents("contract clause")

            # Detect contradictions
            result = await legal.detect_contradictions(["doc:1", "doc:2"])

            # Get case bundle
            result = await legal.get_bordereau("case:123")
    """

    def __init__(
        self,
        url: Optional[str] = None,
        transport: str = "sse",
        timeout: Optional[int] = None,
        max_inflight: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize LegalOps client.

        Args:
            url: MCP endpoint URL (default from config)
            transport: Transport type ("sse" or "http")
            timeout: Connection timeout in seconds (default from config)
            max_inflight: Max parallel calls (default from config)
            max_retries: Max retries on transport errors (default from config)
        """
        if url is None:
            url = config.legal_sse_url if transport == "sse" else config.legal_http_url
        self.url = url
        self.transport = transport
        self.timeout = timeout or int(config.mcp_timeout)
        self.max_inflight = max_inflight or config.legal_max_inflight
        self.max_retries = max_retries or config.mcp_max_retries
        self._session: Optional[MCPSession] = None
        self._context_manager = None

    async def __aenter__(self) -> "LegalOpsClient":
        """Connect to LegalOps MCP server."""
        if self.transport == "sse":
            self._context_manager = connect_sse(
                self.url, self.timeout, self.max_inflight, self.max_retries
            )
        else:
            self._context_manager = connect_http(
                self.url, self.timeout, self.max_inflight, self.max_retries
            )
        self._session = await self._context_manager.__aenter__()

        # Check required capabilities
        await self._session.assert_capabilities(LEGAL_REQUIRED_TOOLS)

        # Log available tools
        tools_count = len(self._session.tools_cache)
        logger.info(f"Connected to LegalOps at {self.url} ({tools_count} tools)")
        return self

    async def __aexit__(self, *args) -> None:
        """Disconnect from LegalOps MCP server."""
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(*args)
            except RuntimeError as e:
                # Handle anyio cancel scope task boundary issue
                if "cancel scope" in str(e):
                    logger.debug(f"Ignoring cancel scope warning on disconnect: {e}")
                else:
                    raise
        self._session = None
        self._context_manager = None
        logger.debug("Disconnected from LegalOps")

    @property
    def session(self) -> MCPSession:
        """Get active session (raises if not connected)."""
        if self._session is None:
            raise RuntimeError("Not connected. Use 'async with LegalOpsClient()' context.")
        return self._session

    # ─────────────────────────────────────────────────────────────────────────
    # L2: Document Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def ingest_pdf(
        self,
        pdf_path: str,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Ingest a PDF document.

        Args:
            pdf_path: Path to PDF file
            project_id: Project ID (uses default if not specified)

        Returns:
            Ingestion result with document_id
        """
        args = {"pdf_path": pdf_path, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("ingest_pdf", args)

    async def ingest_text(
        self,
        content: str,
        title: str,
        project_id: Optional[str] = None,
        ner_mode: str = "quick",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Ingest text content.

        Args:
            content: Text content
            title: Document title (becomes filename)
            project_id: Project ID
            ner_mode: Entity extraction mode ("quick", "full", "off")

        Returns:
            Ingestion result with document_id
        """
        args = {
            "text": content,  # API uses 'text' not 'content'
            "filename": title,  # API uses 'filename' not 'title'
            "ner_mode": ner_mode,
            **kwargs,
        }
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("ingest_text", args)

    async def search_documents(
        self,
        query: str,
        project_id: Optional[str] = None,
        limit: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search documents using hybrid search.

        Args:
            query: Search query
            project_id: Limit to project
            limit: Max results

        Returns:
            Search results with documents
        """
        args = {"query": query, "limit": limit, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("search_documents", args)

    async def smart_search(
        self,
        query: str,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Smart search with automatic query expansion.

        Args:
            query: Natural language query
            project_id: Limit to project

        Returns:
            Enhanced search results
        """
        args = {"query": query, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("smart_search", args)

    async def ask_question(
        self,
        question: str,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Ask a question and get RAG-powered answer.

        Args:
            question: Question in natural language
            project_id: Limit to project

        Returns:
            Answer with sources
        """
        args = {"question": question, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("ask_question", args)

    async def get_document(
        self,
        document_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get document by ID."""
        return await self.session.call_tool("get_document", {
            "document_id": document_id,
            **kwargs,
        })

    async def list_documents(
        self,
        project_id: Optional[str] = None,
        limit: int = 50,
        **kwargs,
    ) -> Dict[str, Any]:
        """List documents in project."""
        args = {"limit": limit, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("list_documents", args)

    async def list_projects(self, **kwargs) -> Dict[str, Any]:
        """List all projects."""
        return await self.session.call_tool("list_projects", kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # L3: Entity Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def extract_entities(
        self,
        document_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Extract named entities from document.

        Args:
            document_id: Document to process

        Returns:
            Extracted entities (persons, organizations, dates, etc.)
        """
        return await self.session.call_tool("extract_entities", {
            "document_id": document_id,
            **kwargs,
        })

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search for entities.

        Args:
            query: Search query
            entity_type: Filter by type (person, organization, etc.)
            project_id: Limit to project

        Returns:
            Matching entities
        """
        args = {"query": query, **kwargs}
        if entity_type:
            args["entity_type"] = entity_type
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("search_entities", args)

    async def get_entity_graph(
        self,
        entity_id: str,
        depth: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get entity relationship graph.

        Args:
            entity_id: Starting entity
            depth: Graph traversal depth

        Returns:
            Entity graph with relations
        """
        return await self.session.call_tool("get_entity_graph", {
            "entity_id": entity_id,
            "depth": depth,
            **kwargs,
        })

    async def search_by_party(
        self,
        party_name: str,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search documents by party name.

        Args:
            party_name: Name of party to search
            project_id: Limit to project

        Returns:
            Documents involving this party
        """
        args = {"party_name": party_name, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("search_by_party", args)

    # ─────────────────────────────────────────────────────────────────────────
    # L4: Analysis (Contradictions)
    # ─────────────────────────────────────────────────────────────────────────

    async def detect_contradictions(
        self,
        document_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Detect contradictions between documents.

        Args:
            document_ids: Specific documents to compare
            project_id: Scan entire project

        Returns:
            Detected contradictions with severity
        """
        args = {**kwargs}
        if document_ids:
            args["document_ids"] = document_ids
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("detect_contradictions", args)

    async def get_contradictions(
        self,
        project_id: Optional[str] = None,
        status: str = "open",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get existing contradictions.

        Args:
            project_id: Filter by project
            status: Filter by status (open, reviewed, resolved)

        Returns:
            List of contradictions
        """
        args = {"status": status, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("get_contradictions", args)

    async def review_contradiction(
        self,
        contradiction_id: str,
        status: str,
        notes: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Review and update contradiction status.

        Args:
            contradiction_id: Contradiction to review
            status: New status
            notes: Review notes

        Returns:
            Updated contradiction
        """
        args = {
            "contradiction_id": contradiction_id,
            "status": status,
            **kwargs,
        }
        if notes:
            args["notes"] = notes
        return await self.session.call_tool("review_contradiction", args)

    async def compare_documents(
        self,
        doc_id_1: str,
        doc_id_2: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compare two documents for differences.

        Args:
            doc_id_1: First document
            doc_id_2: Second document

        Returns:
            Comparison results with differences
        """
        return await self.session.call_tool("compare_documents", {
            "doc_id_1": doc_id_1,
            "doc_id_2": doc_id_2,
            **kwargs,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # L5: Evidence
    # ─────────────────────────────────────────────────────────────────────────

    async def find_evidence(
        self,
        claim: str,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Find evidence for a claim.

        Args:
            claim: Claim to find evidence for
            project_id: Limit to project

        Returns:
            Supporting and contradicting evidence
        """
        args = {"claim": claim, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("find_evidence", args)

    async def analyze_coverage(
        self,
        text: str,
        project_id: Optional[str] = None,
        split_mode: str = "paragraph",
        min_score: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze evidence coverage for a text.

        Splits text into claims and finds evidence for each.

        Args:
            text: Text to analyze (draft, claims, etc.)
            project_id: Project to search evidence in
            split_mode: How to split text (paragraph, sentence, line)
            min_score: Minimum score for "covered" status

        Returns:
            Coverage analysis with per-claim status
        """
        args = {
            "text": text,
            "split_mode": split_mode,
            "min_score": min_score,
            **kwargs,
        }
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("analyze_coverage", args)

    async def audit_draft(
        self,
        text: str,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full audit of a draft with evidence coverage.

        Args:
            text: Draft text to audit
            project_id: Project to audit against

        Returns:
            Audit results with recommendations
        """
        args = {"text": text, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("audit_draft", args)

    async def get_project_readiness(
        self,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Check project readiness for evidence analysis.

        Args:
            project_id: Project to check

        Returns:
            Readiness status and recommendations
        """
        args = {**kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("get_project_readiness", args)

    async def get_citation_format(
        self,
        document_id: str,
        page_number: Optional[int] = None,
        style: str = "french_court",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get formatted legal citation for a document.

        Args:
            document_id: Document to cite
            page_number: Optional page number
            style: Citation style (french_court, academic, short)

        Returns:
            Formatted citation string
        """
        args = {"document_id": document_id, "style": style, **kwargs}
        if page_number is not None:
            args["page_number"] = page_number
        return await self.session.call_tool("get_citation_format", args)

    async def compare_documents_for_contradictions(
        self,
        doc_a_id: str,
        doc_b_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compare two documents for contradictions.

        Deep comparison finding specific contradicting statements.

        Args:
            doc_a_id: First document ID
            doc_b_id: Second document ID

        Returns:
            List of contradictions with statements from each doc
        """
        return await self.session.call_tool("compare_documents_for_contradictions", {
            "doc_a_id": doc_a_id,
            "doc_b_id": doc_b_id,
            **kwargs,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # L6: Drafting
    # ─────────────────────────────────────────────────────────────────────────

    async def annotate_text(
        self,
        text: str,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Annotate text with citations.

        Args:
            text: Text to annotate
            project_id: Project for source lookup

        Returns:
            Annotated text with citations
        """
        args = {"text": text, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("annotate_text", args)

    async def generate_skeleton(
        self,
        case_id: str,
        template: str = "conclusions",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate document skeleton.

        Args:
            case_id: Case to generate for
            template: Template type (conclusions, etc.)

        Returns:
            Generated skeleton
        """
        return await self.session.call_tool("generate_skeleton", {
            "case_id": case_id,
            "template": template,
            **kwargs,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # L7: Bundle / Case Management
    # ─────────────────────────────────────────────────────────────────────────

    async def create_case(
        self,
        name: str,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a new case.

        Args:
            name: Case name
            project_id: Parent project

        Returns:
            Created case details
        """
        args = {"name": name, **kwargs}
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("create_case", args)

    async def add_piece(
        self,
        case_id: str,
        document_id: str,
        piece_number: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add a piece to a case.

        Args:
            case_id: Case ID
            document_id: Document to add
            piece_number: Optional piece number

        Returns:
            Added piece details
        """
        args = {
            "case_id": case_id,
            "document_id": document_id,
            **kwargs,
        }
        if piece_number is not None:
            args["piece_number"] = piece_number
        return await self.session.call_tool("add_piece", args)

    async def list_case_pieces(
        self,
        case_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """List all pieces in a case."""
        return await self.session.call_tool("list_case_pieces", {
            "case_id": case_id,
            **kwargs,
        })

    async def add_piece_to_case(
        self,
        case_id: str,
        document_id: str,
        label: Optional[str] = None,
        piece_no: Optional[int] = None,
        pages_used: str = "all",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add a document as a numbered piece to a case.

        Args:
            case_id: Case ID
            document_id: Document ID to add
            label: Piece label/description (auto from doc if not specified)
            piece_no: Piece number (auto-assigned if not specified)
            pages_used: Which pages to include ("all", "1-5", "1,3,5")

        Returns:
            Created piece details including piece_no
        """
        args: Dict[str, Any] = {
            "case_id": case_id,
            "document_id": document_id,
            "pages_used": pages_used,
            **kwargs,
        }
        if label:
            args["label"] = label
        if piece_no is not None:
            args["piece_no"] = piece_no

        return await self.session.call_tool("add_piece", args)

    async def get_bordereau(
        self,
        case_id: str,
        format: str = "markdown",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get bordereau (document inventory) for a case.

        Args:
            case_id: Case ID
            format: Output format (markdown, html, pdf)

        Returns:
            Bordereau content
        """
        return await self.session.call_tool("get_bordereau", {
            "case_id": case_id,
            "format": format,
            **kwargs,
        })

    async def get_timeline(
        self,
        case_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get timeline for a case.

        Args:
            case_id: Case ID

        Returns:
            Timeline with events
        """
        return await self.session.call_tool("get_timeline", {
            "case_id": case_id,
            **kwargs,
        })

    async def export_bundle(
        self,
        case_id: str,
        format: str = "zip",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Export case bundle.

        Args:
            case_id: Case to export
            format: Export format (zip, pdf)

        Returns:
            Export result with path
        """
        return await self.session.call_tool("export_bundle", {
            "case_id": case_id,
            "format": format,
            **kwargs,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Export Validation & Integrity (L7 Meta)
    # ─────────────────────────────────────────────────────────────────────────

    async def validate_export(
        self,
        case_id: Optional[str] = None,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate case/project for export completeness.

        Checks for missing pieces, unresolved contradictions, coverage gaps.

        Args:
            case_id: Case ID (optional)
            project_id: Project ID (optional)

        Returns:
            Validation results with is_valid, is_ready, warnings, errors
        """
        args: Dict[str, Any] = {**kwargs}
        if case_id:
            args["case_id"] = case_id
        if project_id:
            args["project_id"] = project_id
        return await self.session.call_tool("validate_export", args)

    async def get_case_state(
        self,
        case_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get case state and blocking issues for next transition.

        Args:
            case_id: Case ID

        Returns:
            State info with current_state, possible_transitions, blocking_issues
        """
        return await self.session.call_tool("get_case_state", {
            "case_id": case_id,
            **kwargs,
        })

    async def transition_case_state(
        self,
        case_id: str,
        to_state: str,
        force: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transition case to a new state.

        Args:
            case_id: Case ID
            to_state: Target state (evidence_ready, reviewed, finalized, exported, filed)
            force: Skip blocking issue checks (dangerous)

        Returns:
            Transition result with success, blocking_issues
        """
        return await self.session.call_tool("transition_case_state", {
            "case_id": case_id,
            "to_state": to_state,
            "force": force,
            **kwargs,
        })

    async def create_export_hash(
        self,
        case_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create deterministic hash for case export.

        Generates content hash for tamper detection and court verification.

        Args:
            case_id: Case ID

        Returns:
            Export manifest with content_hash, pieces, verification_info
        """
        return await self.session.call_tool("create_export_hash", {
            "case_id": case_id,
            **kwargs,
        })

    async def verify_export_integrity(
        self,
        content_hash: str,
        pieces: List[Dict[str, Any]],
        tool_version: str,
        timestamp: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Verify export integrity by recomputing hash.

        Args:
            content_hash: Expected hash from manifest
            pieces: Piece list with file hashes
            tool_version: Tool version used
            timestamp: Export timestamp (ISO format)

        Returns:
            Verification result with is_valid, verdict (INTEGRITY_OK or TAMPERED)
        """
        return await self.session.call_tool("verify_export_integrity", {
            "content_hash": content_hash,
            "pieces": pieces,
            "tool_version": tool_version,
            "timestamp": timestamp,
            **kwargs,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Meta
    # ─────────────────────────────────────────────────────────────────────────

    async def get_server_info(self, **kwargs) -> Dict[str, Any]:
        """Get LegalOps server info and capabilities."""
        return await self.session.call_tool("get_server_info", kwargs)

    async def health_check(self) -> Dict[str, Any]:
        """Check LegalOps MCP server health."""
        return await self.session.health_check()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available LegalOps tools."""
        return await self.session.list_tools()
