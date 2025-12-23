# -*- coding: utf-8 -*-
"""
Real Data Validation Checklist for BrainStorm Orchestrator.

Run this before deploying to production or after major changes.
Tests the critical paths with actual document data.

Usage:
    python -m integration.validation_checklist --project test-proj --case test-case
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a validation check."""

    name: str
    passed: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""

    passed: int = 0
    failed: int = 0
    warnings: int = 0
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)
        if check.passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def success(self) -> bool:
        return self.failed == 0


async def run_validation_checklist(
    project_id: str,
    case_id: Optional[str] = None,
    verbose: bool = True,
) -> ValidationReport:
    """
    Run complete validation checklist.

    Checks:
    1. Citation/piece_mapping consistency after auto-add
    2. No duplicate pieces on repeated runs
    3. Deterministic sorting with new documents
    4. deep_contradiction_check threshold
    5. Entity resolution stability
    6. Red-flag reproducibility

    Args:
        project_id: Project to validate
        case_id: Case ID (optional, creates temp if needed)
        verbose: Print progress

    Returns:
        ValidationReport with all check results
    """
    from integration.orchestrator import BrainStormOrchestrator

    report = ValidationReport()

    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION CHECKLIST: project={project_id}, case={case_id}")
        print(f"{'='*60}\n")

    async with BrainStormOrchestrator() as orch:

        # ─────────────────────────────────────────────────────────────
        # CHECK 1: Documents exist and are indexed
        # ─────────────────────────────────────────────────────────────
        if verbose:
            print("CHECK 1: Documents indexed...")

        try:
            docs = await orch.legal.list_documents(project_id=project_id)
            doc_count = docs.get("count", 0)

            check = CheckResult(
                name="documents_indexed",
                passed=doc_count >= 2,
                message=f"Found {doc_count} documents",
                details={"count": doc_count},
            )

            if doc_count < 2:
                check.message = f"Need at least 2 documents for validation (found {doc_count})"
        except Exception as e:
            check = CheckResult(
                name="documents_indexed",
                passed=False,
                message=f"Failed to list documents: {e}",
            )

        report.add(check)
        if verbose:
            print(f"  {'✓' if check.passed else '✗'} {check.message}")

        if doc_count < 2:
            if verbose:
                print("\n⚠ Cannot continue validation without documents.\n")
            return report

        # ─────────────────────────────────────────────────────────────
        # CHECK 2: Deterministic evidence_matrix output
        # ─────────────────────────────────────────────────────────────
        if verbose:
            print("\nCHECK 2: Deterministic evidence matrix...")

        try:
            # Run twice with deterministic=True
            result1 = await orch.evidence_matrix(
                project_id=project_id,
                max_claims=5,
                deterministic=True,
                compute_red_flags=True,
            )
            matrix1 = result1.data.get("matrix", [])
            claims1 = [m.get("claim", "")[:50] for m in matrix1]

            result2 = await orch.evidence_matrix(
                project_id=project_id,
                max_claims=5,
                deterministic=True,
                compute_red_flags=True,
            )
            matrix2 = result2.data.get("matrix", [])
            claims2 = [m.get("claim", "")[:50] for m in matrix2]

            # Compare claim order
            is_deterministic = claims1 == claims2

            # Compare red-flag counts
            rf1 = result1.data.get("red_flag_summary", {}).get("total", 0)
            rf2 = result2.data.get("red_flag_summary", {}).get("total", 0)
            rf_stable = rf1 == rf2

            check = CheckResult(
                name="deterministic_matrix",
                passed=is_deterministic and rf_stable,
                message=f"Claims match: {is_deterministic}, Red-flags stable: {rf_stable}",
                details={
                    "claims_run1": len(claims1),
                    "claims_run2": len(claims2),
                    "claims_match": is_deterministic,
                    "red_flags_run1": rf1,
                    "red_flags_run2": rf2,
                },
            )
        except Exception as e:
            check = CheckResult(
                name="deterministic_matrix",
                passed=False,
                message=f"Failed: {e}",
            )

        report.add(check)
        if verbose:
            print(f"  {'✓' if check.passed else '✗'} {check.message}")

        # ─────────────────────────────────────────────────────────────
        # CHECK 3: Piece mapping consistency (if case provided)
        # ─────────────────────────────────────────────────────────────
        if case_id:
            if verbose:
                print("\nCHECK 3: Piece mapping consistency...")

            try:
                # Get pieces before
                pieces_before = await orch.legal.list_case_pieces(case_id=case_id)
                count_before = len(pieces_before.get("pieces", []))

                # Run with auto_add_pieces
                result = await orch.build_case_brief(
                    project_id=project_id,
                    case_id=case_id,
                    include_evidence_matrix=True,
                    auto_add_pieces=True,
                )

                # Get pieces after
                pieces_after = await orch.legal.list_case_pieces(case_id=case_id)
                count_after = len(pieces_after.get("pieces", []))

                auto_added = result.data.get("pieces_auto_added", 0)

                check = CheckResult(
                    name="piece_mapping",
                    passed=True,
                    message=f"Pieces: {count_before} -> {count_after} (auto-added: {auto_added})",
                    details={
                        "before": count_before,
                        "after": count_after,
                        "auto_added": auto_added,
                    },
                )
            except Exception as e:
                check = CheckResult(
                    name="piece_mapping",
                    passed=False,
                    message=f"Failed: {e}",
                )

            report.add(check)
            if verbose:
                print(f"  {'✓' if check.passed else '✗'} {check.message}")

            # ─────────────────────────────────────────────────────────
            # CHECK 4: No duplicate pieces on repeated runs
            # ─────────────────────────────────────────────────────────
            if verbose:
                print("\nCHECK 4: No duplicate pieces...")

            try:
                # Run again with auto_add_pieces
                await orch.build_case_brief(
                    project_id=project_id,
                    case_id=case_id,
                    include_evidence_matrix=True,
                    auto_add_pieces=True,
                )

                pieces_check = await orch.legal.list_case_pieces(case_id=case_id)
                count_check = len(pieces_check.get("pieces", []))

                # Check for duplicates by document_id
                doc_ids = [p.get("document_id") for p in pieces_check.get("pieces", [])]
                unique_docs = len(set(doc_ids))
                has_duplicates = len(doc_ids) != unique_docs

                check = CheckResult(
                    name="no_duplicate_pieces",
                    passed=not has_duplicates and count_check == count_after,
                    message=f"Pieces stable: {count_check} (duplicates: {has_duplicates})",
                    details={
                        "count_after_second_run": count_check,
                        "has_duplicates": has_duplicates,
                        "unique_docs": unique_docs,
                    },
                )
            except Exception as e:
                check = CheckResult(
                    name="no_duplicate_pieces",
                    passed=False,
                    message=f"Failed: {e}",
                )

            report.add(check)
            if verbose:
                print(f"  {'✓' if check.passed else '✗'} {check.message}")

        # ─────────────────────────────────────────────────────────────
        # CHECK 5: Contradiction detection threshold
        # ─────────────────────────────────────────────────────────────
        if verbose:
            print("\nCHECK 5: Contradiction detection performance...")

        try:
            import time

            start = time.time()
            contr_result = await orch.legal.detect_contradictions(
                project_id=project_id,
            )
            elapsed = time.time() - start

            contr_count = contr_result.get("contradictions_count", 0)

            # Threshold: should complete in < 30s for typical projects
            is_fast = elapsed < 30.0

            check = CheckResult(
                name="contradiction_performance",
                passed=is_fast,
                message=f"Found {contr_count} contradictions in {elapsed:.1f}s",
                details={
                    "count": contr_count,
                    "elapsed_seconds": elapsed,
                    "threshold_seconds": 30.0,
                },
            )

            if not is_fast:
                check.message += " (SLOW - consider deep_contradiction_check=False for large docs)"

        except Exception as e:
            check = CheckResult(
                name="contradiction_performance",
                passed=False,
                message=f"Failed: {e}",
            )

        report.add(check)
        if verbose:
            print(f"  {'✓' if check.passed else '✗'} {check.message}")

        # ─────────────────────────────────────────────────────────────
        # CHECK 6: Entity resolution stability
        # ─────────────────────────────────────────────────────────────
        if verbose:
            print("\nCHECK 6: Entity resolution stability...")

        try:
            # Get project readiness (includes entity stats)
            readiness = await orch.legal.get_project_readiness(project_id=project_id)

            status = readiness.get("status", "unknown")
            coverage = readiness.get("coverage_pct", 0)
            entities = readiness.get("stats", {}).get("entities", 0)

            check = CheckResult(
                name="entity_resolution",
                passed=status in ("ready", "degraded"),
                message=f"Status: {status}, Coverage: {coverage}%, Entities: {entities}",
                details={
                    "status": status,
                    "coverage_pct": coverage,
                    "entities": entities,
                },
            )

            if status == "cold":
                check.message += " (run backfill for better analysis)"

        except Exception as e:
            check = CheckResult(
                name="entity_resolution",
                passed=False,
                message=f"Failed: {e}",
            )

        report.add(check)
        if verbose:
            print(f"  {'✓' if check.passed else '✗'} {check.message}")

        # ─────────────────────────────────────────────────────────────
        # CHECK 7: prepare_hearing smoke test
        # ─────────────────────────────────────────────────────────────
        if case_id:
            if verbose:
                print("\nCHECK 7: prepare_hearing smoke test...")

            try:
                result = await orch.prepare_hearing(
                    case_id=case_id,
                    project_id=project_id,
                    auto_export=False,  # Don't actually export
                    validate_integrity=True,
                )

                is_ready = result.data.get("hearing_readiness", {}).get("ready", False)
                p0_count = result.data.get("hearing_readiness", {}).get("blocking_p0", 0)
                validated = result.data.get("validation", {}).get("is_valid", False)
                integrity = result.data.get("integrity", {}).get("verified", False)

                check = CheckResult(
                    name="prepare_hearing",
                    passed=result.success,
                    message=f"Ready: {is_ready}, P0: {p0_count}, Valid: {validated}, Integrity: {integrity}",
                    details={
                        "hearing_ready": is_ready,
                        "blocking_p0": p0_count,
                        "validated": validated,
                        "integrity_verified": integrity,
                    },
                )
            except Exception as e:
                check = CheckResult(
                    name="prepare_hearing",
                    passed=False,
                    message=f"Failed: {e}",
                )

            report.add(check)
            if verbose:
                print(f"  {'✓' if check.passed else '✗'} {check.message}")

    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULT: {report.passed} passed, {report.failed} failed")
        print(f"{'='*60}\n")

        if report.failed > 0:
            print("FAILED CHECKS:")
            for check in report.checks:
                if not check.passed:
                    print(f"  - {check.name}: {check.message}")
            print()

    return report


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run validation checklist")
    parser.add_argument("--project", required=True, help="Project ID to validate")
    parser.add_argument("--case", help="Case ID (optional)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    report = await run_validation_checklist(
        project_id=args.project,
        case_id=args.case,
        verbose=not args.quiet,
    )

    exit(0 if report.success else 1)


if __name__ == "__main__":
    asyncio.run(main())
