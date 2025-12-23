# -*- coding: utf-8 -*-
"""Test prepare_hearing with new features."""

import asyncio
import json
import sys

async def main():
    from integration.orchestrator import BrainStormOrchestrator

    print("=" * 60)
    print("TEST: prepare_hearing with new features")
    print("=" * 60)

    async with BrainStormOrchestrator() as orch:
        # Step 1: Create test project
        print("\n[1] Creating test project...")
        project_id = f"proj:test-hearing-{int(asyncio.get_event_loop().time())}"

        # Step 2: Ingest test documents
        print("\n[2] Ingesting test documents...")

        doc1 = await orch.legal.ingest_text(
            project_id=project_id,
            content="""
            CONTRAT DE VENTE - Référence CV-2024-001
            Date: 15 janvier 2024

            Entre les parties:
            - VENDEUR: Société ALPHA SARL, représentée par M. Jean DUPONT
            - ACHETEUR: Société BETA SA, représentée par Mme Marie MARTIN

            Article 1: Objet
            Le vendeur cède à l'acheteur un lot de 500 unités de marchandises
            au prix unitaire de 100 EUR, soit un total de 50.000 EUR.

            Article 2: Livraison
            La livraison sera effectuée au plus tard le 1er mars 2024.

            Article 3: Paiement
            Le paiement sera effectué à 30 jours fin de mois.
            """,
            title="Contrat de vente CV-2024-001",
            doc_type="contract",
        )
        print(f"  - Doc 1: {doc1.get('document_id', 'error')}")

        doc2 = await orch.legal.ingest_text(
            project_id=project_id,
            content="""
            PROCÈS-VERBAL DE LIVRAISON
            Date: 5 mars 2024

            Référence contrat: CV-2024-001

            La société BETA SA confirme avoir reçu ce jour:
            - 450 unités de marchandises (au lieu des 500 prévues)
            - État: 50 unités endommagées

            Réserves émises:
            1. Livraison partielle (50 unités manquantes)
            2. Défaut de qualité sur 50 unités
            3. Retard de livraison de 4 jours

            Signé: Mme Marie MARTIN
            """,
            title="PV Livraison - Mars 2024",
            doc_type="evidence",
        )
        print(f"  - Doc 2: {doc2.get('document_id', 'error')}")

        doc3 = await orch.legal.ingest_text(
            project_id=project_id,
            content="""
            COURRIER DE MISE EN DEMEURE
            Date: 15 mars 2024

            De: BETA SA
            À: ALPHA SARL

            Objet: Mise en demeure - Contrat CV-2024-001

            Madame, Monsieur,

            Par la présente, nous vous mettons en demeure de:
            1. Livrer les 50 unités manquantes sous 8 jours
            2. Remplacer les 50 unités défectueuses
            3. Nous indemniser pour le préjudice subi

            À défaut, nous nous réservons le droit d'engager toute action judiciaire.

            Veuillez agréer...

            Mme Marie MARTIN
            Directrice Générale
            """,
            title="Mise en demeure - 15 mars 2024",
            doc_type="correspondence",
        )
        print(f"  - Doc 3: {doc3.get('document_id', 'error')}")

        # Step 3: Create a case
        print("\n[3] Creating case...")
        case_result = await orch.legal.create_case(
            name="Litige CV-2024-001 - BETA vs ALPHA",
            project_id=project_id,
        )
        # Extract case_id - multiple fallbacks
        case_id = case_result.get("case_id") or case_result.get("id", "")
        if not case_id:
            raw = case_result.get("raw")
            if isinstance(raw, dict):
                case_id = raw.get("id", "")
            elif isinstance(raw, list) and raw:
                case_id = raw[0].get("id", "") if isinstance(raw[0], dict) else ""
        if not case_id:
            case_id = f"case:test-{int(asyncio.get_event_loop().time())}"
        print(f"  - Case ID: {case_id}")

        # Step 4: Run prepare_hearing
        print("\n[4] Running prepare_hearing (max_doc_pairs=50)...")
        result = await orch.prepare_hearing(
            case_id=case_id,
            project_id=project_id,
            questions=["Quelles sont les violations du contrat?"],
            auto_export=False,  # Don't actually export
            validate_integrity=True,
            generate_hearing_pack=True,
            max_doc_pairs=50,
        )

        print(f"\n  Success: {result.success}")
        print(f"  Trace ID: {result.trace_id}")

        # Step 5: Get envelope
        print("\n[5] Testing to_envelope()...")
        envelope = result.to_envelope()

        print(f"\n  Envelope structure:")
        print(f"    ok: {envelope.get('ok')}")
        print(f"    workflow: {envelope.get('workflow')}")
        print(f"    trace_id: {envelope.get('trace_id')}")
        print(f"    project_id: {envelope.get('project_id')}")
        print(f"    case_id: {envelope.get('case_id')}")

        # Check inputs captured
        inputs = envelope.get("inputs", {})
        print(f"\n  Inputs captured:")
        print(f"    max_doc_pairs: {inputs.get('max_doc_pairs')}")
        print(f"    generate_hearing_pack: {inputs.get('generate_hearing_pack')}")

        # Check outputs
        outputs = envelope.get("outputs", {})
        print(f"\n  Outputs:")
        print(f"    hearing_ready: {outputs.get('hearing_readiness', {}).get('ready')}")
        print(f"    blocking_p0: {outputs.get('hearing_readiness', {}).get('blocking_p0')}")
        print(f"    integrity_verified: {outputs.get('integrity', {}).get('verified')}")

        # Check artifacts (dict format: artifacts, integrity, export_bundle)
        artifacts = envelope.get("artifacts", {})
        print(f"\n  Artifacts: {len(artifacts)} keys")
        for key, val in artifacts.items():
            if val:
                if isinstance(val, dict):
                    print(f"    - {key}: {list(val.keys())[:5]}")
                else:
                    print(f"    - {key}: {type(val).__name__}")
            else:
                print(f"    - {key}: (empty)")

        # Check action plan
        action_plan = envelope.get("action_plan", [])
        print(f"\n  Action Plan: {len(action_plan)} items")
        for ap in action_plan[:5]:
            print(f"    - [{ap.get('priority')}] {ap.get('action', '')[:50]}")

        # Also check result.data for action_plan directly
        direct_ap = result.data.get("action_plan", [])
        if len(direct_ap) != len(action_plan):
            print(f"    (direct in result.data: {len(direct_ap)} items)")

        # Check timings
        timings = envelope.get("timings_ms", {})
        print(f"\n  Timings (ms):")
        for step, ms in list(timings.items())[:5]:
            print(f"    - {step}: {ms}")

        # Check warnings
        warnings = envelope.get("warnings", [])
        if warnings:
            print(f"\n  Warnings: {len(warnings)}")
            for w in warnings[:3]:
                print(f"    - {w}")

        # Check errors
        errors = envelope.get("errors", [])
        if errors:
            print(f"\n  Errors: {len(errors)}")
            for e in errors[:3]:
                print(f"    - {e}")

        # Debug: check brief data
        brief = outputs.get("brief", {})
        print(f"\n  Brief stats:")
        print(f"    - evidence_matrix_count: {brief.get('evidence_matrix_count', 'N/A')}")
        print(f"    - contradictions_count: {brief.get('contradictions_count', 'N/A')}")
        print(f"    - action_plan_p0: {brief.get('action_plan_p0', 'N/A')}")

        # Debug: integrity error
        integrity = artifacts.get("integrity", {})
        if integrity.get("error"):
            print(f"\n  Integrity error: {integrity.get('error')}")

        # Check hearing pack
        hearing_pack = result.data.get("hearing_pack")
        if hearing_pack:
            print(f"\n[6] Hearing Pack generated:")
            print(f"    document_id: {hearing_pack.get('document_id')}")
            print(f"    piece_id: {hearing_pack.get('piece_id', 'N/A')}")
            content_preview = hearing_pack.get("content", "")[:200]
            print(f"    content preview:\n{content_preview}...")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

        # Return success
        return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
