# -*- coding: utf-8 -*-
"""
Load Testing Suite for BrainStorm v1.2.0

Uses real legal documents from OlgaFinal dataset:
- 5,750 PDFs
- 5,808 TXT (extracted text)

Test scenarios:
- 100 docs: < 60s prepare_hearing
- 500 docs: < 180s prepare_hearing
- 1000 entities: < 5s graph query
- 10 parallel calls: 100% success
"""
