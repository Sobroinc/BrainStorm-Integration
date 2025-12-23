# -*- coding: utf-8 -*-
"""
Провайдеры для интеграции с Agent.CentralHub.

Провайдеры абстрагируют MCP клиенты в интерфейсы,
понятные CentralHub (memories, services).
"""

from .memory_provider import BraineMemoryProvider
from .legal_provider import LegalOpsProvider

__all__ = [
    "BraineMemoryProvider",
    "LegalOpsProvider",
]
