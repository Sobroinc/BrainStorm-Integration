# -*- coding: utf-8 -*-
"""
Root conftest for Integration tests.

Sets up Python path for 'integration' module imports.
"""

import sys
from pathlib import Path
import types

# Get paths
_integration_root = Path(__file__).parent
_brainstorm_root = _integration_root.parent

# Add Integration folder to path (for direct imports)
if str(_integration_root) not in sys.path:
    sys.path.insert(0, str(_integration_root))

# Add BrainStorm Moduls to path
if str(_brainstorm_root) not in sys.path:
    sys.path.insert(0, str(_brainstorm_root))


def pytest_configure(config):
    """Set up 'integration' module alias after path is configured."""
    if "integration" in sys.modules:
        return

    # Create 'integration' namespace
    integration = types.ModuleType("integration")
    integration.__path__ = [str(_integration_root)]
    integration.__file__ = str(_integration_root / "__init__.py")

    # Import submodules directly (path is already set up)
    import orchestrator
    import mcp_client
    import mcp_client.sse_session

    # Assign to namespace
    integration.orchestrator = orchestrator
    integration.mcp_client = mcp_client
    integration.BrainStormOrchestrator = orchestrator.BrainStormOrchestrator
    integration.connect_sse = mcp_client.connect_sse
    integration.connect_http = mcp_client.connect_http

    # Register in sys.modules
    sys.modules["integration"] = integration
    sys.modules["integration.orchestrator"] = orchestrator
    sys.modules["integration.mcp_client"] = mcp_client
    sys.modules["integration.mcp_client.sse_session"] = mcp_client.sse_session
