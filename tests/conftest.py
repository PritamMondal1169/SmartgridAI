"""Global pytest configuration for SmartgridAI test suite."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Ensure project root and src are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "tier1: Tier 1 Feature Coverage tests")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary & Corner Case tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-Feature Integration tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-World Application Scenario tests")
    config.addinivalue_line("markers", "data: Data preprocessing and quality gate tests")
    config.addinivalue_line("markers", "api: FastAPI backend endpoints and envelope tests")
    config.addinivalue_line("markers", "models: Machine learning models M1-M5 contract tests")
    config.addinivalue_line("markers", "vision: Computer vision infrastructure inspection tests")
    config.addinivalue_line("markers", "security: Authentication, RBAC, and rate limiting tests")
    config.addinivalue_line("markers", "slow: Compute-intensive or long-running simulation tests")
