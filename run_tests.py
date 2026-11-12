#!/usr/bin/env python3
"""SmartgridAI Standalone E2E Test Runner.

Provides a unified CLI interface for running the requirement-driven E2E test suite
across Tiers 1-4, parsing results, and emitting standardized test_report.json.

Usage:
    python run_tests.py --tier all
    python run_tests.py --tier 1
    python run_tests.py --tier 2 --report reports/tier2_report.json
    python run_tests.py --marker "data" -v
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)


def check_environment() -> list[str]:
    """Check required packages and runtime dependencies."""
    missing: list[str] = []
    packages = ["pytest", "fastapi", "pydantic", "httpx", "starlette", "pandas", "numpy"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="SmartgridAI E2E Test Suite Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        "-t",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Test tier to execute",
    )
    parser.add_argument(
        "--marker",
        "-m",
        default=None,
        help="Pytest marker filter expression (e.g. 'data', 'api', 'models')",
    )
    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help="Skip compute-heavy long tests",
    )
    parser.add_argument(
        "--report",
        "-r",
        default="test_report.json",
        help="Output path for JSON test report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Enable verbose test output",
    )
    return parser.parse_args()


class JsonReportPlugin:
    """Pytest plugin collecting structured test results."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.summary: dict[str, int] = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        }

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when == "call":
            self.summary["total"] += 1
            if report.passed:
                self.summary["passed"] += 1
            elif report.failed:
                self.summary["failed"] += 1
                self.results.append({
                    "nodeid": report.nodeid,
                    "outcome": "failed",
                    "duration": getattr(report, "duration", 0.0),
                    "longrepr": str(report.longrepr) if report.longrepr else "",
                })
            elif report.skipped:
                self.summary["skipped"] += 1
        elif report.when in ("setup", "teardown") and report.failed:
            self.summary["errors"] += 1
            self.results.append({
                "nodeid": report.nodeid,
                "outcome": f"error_{report.when}",
                "duration": getattr(report, "duration", 0.0),
                "longrepr": str(report.longrepr) if report.longrepr else "",
            })


def main() -> int:
    """Execute test suite based on CLI configuration."""
    args = parse_args()
    missing_deps = check_environment()
    if missing_deps:
        print(f"[ERROR] Missing required test dependencies: {', '.join(missing_deps)}", file=sys.stderr)
        print("Please install them in the virtual environment.", file=sys.stderr)
        return 2

    import pytest

    # Select target test files based on tier
    tier_map = {
        "1": ["tests/e2e/test_tier1_features.py"],
        "2": ["tests/e2e/test_tier2_boundaries.py"],
        "3": ["tests/e2e/test_tier3_integration.py"],
        "4": ["tests/e2e/test_tier4_workloads.py"],
        "all": [
            "tests/e2e/test_tier1_features.py",
            "tests/e2e/test_tier2_boundaries.py",
            "tests/e2e/test_tier3_integration.py",
            "tests/e2e/test_tier4_workloads.py",
        ],
    }

    test_targets = tier_map[args.tier]
    existing_targets = [t for t in test_targets if (PROJECT_ROOT / t).exists()]
    if not existing_targets:
        print(f"[ERROR] None of the requested test files exist: {test_targets}", file=sys.stderr)
        return 2

    pytest_args = list(existing_targets)
    if args.verbose:
        pytest_args.append("-v")
    if args.marker:
        pytest_args.extend(["-m", args.marker])
    if args.fast:
        pytest_args.extend(["-m", "not slow"])

    pytest_args.extend(["--tb=short", "-ra"])

    plugin = JsonReportPlugin()
    start_time = time.time()
    print("=" * 80)
    print(f"  SmartgridAI E2E Test Runner — Tier: {args.tier.upper()}  ")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}  ")
    print(f"  Target files: {existing_targets}  ")
    print("=" * 80)

    exit_code = pytest.main(pytest_args, plugins=[plugin])
    duration = time.time() - start_time

    # Generate JSON report
    status = "PASS" if (exit_code == 0 and plugin.summary["failed"] == 0 and plugin.summary["errors"] == 0) else "FAIL"
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": args.tier,
        "duration_seconds": round(duration, 3),
        "summary": plugin.summary,
        "status": status,
        "failures": plugin.results,
    }

    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"  EXECUTION SUMMARY: {status}")
    print(f"  Total: {plugin.summary['total']} | Passed: {plugin.summary['passed']} | "
          f"Failed: {plugin.summary['failed']} | Skipped: {plugin.summary['skipped']} | "
          f"Duration: {duration:.2f}s")
    print(f"  Report written to: {report_path}")
    print("=" * 80)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
