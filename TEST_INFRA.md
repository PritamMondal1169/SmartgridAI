# TEST_INFRA.md — SmartgridAI Test Infrastructure & Quality Assurance Specification

## 1. Overview & Architectural Principles

SmartgridAI is an end-to-end coursework demonstration platform integrating physical smart-meter telemetry (D1), national grid demand & renewables (D2), and distribution infrastructure computer vision (D3). 

This document defines the test architecture, execution harness, and verification standards for the opaque-box, requirement-driven test suite.

### Key Quality Principles
1. **Requirement-Driven Opaque-Box Verification**: Tests validate observable contracts, mathematical properties, physical data invariants, and HTTP response envelopes without coupling to private internal implementation details.
2. **Three Decoupled Planes**:
   - **Data & Preprocessing Engine**: Ingests D1/D2/D3 datasets, enforces physical contracts ($P_{\text{kW}} = t_{\text{kWh}} \times 20$, outage mask $V \le 5\text{V}$ or $I \le 0.01\text{A}$), 5 quality admission gates, and 70/15/15 chronological splitting with SHA-256 signed manifests.
   - **Machine Learning & Analytics Engine (M1–M5)**: Verifies model architecture contracts, multi-seed metrics ($\ge 3$ seeds, mean $\pm$ std), and non-negotiable Definitions of Done (DoD).
   - **Application & Delivery Plane**: Verifies FastAPI REST services (15 endpoints, standardized `{"data", "meta"}` or `{"error"}` envelopes, JWT auth, RBAC, rate limiting) and frontend visualization contracts.
3. **Non-Negotiable Baseline Floor**: No deep neural model prediction is evaluated without its co-located baseline (M1 MLP vs Ridge Regression, M2 fixed vs default, M3 CNN vs Majority-Class, M4 LSTM/GRU vs Seasonal-Naive and Persistence, M5 VAE vs Daily Z-Score).
4. **Deterministic and Self-Contained**: Every test sets up its own isolated state via parameterizable fixtures, does not depend on test execution order, and uses deterministic random seeds.

---

## 2. Directory Layout & Organization

The test suite is organized into standard pytest suites under `tests/`:

```
c:\Users\Pritam\OneDrive\Desktop\SmartgridAI\
├── tests/
│   ├── conftest.py                   # Root pytest hooks, custom markers, path setup
│   ├── unit/                         # Unit tests for isolated mathematical functions
│   └── e2e/
│       ├── conftest.py               # E2E fixtures: data factories, auth, client, oracles
│       ├── test_tier1_features.py    # Tier 1: Isolated Feature Coverage (24 tests)
│       ├── test_tier2_boundaries.py  # Tier 2: Boundary & Corner Cases (20 tests)
│       ├── test_tier3_integration.py # Tier 3: Cross-Feature Combinations (5 tests)
│       └── test_tier4_workloads.py   # Tier 4: Real-World Demonstration Workloads (5 tests)
├── run_tests.py                      # Standalone CLI test runner
├── TEST_INFRA.md                     # Test infrastructure specification (this file)
└── TEST_READY.md                     # Verification report and gate status
```

---

## 3. Environment & Prerequisites

### Runtime Requirements
- **Python**: 3.11+
- **Test Framework**: `pytest >= 8.0`, `pytest-cov`, `pytest-asyncio`
- **Application & Client Dependencies**: `fastapi`, `starlette`, `pydantic >= 2.0`, `httpx`, `requests`
- **Scientific & Data Libraries**: `numpy`, `pandas`, `pyarrow`, `scipy`, `pillow`, `openpyxl`
- **Authentication**: HMAC-SHA256 JWT validation

### Execution Command
Using the project Python environment:
```powershell
c:\Users\Pritam\OneDrive\Desktop\SmartgridAI\.venv\Scripts\python.exe run_tests.py --tier all
```
Or directly via pytest:
```powershell
c:\Users\Pritam\OneDrive\Desktop\SmartgridAI\.venv\Scripts\python.exe -m pytest tests/e2e -v
```

---

## 4. Test Runner Architecture (`run_tests.py`)

`run_tests.py` provides a standardized command-line interface for local development, CI/CD pipelines, and multi-agent verification.

### CLI Parameters
| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--tier` | `-t` | Select tier to run (`1`, `2`, `3`, `4`, or `all`) | `all` |
| `--marker` | `-m` | Custom pytest marker expression (e.g. `data`, `api`, `models`) | `None` |
| `--report` | `-r` | Output file path for JSON test report | `test_report.json` |
| `--fast` | `-f` | Skip compute-heavy tests | `False` |
| `--verbose` | `-v` | Enable detailed test output | `True` |

### Exit Code Protocol
- `0`: All tests passed cleanly (100% pass rate).
- `1`: One or more tests failed or test runner encountered an execution failure.
- `2`: Configuration or environment error (missing packages, invalid arguments).

### Machine-Readable Report Schema (`test_report.json`)
```json
{
  "timestamp": "2026-11-11T22:00:00Z",
  "tier": "all",
  "duration_seconds": 12.45,
  "summary": {
    "total": 54,
    "passed": 54,
    "failed": 0,
    "skipped": 0,
    "errors": 0
  },
  "status": "PASS",
  "failures": []
}
```

---

## 5. 4-Tier Test Suite Specifications

### Tier 1: Feature Coverage Suite (`test_tier1_features.py`)
Provides isolated equivalence-class tests across all functional areas:
1. **Data Preprocessing Engine** (7 tests):
   - Physical active power conversion: $P_{\text{kW}} = t_{\text{kWh}} \times 20$.
   - Outage masking dual condition: $V \le 5\text{V}$ OR $I \le 0.01\text{A} \implies P_{\text{kW}} = \text{NaN}, \text{is\_outage} = \text{True}$.
   - Gate 1: 30-day continuous span validation.
   - Gate 2: 80% 15-minute bin coverage validation.
   - Gate 3: 40% outage fraction cap.
   - Gates 4 & 5: Scrubber eliminating duplicates and negative/impossible values.
   - Chronological 70/15/15 splitting with SHA-256 signed `split_manifest.json`.
2. **Model Training & Evaluation Contracts M1–M5** (6 tests):
   - M1: XOR solves in $\ge 19/20$ seeds with hidden layer, $\le 2/20$ without; MLP beats Ridge MAE.
   - M2: 5-layer sigmoid gradient norm decays $> 2$ orders of magnitude; exploding gradients mitigated by norm clipping.
   - M3: Pole Lean CNN achieves macro-F1 $\ge 0.70$ vs majority class floor; CPU p95 $< 4$s.
   - M3: Pole Detection YOLO achieves mAP@0.5 $\ge 0.85$; Vegetation U-Net achieves mIoU $\ge 0.55$.
   - M4: LSTM and GRU forecasters beat Seasonal-Naive MASE on $\ge 70\%$ of admitted meters.
   - M5: WGAN-GP downstream utility within 15% of real training; VAE anomaly F1 $\ge 0.65$ at p99 threshold.
3. **Backend API Endpoints & Envelopes** (6 tests):
   - Uniform response envelopes: `{"data", "meta"}` on success, `{"error": {"code", "message", "detail"}}` on failure.
   - JWT authentication and role-based access control (`viewer`, `analyst`, `admin`).
   - Analytics routes: `/api/analytics/overview` and `/api/analytics/series`.
   - Forecast routes: `/api/forecast/{arch}` with baseline overlays.
   - Vision routes: `/api/infrastructure/classify`, `/detect`, `/segment`.
   - Generative & Anomaly routes: `/api/generative/gan`, `/api/generative/vae`, `/api/anomalies`.
4. **Dashboard Contracts & Exports** (5 tests):
   - Co-located visual baseline display contract.
   - System KPI overview telemetry cards.
   - Field inspection triage queue schema.
   - M2 Dynamics Plotly gradient norm schema.
   - Data Export Center file downloads (`json`, `csv`, `parquet`).

### Tier 2: Boundary & Corner Cases (`test_tier2_boundaries.py`)
Adversarial, extreme, and edge input testing:
1. **Physical Data Boundaries** (5 tests):
   - Exact voltage and current outage boundaries ($V=5.000$ vs $5.001$, $I=0.010$ vs $0.010001$).
   - Legitimate zero consumption ($P=0, \text{outage}=\text{False}$) vs negative power and $10,000$ kW spikes.
   - Quality gate exact boundaries ($29.989$d vs $30.0$d, $79.965\%$ vs $80.000\%$, $40.000\%$ vs $40.034\%$).
   - Calendar edge cases: Leap Year Feb 29 (96 bins), IST UTC+05:30 offset, year rollover.
   - Mixed delimiters (TAB vs COMMA), empty files (0 bytes), missing headers.
2. **Vision & Image Upload Boundaries** (5 tests):
   - 10 MB file cap: $10,485,760$ bytes accepted, $10,485,761$ bytes rejected with HTTP 413/400.
   - Dimension boundary: $4096 \times 4096$ accepted, $4097 \times 4096$ rejected/downscaled.
   - Magic byte verification: Text/executable renamed to `.jpg` rejected; genuine JPEG accepted.
   - EXIF GPS metadata stripped before persistence.
   - Extreme image dimensions ($1 \times 1$ px rejected, all-black/all-white images processed cleanly).
3. **Backend Security, Auth & Rate Limiting** (5 tests):
   - Rate limit: 60 req/min general limit (request 60 passes, 61 returns HTTP 429 with `Retry-After`).
   - Rate limit: 10 uploads/min vision limit.
   - JWT expiration boundary (23h59m valid, 24h01m returns HTTP 401).
   - RBAC violation: Viewer calling forecast, analyst calling training compare, forged token.
   - Parameter validation: Negative history, 99999h horizon, 1 MB buffer overflow payload.
4. **Model Inference Edge Cases** (5 tests):
   - 100% outage history window (all NaNs) handled cleanly without unhandled crash.
   - VAE anomaly scoring on flatline zero and extreme constant load triggers anomaly detection.
   - Scale-free MASE handles flatline series with zero seasonal difference without ZeroDivisionError.
   - XOR model evaluates continuous float inputs smoothly.
   - Training compare endpoint validates optimizer and seed list constraints (HTTP 422 on empty).

### Tier 3: Cross-Feature Combinations (`test_tier3_integration.py`)
Pairwise and end-to-end pipeline workflows:
1. `TEST-T3-XFEAT-01`: Raw CSV ingestion $\to$ physical conversion $\to$ 5 quality gates $\to$ 70/15/15 chronological split $\to$ SHA-256 signed `split_manifest.json` $\to$ partitioned Parquet storage.
2. `TEST-T3-XFEAT-02`: Partitioned Parquet series $\to$ Model inference engine $\to$ Standardized API response envelope with dataset hash.
3. `TEST-T3-XFEAT-03`: Multi-module deep learning models (M1, M4, M5) paired with co-located baselines in model evaluation registry.
4. `TEST-T3-XFEAT-04`: 4-stage middleware pipeline precedence: 401 (Auth) $\to$ 403 (RBAC) $\to$ 429 (Rate Limit) $\to$ 422 (Schema Validation) $\to$ 200/202.
5. `TEST-T3-XFEAT-05`: Async background training job lifecycle (`POST /api/training/compare` $\to$ HTTP 202 $\to$ polling `/api/jobs/{id}` from `queued` to `running` to `succeeded` $\to$ registry export).

### Tier 4: Real-World Demonstration Workloads (`test_tier4_workloads.py`)
High-level operational scenarios and persona workflows:
1. `TEST-T4-SCEN-01`: Complete End-to-End Grid Monitoring Workflow (Ingestion $\to$ Quality filtering $\to$ VAE anomaly stream detection $\to$ Alert creation $\to$ Analyst investigation).
2. `TEST-T4-SCEN-02`: 8-Minute PRD Acceptance Demonstration Flow (Overview $\to$ Forecast with baseline overlay $\to$ Pole lean classification $\to$ Anomaly timeline $\to$ Optimizer compare $\to$ Gradient norm decay plot).
3. `TEST-T4-SCEN-03`: Storm & Severe Outage Event Triage Workflow (Monsoon surge $\to$ Outage masking $\to$ Drone inspection upload $\to$ Lean classification & vegetation segmentation $\to$ Triage prioritization).
4. `TEST-T4-SCEN-04`: Automated Model Retraining, Hash Checksum & Metric Tracking Cycle (Telemetry tranche $\to$ Manifest update $\to$ Hash checksum verification $\to$ 3-seed retraining $\to$ Baseline check $\to$ Registry recording).
5. `TEST-T4-SCEN-05`: Multi-Meter Portfolio Demand Forecasting (10 diverse meters $\to$ 7-day sliding windows $\to$ 24h forecast $\to$ MASE evaluation $\to \ge 70\%$ beat Seasonal-Naive).

---

## 6. Fixtures & Test Data Architecture (`tests/e2e/conftest.py`)

The test suite relies on reusable, parameterizable factories:
- `test_workspace`: Creates an isolated directory tree with `raw/d1`, `raw/d2`, `processed`, `exports`, and `reports`.
- `d1_csv_factory`: Generates synthetic D1 smart meter CSV/TAB files with configurable parameters (days, outage fraction, missing bins, delimiters, duplicates, negative readings, spikes).
- `d2_xlsx_factory`: Generates synthetic D2 Excel workbooks with `Sheet1` and `Sheet2` SCADA generation and demand columns.
- `sample_images`: Synthesizes valid JPEG/PNG images, spoofed extension files, oversized images ($> 10$ MB or $> 4096$ px), and EXIF GPS-tagged images.
- `jwt_helper` & `auth_headers`: Signs HMAC-SHA256 tokens for `viewer`, `analyst`, `admin`, expired tokens, and invalid tokens.
- `api_client`: Starlette/FastAPI `TestClient` exercising the complete HTTP middleware and routing stack.
- `reference_pipeline`: Authoritative data processing and quality gate algorithms used as oracle references for contract validation.

---

## 7. Definition of Done & Quality Gates

The test suite enforces strict numerical thresholds defined in `docs/PRD.md` and `PROJECT.md`:
| Metric | Area | Minimum Threshold |
|--------|------|-------------------|
| XOR Convergence | M1 | $\ge 19/20$ seeds with hidden, $\le 2/20$ without |
| MLP vs Ridge MAE | M1 | $\text{MAE}_{\text{MLP}} < \text{MAE}_{\text{Ridge}}$ |
| Sigmoid Gradient Norm Decay | M2 | $\frac{\|\nabla_{W1}\|_2}{\|\nabla_{W5}\|_2} > 100$ ($> 2$ orders of magnitude) |
| Pole Lean Classification | M3 | $\text{Macro-F1} \ge 0.70$ on held-out test split |
| Pole Detection | M3 | $\text{mAP@0.5} \ge 0.85$ |
| Vegetation Segmentation | M3 | $\text{mIoU} \ge 0.55$ on 15% train carve-out |
| CPU Inference Latency | M3 | $\text{p95} < 4.0\text{ seconds}$ |
| Forecaster MASE Beat Rate | M4 | LSTM and GRU beat Seasonal-Naive on $\ge 70\%$ of admitted meters |
| Forecaster CPU Latency | M4 | $\text{p95} < 1.5\text{ seconds}$ |
| WGAN Downstream Utility | M5 | $\frac{|\text{MAE}_{\text{synthetic}} - \text{MAE}_{\text{real}}|}{\text{MAE}_{\text{real}}} \le 0.15$ |
| VAE Anomaly Detection | M5 | $\text{Macro-F1} \ge 0.65$ at p99 threshold on injected perturbations |
| Overall E2E Pass Rate | All | Exactly 100% (54 / 54 tests passing) |

---

## 8. Troubleshooting & Debugging Guide

1. **Test Failure in Response Envelope**:
   - Check if response returns `{"data", "meta"}` for success or `{"error": {"code", "message", "detail"}}` for failure.
   - Ensure `"meta"` includes `"latency_ms"`, `"model_version"`, and `"dataset_hash"`.
2. **Rate Limiting Cascades**:
   - Rate limit tests increment request counters. Ensure the rate limiter fixture resets or uses dedicated client IP addresses per test.
3. **Outage Formula Discrepancy**:
   - Ensure active power conversion applies $P_{\text{kW}} = t_{\text{kWh}} \times 20$.
   - Confirm outage condition checks $V \le 5\text{V}$ OR $I \le 0.01\text{A}$ and sets power to `NaN` (never `0.0`).
