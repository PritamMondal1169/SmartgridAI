# Project: SmartgridAI

## Architecture
SmartgridAI is an end-to-end coursework demonstration platform integrating physical smart-meter telemetry (D1), national grid demand & renewables (D2), and distribution infrastructure computer vision (D3). The system is organized into three decoupled planes:
1. **Data & Preprocessing Engine**: Ingests multi-format smart meter CSVs (handling comma delimiters, timestamp normalization to UTC, $P_{\text{kW}} = t_{\text{kWh}} \times 20$, outage masking $V \le 5\text{V}$ or $I \le 0.01\text{A}$), enforces strict 5-gate quality admission, generates `quality_report.json`, and records chronological 70/15/15 splits in signed `split_manifest.json`.
2. **Machine Learning & Analytics Engine (M1–M5)**:
   - Self-contained, executable Google Colab GPU notebooks configured for cloud GPU training with multi-seed logging and exported weights/metrics.
   - Lightweight local inference engine loading exported weights for FastAPI demonstration.
   - Five syllabus modules with mandatory co-located baselines:
     - M1: 2D Perceptron, 2-2-1 XOR MLP, Demand MLP Regressor vs Ridge Regression baseline.
     - M2: Fixed DNN, SGD/Adam/RMSprop sweep, Random/Xavier/He initializations, Dropout/BatchNorm/None regularizers, per-layer gradient norm logging, vanishing/exploding gradient reproduction & gradient clipping.
     - M3: Pole Lean CNN classification (macro-F1 $\ge 0.70$), Pole Detection YOLO (mAP@0.5 $\ge 0.85$), Vegetation-Conductor segmentation (mIoU $\ge 0.55$), CPU p95 $< 4$s.
     - M4: 24h sequence forecasting with LSTM & GRU beating Seasonal-Naive MASE on $\ge 70\%$ of admitted meters, plus Persistence and LightGBM baselines.
     - M5: WGAN-GP 24h demand profile generation (utility within 15% of real training) and VAE anomaly detection with 5-perturbation injection engine (F1 $\ge 0.65$ at p99 threshold).
3. **Application & Delivery Plane**:
   - Thin FastAPI backend (`backend/`) exposing 15 REST endpoints, standardized response envelope `{"data", "meta"}` or `{"error"}`, JWT auth, RBAC, and rate limiting.
   - Interactive React 18 + Vite + Tailwind dashboard (`frontend/`) with dark theme, Recharts series charts, XOR canvas, Plotly gradient decay, inspection queue, and anomaly timeline.
   - Requirement-driven E2E test suite (`tests/e2e/`) validating 100% contract compliance across all tiers.

---

## Feature Inventory
Every discovered feature assigned to a milestone (100% mapped):

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Multi-Format File Reader | Ingests D1 CSVs handling comma delimiters, parses column schema index 0-5 | M1_DATA | survey |
| 2 | Physical Power Converter | Converts 3-minute interval energy $t_{\text{kWh}}$ to active power $P_{\text{kW}} = t_{\text{kWh}} \times 20$ | M1_DATA | survey |
| 3 | Outage Masker | Flags grid outages when $V \le 5\text{V}$ or $I \le 0.01\text{A}$, sets power to NaN, sets `is_outage=True` | M1_DATA | survey |
| 4 | 15-Min Resampler | Resamples cleaned 3-minute readings to uniform 15-minute intervals for demand forecasting | M1_DATA | survey |
| 5 | Span Validator | Ensures meter possesses $\ge 30$ continuous days of data | M1_DATA | survey |
| 6 | Bin Coverage Validator | Checks that $\ge 80\%$ of 15-minute time bins are populated | M1_DATA | survey |
| 7 | Outage Fraction Filter | Verifies that total outage fraction does not exceed $40\%$ of the span | M1_DATA | survey |
| 8 | Deduplication & Anomaly Scrubber | Ensures 0 duplicate timestamps and 0 negative/impossible values after cleaning | M1_DATA | survey |
| 9 | Quality Report Generator | Compiles admission summary, admission rate (target 60-70%), and per-meter rejection reasons | M1_DATA | survey |
| 10 | Chronological Splitter | Splits admitted series strictly chronologically per meter into 70% train, 15% valid, 15% test splits | M1_DATA | survey |
| 11 | Split Manifest Signer | Hashes raw source files with SHA-256 and records split date/time boundaries and hashes | M1_DATA | survey |
| 12 | Parquet Partitioned Storage | Persists processed series partitioned by `city=/year=/meter=/` | M1_DATA | survey |
| 13 | PD-Defect Deduplicator | Runs perceptual hashing (pHash) across train and valid image sets to prevent image leakage | M1_DATA | survey |
| 14 | Segmentation Carve-Out | Generates a 15% validation carve-out from training set for Vegetation–Conductor segmentation | M1_DATA | survey |
| 15 | D2 Grid Demand Ingestion | Ingests D2 Sheet2 SCADA generation and demand time series from `.xlsx` files | M1_DATA | survey |
| 16 | 2D Perceptron Toy Lab | Simulates single-layer perceptron on linearly separable 2-feature toy data | M2_SHALLOW_DYNAMICS | survey |
| 17 | XOR 2-2-1 MLP Solver | Trains 2-2-1 MLP on XOR over 20 random seeds ($\ge 19/20$ solve with hidden, $\le 2/20$ without) | M2_SHALLOW_DYNAMICS | survey |
| 18 | 15-Min Demand MLP Regressor | Trains MLP neural network regressor on 15-minute demand lag features | M2_SHALLOW_DYNAMICS | survey |
| 19 | Ridge Regression Baseline | Computes L2-regularized linear baseline on identical features as the MLP regressor | M2_SHALLOW_DYNAMICS | survey |
| 20 | Fixed DNN Benchmark | Trains fixed multi-layer DNN architecture across hyperparameter combinations | M2_SHALLOW_DYNAMICS | survey |
| 21 | Optimizer Sweep | Compares SGD, Adam, and RMSprop optimizers under identical inits across 3 seeds (mean $\pm$ std) | M2_SHALLOW_DYNAMICS | survey |
| 22 | Initializer Sweep | Evaluates Random Normal, Xavier (Glorot), and He (Kaiming) initializations | M2_SHALLOW_DYNAMICS | survey |
| 23 | Regularizer Sweep | Benchmarks Dropout vs. Batch Normalization vs. No Regularization | M2_SHALLOW_DYNAMICS | survey |
| 24 | Layer Gradient-Norm Logger | Computes and logs the $L_2$ norm of gradients for each layer across training epochs | M2_SHALLOW_DYNAMICS | survey |
| 25 | Vanishing Gradient Reproducer | Demonstrates gradient decay $> 2$ orders of magnitude on a 5-layer sigmoid stack | M2_SHALLOW_DYNAMICS | survey |
| 26 | Exploding Gradient & Clip Demo | Reproduces exploding gradients via high learning rate and verifies stability via gradient norm clipping | M2_SHALLOW_DYNAMICS | survey |
| 27 | Pole Lean CNN Classifier | 3-class CNN classifying pole uprightness (Straight, Leaned, Severe Tilt) achieving macro-F1 $\ge 0.70$ | M3_VISION | survey |
| 28 | Majority-Class Vision Baseline | Evaluates baseline predicting majority class on validation/test sets | M3_VISION | survey |
| 29 | Pole Detection YOLO | Detects distribution poles in field images achieving mAP@0.5 $\ge 0.85$ | M3_VISION | survey |
| 30 | Vegetation–Conductor U-Net | Segments vegetation encroaching on power conductors achieving mIoU $\ge 0.55$ | M3_VISION | survey |
| 31 | Image Ingestion Guard | Validates magic bytes (JPEG/PNG), enforces 10 MB and 4096 px caps, strips EXIF data | M3_VISION | survey |
| 32 | Multi-Step Window Builder | Generates 7-day input history (672 bins) and 24-hour horizon (96 bins) sliding windows | M4_SEQ_GEN_ANOMALY | survey |
| 33 | Per-Meter Scaler | Fits robust scaler/min-max per meter on train set; applies transform to valid/test | M4_SEQ_GEN_ANOMALY | survey |
| 34 | Global Sequence Forecaster | Trains shared LSTM & GRU models with meter embeddings and outage loss masking | M4_SEQ_GEN_ANOMALY | survey |
| 35 | Persistence Baseline | Evaluates naive day-ahead persistence baseline: $\hat{y}_t = y_{t-24h}$ | M4_SEQ_GEN_ANOMALY | survey |
| 36 | Seasonal-Naive Baseline | Evaluates week-ahead seasonal-naive baseline: $\hat{y}_t = y_{t-168h}$ | M4_SEQ_GEN_ANOMALY | survey |
| 37 | LightGBM Tabular Baseline | Trains gradient boosted tree baseline on calendar and auto-regressive lag features | M4_SEQ_GEN_ANOMALY | survey |
| 38 | MASE & Metric Evaluator | Computes scale-free MASE, MAE, RMSE, and MAPE per forecast horizon ($h=1\dots 24$) | M4_SEQ_GEN_ANOMALY | survey |
| 39 | Admission Meter Beat Evaluator | Verifies whether LSTM/GRU beats Seasonal-Naive MASE across $\ge 70\%$ of admitted meters | M4_SEQ_GEN_ANOMALY | survey |
| 40 | WGAN-GP Demand Synthesizer | Generates synthetic 24-hour daily demand curves using WGAN-GP | M4_SEQ_GEN_ANOMALY | survey |
| 41 | MMD & Discriminator Scorer | Computes Maximum Mean Discrepancy (MMD) and discriminator loss on real vs synthetic data | M4_SEQ_GEN_ANOMALY | survey |
| 42 | Train-on-Synthetic Utility Test | Trains M4 LSTM on synthetic data; evaluates test MAE within 15% of real training | M4_SEQ_GEN_ANOMALY | survey |
| 43 | VAE Demand Model | Reconstructs 24-hour demand windows; computes reconstruction error for anomaly scoring | M4_SEQ_GEN_ANOMALY | survey |
| 44 | Perturbation Injection Engine | Injects 5 synthetic anomalies (magnitude scale, load shift, flatline, dropout, phase shift) | M4_SEQ_GEN_ANOMALY | survey |
| 45 | Anomaly Metric Evaluator | Computes Precision, Recall, and F1 at validation-derived p99 threshold (DoD F1 $\ge 0.65$) | M4_SEQ_GEN_ANOMALY | survey |
| 46 | Daily Z-Score Baseline | Computes baseline z-score on daily aggregated consumption totals | M4_SEQ_GEN_ANOMALY | survey |
| 47 | FastAPI Server & Envelope | Serves REST API wrapping all outputs in standardized `{"data", "meta"}` or `{"error"}` envelopes | M5_API_BACKEND | survey |
| 48 | JWT Authentication & RBAC | Manages token issue/refresh, enforces roles (`viewer`, `analyst`, `admin`) | M5_API_BACKEND | survey |
| 49 | Rate Limiter Middleware | Enforces 60 req/min for general API and 10 uploads/min for vision endpoints | M5_API_BACKEND | survey |
| 50 | `/api/analytics/overview` | Returns system-level KPI totals (total meters, consumption, admitted count, active anomalies) | M5_API_BACKEND | survey |
| 51 | `/api/analytics/series` | Streams historical consumption series for a given meter with outage annotations | M5_API_BACKEND | survey |
| 52 | `/api/ann/predict` | Computes M1 MLP demand prediction alongside Ridge baseline prediction | M5_API_BACKEND | survey |
| 53 | `/api/ann/xor` | Runs live XOR simulation returning decision boundary, weights, and convergence | M5_API_BACKEND | survey |
| 54 | `/api/training/compare` | Enqueues M2 background training comparison job across optimizers and seeds | M5_API_BACKEND | survey |
| 55 | `/api/jobs/{id}` | Returns real-time status and output of background training jobs | M5_API_BACKEND | survey |
| 56 | `/api/infrastructure/classify` | Runs M3 CNN pole lean classification on uploaded image | M5_API_BACKEND | survey |
| 57 | `/api/infrastructure/detect` | Runs M3 YOLO pole detection returning bounding boxes and labels | M5_API_BACKEND | survey |
| 58 | `/api/infrastructure/segment` | Runs M3 U-Net segmentation returning conductor/vegetation polygons | M5_API_BACKEND | survey |
| 59 | `/api/forecast/{arch}` | Generates 24-hour sequence forecast with Persistence & Seasonal-Naive overlays | M5_API_BACKEND | survey |
| 60 | `/api/generative/gan` | Generates $N$ synthetic daily load profiles from WGAN-GP generator | M5_API_BACKEND | survey |
| 61 | `/api/generative/vae` | Computes VAE reconstruction and anomaly likelihood score for input window | M5_API_BACKEND | survey |
| 62 | `/api/anomalies` | Lists historical detected and injected anomalies with pagination | M5_API_BACKEND | survey |
| 63 | `/api/models/metrics` | Serves model registry benchmark metrics with co-located baseline rows | M5_API_BACKEND | survey |
| 64 | `/api/export/{artifact}` | Exports datasets, quality reports, or metric tables in CSV, JSON, or Parquet | M5_API_BACKEND | survey |
| 65 | Database Relational Schema | Manages SQLite/PostgreSQL schema for users, datasets, models, experiments, jobs | M5_API_BACKEND | survey |
| 66 | Model Registry & Provenance | Records model architecture, weights path, config hash, dataset hash, seed, metrics | M5_API_BACKEND | survey |
| 67 | React + Vite UI Shell | Responsive single-page application with dark charcoal theme and electric blue/green accents | M6_FRONTEND_UI | survey |
| 68 | Overview KPI Dashboard | Displays high-level grid telemetry: total meters, daily consumption, open alerts | M6_FRONTEND_UI | survey |
| 69 | Time-Series Analytics Console | Interactive Recharts consumption graph with zoom, resolution toggle, and outage bars | M6_FRONTEND_UI | survey |
| 70 | M1 Shallow Network Lab | Interactive XOR decision boundary canvas and MLP vs Ridge demand comparison card | M6_FRONTEND_UI | survey |
| 71 | M2 Training Dynamics Lab | Plotly visualization of per-layer gradient norms, vanishing/exploding curves | M6_FRONTEND_UI | survey |
| 72 | M3 Inspection Triage Queue | Triage queue for field inspection images with YOLO bounding box overlays and lean badges | M6_FRONTEND_UI | survey |
| 73 | M4 Forecast Console | 24-hour forecast chart displaying LSTM/GRU predictions with Seasonal-Naive baseline overlay | M6_FRONTEND_UI | survey |
| 74 | M5 Anomaly & Generative Lab | Synthetic load generator inspector and VAE anomaly score timeline with p99 threshold bar | M6_FRONTEND_UI | survey |
| 75 | Model Registry & Metric Table | Searchable table displaying all experiments, config hashes, seeds, and baseline comparison rows | M6_FRONTEND_UI | survey |
| 76 | Data Export Center | Export modal allowing single-click download of quality reports and split manifests | M6_FRONTEND_UI | survey |
| 77 | Requirement-Driven Test Runner | Automated opaque-box test harness exercising all Tiers 1-4 | E2E_TESTING_TRACK | survey |
| 78 | Tier 1 Feature Coverage Suite | $\ge 5$ isolated tests per feature on equivalence class representatives | E2E_TESTING_TRACK | survey |
| 79 | Tier 2 Boundary & Corner Suite | Boundary, edge, and error-prone inputs per feature | E2E_TESTING_TRACK | survey |
| 80 | Tier 3 Cross-Feature Suite | Pairwise integration tests across data, models, API, and UI | E2E_TESTING_TRACK | survey |
| 81 | Tier 4 Real-World Workload Suite | End-to-end 8-minute demonstration script and end-user workflows | E2E_TESTING_TRACK | survey |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | M1_DATA | Data Processing & Quality Assurance (D1, D2, D3 contracts, quality_report.json, split_manifest.json, docs/DATA_CONTRACT.md, docs/CREDITS.md) | none | DONE |
| M2 | M2_SHALLOW_DYNAMICS | Model Training M1 & M2 (Perceptron, XOR, Demand MLP vs Ridge; Dynamics sweeps, gradient norms, vanishing/clipping) | M1 | PLANNED |
| M3 | M3_VISION | Model Training M3 (Pole Lean CNN, Pole Detection YOLO, Vegetation U-Net, CPU latency optimization) | M1 | PLANNED |
| M4 | M4_SEQ_GEN_ANOMALY | Model Training M4 & M5 (LSTM/GRU forecasting vs Baselines, WGAN-GP generator, VAE anomaly detection with 5-perturbation injection) | M1 | PLANNED |
| M5 | M5_API_BACKEND | FastAPI REST Backend Service (15 endpoints, envelope, auth, rate limiting, async jobs, local inference loading exported weights) | M1, M2, M3, M4 | PLANNED |
| M6 | M6_FRONTEND_UI | React + Vite + Tailwind Dashboard (Dark theme, KPI overview, series charts, M1-M5 interactive experiment labs, triage queue, exports) | M5 | PLANNED |
| M_E2E | E2E_TESTING_TRACK | E2E Test Suite (Harness, Tiers 1-4 test suites, TEST_INFRA.md, TEST_READY.md) | parallel track | DONE |
| M_FINAL | FINAL_E2E_VERIFICATION | 100% E2E test pass across all tiers + Phase 2 adversarial coverage hardening | M1-M6, M_E2E | PLANNED |

---

## Interface Contracts

### Data Preprocessing ↔ Models & Storage
- Processed Series: Parquet files partitioned by `city=/year=/meter=/` or standardized tables with columns:
  - `timestamp`: UTC ISO-8601 string
  - `meter_id`: string
  - `power_kw`: float (active power converted via $P_{\text{kW}} = t_{\text{kWh}} \times 20$)
  - `voltage_v`: float
  - `current_a`: float
  - `freq_hz`: float
  - `is_outage`: boolean ($V \le 5$ or $I \le 0.01$)
- `split_manifest.json`:
  ```json
  {
    "generated_at": "ISO-8601",
    "source_file_sha256": { "<filename>": "<sha256>" },
    "splits": {
      "<meter_id>": {
        "train": { "start": "ISO", "end": "ISO", "rows": 1234 },
        "valid": { "start": "ISO", "end": "ISO", "rows": 264 },
        "test": { "start": "ISO", "end": "ISO", "rows": 264 }
      }
    }
  }
  ```
- `quality_report.json`:
  ```json
  {
    "total_meters_evaluated": 84,
    "admitted_meters_count": 55,
    "admission_rate": 0.655,
    "failing_meters": {
      "<meter_id>": {
        "failing_gates": ["outage_fraction_exceeded"],
        "outage_fraction": 0.48
      }
    }
  }
  ```

### Models ↔ FastAPI Backend
- Model Registry Export Contract (`exports/`):
  - Model weights: `.pt` or `.onnx` or weights dictionary
  - `model_card.json` / `metrics.json`:
    ```json
    {
      "model_name": "m4_lstm_forecaster",
      "module": "M4",
      "architecture": "LSTM(hidden=64, layers=2, embed=16)",
      "config_hash": "sha256_hash",
      "dataset_hash": "sha256_hash",
      "seed": 42,
      "metrics": {
        "mase": 0.812,
        "mae": 0.045,
        "rmse": 0.078
      },
      "baseline": {
        "name": "seasonal_naive",
        "mase": 1.000,
        "mae": 0.061
      },
      "hardware": "Tesla T4 GPU",
      "wall_clock_s": 145.2
    }
    ```

### FastAPI Backend ↔ React Frontend
- Global JSON Envelope:
  ```json
  {
    "data": { ... },
    "meta": {
      "model_version": "v1.0.0",
      "dataset_hash": "a1b2c3d4...",
      "latency_ms": 42.5
    }
  }
  ```
- Error Envelope:
  ```json
  {
    "error": {
      "code": "BAD_REQUEST",
      "message": "Human-readable description",
      "detail": { ... }
    }
  }
  ```

---

## Code Layout

```
c:\Users\Pritam\OneDrive\Desktop\SmartgridAI\
├── data/
│   └── pd-defect/                    # D3 vision dataset
├── archive (1)/                      # D1 raw smart meter CSVs
├── Electricity Demand.../            # D2 raw grid demand XLSXs
├── docs/
│   ├── PRD.md                        # Product requirements
│   ├── DATA_CONTRACT.md              # Physical data contract
│   ├── MODEL_CARDS.md                # Model architecture & evaluation cards
│   └── CREDITS.md                    # Attribution notices
├── src/
│   └── smartgrid/
│       ├── __init__.py
│       ├── data/
│       │   ├── contract.py           # Physical formulas & validation
│       │   ├── ingestion.py          # D1, D2, D3 readers
│       │   ├── quality_gates.py      # 5-gate admission engine
│       │   └── splitter.py           # Chronological 70/15/15 splitter
│       └── models/
│           ├── inference.py          # Lightweight local model inference
│           ├── baselines.py          # Ridge, naive, majority-class baselines
│           └── evaluation.py         # MASE, macro-F1, mAP, mIoU calculators
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI application
│   │   ├── config.py                 # App settings
│   │   ├── routes/                   # Endpoint routers (analytics, ann, forecast, etc.)
│   │   ├── schemas/                  # Pydantic v2 request/response schemas
│   │   └── services/                 # Business logic & job queue
│   └── requirements.txt
├── notebooks/
│   ├── M1_shallow.ipynb              # Perceptron, XOR MLP, Demand MLP vs Ridge
│   ├── M2_dynamics.ipynb             # Optimizers, inits, regularizers, gradient norms
│   ├── M3_vision.ipynb               # Pole lean CNN, YOLO detection, U-Net
│   ├── M4_sequence.ipynb             # Multi-step LSTM/GRU vs Baselines
│   └── M5_generative_anomaly.ipynb   # WGAN-GP & VAE anomaly injection
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── components/               # KPI cards, charts, experiment labs
│       └── pages/                    # Overview, Forecast, Infrastructure, Dynamics, etc.
├── tests/
│   ├── unit/                         # Fast unit tests for contracts and logic
│   └── e2e/                          # Requirement-driven E2E test suite (Tiers 1-4)
├── split_manifest.json               # Generated split boundaries & hashes
└── quality_report.json               # Generated meter admission report
```
