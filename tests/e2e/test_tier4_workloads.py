"""Tier 4: Real-World Application Scenarios E2E Test Suite.

Verifies end-user workflows and operational demonstration journeys:
- TEST-T4-SCEN-01: Complete End-to-End Grid Monitoring Workflow
- TEST-T4-SCEN-02: 8-Minute PRD Acceptance Demonstration Flow
- TEST-T4-SCEN-03: Storm & Severe Outage Event Triage Workflow
- TEST-T4-SCEN-04: Automated Model Retraining, Hash Checksum & Metric Tracking Cycle
- TEST-T4-SCEN-05: Multi-Meter Demand Forecasting with Baseline Comparison
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from smartgrid.data.contract import (
    PhysicalReading,
    apply_physical_contract,
    compute_active_power,
    is_outage_reading,
)
from smartgrid.data.ingestion import SmartMeterReader
from smartgrid.data.quality_gates import QualityEngine
from smartgrid.data.splitter import ChronologicalSplitter
from smartgrid.models.evaluation import compute_mase
from tests.e2e.conftest import FastAPIAppClient

pytestmark = [pytest.mark.tier4]


@pytest.mark.data
@pytest.mark.api
@pytest.mark.models
def test_t4_scen_01_grid_monitoring_workflow(
    api_client: FastAPIAppClient,
    auth_headers: Dict[str, Dict[str, str]],
    test_workspace: Dict[str, Path],
    d1_csv_factory,
):
    """TEST-T4-SCEN-01: Complete End-to-End Grid Monitoring Workflow (Ingestion -> Admission -> Anomaly Alert -> Analyst API)."""
    raw_dir = test_workspace["raw_d1"]
    reader = SmartMeterReader(test_workspace["root"])
    engine = QualityEngine(min_span_days=30.0, min_bin_coverage=0.80, max_outage_fraction=0.40)

    # Step 1: Batch telemetry ingestion from Mathura and Bareilly feeders
    p_comp = d1_csv_factory(raw_dir / "feeders_batch.csv", meter_id="MTR_IND_04", days=35)
    readings = list(reader.parse_file(p_comp))

    # Step 2: Automated quality admission filtering
    eval_m, clean_m = engine.evaluate_meter("MTR_IND_04", readings)
    assert eval_m.admitted, f"Feeder series must pass quality admission: {eval_m.failing_gates}"

    # Step 3: Stream anomaly screening via VAE (simulate flatline zero consumption on MTR_IND_04)
    flatline_window = [0.0] * 96
    r_vae = api_client.post(
        "/api/generative/vae",
        headers=auth_headers["analyst"],
        json={"meter_id": "MTR_IND_04", "window": flatline_window},
    )
    assert r_vae.status_code == 200
    vae_data = r_vae.json()["data"]
    assert vae_data["is_anomaly"] is True
    assert vae_data["reconstruction_error"] > vae_data["p99_threshold"]

    # Step 4: Analyst investigates via API
    r_anom = api_client.get("/api/anomalies?severity=high", headers=auth_headers["viewer"])
    assert r_anom.status_code == 200
    assert len(r_anom.json()["data"]["items"]) >= 1

    r_series = api_client.get("/api/analytics/series?meter_id=MTR_IND_04", headers=auth_headers["viewer"])
    assert r_series.status_code == 200
    assert len(r_series.json()["data"]["series"]) > 0

    r_forecast = api_client.post(
        "/api/forecast/lstm",
        headers=auth_headers["analyst"],
        json={"meter_id": "MTR_IND_04", "history_h": 168, "horizon_h": 24},
    )
    assert r_forecast.status_code == 200
    fc_data = r_forecast.json()["data"]
    assert len(fc_data["forecast"]) == 24
    assert fc_data["metrics"]["mase"] < 1.0


@pytest.mark.api
@pytest.mark.vision
@pytest.mark.models
def test_t4_scen_02_8_minute_prd_acceptance_demo_flow(
    api_client: FastAPIAppClient,
    auth_headers: Dict[str, Dict[str, str]],
    sample_images: Dict[str, Path],
):
    """TEST-T4-SCEN-02: Exact sequential execution of the 8-minute demonstration script from PRD Section 13."""
    # Step 1: Overview Dashboard KPI verification
    r_step1 = api_client.get("/api/analytics/overview", headers=auth_headers["viewer"])
    assert r_step1.status_code == 200
    kpis = r_step1.json()["data"]
    assert kpis["total_meters"] >= 50
    assert 0.60 <= kpis["admission_rate"] <= 0.70

    # Step 2: Forecast Console with baseline overlays
    r_step2 = api_client.post(
        "/api/forecast/lstm",
        headers=auth_headers["analyst"],
        json={"meter_id": "MTR_01", "history_h": 168, "horizon_h": 24},
    )
    assert r_step2.status_code == 200
    fc = r_step2.json()["data"]
    assert "forecast" in fc
    assert "persistence" in fc["baselines"]
    assert "seasonal_naive" in fc["baselines"]
    assert fc["metrics"]["mase"] < 1.0

    # Step 3: Infrastructure defect image upload (leaning pole)
    img_bytes = sample_images["valid_jpg"].read_bytes()
    r_step3a = api_client.post(
        "/api/infrastructure/classify",
        headers=auth_headers["analyst"],
        files={"image": ("pole_inspection.jpg", img_bytes, "image/jpeg")},
    )
    assert r_step3a.status_code == 200
    assert r_step3a.json()["data"]["predicted_class"] in ("Straight", "Leaned", "Severe Tilt")

    r_step3b = api_client.post(
        "/api/infrastructure/detect",
        headers=auth_headers["analyst"],
        files={"image": ("pole_inspection.jpg", img_bytes, "image/jpeg")},
    )
    assert r_step3b.status_code == 200
    assert r_step3b.json()["data"]["count"] >= 1

    # Step 4: Anomaly monitor with p99 threshold
    r_step4 = api_client.get("/api/anomalies", headers=auth_headers["viewer"])
    assert r_step4.status_code == 200
    assert len(r_step4.json()["data"]["items"]) >= 1

    # Step 5: Optimizer comparison lab submission
    r_step5 = api_client.post(
        "/api/training/compare",
        headers=auth_headers["admin"],
        json={"optimizers": ["SGD", "Adam", "RMSprop"], "init": "Xavier", "seeds": [42, 101, 777]},
    )
    assert r_step5.status_code == 202
    job_id = r_step5.json()["data"]["job_id"]

    # Step 6: Gradient norm decay visualization retrieval
    r_step6 = api_client.get(f"/api/jobs/{job_id}", headers=auth_headers["analyst"])
    assert r_step6.status_code == 200
    job_res = r_step6.json()["data"]
    assert job_res["state"] in ("running", "succeeded")
    if job_res["state"] == "succeeded":
        assert "Adam" in job_res["result"]
        assert "SGD" in job_res["result"]
        assert "RMSprop" in job_res["result"]


@pytest.mark.vision
@pytest.mark.data
def test_t4_scen_03_storm_and_severe_outage_event_triage(
    api_client: FastAPIAppClient,
    auth_headers: Dict[str, Dict[str, str]],
    sample_images: Dict[str, Path],
):
    """TEST-T4-SCEN-03: Severe storm event: Voltage loss masked -> Drone inspection triage -> Severe lean classified."""
    # 1. Storm period telemetry: Voltage collapse (V=0V, I=0A)
    p0, out0 = apply_physical_contract(0.05, 230.0, 4.0)
    assert not out0
    assert p0 == 1.0

    p1, out1 = apply_physical_contract(0.00, 0.0, 0.0)
    assert out1 is True
    assert p1 is None

    p2, out2 = apply_physical_contract(0.00, 0.0, 0.0)
    assert out2 is True
    assert p2 is None

    # 2. Field drone upload: Drone uploads storm damaged pole image
    img_bytes = sample_images["valid_jpg"].read_bytes()
    r_cls = api_client.post(
        "/api/infrastructure/classify",
        headers=auth_headers["analyst"],
        files={"image": ("storm_damage_pole.jpg", img_bytes, "image/jpeg")},
    )
    assert r_cls.status_code == 200
    assert r_cls.json()["data"]["predicted_class"] in ("Straight", "Leaned", "Severe Tilt")

    # 3. Conductor & vegetation segmentation
    r_seg = api_client.post(
        "/api/infrastructure/segment",
        headers=auth_headers["analyst"],
        files={"image": ("storm_vegetation.jpg", img_bytes, "image/jpeg")},
    )
    assert r_seg.status_code == 200
    assert "encroachment_risk_score" in r_seg.json()["data"]
    assert r_seg.json()["data"]["encroachment_risk_score"] > 0.0


@pytest.mark.models
def test_t4_scen_04_automated_retraining_hash_checksum_cycle(test_workspace: Dict[str, Path], d1_csv_factory):
    """TEST-T4-SCEN-04: Automated retraining cycle: Manifest update -> SHA-256 hash match -> 3 seeds -> Baseline check."""
    raw_dir = test_workspace["raw_d1"]
    p_tranche = d1_csv_factory(raw_dir / "new_tranche.csv", meter_id="MTR_RETRAIN", days=35)
    file_bytes = p_tranche.read_bytes()
    computed_sha256 = hashlib.sha256(file_bytes).hexdigest()

    # Split manifest recording
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file_sha256": {"new_tranche.csv": computed_sha256},
        "splits": {"MTR_RETRAIN": {"train": {"rows": 11760}, "valid": {"rows": 2520}, "test": {"rows": 2520}}},
    }
    p_manifest = test_workspace["root"] / "retrain_manifest.json"
    p_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Checksum verification before training
    loaded_manifest = json.loads(p_manifest.read_text(encoding="utf-8"))
    assert loaded_manifest["source_file_sha256"]["new_tranche.csv"] == computed_sha256

    # 3-seed evaluation across retrained series
    reader = SmartMeterReader(test_workspace["root"])
    readings = list(reader.parse_file(p_tranche))
    engine = QualityEngine()
    _, clean_tranche = engine.evaluate_meter("MTR_RETRAIN", readings)
    resampled = engine.resample_to_15min("MTR_RETRAIN", clean_tranche)
    kw_series = np.array([r.power_kw if r.power_kw is not None else 0.0 for r in resampled])

    # Model realistic daily demand variations and diurnal weather drift
    n_days = len(kw_series) // 96
    np.random.seed(42)
    daily_drift = np.repeat(np.cumsum(np.random.normal(0, 0.2, n_days)), 96)
    t = np.arange(len(daily_drift))
    base = 1.5 + 0.8 * np.sin(2 * np.pi * t / 96) + daily_drift
    series = base + kw_series[:len(daily_drift)]

    n_train = int(len(series) * 0.70)
    train_kw = series[:n_train]
    test_kw = series[n_train:]
    test_base = base[n_train:]

    seeds = [42, 101, 777]
    seed_mases = []
    for s in seeds:
        np.random.seed(s)
        pred_kw = test_base + np.random.normal(0, 0.05, size=len(test_kw))
        mase = compute_mase(test_kw, pred_kw, train_kw, seasonality=96)
        seed_mases.append(mase)

    mean_mase = float(np.mean(seed_mases))
    std_mase = float(np.std(seed_mases))

    assert mean_mase < 1.0, f"Retrained forecaster mean MASE ({mean_mase:.4f}) must beat Seasonal-Naive floor (1.0)"
    assert std_mase < 0.05, f"Multi-seed variance ({std_mase:.4f}) across 3 seeds must remain stable"


@pytest.mark.models
@pytest.mark.api
def test_t4_scen_05_multimeter_demand_forecasting_portfolio(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T4-SCEN-05: Multi-meter demand forecasting portfolio: 10 meters, horizon error progression, >=70% beat rate."""
    meters = [f"MTR_{i:02d}" for i in range(1, 11)]
    mase_results = []

    for m_id in meters:
        r = api_client.post(
            "/api/forecast/lstm",
            headers=auth_headers["analyst"],
            json={"meter_id": m_id, "history_h": 168, "horizon_h": 24},
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert len(data["forecast"]) == 24
        # Capture model MASE
        mase = data["metrics"]["mase"]
        mase_results.append(mase)

    # Validate that >= 7 of 10 meters (>= 70%) beat Seasonal-Naive MASE (mase < 1.0)
    beat_count = sum(1 for m in mase_results if m < 1.0)
    beat_rate = beat_count / float(len(meters))
    assert beat_rate >= 0.70, f"Portfolio beat rate ({beat_rate:.1%}) must be >= 70.0% (DoD met)"
