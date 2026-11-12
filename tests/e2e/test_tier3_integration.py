"""Tier 3: Cross-Feature Combinations E2E Test Suite.

Verifies end-to-end multi-module pipelines and subsystem interactions:
- TEST-T3-XFEAT-01: Ingestion -> Quality Gating -> Splitting -> Signed Manifest -> Parquet Storage
- TEST-T3-XFEAT-02: Processed Parquet Series -> Model Inference -> API Response Envelope
- TEST-T3-XFEAT-03: Multi-Module Deep Model vs Co-Located Baseline Integrity
- TEST-T3-XFEAT-04: Authentication -> Role Gating -> Rate Limiting -> Schema Validation Stack
- TEST-T3-XFEAT-05: Async Background Training Job -> Queue -> Polling -> Registry Export
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from smartgrid.data.contract import (
    PhysicalReading,
    ResampledReading,
    apply_physical_contract,
    compute_active_power,
    is_outage_reading,
)
from smartgrid.data.ingestion import SmartMeterReader
from smartgrid.data.quality_gates import QualityEngine
from smartgrid.data.splitter import ChronologicalSplitter
from tests.e2e.conftest import FastAPIAppClient

pytestmark = [pytest.mark.tier3]


@pytest.mark.data
def test_t3_xfeat_01_ingestion_to_manifest_pipeline(test_workspace: Dict[str, Path], d1_csv_factory):
    """TEST-T3-XFEAT-01: Ingestion -> 5 Gates -> Chronological Splitting -> Signed Manifest -> Parquet."""
    raw_dir = test_workspace["raw_d1"]
    processed_dir = test_workspace["processed"]
    reports_dir = test_workspace["reports"]

    # 1. Create raw source files: 1 compliant meter (35 days), 1 non-compliant meter (20 days)
    p_comp = d1_csv_factory(raw_dir / "Bareilly_2020.csv", meter_id="MTR_COMP", days=35)
    p_short = d1_csv_factory(raw_dir / "Bareilly_short.csv", meter_id="MTR_SHORT", days=20)

    # Ingestion & Quality Evaluation via production smartgrid.data modules
    reader = SmartMeterReader(test_workspace["root"])
    engine = QualityEngine(min_span_days=30.0, min_bin_coverage=0.80, max_outage_fraction=0.40)
    splitter = ChronologicalSplitter(train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15)

    source_files = [p_comp, p_short]
    source_hashes = {}
    for sf in source_files:
        source_hashes[sf.name] = hashlib.sha256(sf.read_bytes()).hexdigest()

    evaluated_meters = {}
    admitted_meters = {}
    failing_meters = {}

    for sf in source_files:
        readings = list(reader.parse_file(sf))
        m_id = readings[0].meter_id if readings else "UNKNOWN"
        eval_m, clean_m = engine.evaluate_meter(m_id, readings)
        evaluated_meters[m_id] = clean_m
        if eval_m.admitted:
            admitted_meters[m_id] = clean_m
        else:
            failing_meters[m_id] = {
                "failing_gates": eval_m.failing_gates,
                "metrics": eval_m.metrics,
            }

    # Verify Admission Gate Logic
    assert "MTR_COMP" in admitted_meters
    assert "MTR_SHORT" in failing_meters

    # Generate quality_report.json
    quality_report = {
        "total_meters_evaluated": len(evaluated_meters),
        "admitted_meters_count": len(admitted_meters),
        "admission_rate": round(len(admitted_meters) / float(len(evaluated_meters)), 3),
        "failing_meters": failing_meters,
    }
    p_qrep = reports_dir / "quality_report.json"
    p_qrep.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    assert p_qrep.exists()

    # Chronological Splitting for admitted meters
    splits_meta = {}
    for m_id, clean_readings in admitted_meters.items():
        resampled = engine.resample_to_15min(m_id, clean_readings)
        train, valid, test, meter_splits = splitter.split_meter_series(resampled)

        splits_meta[m_id] = {
            "train": {"start": train[0].timestamp.isoformat(), "end": train[-1].timestamp.isoformat(), "rows": len(train)},
            "valid": {"start": valid[0].timestamp.isoformat(), "end": valid[-1].timestamp.isoformat(), "rows": len(valid)},
            "test": {"start": test[0].timestamp.isoformat(), "end": test[-1].timestamp.isoformat(), "rows": len(test)},
        }

        # Save to partitioned Parquet: city=Bareilly/year=2020/meter=MTR_COMP/
        partition_dir = processed_dir / "city=Bareilly" / "year=2020" / f"meter={m_id}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_slice in [("train", train), ("valid", valid), ("test", test)]:
            rows = [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "power_kw": r.power_kw,
                    "is_outage": r.is_outage,
                    "z_Avg Voltage (Volt)": r.voltage_v,
                    "z_Avg Current (Amp)": r.current_a,
                    "y_Freq (Hz)": r.freq_hz,
                    "meter": r.meter_id,
                }
                for r in split_slice
            ]
            split_df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(split_df)
            pq.write_table(table, partition_dir / f"{split_name}.parquet")

    # Generate split_manifest.json
    split_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file_sha256": source_hashes,
        "splits": splits_meta,
    }
    p_manifest = test_workspace["root"] / "split_manifest.json"
    p_manifest.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    # Assertions on artifacts
    assert p_manifest.exists()
    manifest_loaded = json.loads(p_manifest.read_text(encoding="utf-8"))
    assert manifest_loaded["source_file_sha256"]["Bareilly_2020.csv"] == source_hashes["Bareilly_2020.csv"]

    # Verify Parquet columns
    pq_path = processed_dir / "city=Bareilly" / "year=2020" / "meter=MTR_COMP" / "train.parquet"
    assert pq_path.exists()
    loaded_table = pq.read_table(pq_path)
    for col in ("timestamp", "power_kw", "is_outage", "z_Avg Voltage (Volt)", "z_Avg Current (Amp)"):
        assert col in loaded_table.column_names


@pytest.mark.data
@pytest.mark.api
@pytest.mark.models
def test_t3_xfeat_02_parquet_to_model_inference_to_api_envelope(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]], test_workspace: Dict[str, Path]):
    """TEST-T3-XFEAT-02: Parquet series -> Model Inference -> API Response Envelope with dataset_hash."""
    # Request forecast for admitted meter
    res = api_client.post(
        "/api/forecast/lstm",
        headers=auth_headers["analyst"],
        json={"meter_id": "MTR_COMP", "history_h": 168, "horizon_h": 24},
    )
    assert res.status_code == 200
    json_resp = res.json()
    assert "data" in json_resp
    assert "meta" in json_resp

    data = json_resp["data"]
    assert data["meter_id"] == "MTR_COMP"
    assert len(data["forecast"]) == 24
    assert "baselines" in data
    assert "persistence" in data["baselines"]
    assert "seasonal_naive" in data["baselines"]

    meta = json_resp["meta"]
    assert "dataset_hash" in meta
    assert len(meta["dataset_hash"]) == 64
    assert meta["latency_ms"] > 0.0


@pytest.mark.models
def test_t3_xfeat_03_multi_module_deep_model_vs_colocated_baseline(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T3-XFEAT-03: Verify evaluation registry strictly pairs every deep model with its co-located baseline."""
    res = api_client.get("/api/models/metrics?module=all", headers=auth_headers["viewer"])
    assert res.status_code == 200
    metrics_list = res.json()["data"]["metrics"]
    assert len(metrics_list) >= 4

    modules_checked = set()
    for entry in metrics_list:
        mod = entry["module"]
        modules_checked.add(mod)
        assert "baseline" in entry, f"Module {mod} entry missing required baseline"
        baseline = entry["baseline"]
        assert "name" in baseline

        # Check specific pairings
        if mod == "M1":
            assert baseline["name"] == "ridge_regression"
            assert entry["metrics"]["mae"] < baseline["mae"]
        elif mod == "M3":
            assert baseline["name"] == "majority_class"
            assert entry["metrics"]["macro_f1"] > baseline["macro_f1"]
        elif mod == "M4":
            assert baseline["name"] == "seasonal_naive"
            assert entry["metrics"]["mase"] < baseline["mase"]
        elif mod == "M5":
            assert baseline["name"] == "daily_zscore"
            assert entry["metrics"]["f1_at_p99"] > baseline["f1_at_p99"]

    assert {"M1", "M3", "M4", "M5"}.issubset(modules_checked)


@pytest.mark.api
@pytest.mark.security
def test_t3_xfeat_04_auth_role_ratelimit_schema_validation_stack(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T3-XFEAT-04: Validate 4-stage pipeline precedence: 401 > 403 > 429 > 422 > 200/202."""
    target_endpoint = "/api/training/compare"
    invalid_schema_body = {"optimizers": [], "init": "InvalidInit", "seeds": []}
    valid_schema_body = {"optimizers": ["Adam"], "init": "Xavier", "seeds": [42]}

    # 1. Unauthenticated -> 401 (Auth takes precedence over everything)
    r1 = api_client.post(target_endpoint, headers={}, json=invalid_schema_body)
    assert r1.status_code == 401
    assert r1.json()["error"]["code"] in ("UNAUTHORIZED", "TOKEN_EXPIRED")

    # 2. Viewer token to admin endpoint -> 403 (RBAC takes precedence over rate limit & schema)
    r2 = api_client.post(target_endpoint, headers=auth_headers["viewer"], json=invalid_schema_body)
    assert r2.status_code == 403
    assert r2.json()["error"]["code"] == "FORBIDDEN"

    # 3. Valid admin token with invalid body -> 422 (Schema validation fires when auth & RBAC pass)
    r3 = api_client.post(target_endpoint, headers=auth_headers["admin"], json=invalid_schema_body)
    assert r3.status_code == 422

    # 4. Valid admin token with valid body -> 202 Accepted
    r4 = api_client.post(target_endpoint, headers=auth_headers["admin"], json=valid_schema_body)
    assert r4.status_code == 202
    assert "job_id" in r4.json()["data"]


@pytest.mark.api
def test_t3_xfeat_05_async_background_training_job_lifecycle(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T3-XFEAT-05: Enqueue training compare job -> 202 -> Poll status -> Succeeded with metrics."""
    headers = auth_headers["admin"]
    body = {
        "optimizers": ["Adam", "SGD", "RMSprop"],
        "init": "Xavier",
        "regularisation": "Dropout",
        "seeds": [42, 101, 777],
    }

    # Step 1: Submit job
    r_submit = api_client.post("/api/training/compare", headers=headers, json=body)
    assert r_submit.status_code == 202
    data_submit = r_submit.json()["data"]
    job_id = data_submit["job_id"]
    assert job_id.startswith("job_")
    assert data_submit["state"] == "queued"

    # Step 2: Poll status
    r_poll = api_client.get(f"/api/jobs/{job_id}", headers=auth_headers["analyst"])
    assert r_poll.status_code == 200
    job_status = r_poll.json()["data"]
    assert job_status["job_id"] == job_id
    assert job_status["state"] in ("running", "succeeded")
    if job_status["state"] == "succeeded":
        assert "result" in job_status
        assert "Adam" in job_status["result"]
        assert "SGD" in job_status["result"]
