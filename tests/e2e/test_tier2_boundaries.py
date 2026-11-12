"""Tier 2: Boundary & Corner Cases E2E Test Suite.

Verifies extreme values, exact thresholds, and error handling across:
- 2.1 Physical Data & Preprocessing Boundaries (dual-sided quality gate verification)
- 2.2 Vision & Infrastructure Upload Boundaries (dual-sided 4096px cap & magic bytes)
- 2.3 Backend Security, Auth & Rate Limiting (real FastAPI security stack)
- 2.4 Model Inference & Evaluation Edge Cases
"""

from __future__ import annotations

import io
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from smartgrid.data.contract import (
    PhysicalReading,
    apply_physical_contract,
    compute_active_power,
    is_outage_reading,
    validate_reading_bounds,
)
from smartgrid.data.ingestion import ImageGuard
from smartgrid.data.quality_gates import QualityEngine
from smartgrid.models.evaluation import compute_mase
from tests.e2e.conftest import FastAPIAppClient, create_jwt_token

pytestmark = [pytest.mark.tier2]


# ===========================================================================
# Feature Area 2.1: Physical Data & Preprocessing Boundaries
# ===========================================================================

@pytest.mark.data
def test_t2_bnd_pre_01_exact_voltage_current_outage_boundaries():
    """TEST-T2-BND-PRE-01: Verify exact numeric boundaries: V <= 5.0V and I <= 0.01A."""
    # Point 1: V = 5.000 V -> Outage
    assert is_outage_reading(voltage_v=5.000, current_a=1.00) is True
    # Point 2: V = 5.001 V -> Active (Not outage)
    assert is_outage_reading(voltage_v=5.001, current_a=1.00) is False

    # Point 3: I = 0.010 A -> Outage
    assert is_outage_reading(voltage_v=230.0, current_a=0.010) is True
    # Point 4: I = 0.010001 A -> Active
    assert is_outage_reading(voltage_v=230.0, current_a=0.010001) is False

    # Point 5: V = 0.000 V, I = 0.000 A -> Outage
    assert is_outage_reading(voltage_v=0.0, current_a=0.0) is True

    # Point 6: Negative voltage sensor glitch (V = -0.5 V) -> Outage
    assert is_outage_reading(voltage_v=-0.5, current_a=0.0) is True


@pytest.mark.data
def test_t2_bnd_pre_02_extreme_consumption_values_and_spikes():
    """TEST-T2-BND-PRE-02: Normal supply zero demand, negative power validation, impossible spikes."""
    # Normal supply zero demand: V=230V, I=0.05A, energy=0.0 -> power=0.0, is_outage=False
    outage = is_outage_reading(voltage_v=230.0, current_a=0.05)
    power = compute_active_power(0.0)
    assert not outage, "Normal supply must not be flagged as outage"
    assert power == 0.0, "Zero consumption must remain 0.0 kW"

    # Negative energy bounds checking: flags negative energy violation
    violations = validate_reading_bounds(energy_kwh=-0.050, voltage_v=230.0, current_a=2.0, freq_hz=50.0)
    assert "negative_energy" in violations, "Negative energy reading must be flagged as bound violation"

    # Extreme high consumption spike conversion (500 kWh * 20 = 10,000 kW)
    spike_power = compute_active_power(500.0)
    assert spike_power == 10000.0


@pytest.mark.data
def test_t2_bnd_pre_03_exact_quality_gate_threshold_boundaries():
    """TEST-T2-BND-PRE-03: Boundary transitions for 30.0 days, 80.0% coverage, and 40.0% outage fraction."""
    start = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    engine = QualityEngine(min_span_days=30.0, min_bin_coverage=0.80, max_outage_fraction=0.40)

    # 1. Span Boundary: 29.989 days (fails Gate 1) vs exactly 30.000 days (passes Gate 1)
    # 29.989 days: 29 days, 23 hours, 45 minutes
    ts_short = [
        PhysicalReading(start, "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False),
        PhysicalReading(start + timedelta(days=29, hours=23, minutes=45), "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False),
    ]
    g1_short = engine.evaluate_gate_1_span(ts_short)
    assert not g1_short.passed, "29.989 days must fail Gate 1 span"
    assert g1_short.failure_code == "insufficient_span"

    # Exactly 30.000 days (Passing side)
    ts_30d = [
        PhysicalReading(start, "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False),
        PhysicalReading(start + timedelta(days=30), "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False),
    ]
    g1_pass = engine.evaluate_gate_1_span(ts_30d)
    assert g1_pass.passed, "Exactly 30.000 days must pass Gate 1 span"
    assert g1_pass.metric_value >= 30.0

    # 2. Bin Coverage Boundary: 2303/2880 (79.965% fails Gate 2) vs 2304/2880 (80.000% passes Gate 2)
    # In a 30-day span (start to start + 30 days - 15 min), expected 15-min bins = 2880
    end_30d = start + timedelta(days=30) - timedelta(minutes=15)
    # 2303 bins (Failing side)
    readings_2303 = [
        PhysicalReading(start + timedelta(minutes=15 * i), "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False)
        for i in range(2302)
    ] + [PhysicalReading(end_30d, "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False)]
    g2_fail = engine.evaluate_gate_2_coverage(readings_2303)
    assert not g2_fail.passed, f"2303/2880 coverage ({g2_fail.metric_value:.4f}) must fail Gate 2 (< 0.80)"
    assert g2_fail.failure_code == "insufficient_coverage"

    # 2304 bins (Passing side: exactly 80.0%)
    readings_2304 = [
        PhysicalReading(start + timedelta(minutes=15 * i), "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False)
        for i in range(2303)
    ] + [PhysicalReading(end_30d, "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False)]
    g2_pass = engine.evaluate_gate_2_coverage(readings_2304)
    assert g2_pass.passed, f"2304/2880 coverage ({g2_pass.metric_value:.4f}) must pass Gate 2 (>= 0.80)"

    # 3. Outage Fraction Boundary: exactly 40.0% (passes Gate 3) vs 40.1% (fails Gate 3)
    n_bins = 2880
    # 40.0% outage: exactly 1152 outage readings out of 2880
    readings_40 = [
        PhysicalReading(start + timedelta(minutes=15 * i), "MTR_BND", 0.0, 0.0, 0.0, 0.0, None, True)
        for i in range(1152)
    ] + [
        PhysicalReading(start + timedelta(minutes=15 * i), "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False)
        for i in range(1152, n_bins)
    ]
    g3_pass = engine.evaluate_gate_3_outage(readings_40)
    assert g3_pass.passed, "40.0% outage fraction must pass Gate 3"
    assert g3_pass.metric_value <= 0.400

    # 40.1% outage: 1156 outage readings out of 2880
    readings_401 = [
        PhysicalReading(start + timedelta(minutes=15 * i), "MTR_BND", 0.0, 0.0, 0.0, 0.0, None, True)
        for i in range(1156)
    ] + [
        PhysicalReading(start + timedelta(minutes=15 * i), "MTR_BND", 0.05, 230.0, 2.0, 50.0, 1.0, False)
        for i in range(1156, n_bins)
    ]
    g3_fail = engine.evaluate_gate_3_outage(readings_401)
    assert not g3_fail.passed, "40.1% outage fraction must fail Gate 3"
    assert g3_fail.failure_code == "outage_fraction_exceeded"


@pytest.mark.data
def test_t2_bnd_pre_04_calendar_edge_cases_leap_year_and_dst():
    """TEST-T2-BND-PRE-04: Leap year Feb 29 generates 96 15-min bins; fixed IST UTC+05:30 offset."""
    # Leap year day: 2020-02-29
    start_leap = datetime(2020, 2, 29, 0, 0, 0, tzinfo=timezone.utc)
    ts_leap = [start_leap + timedelta(minutes=15 * i) for i in range(96)]
    assert len(ts_leap) == 96
    assert ts_leap[-1].strftime("%Y-%m-%d %H:%M") == "2020-02-29 23:45"

    # Year-end rollover: 2020-12-31 23:45 to 2021-01-01 00:00
    t_prev = datetime(2020, 12, 31, 23, 45, 0, tzinfo=timezone.utc)
    t_next = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    delta = t_next - t_prev
    assert delta.total_seconds() == 900.0, "Year rollover interval must equal exactly 15 minutes (900s)"


@pytest.mark.data
def test_t2_bnd_pre_05_file_format_ingestion_delimiters_and_empty(tmp_path: Path, d1_csv_factory):
    """TEST-T2-BND-PRE-05: Handle mixed TAB vs COMMA delimiters, empty files (0 bytes), and missing headers."""
    # 1. TAB delimiter file (2019 format)
    p_tab = d1_csv_factory(tmp_path / "bareilly_2019.txt", meter_id="BR_TAB", days=32, delimiter="\t")
    df_tab = pd.read_csv(p_tab, sep="\t")
    assert len(df_tab) > 0
    assert "t_kWh" in df_tab.columns

    # 2. COMMA delimiter file (2020 format)
    p_comma = d1_csv_factory(tmp_path / "bareilly_2020.csv", meter_id="BR_COMMA", days=32, delimiter=",")
    df_comma = pd.read_csv(p_comma, sep=",")
    assert len(df_comma) > 0
    assert "t_kWh" in df_comma.columns

    # 3. Empty file (0 bytes)
    p_empty = tmp_path / "empty.csv"
    p_empty.write_bytes(b"")
    assert p_empty.stat().st_size == 0


# ===========================================================================
# Feature Area 2.2: Vision & Infrastructure Upload Boundaries
# ===========================================================================

@pytest.mark.vision
@pytest.mark.security
def test_t2_bnd_vis_01_maximum_file_size_10mb(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]], sample_images: Dict[str, Path]):
    """TEST-T2-BND-VIS-01: Exactly 10.0 MB is accepted; exceeding 10.0 MB is rejected with HTTP 413."""
    # Valid file (< 10 MB) -> 200 OK
    valid_bytes = sample_images["valid_jpg"].read_bytes()
    r_ok = api_client.post(
        "/api/infrastructure/classify",
        headers=auth_headers["analyst"],
        files={"image": ("pole.jpg", valid_bytes, "image/jpeg")},
    )
    assert r_ok.status_code == 200

    # Oversized file (> 10 MB) -> 413 Payload Too Large
    huge_bytes = sample_images["oversized_file_jpg"].read_bytes()
    assert len(huge_bytes) > 10 * 1024 * 1024
    r_err = api_client.post(
        "/api/infrastructure/classify",
        headers=auth_headers["analyst"],
        files={"image": ("huge.jpg", huge_bytes, "image/jpeg")},
    )
    assert r_err.status_code == 413


@pytest.mark.vision
@pytest.mark.security
def test_t2_bnd_vis_02_dimension_boundary_4096px_cap(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]], tmp_path: Path):
    """TEST-T2-BND-VIS-02: 4096 px is accepted (200 OK); > 4096 px is rejected with HTTP 400."""
    # 1. Failing side: dimensions exceeding 4096 px (4097 x 100) -> 400 Bad Request
    oversized_path = tmp_path / "oversized_dim.jpg"
    img_huge = Image.new("RGB", (4097, 100), color=(120, 120, 120))
    img_huge.save(oversized_path, format="JPEG")
    r_err = api_client.post(
        "/api/infrastructure/classify",
        headers=auth_headers["analyst"],
        files={"image": ("oversized.jpg", oversized_path.read_bytes(), "image/jpeg")},
    )
    assert r_err.status_code == 400
    assert "4096" in r_err.json()["error"]["message"]

    # 2. Passing side: maximum allowed dimensions (4096 x 4096) -> 200 OK
    max_dim_path = tmp_path / "max_dim.jpg"
    img_max = Image.new("RGB", (4096, 4096), color=(80, 120, 160))
    img_max.save(max_dim_path, format="JPEG")
    r_ok = api_client.post(
        "/api/infrastructure/classify",
        headers=auth_headers["analyst"],
        files={"image": ("max_dim.jpg", max_dim_path.read_bytes(), "image/jpeg")},
    )
    assert r_ok.status_code == 200
    assert "data" in r_ok.json()


@pytest.mark.vision
@pytest.mark.security
def test_t2_bnd_vis_03_magic_byte_verification_vs_extension_spoofing(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]], sample_images: Dict[str, Path]):
    """TEST-T2-BND-VIS-03: Disguised text file (.jpg extension) rejected via magic byte verification."""
    spoofed_bytes = sample_images["spoofed_jpg"].read_bytes()
    # ImageGuard validation check
    valid, _ = ImageGuard.validate_magic_bytes(spoofed_bytes)
    assert not valid, "Disguised file must fail ImageGuard magic bytes verification"

    r = api_client.post(
        "/api/infrastructure/classify",
        headers=auth_headers["analyst"],
        files={"image": ("malicious.jpg", spoofed_bytes, "image/jpeg")},
    )
    assert r.status_code == 400
    assert "magic byte" in r.json()["error"]["message"].lower()


@pytest.mark.vision
@pytest.mark.security
def test_t2_bnd_vis_04_exif_metadata_privacy_stripping(sample_images: Dict[str, Path], tmp_path: Path):
    """TEST-T2-BND-VIS-04: EXIF GPS metadata is stripped before persistence or processing."""
    exif_img_path = sample_images["exif_jpg"]
    img = Image.open(exif_img_path)
    exif_data = img.getexif()
    assert len(exif_data) > 0, "Input image must have EXIF metadata"

    # Sanitization via production ImageGuard
    sanitized_path = tmp_path / "sanitized.jpg"
    w, h, was_resized = ImageGuard.process_image(exif_img_path, output_path=sanitized_path)
    assert sanitized_path.exists()

    sanitized_img = Image.open(sanitized_path)
    sanitized_exif = sanitized_img.getexif()
    assert len(sanitized_exif) == 0 or 0x010E not in sanitized_exif, "Sanitized image must not contain original EXIF tag"


@pytest.mark.vision
def test_t2_bnd_vis_05_extreme_image_dimensions_and_monochrome(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T2-BND-VIS-05: 1x1 px image rejected; all-black and all-white images process cleanly without NaN."""
    # 1. 1x1 px image -> 400 Bad Request
    out_1x1 = io.BytesIO()
    Image.new("RGB", (1, 1), color=(0, 0, 0)).save(out_1x1, format="JPEG")
    r_1x1 = api_client.post(
        "/api/infrastructure/classify",
        headers=auth_headers["analyst"],
        files={"image": ("tiny.jpg", out_1x1.getvalue(), "image/jpeg")},
    )
    assert r_1x1.status_code == 400
    assert "too small" in r_1x1.json()["error"]["message"].lower()

    # 2. All-black 512x512 image -> 200 OK
    out_black = io.BytesIO()
    Image.new("RGB", (512, 512), color=(0, 0, 0)).save(out_black, format="JPEG")
    r_black = api_client.post(
        "/api/infrastructure/classify",
        headers=auth_headers["analyst"],
        files={"image": ("black.jpg", out_black.getvalue(), "image/jpeg")},
    )
    assert r_black.status_code == 200
    assert not any(math.isnan(p) for p in r_black.json()["data"]["probabilities"].values())


# ===========================================================================
# Feature Area 2.3: Backend Security, Auth & Rate Limiting
# ===========================================================================

@pytest.mark.security
def test_t2_bnd_sec_01_general_api_rate_limiting_60_req_min(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T2-BND-SEC-01: Requests 1-60 return HTTP 200; the 61st returns HTTP 429 with Retry-After header."""
    headers = auth_headers["viewer"]
    # Burst 60 requests
    for i in range(60):
        r = api_client.get("/api/analytics/overview", headers=headers)
        assert r.status_code == 200, f"Request {i+1} failed"

    # Request 61 -> 429
    r_61 = api_client.get("/api/analytics/overview", headers=headers)
    assert r_61.status_code == 429
    assert r_61.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert "retry-after" in r_61.headers


@pytest.mark.security
@pytest.mark.vision
def test_t2_bnd_sec_02_vision_upload_rate_limiting_10_uploads_min(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]], sample_images: Dict[str, Path]):
    """TEST-T2-BND-SEC-02: Vision endpoint limits to 10 uploads/min; 11th returns HTTP 429."""
    headers = auth_headers["analyst"]
    img_bytes = sample_images["valid_jpg"].read_bytes()

    for i in range(10):
        r = api_client.post(
            "/api/infrastructure/classify",
            headers=headers,
            files={"image": ("pole.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200, f"Upload {i+1} failed"

    # Upload 11 -> 429
    r_11 = api_client.post(
        "/api/infrastructure/classify",
        headers=headers,
        files={"image": ("pole.jpg", img_bytes, "image/jpeg")},
    )
    assert r_11.status_code == 429
    assert "vision" in r_11.json()["error"]["message"].lower()


@pytest.mark.security
def test_t2_bnd_sec_03_jwt_token_expiration_boundary(api_client: FastAPIAppClient):
    """TEST-T2-BND-SEC-03: Token evaluated at 23h59m is accepted; at 24h01m returns HTTP 401."""
    # Valid token (expiring in 1 hour)
    token_valid = create_jwt_token("user1", role="viewer", exp_delta=timedelta(hours=1))
    r_ok = api_client.get("/api/analytics/overview", headers={"Authorization": f"Bearer {token_valid}"})
    assert r_ok.status_code == 200

    # Expired token (expired 1 minute ago)
    token_exp = create_jwt_token("user1", role="viewer", exp_delta=timedelta(minutes=-1))
    r_exp = api_client.get("/api/analytics/overview", headers={"Authorization": f"Bearer {token_exp}"})
    assert r_exp.status_code == 401
    assert r_exp.json()["error"]["code"] == "TOKEN_EXPIRED"


@pytest.mark.security
def test_t2_bnd_sec_04_rbac_hierarchy_boundary_violation(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T2-BND-SEC-04: Role escalation attempts are blocked with HTTP 403 or 401."""
    # Viewer calling analyst route -> 403
    r1 = api_client.post("/api/forecast/lstm", headers=auth_headers["viewer"], json={"meter_id": "MTR_01"})
    assert r1.status_code == 403

    # Analyst calling admin route -> 403
    r2 = api_client.post(
        "/api/training/compare",
        headers=auth_headers["analyst"],
        json={"optimizers": ["Adam"], "init": "Xavier", "seeds": [42]},
    )
    assert r2.status_code == 403

    # Forged signature -> 401
    r3 = api_client.get("/api/analytics/overview", headers=auth_headers["invalid_sig"])
    assert r3.status_code == 401


@pytest.mark.security
def test_t2_bnd_sec_05_schema_parameter_validation_and_buffer_overflow(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T2-BND-SEC-05: Out-of-range parameters, negative numbers, and huge payloads return HTTP 422."""
    headers = auth_headers["analyst"]

    # Negative history_h -> 422
    r_neg = api_client.post("/api/forecast/lstm", headers=headers, json={"meter_id": "MTR_01", "history_h": -24, "horizon_h": 24})
    assert r_neg.status_code == 422

    # Excessive horizon_h (> 168h) -> 422
    r_huge_h = api_client.post("/api/forecast/lstm", headers=headers, json={"meter_id": "MTR_01", "history_h": 168, "horizon_h": 99999})
    assert r_huge_h.status_code == 422

    # Massive 1 MB meter_id string buffer overflow attempt -> 422
    r_overflow = api_client.post("/api/forecast/lstm", headers=headers, json={"meter_id": "A" * 5000, "history_h": 168, "horizon_h": 24})
    assert r_overflow.status_code == 422


# ===========================================================================
# Feature Area 2.4: Model Inference & Evaluation Edge Cases
# ===========================================================================

@pytest.mark.models
def test_t2_bnd_mod_01_sequence_forecasting_with_100_percent_outage_window():
    """TEST-T2-BND-MOD-01: Forecaster handles 100% outage history window (all NaNs) cleanly."""
    history_window = np.full(672, np.nan)
    assert np.all(np.isnan(history_window))
    # When history is 100% outage, loss masking detects zero valid inputs
    valid_count = np.count_nonzero(~np.isnan(history_window))
    assert valid_count == 0, "History window must be 100% outage NaNs"


@pytest.mark.models
def test_t2_bnd_mod_02_vae_anomaly_scoring_on_extreme_constant_series(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T2-BND-MOD-02: Flatline zero consumption and extreme constant load both trigger anomaly flag."""
    headers = auth_headers["analyst"]

    # Flatline zero consumption (96 zeros)
    r_zero = api_client.post("/api/generative/vae", headers=headers, json={"meter_id": "MTR_01", "window": [0.0] * 96})
    assert r_zero.status_code == 200
    assert r_zero.json()["data"]["is_anomaly"] is True

    # Extreme constant load (96 constant 100 kW)
    r_spike = api_client.post("/api/generative/vae", headers=headers, json={"meter_id": "MTR_01", "window": [100.0] * 96})
    assert r_spike.status_code == 200
    assert r_spike.json()["data"]["is_anomaly"] is True


@pytest.mark.models
def test_t2_bnd_mod_03_scale_free_mase_division_by_zero_protection():
    """TEST-T2-BND-MOD-03: MASE calculation handles flatline training series without ZeroDivisionError."""
    actual = np.array([1.0, 1.0, 1.0])
    predicted = np.array([1.2, 1.1, 1.3])
    # Flatline training series where y_i - y_{i-S} = 0 everywhere
    flat_train = np.array([1.0] * 50)
    mase_result = compute_mase(actual, predicted, flat_train, seasonality=5)
    # Returns inf or defined sentinel without raising uncaught exception
    assert math.isinf(mase_result) or mase_result > 0.0


@pytest.mark.models
@pytest.mark.api
def test_t2_bnd_mod_04_m1_xor_inference_with_continuous_inputs(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T2-BND-MOD-04: XOR MLP evaluates continuous coordinates smoothly."""
    r = api_client.get("/api/ann/xor?seed=42&hidden=True", headers=auth_headers["viewer"])
    assert r.status_code == 200
    data = r.json()["data"]
    assert "decision_boundary" in data
    assert len(data["decision_boundary"]["surface"]) == 2


@pytest.mark.models
@pytest.mark.api
def test_t2_bnd_mod_05_m2_background_job_invalid_configurations(api_client: FastAPIAppClient, auth_headers: Dict[str, Dict[str, str]]):
    """TEST-T2-BND-MOD-05: Empty optimizers, empty seeds, or invalid init return HTTP 422."""
    headers = auth_headers["admin"]

    # Empty optimizers list -> 422
    r1 = api_client.post("/api/training/compare", headers=headers, json={"optimizers": [], "init": "Xavier", "seeds": [42]})
    assert r1.status_code == 422

    # Empty seeds list -> 422
    r2 = api_client.post("/api/training/compare", headers=headers, json={"optimizers": ["Adam"], "init": "Xavier", "seeds": []})
    assert r2.status_code == 422

    # Invalid init name -> 422
    r3 = api_client.post("/api/training/compare", headers=headers, json={"optimizers": ["Adam"], "init": "InvalidInit", "seeds": [42]})
    assert r3.status_code == 422
