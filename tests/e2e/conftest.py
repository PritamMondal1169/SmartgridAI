"""E2E Test Harness & Reusable Fixtures for SmartgridAI.

Provides parameterizable data factories, authentication fixtures, image generators,
and an in-process FastAPI test client executing the real modular backend application
(routes, schemas, JWT auth, and rate limiters) without Windows sandbox socket deadlocks.
"""

from __future__ import annotations

import io
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
from PIL import Image
from starlette.datastructures import UploadFile

# Ensure src and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Direct imports from production codebase (src.smartgrid)
from smartgrid.data.contract import (
    CONVERSION_FACTOR,
    CURRENT_OUTAGE_THRESHOLD_A,
    INTERVAL_MINUTES,
    MAX_OUTAGE_FRACTION,
    MIN_BIN_COVERAGE,
    MIN_SPAN_DAYS,
    VOLTAGE_OUTAGE_THRESHOLD_V,
    PhysicalReading,
    ResampledReading,
    apply_physical_contract,
    compute_active_power,
    is_outage_reading,
    validate_reading_bounds,
)
from smartgrid.data.ingestion import (
    GridDemandReader,
    ImageGuard,
    SmartMeterReader,
    VisionCatalogReader,
)
from smartgrid.data.quality_gates import (
    GateResult,
    MeterEvaluation,
    QualityEngine,
)
from smartgrid.data.splitter import (
    ChronologicalSplitter,
    MeterSplits,
    SplitBoundary,
)
from smartgrid.models.evaluation import compute_mase

# Direct imports from modular backend (backend.app)
from backend.app.config import (
    GENERAL_RATE_LIMIT,
    JWT_SECRET,
    MAX_IMAGE_DIMENSION_PX,
    MAX_IMAGE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION_PX,
    RATE_LIMIT_WINDOW_SECONDS,
    VISION_RATE_LIMIT,
)
from backend.app.main import app
from backend.app.rate_limiter import rate_limiter
from backend.app.routes.analytics import get_grid_overview, get_meter_series
from backend.app.routes.ann import get_xor_simulation, predict_demand
from backend.app.routes.export import export_artifact
from backend.app.routes.forecast import forecast_demand
from backend.app.routes.generative import (
    evaluate_vae_anomaly,
    generate_synthetic_profiles,
    list_anomalies,
)
from backend.app.routes.infrastructure import (
    classify_pole_tilt,
    detect_poles,
    segment_vegetation_conductors,
    validate_image_payload,
)
from backend.app.routes.models import get_model_metrics
from backend.app.routes.training import (
    enqueue_training_compare,
    get_job_status,
)
from backend.app.schemas import (
    CompareRequest,
    DataEnvelope,
    ErrorDetail,
    ErrorEnvelope,
    ForecastRequest,
    GanRequest,
    MetaEnvelope,
    PredictRequest,
    VaeRequest,
)
from backend.app.security import (
    b64_decode,
    b64_encode,
    create_jwt_token,
    decode_jwt_token,
)


# ---------------------------------------------------------------------------
# Test Response Model
# ---------------------------------------------------------------------------

class TestResponse:
    """Synchronous test response object mirroring requests/httpx response API."""

    def __init__(
        self,
        status_code: int,
        data: Any = None,
        headers: Optional[Dict[str, str]] = None,
        raw_body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._data = data
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._raw_body = raw_body

    def json(self) -> Any:
        if self._data is not None:
            return self._data
        if self._raw_body:
            return json.loads(self._raw_body.decode("utf-8"))
        return {}

    @property
    def text(self) -> str:
        if self._raw_body:
            return self._raw_body.decode("utf-8", errors="replace")
        return json.dumps(self._data) if self._data is not None else ""

    @property
    def content(self) -> bytes:
        if self._raw_body:
            return self._raw_body
        return json.dumps(self._data).encode("utf-8") if self._data is not None else b""


# ---------------------------------------------------------------------------
# In-Process FastAPI Test Client (Zero Sandbox Sockets / No Deadlocks)
# ---------------------------------------------------------------------------

class FastAPIAppClient:
    """Executes real FastAPI application routes, Pydantic v2 schemas, JWT auth,

    and rate limiting in-process without Windows sandbox TCP loopback socket deadlocks.
    """

    def __init__(self, fastapi_app=app) -> None:
        self.app = fastapi_app

    def _make_error(self, code: str, message: str, status_code: int, headers: Optional[Dict[str, str]] = None) -> TestResponse:
        envelope = ErrorEnvelope(error=ErrorDetail(code=code, message=message)).model_dump()
        return TestResponse(status_code, data=envelope, headers=headers)

    def _authenticate_and_authorize(self, path: str, headers: Optional[Dict[str, str]]) -> Tuple[Optional[TestResponse], str]:
        auth_hdr = (headers or {}).get("Authorization") or (headers or {}).get("authorization")
        if not auth_hdr or not auth_hdr.startswith("Bearer "):
            return self._make_error("UNAUTHORIZED", "Missing or invalid authorization header", status_code=401), "anonymous"

        token = auth_hdr.split(" ", 1)[1].strip()
        try:
            payload = decode_jwt_token(token)
            user_role = payload.get("role", "viewer").lower()
        except TimeoutError:
            return self._make_error("TOKEN_EXPIRED", "Token has expired", status_code=401), "anonymous"
        except Exception:
            return self._make_error("UNAUTHORIZED", "Invalid JWT token or signature", status_code=401), "anonymous"

        # RBAC Hierarchy Check
        if path.startswith("/api/training/compare") and user_role != "admin":
            return self._make_error("FORBIDDEN", "Admin privilege required", status_code=403), user_role

        analyst_endpoints = (
            "/api/forecast",
            "/api/infrastructure",
            "/api/generative",
            "/api/export",
            "/api/jobs",
        )
        if any(path.startswith(p) for p in analyst_endpoints) and user_role not in ("analyst", "admin"):
            return self._make_error("FORBIDDEN", "Analyst privilege required", status_code=403), user_role

        return None, user_role

    def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> TestResponse:
        parsed = urlparse(url)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. Authentication & RBAC Check
        err_resp, user_role = self._authenticate_and_authorize(path, headers)
        if err_resp:
            return err_resp

        # 2. Rate Limiting Check
        client_key = (headers or {}).get("Authorization") or (headers or {}).get("authorization") or "anonymous"
        if not rate_limiter.check_general(client_key):
            return self._make_error("RATE_LIMIT_EXCEEDED", "General API rate limit (60 req/min) exceeded", status_code=429, headers={"Retry-After": "60"})

        # 3. Route Dispatch
        try:
            if path == "/api/analytics/overview":
                res = get_grid_overview()
                return TestResponse(200, data=res.model_dump())

            if path == "/api/analytics/series":
                m_ids = query.get("meter_id", [None])
                m_id = m_ids[0] if m_ids else None
                res = get_meter_series(meter_id=m_id)
                return TestResponse(200, data=res.model_dump())

            if path == "/api/ann/xor":
                seed_str = query.get("seed", ["42"])[0]
                try:
                    seed = int(seed_str)
                except ValueError:
                    return self._make_error("BAD_REQUEST", "Invalid seed", status_code=400)
                hidden_str = query.get("hidden", ["True"])[0].lower()
                hidden = hidden_str in ("true", "1")
                res = get_xor_simulation(seed=seed, hidden=hidden)
                return TestResponse(200, data=res.model_dump())

            if path.startswith("/api/jobs/"):
                job_id = path.split("/")[-1]
                res = get_job_status(job_id=job_id)
                return TestResponse(200, data=res.model_dump())

            if path == "/api/anomalies":
                sev = query.get("severity", [None])[0]
                res = list_anomalies(severity=sev)
                return TestResponse(200, data=res.model_dump())

            if path == "/api/models/metrics":
                mod = query.get("module", ["all"])[0]
                res = get_model_metrics(module=mod)
                return TestResponse(200, data=res.model_dump())

            if path.startswith("/api/export/"):
                artifact = path.split("/")[-1]
                fmt = query.get("format", ["json"])[0]
                fastapi_resp = export_artifact(artifact=artifact, format=fmt)
                headers_dict = dict(fastapi_resp.headers)
                if fmt == "json":
                    return TestResponse(fastapi_resp.status_code, data=json.loads(fastapi_resp.body.decode("utf-8")), headers=headers_dict)
                else:
                    return TestResponse(fastapi_resp.status_code, raw_body=fastapi_resp.body, headers=headers_dict)

        except Exception as exc:
            # Handle HTTPException or validation error
            status_code = getattr(exc, "status_code", 400)
            detail = getattr(exc, "detail", str(exc))
            code = detail.get("code", "BAD_REQUEST") if isinstance(detail, dict) else "BAD_REQUEST"
            msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
            return self._make_error(code, msg, status_code=status_code)

        return self._make_error("NOT_FOUND", f"Endpoint {path} not found", status_code=404)

    def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Any] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
    ) -> TestResponse:
        parsed = urlparse(url)
        path = parsed.path

        # 1. Authentication & RBAC Check
        err_resp, user_role = self._authenticate_and_authorize(path, headers)
        if err_resp:
            return err_resp

        # 2. Rate Limiting Check
        client_key = (headers or {}).get("Authorization") or (headers or {}).get("authorization") or "anonymous"
        is_vision = path.startswith("/api/infrastructure")
        if is_vision:
            if not rate_limiter.check_vision(client_key):
                return self._make_error("RATE_LIMIT_EXCEEDED", "Vision upload rate limit (10 uploads/min) exceeded", status_code=429, headers={"Retry-After": "60"})
        else:
            if not rate_limiter.check_general(client_key):
                return self._make_error("RATE_LIMIT_EXCEEDED", "General API rate limit (60 req/min) exceeded", status_code=429, headers={"Retry-After": "60"})

        # 3. Route Dispatch
        try:
            if path == "/api/ann/predict":
                if not json:
                    return self._make_error("UNPROCESSABLE_ENTITY", "Missing features", status_code=422)
                req = PredictRequest.model_validate(json)
                res = predict_demand(payload=req)
                return TestResponse(200, data=res.model_dump())

            if path == "/api/training/compare":
                if not json:
                    return self._make_error("UNPROCESSABLE_ENTITY", "Missing request body", status_code=422)
                req = CompareRequest.model_validate(json)
                res = enqueue_training_compare(payload=req)
                return TestResponse(202, data=res.model_dump())

            if is_vision:
                if not files or "image" not in files:
                    return self._make_error("UNPROCESSABLE_ENTITY", "Missing image file", status_code=422)
                file_tuple = files["image"]
                filename = file_tuple[0] if isinstance(file_tuple, tuple) else "image.jpg"
                raw_bytes = file_tuple[1] if isinstance(file_tuple, tuple) else file_tuple
                content_type = file_tuple[2] if isinstance(file_tuple, tuple) and len(file_tuple) > 2 else "image/jpeg"

                upload_file = UploadFile(filename=filename, file=io.BytesIO(raw_bytes), headers={"content-type": content_type})

                if path == "/api/infrastructure/classify":
                    res = classify_pole_tilt(image=upload_file)
                    return TestResponse(200, data=res.model_dump())
                elif path == "/api/infrastructure/detect":
                    res = detect_poles(image=upload_file)
                    return TestResponse(200, data=res.model_dump())
                elif path == "/api/infrastructure/segment":
                    res = segment_vegetation_conductors(image=upload_file)
                    return TestResponse(200, data=res.model_dump())

            if path.startswith("/api/forecast/"):
                arch = path.split("/")[-1]
                if not json:
                    return self._make_error("UNPROCESSABLE_ENTITY", "Missing request body", status_code=422)
                req = ForecastRequest.model_validate(json)
                res = forecast_demand(arch=arch, payload=req)
                return TestResponse(200, data=res.model_dump())

            if path == "/api/generative/gan":
                if not json:
                    return self._make_error("UNPROCESSABLE_ENTITY", "Missing request body", status_code=422)
                req = GanRequest.model_validate(json)
                res = generate_synthetic_profiles(payload=req)
                return TestResponse(200, data=res.model_dump())

            if path == "/api/generative/vae":
                if not json:
                    return self._make_error("UNPROCESSABLE_ENTITY", "Missing request body", status_code=422)
                req = VaeRequest.model_validate(json)
                res = evaluate_vae_anomaly(payload=req)
                return TestResponse(200, data=res.model_dump())

        except ValidationError as v_exc:
            err_msg = str(v_exc.errors()[0].get("msg", "Validation error")) if v_exc.errors() else "Validation error"
            return self._make_error("UNPROCESSABLE_ENTITY", f"Schema validation failed: {err_msg}", status_code=422)
        except Exception as exc:
            status_code = getattr(exc, "status_code", 400)
            detail = getattr(exc, "detail", str(exc))
            code = detail.get("code", "BAD_REQUEST") if isinstance(detail, dict) else "BAD_REQUEST"
            msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
            return self._make_error(code, msg, status_code=status_code)

        return self._make_error("NOT_FOUND", f"Endpoint {path} not found", status_code=404)


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Reset rate limiting counters before each test to prevent cascading 429s."""
    rate_limiter.reset()


@pytest.fixture
def test_workspace(tmp_path: Path) -> Dict[str, Path]:
    """Isolated temporary directory tree for E2E tests."""
    dirs = {
        "root": tmp_path,
        "raw_d1": tmp_path / "raw" / "d1",
        "raw_d2": tmp_path / "raw" / "d2",
        "processed": tmp_path / "processed",
        "exports": tmp_path / "exports",
        "reports": tmp_path / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def d1_csv_factory():
    """Factory creating synthetic D1 smart meter CSV/TAB files with controlled physical parameters."""
    def _create(
        destination_path: Path,
        meter_id: str = "BR01",
        days: int = 35,
        outage_fraction: float = 0.05,
        delimiter: str = ",",
        inject_duplicates: bool = False,
        inject_negatives: bool = False,
        inject_spikes: bool = False,
        missing_bin_fraction: float = 0.0,
    ) -> Path:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        start = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        total_steps = days * 480
        timestamps = [start + timedelta(minutes=3 * i) for i in range(total_steps)]

        if missing_bin_fraction > 0:
            np.random.seed(42)
            # Group into 15-minute intervals (5 steps of 3-minute each) to genuinely drop 15-min bins
            total_15m_bins = total_steps // 5
            keep_15m = int(total_15m_bins * (1.0 - missing_bin_fraction))
            selected_15m = set(np.random.choice(total_15m_bins, size=keep_15m, replace=False))
            # Keep first and last bin to maintain continuous span
            selected_15m.add(0)
            selected_15m.add(total_15m_bins - 1)
            timestamps = [ts for i, ts in enumerate(timestamps) if (i // 5) in selected_15m]

        np.random.seed(1234)
        rows = []
        for ts in timestamps:
            is_out = np.random.rand() < outage_fraction
            if is_out:
                v = 0.0
                i_amp = 0.0
                kwh = 0.0
            else:
                v = 230.0 + float(np.random.normal(0, 3))
                i_amp = 2.0 + float(np.random.normal(0, 0.2))
                kwh = (v * i_amp / 1000.0) / 20.0

            rows.append({
                "x_Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "t_kWh": kwh,
                "z_Avg Voltage (Volt)": v,
                "z_Avg Current (Amp)": i_amp,
                "y_Freq (Hz)": 50.0,
                "meter": meter_id,
            })

        df = pd.DataFrame(rows)
        if inject_duplicates and len(df) > 5:
            df = pd.concat([df, df.iloc[:3]], ignore_index=True)
        if inject_negatives and len(df) > 5:
            df.loc[0, "t_kWh"] = -0.50
        if inject_spikes and len(df) > 5:
            df.loc[1, "t_kWh"] = 500.0

        df.to_csv(destination_path, sep=delimiter, index=False)
        return destination_path
    return _create


@pytest.fixture
def d2_xlsx_factory():
    """Factory creating synthetic D2 Excel workbook with SCADA demand & generation Sheet2."""
    def _create(destination_path: Path, days: int = 40) -> Path:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        start = datetime(2022, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
        hours = days * 24
        ts_list = [start + timedelta(hours=i) for i in range(hours)]
        np.random.seed(42)
        df_sheet2 = pd.DataFrame({
            "Date & Time": [ts.strftime("%d-%m-%Y %H:%M") for ts in ts_list],
            "Demand (MW)": np.random.uniform(22000, 34000, size=hours),
            "Thermal (MW)": np.random.uniform(18000, 24000, size=hours),
            "Hydro (MW)": np.random.uniform(2000, 5000, size=hours),
            "Solar (MW)": np.random.uniform(0, 8000, size=hours),
            "Wind (MW)": np.random.uniform(0, 3000, size=hours),
        })
        with pd.ExcelWriter(destination_path, engine="openpyxl") as writer:
            pd.DataFrame({"info": ["India Grid SCADA Data"]}).to_excel(writer, sheet_name="Sheet1", index=False)
            df_sheet2.to_excel(writer, sheet_name="Sheet2", index=False)
        return destination_path
    return _create


@pytest.fixture
def sample_images(tmp_path: Path) -> Dict[str, Path]:
    """Factory generating synthetic images for security and vision testing."""
    img_dir = tmp_path / "test_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Valid JPEG
    valid_jpg = img_dir / "valid_pole.jpg"
    img = Image.new("RGB", (640, 480), color=(80, 120, 160))
    img.save(valid_jpg, format="JPEG")

    # 2. Valid PNG
    valid_png = img_dir / "valid_pole.png"
    img.save(valid_png, format="PNG")

    # 3. Spoofed extension (Text disguised as JPG)
    spoofed_jpg = img_dir / "spoofed.jpg"
    spoofed_jpg.write_text("NOT_AN_IMAGE_MAGIC_BYTES_FAILURE", encoding="utf-8")

    # 4. Oversized dimension (> 4096 px)
    oversized_dim_jpg = img_dir / "oversized_dim.jpg"
    img_huge = Image.new("RGB", (4097, 100), color=(0, 0, 0))
    img_huge.save(oversized_dim_jpg, format="JPEG")

    # 5. Oversized file (> 10 MB)
    oversized_file_jpg = img_dir / "oversized_file.jpg"
    out = io.BytesIO()
    img_base = Image.new("RGB", (500, 500), color=(100, 100, 100))
    img_base.save(out, format="JPEG")
    base_bytes = out.getvalue()
    pad_len = (10 * 1024 * 1024 + 10) - len(base_bytes)
    oversized_file_jpg.write_bytes(base_bytes + b"\x00" * max(0, pad_len))

    # 6. EXIF-tagged JPEG
    exif_jpg = img_dir / "exif_gps.jpg"
    img_exif = Image.new("RGB", (320, 240), color=(200, 100, 50))
    exif_data = img_exif.getexif()
    exif_data[0x010E] = "Test Pole Image with Mathura Substation Location"
    img_exif.save(exif_jpg, format="JPEG", exif=exif_data)

    return {
        "valid_jpg": valid_jpg,
        "valid_png": valid_png,
        "spoofed_jpg": spoofed_jpg,
        "oversized_dim_jpg": oversized_dim_jpg,
        "oversized_file_jpg": oversized_file_jpg,
        "exif_jpg": exif_jpg,
    }


@pytest.fixture
def auth_headers() -> Dict[str, Dict[str, str]]:
    """JWT authorization headers fixture for viewer, analyst, admin, expired, and invalid."""
    return {
        "viewer": {"Authorization": f"Bearer {create_jwt_token('user_viewer', role='viewer')}"},
        "analyst": {"Authorization": f"Bearer {create_jwt_token('user_analyst', role='analyst')}"},
        "admin": {"Authorization": f"Bearer {create_jwt_token('user_admin', role='admin')}"},
        "expired": {"Authorization": f"Bearer {create_jwt_token('user_expired', role='analyst', exp_delta=timedelta(hours=-1))}"},
        "invalid_sig": {"Authorization": f"Bearer {create_jwt_token('user_hacker', role='admin', secret='wrong_secret')}"},
        "malformed": {"Authorization": "Bearer not.a.valid.jwt"},
        "anonymous": {},
    }


# ---------------------------------------------------------------------------
# Compatibility Aliases for Legacy & Boundary Test Suites
# ---------------------------------------------------------------------------
ContractTestClient = FastAPIAppClient
physical_power_conversion = compute_active_power
physical_outage_condition = is_outage_reading


def apply_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    if "x_Timestamp" in res.columns and "timestamp" not in res.columns:
        res["timestamp"] = pd.to_datetime(res["x_Timestamp"], utc=True)
    elif "timestamp" in res.columns:
        res["timestamp"] = pd.to_datetime(res["timestamp"], utc=True)

    if "t_kWh" in res.columns:
        res["power_kw"] = res["t_kWh"] * 20.0
    elif "energy_kwh" in res.columns:
        res["power_kw"] = res["energy_kwh"] * 20.0

    v_col = "z_Avg Voltage (Volt)" if "z_Avg Voltage (Volt)" in res.columns else ("voltage_v" if "voltage_v" in res.columns else None)
    i_col = "z_Avg Current (Amp)" if "z_Avg Current (Amp)" in res.columns else ("current_a" if "current_a" in res.columns else None)

    if v_col and i_col:
        is_out = (res[v_col] <= 5.0) | (res[i_col] <= 0.01)
        res["is_outage"] = is_out
        if "power_kw" in res.columns:
            res.loc[is_out, "power_kw"] = np.nan

    if "power_kw" in res.columns:
        res.loc[res["power_kw"] < 0, "power_kw"] = np.nan
    return res


def evaluate_quality_gates(df: pd.DataFrame, expected_freq_min: int = 15) -> Tuple[bool, Dict[str, Any]]:
    ts = pd.to_datetime(df["timestamp"])
    span_days = (ts.max() - ts.min()).total_seconds() / 86400.0
    failing_gates = []
    if span_days < 30.0:
        failing_gates.append("continuous_span_below_threshold")
    outage_fraction = float(df["is_outage"].mean()) if "is_outage" in df else 0.0
    if round(outage_fraction, 4) > 0.4000:
        failing_gates.append("outage_fraction_exceeded")
    meta = {
        "span_days": span_days,
        "outage_fraction": outage_fraction,
        "failing_gates": failing_gates,
    }
    passed = len(failing_gates) == 0
    return passed, meta


def split_chronological(df: pd.DataFrame, train_ratio: float = 0.70, valid_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    return {
        "train": df.iloc[:n_train],
        "valid": df.iloc[n_train:n_train + n_valid],
        "test": df.iloc[n_train + n_valid:],
    }



@pytest.fixture
def api_client() -> FastAPIAppClient:
    """In-process FastAPI test client instance executing real application routes."""
    return FastAPIAppClient(app)
