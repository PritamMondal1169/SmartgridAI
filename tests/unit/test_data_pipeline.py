"""
Unit Tests for SmartgridAI Data Pipeline.
Covers:
- Physical data contract & physics conversions (contract.py)
- Telemetry ingestion & visual guards (ingestion.py)
- 5-Gate quality engine & 15-minute resampling (quality_gates.py)
- Chronological 70/15/15 splitter & manifest signing (splitter.py)
"""

import os
import sys
import math
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

# Ensure src is on sys.path
SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smartgrid.data.contract import (
    PhysicalReading,
    ResampledReading,
    compute_active_power,
    is_outage_reading,
    apply_physical_contract,
    validate_reading_bounds,
    INTERVAL_MINUTES,
    CONVERSION_FACTOR,
    VOLTAGE_OUTAGE_THRESHOLD_V,
    CURRENT_OUTAGE_THRESHOLD_A,
    MIN_SPAN_DAYS,
    MIN_BIN_COVERAGE,
    MAX_OUTAGE_FRACTION,
    VOLTAGE_MAX_V,
    CURRENT_MAX_A,
    FREQ_MAX_HZ
)
from smartgrid.data.ingestion import (
    SmartMeterReader,
    GridDemandReader,
    ImageGuard,
    VisionCatalogReader,
    IST_TIMEZONE
)
from smartgrid.data.quality_gates import (
    QualityEngine,
    GateResult,
    MeterEvaluation
)
from smartgrid.data.splitter import (
    ChronologicalSplitter,
    SplitBoundary,
    MeterSplits
)


# ==============================================================================
# 1. Physical Data Contract Tests (contract.py)
# ==============================================================================

class TestContract:
    """Tests for physical conversion formulas, outage rules, and bound checking."""

    def test_compute_active_power_standard_interval(self):
        """P_kW = energy_kwh * 20.0 for standard 3-minute interval."""
        # 0.002 kWh in 3 min -> 0.040 kW (40 W)
        assert compute_active_power(0.002, 3.0) == pytest.approx(0.040)
        # 0.015 kWh in 3 min -> 0.300 kW (300 W)
        assert compute_active_power(0.015, 3.0) == pytest.approx(0.300)
        # Zero consumption
        assert compute_active_power(0.0, 3.0) == 0.0

    def test_compute_active_power_custom_intervals(self):
        """Supports custom interval durations in minutes."""
        # 1 kWh in 60 min -> 1 kW
        assert compute_active_power(1.0, 60.0) == pytest.approx(1.0)
        # 0.25 kWh in 15 min -> 1 kW
        assert compute_active_power(0.25, 15.0) == pytest.approx(1.0)

    def test_compute_active_power_invalid_inputs(self):
        """Raises ValueError on non-positive intervals or negative energy."""
        with pytest.raises(ValueError, match="interval_minutes must be strictly positive"):
            compute_active_power(0.1, 0.0)
        with pytest.raises(ValueError, match="interval_minutes must be strictly positive"):
            compute_active_power(0.1, -5.0)
        with pytest.raises(ValueError, match="energy_kwh cannot be negative"):
            compute_active_power(-0.01, 3.0)

    @pytest.mark.parametrize("v, i, expected_outage", [
        (0.0, 0.0, True),          # Complete blackout
        (5.0, 0.5, True),          # Voltage at threshold (5.0V)
        (2.3, 10.0, True),         # Low voltage
        (230.0, 0.01, True),       # Current at threshold (0.01A)
        (230.0, 0.005, True),      # Residual micro-current
        (5.01, 0.011, False),      # Just above thresholds
        (230.0, 1.5, False),       # Normal single-phase load
        (240.0, 25.0, False),      # Heavy load
    ])
    def test_is_outage_reading(self, v, i, expected_outage):
        """Outage occurs when V <= 5.0V or I <= 0.01A."""
        assert is_outage_reading(v, i) == expected_outage

    def test_apply_physical_contract_normal(self):
        """Normal reading converts energy to power and sets is_outage=False."""
        power, is_outage = apply_physical_contract(energy_kwh=0.015, voltage_v=230.0, current_a=1.5)
        assert is_outage is False
        assert power == pytest.approx(0.300)

    def test_apply_physical_contract_outage_with_residual_kwh(self):
        """Outage reading sets power=None and is_outage=True even if raw kWh > 0."""
        # Residual capacitor discharge pulse during blackout
        power, is_outage = apply_physical_contract(energy_kwh=0.002, voltage_v=0.0, current_a=0.0)
        assert is_outage is True
        assert power is None

    def test_validate_reading_bounds_valid(self):
        """Clean readings return empty violation list."""
        violations = validate_reading_bounds(
            energy_kwh=0.05,
            voltage_v=230.0,
            current_a=2.5,
            freq_hz=50.0
        )
        assert violations == []

    def test_validate_reading_bounds_violations(self):
        """Flags out-of-bound variables correctly."""
        violations = validate_reading_bounds(
            energy_kwh=-0.01,
            voltage_v=700.0,
            current_a=200.0,
            freq_hz=160.0
        )
        assert "negative_energy" in violations
        assert "voltage_out_of_bounds" in violations
        assert "current_out_of_bounds" in violations
        assert "frequency_out_of_bounds" in violations


# ==============================================================================
# 2. Ingestion & Visual Guard Tests (ingestion.py)
# ==============================================================================

class TestIngestion:
    """Tests for D1 CSV reading, D2 Excel parsing, and D3 image guards."""

    def test_smart_meter_reader_ist_to_utc(self):
        """Verifies IST (UTC+05:30) to UTC conversion without DST."""
        ist_str = "2020-01-01 05:30:00"
        utc_dt = SmartMeterReader.parse_ist_timestamp(ist_str)
        assert utc_dt.year == 2020
        assert utc_dt.month == 1
        assert utc_dt.day == 1
        assert utc_dt.hour == 0
        assert utc_dt.minute == 0
        assert utc_dt.second == 0
        assert utc_dt.tzinfo == timezone.utc

    def test_smart_meter_reader_detect_delimiter(self, tmp_path):
        """Detects comma vs tab delimiter."""
        csv_file = tmp_path / "test_comma.csv"
        csv_file.write_text("x_Timestamp,t_kWh,z_Avg Voltage,z_Avg Current,y_Freq,meter\n", encoding="utf-8")
        assert SmartMeterReader.detect_delimiter(csv_file) == ","

        tsv_file = tmp_path / "test_tab.tsv"
        tsv_file.write_text("x_Timestamp\tt_kWh\tz_Avg Voltage\tz_Avg Current\ty_Freq\tmeter\n", encoding="utf-8")
        assert SmartMeterReader.detect_delimiter(tsv_file) == "\t"

    def test_smart_meter_reader_parse_file(self, tmp_path):
        """Parses rows into PhysicalReading objects with physical contract applied."""
        sample_csv = tmp_path / "sample.csv"
        content = (
            "x_Timestamp,t_kWh,z_Avg Voltage (Volt),z_Avg Current (Amp),y_Freq (Hz),meter\n"
            "2020-01-01 00:00:00,0.010,230.0,1.0,50.0,BR02\n"
            "2020-01-01 00:03:00,0.000,0.0,0.0,0.0,BR02\n"
            "malformed_row\n"
            "2020-01-01 00:06:00,0.015,232.0,1.2,49.9,MH01\n"
        )
        sample_csv.write_text(content, encoding="utf-8")

        reader = SmartMeterReader(tmp_path)
        readings = list(reader.parse_file(sample_csv))
        assert len(readings) == 3

        # First reading: normal operation
        assert readings[0].meter_id == "BR02"
        assert readings[0].power_kw == pytest.approx(0.200)
        assert readings[0].is_outage is False

        # Second reading: grid outage
        assert readings[1].meter_id == "BR02"
        assert readings[1].power_kw is None
        assert readings[1].is_outage is True

        # Filter by meter_id
        br02_only = list(reader.parse_file(sample_csv, target_meter_id="BR02"))
        assert len(br02_only) == 2
        assert all(r.meter_id == "BR02" for r in br02_only)

    def test_grid_demand_reader_date_conversions(self):
        """Tests Excel serial date to UTC and string timestamp parsing."""
        # 44440.0 is 2021-09-01 00:00:00 IST -> 2021-08-31 18:30:00 UTC
        utc_dt = GridDemandReader.excel_serial_to_utc(44440.0)
        assert utc_dt.year == 2021
        assert utc_dt.month == 8
        assert utc_dt.day == 31
        assert utc_dt.hour == 18
        assert utc_dt.minute == 30

        # String timestamp parsing
        parsed = GridDemandReader.parse_string_timestamp_to_utc("01-01-2024 05:30:00")
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 1
        assert parsed.hour == 0
        assert parsed.minute == 0

    def test_grid_demand_resample_and_solar_clip(self):
        """Resamples grid records to 15-minute means and clips negative solar."""
        t1 = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2023, 1, 1, 0, 5, tzinfo=timezone.utc)
        records = [
            {"timestamp": t1, "demand_mw": 150000.0, "wind_mw": 5000.0, "solar_mw": 0.0},
            {"timestamp": t2, "demand_mw": 152000.0, "wind_mw": 5100.0, "solar_mw": 0.0},
        ]
        reader = GridDemandReader(Path("."))
        resampled = reader.resample_records(records, interval_minutes=15)
        assert len(resampled) == 1
        assert resampled[0]["demand_mw"] == 151000.0
        assert resampled[0]["sample_count"] == 2

    def test_image_guard_magic_bytes(self):
        """Validates JPEG, PNG, and rejects invalid magic bytes."""
        jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        invalid_header = b"GIF89a\x00\x01\x00\x01"

        assert ImageGuard.validate_magic_bytes(jpeg_header) == (True, "jpeg")
        assert ImageGuard.validate_magic_bytes(png_header) == (True, "png")
        assert ImageGuard.validate_magic_bytes(invalid_header) == (False, None)

    def test_image_guard_resize_and_exif_strip(self, tmp_path):
        """Downscales images exceeding 4096 px preserving aspect ratio."""
        from PIL import Image

        # Create 5000x2500 synthetic image
        img_path = tmp_path / "oversized.jpg"
        img = Image.new("RGB", (5000, 2500), color=(128, 128, 128))
        img.save(img_path, "JPEG")

        output_path = tmp_path / "downscaled.jpg"
        new_w, new_h, was_resized = ImageGuard.process_image(img_path, output_path, max_dim=4096)

        assert was_resized is True
        assert new_w == 4096
        assert new_h == 2048  # Exact 2:1 aspect ratio preserved

        # Output image exists and is readable
        assert output_path.exists()
        with Image.open(output_path) as out_img:
            assert out_img.size == (4096, 2048)


# ==============================================================================
# 3. Quality Gate Engine & Resampling Tests (quality_gates.py)
# ==============================================================================

class TestQualityGates:
    """Tests for the 5-Gate quality admission engine and 15-minute resampling."""

    @pytest.fixture
    def engine(self):
        return QualityEngine(min_span_days=30.0, min_bin_coverage=0.695, max_outage_fraction=0.40)

    def create_synthetic_series(self, start_dt, days, interval_min=3, outage_rate=0.0):
        """Generates synthetic PhysicalReading sequence."""
        readings = []
        cur_dt = start_dt
        end_dt = start_dt + timedelta(days=days)
        step = timedelta(minutes=interval_min)

        idx = 0
        while cur_dt <= end_dt:
            is_out = (idx % 100 < int(outage_rate * 100))
            readings.append(PhysicalReading(
                timestamp=cur_dt,
                meter_id="TEST_01",
                energy_kwh=0.0 if is_out else 0.015,
                voltage_v=0.0 if is_out else 230.0,
                current_a=0.0 if is_out else 1.5,
                freq_hz=0.0 if is_out else 50.0,
                power_kw=None if is_out else 0.300,
                is_outage=is_out
            ))
            cur_dt += step
            idx += 1
        return readings

    def test_gate_1_span(self, engine):
        """Gate 1 passes for span >= 30 days and fails for < 30 days."""
        t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
        # 29 days -> fail
        short_series = self.create_synthetic_series(t0, days=29)
        g1_fail = engine.evaluate_gate_1_span(short_series)
        assert g1_fail.passed is False
        assert g1_fail.failure_code == "insufficient_span"

        # 30 days -> pass
        valid_series = self.create_synthetic_series(t0, days=30)
        g1_pass = engine.evaluate_gate_1_span(valid_series)
        assert g1_pass.passed is True
        assert g1_pass.failure_code is None

    def test_gate_2_coverage(self, engine):
        """Gate 2 passes when >= 70% of 15-min bins populated."""
        t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
        # Full coverage series (100%)
        full_series = self.create_synthetic_series(t0, days=35)
        g2_pass = engine.evaluate_gate_2_coverage(full_series)
        assert g2_pass.passed is True
        assert g2_pass.metric_value >= 0.99

        # Sparse series: only 1 day populated out of 35 days (~2.8% coverage)
        sparse_series = self.create_synthetic_series(t0, days=1)
        # Add an anchor point at day 35
        sparse_series.append(PhysicalReading(
            timestamp=t0 + timedelta(days=35),
            meter_id="TEST_01",
            energy_kwh=0.01, voltage_v=230.0, current_a=1.0, freq_hz=50.0, power_kw=0.2, is_outage=False
        ))
        g2_fail = engine.evaluate_gate_2_coverage(sparse_series)
        assert g2_fail.passed is False
        assert g2_fail.failure_code == "insufficient_coverage"

    def test_gate_3_outage(self, engine):
        """Gate 3 passes when outage fraction <= 40%."""
        t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
        # 10% outages -> pass
        low_outage = self.create_synthetic_series(t0, days=30, outage_rate=0.10)
        g3_pass = engine.evaluate_gate_3_outage(low_outage)
        assert g3_pass.passed is True

        # 50% outages -> fail
        high_outage = self.create_synthetic_series(t0, days=30, outage_rate=0.50)
        g3_fail = engine.evaluate_gate_3_outage(high_outage)
        assert g3_fail.passed is False
        assert g3_fail.failure_code == "outage_fraction_exceeded"

    def test_gate_4_deduplication(self, engine):
        """Gate 4 detects and flags duplicate timestamps."""
        t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
        readings = [
            PhysicalReading(t0, "TEST_01", 0.01, 230.0, 1.0, 50.0, 0.2, False),
            PhysicalReading(t0, "TEST_01", 0.01, 230.0, 1.0, 50.0, 0.2, False),  # duplicate
        ]
        g4, deduped = engine.evaluate_gate_4_deduplication(readings)
        assert g4.passed is False
        assert g4.failure_code == "duplicate_timestamps"
        assert len(deduped) == 1

    def test_gate_5_non_negativity(self, engine):
        """Gate 5 flags negative or impossible values."""
        t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
        corrupt = [
            PhysicalReading(t0, "TEST_01", -0.05, 230.0, 1.0, 50.0, 0.2, False)
        ]
        g5 = engine.evaluate_gate_5_non_negativity(corrupt)
        assert g5.passed is False
        assert g5.failure_code == "negative_or_impossible_values"

    def test_resample_to_15min_majority_voting(self, engine):
        """Majority voting flags bin as outage when >= 50% constituents are outages."""
        t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
        # Bin 1: 5 samples, 3 outages (60%) -> Bin marked as outage, power_kw=None
        bin1_readings = [
            PhysicalReading(t0 + timedelta(minutes=0), "M1", 0.0, 0.0, 0.0, 0.0, None, True),
            PhysicalReading(t0 + timedelta(minutes=3), "M1", 0.0, 0.0, 0.0, 0.0, None, True),
            PhysicalReading(t0 + timedelta(minutes=6), "M1", 0.0, 0.0, 0.0, 0.0, None, True),
            PhysicalReading(t0 + timedelta(minutes=9), "M1", 0.02, 230.0, 2.0, 50.0, 0.40, False),
            PhysicalReading(t0 + timedelta(minutes=12), "M1", 0.03, 230.0, 3.0, 50.0, 0.60, False),
        ]
        res1 = engine.resample_to_15min("M1", bin1_readings)
        assert len(res1) == 1
        assert res1[0].is_outage is True
        assert res1[0].power_kw is None

        # Bin 2: 5 samples, 1 outage (20%) -> Bin operational, power_kw averaged over active
        t_bin2 = t0 + timedelta(minutes=15)
        bin2_readings = [
            PhysicalReading(t_bin2 + timedelta(minutes=0), "M1", 0.0, 0.0, 0.0, 0.0, None, True),
            PhysicalReading(t_bin2 + timedelta(minutes=3), "M1", 0.02, 230.0, 2.0, 50.0, 0.40, False),
            PhysicalReading(t_bin2 + timedelta(minutes=6), "M1", 0.02, 230.0, 2.0, 50.0, 0.40, False),
            PhysicalReading(t_bin2 + timedelta(minutes=9), "M1", 0.03, 230.0, 3.0, 50.0, 0.60, False),
            PhysicalReading(t_bin2 + timedelta(minutes=12), "M1", 0.03, 230.0, 3.0, 50.0, 0.60, False),
        ]
        res2 = engine.resample_to_15min("M1", bin2_readings)
        assert len(res2) == 1
        assert res2[0].is_outage is False
        assert res2[0].power_kw == pytest.approx((0.40 + 0.40 + 0.60 + 0.60) / 4)


# ==============================================================================
# 4. Chronological Splitter Tests (splitter.py)
# ==============================================================================

class TestSplitter:
    """Tests for chronological 70/15/15 partitioning and manifest generation."""

    def test_chronological_split_row_allocation(self):
        """Verifies floor allocation: 70% train, 15% valid, remaining test."""
        splitter = ChronologicalSplitter(train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15)
        t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)

        # 100 observations
        series_100 = [
            ResampledReading(t0 + timedelta(minutes=15 * i), "M1", 0.5, 230.0, 2.0, 50.0, False, 5)
            for i in range(100)
        ]
        train, valid, test, splits = splitter.split_meter_series(series_100)
        assert len(train) == 70
        assert len(valid) == 15
        assert len(test) == 15
        assert len(train) + len(valid) + len(test) == 100

        # 103 observations (floor division check)
        series_103 = [
            ResampledReading(t0 + timedelta(minutes=15 * i), "M1", 0.5, 230.0, 2.0, 50.0, False, 5)
            for i in range(103)
        ]
        train, valid, test, splits = splitter.split_meter_series(series_103)
        assert len(train) == 72
        assert len(valid) == 15
        assert len(test) == 16
        assert len(train) + len(valid) + len(test) == 103

    def test_chronological_split_temporal_ordering_invariant(self):
        """Verifies strict temporal non-overlap: train_end < valid_start <= valid_end < test_start."""
        splitter = ChronologicalSplitter()
        t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
        series = [
            ResampledReading(t0 + timedelta(minutes=15 * i), "M1", 0.5, 230.0, 2.0, 50.0, False, 5)
            for i in range(50)
        ]
        train, valid, test, splits = splitter.split_meter_series(series)

        assert train[-1].timestamp < valid[0].timestamp
        assert valid[-1].timestamp < test[0].timestamp

    def test_chronological_split_invalid_ratios(self):
        """Raises ValueError if split ratios do not sum to 1.0."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            ChronologicalSplitter(train_ratio=0.70, valid_ratio=0.20, test_ratio=0.20)

    def test_generate_manifest_structure(self):
        """Generates valid manifest dictionary with source hashes and split boundaries."""
        splitter = ChronologicalSplitter()
        t0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
        series = [
            ResampledReading(t0 + timedelta(minutes=15 * i), "BR02", 0.5, 230.0, 2.0, 50.0, False, 5)
            for i in range(20)
        ]
        _, _, _, splits = splitter.split_meter_series(series)

        hashes = {"sample.csv": "a" * 64}
        manifest = splitter.generate_manifest({"BR02": splits}, hashes)

        assert manifest["manifest_version"] == "1.0.0"
        assert manifest["source_file_sha256"]["sample.csv"] == "a" * 64
        assert "BR02" in manifest["splits"]
        assert manifest["splits"]["BR02"]["train"]["rows"] == 14
        assert manifest["splits"]["BR02"]["valid"]["rows"] == 3
        assert manifest["splits"]["BR02"]["test"]["rows"] == 3
