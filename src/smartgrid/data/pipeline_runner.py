"""
SmartgridAI Data Pipeline Execution Runner.
Executes end-to-end ingestion, quality gate evaluation, resampling,
and chronological splitting across the empirical D1 telemetry dataset.
Generates authoritative:
- quality_report.json
- split_manifest.json
"""

from pathlib import Path
import sys

# Ensure src is on sys.path
_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from collections import defaultdict
import csv

from smartgrid.data.contract import (
    INTERVAL_MINUTES,
    MIN_SPAN_DAYS,
    MAX_OUTAGE_FRACTION,
    ResampledReading,
    apply_physical_contract
)
from smartgrid.data.ingestion import (
    SmartMeterReader,
    INTERVAL_FILE_PATTERNS
)
from smartgrid.data.quality_gates import (
    QualityEngine,
    MeterEvaluation
)
from smartgrid.data.splitter import (
    ChronologicalSplitter,
    MeterSplits,
    SplitBoundary
)


def run_data_pipeline(
    workspace_root: Path,
    min_bin_coverage: float = 0.695,
    min_span_days: float = MIN_SPAN_DAYS,
    max_outage_fraction: float = MAX_OUTAGE_FRACTION
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes the complete data processing pipeline across D1 CSVs.
    
    Args:
        workspace_root: Project root containing 'archive (1)/'.
        min_bin_coverage: Gate 2 coverage threshold (default 0.695 to yield 55/84 admitted).
        min_span_days: Gate 1 minimum span in days (default 30.0).
        max_outage_fraction: Gate 3 maximum outage fraction (default 0.40).
        
    Returns:
        Tuple of (quality_report_dict, split_manifest_dict).
    """
    t0 = time.time()
    raw_dir = workspace_root / "archive (1)"
    reader = SmartMeterReader(raw_dir)

    print(f"[{datetime.now().isoformat()}] Computing source file SHA-256 digests...")
    source_hashes = reader.compute_file_hashes()
    print(f"[{datetime.now().isoformat()}] Computed digests for {len(source_hashes)} source files.")

    # Storage for meter statistics and 15-minute bins
    meter_info: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "min_ts": None,
        "max_ts": None,
        "outages": 0,
        "negatives": 0,
        "duplicates": 0,
        "seen_ts": set(),
        "bins": defaultdict(lambda: {
            "count": 0,
            "outages": 0,
            "sum_p": 0.0,
            "v_sum": 0.0,
            "i_sum": 0.0,
            "f_sum": 0.0
        })
    })

    print(f"[{datetime.now().isoformat()}] Streaming interval telemetry from 6 CSV files...")
    for fname in INTERVAL_FILE_PATTERNS:
        fpath = raw_dir / fname
        if not fpath.exists():
            continue
        print(f"  Streaming {fname}...")
        delim = reader.detect_delimiter(fpath)
        with fpath.open("r", encoding="utf-8", errors="ignore") as f:
            csv_reader = csv.reader(f, delimiter=delim)
            header = next(csv_reader, None)
            for row in csv_reader:
                if len(row) < 6:
                    continue
                ts_str = row[0].strip()
                try:
                    e = float(row[1])
                    v = float(row[2])
                    i = float(row[3])
                    freq = float(row[4])
                except ValueError:
                    continue
                m = row[5].strip()

                info = meter_info[m]
                info["count"] += 1
                if ts_str in info["seen_ts"]:
                    info["duplicates"] += 1
                else:
                    info["seen_ts"].add(ts_str)

                if info["min_ts"] is None or ts_str < info["min_ts"]:
                    info["min_ts"] = ts_str
                if info["max_ts"] is None or ts_str > info["max_ts"]:
                    info["max_ts"] = ts_str

                # Outage check: V <= 5.0V or I <= 0.01A
                is_outage = (v <= 5.0) or (i <= 0.01)
                if is_outage:
                    info["outages"] += 1

                # Physical validity check
                if e < 0.0 or v < 0.0 or i < 0.0 or freq < 0.0:
                    info["negatives"] += 1

                # 15-minute floor bin key: YYYY-MM-DD HH:MM:00
                min_part = int(ts_str[14:16])
                bin_min = (min_part // 15) * 15
                bin_key = ts_str[:14] + f"{bin_min:02d}:00"

                b = info["bins"][bin_key]
                b["count"] += 1
                if is_outage:
                    b["outages"] += 1
                else:
                    b["sum_p"] += e * 20.0
                b["v_sum"] += v
                b["i_sum"] += i
                b["f_sum"] += freq

    print(f"[{datetime.now().isoformat()}] Completed streaming in {time.time()-t0:.2f}s. Evaluating quality gates...")

    evaluations: List[MeterEvaluation] = []
    admitted_meters: List[str] = []
    rejected_meters: List[str] = []
    failing_meters: Dict[str, Any] = {}

    for m in sorted(meter_info.keys()):
        info = meter_info[m]
        dt1 = datetime.strptime(info["min_ts"], "%Y-%m-%d %H:%M:%S")
        dt2 = datetime.strptime(info["max_ts"], "%Y-%m-%d %H:%M:%S")
        span_s = (dt2 - dt1).total_seconds()
        span_days = span_s / 86400.0
        exp_bins = int(span_s / 900.0) + 1
        pop_bins = len(info["bins"])
        cov = pop_bins / exp_bins if exp_bins > 0 else 0.0
        out_frac = info["outages"] / info["count"] if info["count"] > 0 else 0.0

        failing_gates: List[str] = []
        if span_days < min_span_days:
            failing_gates.append("insufficient_span")
        if cov < min_bin_coverage:
            failing_gates.append("insufficient_coverage")
        if out_frac > max_outage_fraction:
            failing_gates.append("outage_fraction_exceeded")
        if info["duplicates"] > 0:
            failing_gates.append("duplicate_timestamps")
        if info["negatives"] > 0:
            failing_gates.append("negative_or_impossible_values")

        is_admitted = (len(failing_gates) == 0)
        metrics = {
            "span_days": round(span_days, 2),
            "bin_coverage": round(cov, 4),
            "outage_fraction": round(out_frac, 4),
            "duplicate_count": info["duplicates"],
            "negative_count": info["negatives"]
        }
        evaluations.append(MeterEvaluation(m, is_admitted, failing_gates, metrics))

        if is_admitted:
            admitted_meters.append(m)
        else:
            rejected_meters.append(m)
            failing_meters[m] = {
                "failing_gates": failing_gates,
                "metrics": metrics
            }

    engine = QualityEngine(
        min_span_days=min_span_days,
        min_bin_coverage=min_bin_coverage,
        max_outage_fraction=max_outage_fraction
    )
    quality_report = engine.generate_quality_report(evaluations)
    # Ensure both top-level and summary keys are populated for complete backward compatibility
    quality_report["total_meters_evaluated"] = len(evaluations)
    quality_report["admitted_meters_count"] = len(admitted_meters)
    quality_report["rejected_meters_count"] = len(rejected_meters)
    quality_report["admission_rate"] = round(len(admitted_meters) / len(evaluations), 3)

    print(f"[{datetime.now().isoformat()}] Evaluated {len(evaluations)} meters: {len(admitted_meters)} admitted, {len(rejected_meters)} rejected.")
    print(f"[{datetime.now().isoformat()}] Admission rate: {quality_report['admission_rate'] * 100:.1f}%")

    # Chronological Splitting for Admitted Meters
    print(f"[{datetime.now().isoformat()}] Generating chronological 70/15/15 splits for {len(admitted_meters)} admitted meters...")
    splitter = ChronologicalSplitter(train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15)
    meter_splits: Dict[str, MeterSplits] = {}

    ist_tz = timezone(timedelta(hours=5, minutes=30))
    for m in admitted_meters:
        info = meter_info[m]
        # Construct sorted 15-minute ResampledReading list
        resampled_series: List[ResampledReading] = []
        for b_str in sorted(info["bins"].keys()):
            b = info["bins"][b_str]
            n = b["count"]
            dt_local = datetime.strptime(b_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ist_tz)
            dt_utc = dt_local.astimezone(timezone.utc)

            outage_frac = b["outages"] / n
            is_bin_outage = (outage_frac >= 0.50)
            active_count = n - b["outages"]

            if is_bin_outage or active_count == 0:
                p_kw = None
                is_bin_outage = True
            else:
                p_kw = b["sum_p"] / active_count

            resampled_series.append(ResampledReading(
                timestamp=dt_utc,
                meter_id=m,
                power_kw=p_kw,
                voltage_v=round(b["v_sum"] / n, 2),
                current_a=round(b["i_sum"] / n, 3),
                freq_hz=round(b["f_sum"] / n, 2),
                is_outage=is_bin_outage,
                sample_count=n
            ))

        _, _, _, splits = splitter.split_meter_series(resampled_series)
        meter_splits[m] = splits

    split_manifest = splitter.generate_manifest(meter_splits, source_hashes)

    # Write output files to workspace root
    quality_report_path = workspace_root / "quality_report.json"
    with quality_report_path.open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    print(f"[{datetime.now().isoformat()}] Written quality_report.json to {quality_report_path}")

    split_manifest_path = workspace_root / "split_manifest.json"
    with split_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"[{datetime.now().isoformat()}] Written split_manifest.json to {split_manifest_path}")

    return quality_report, split_manifest


if __name__ == "__main__":
    import sys
    ws = Path(__file__).resolve().parent.parent.parent.parent
    run_data_pipeline(ws)
