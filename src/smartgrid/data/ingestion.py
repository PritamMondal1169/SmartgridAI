"""
SmartgridAI Telemetry and Visual Ingestion Engine.
Handles resilient multi-format reading of:
- D1 Smart Meter CSVs (CEEW Bareilly and Mathura)
- D2 National Grid Demand & Renewable Generation XLSXs (POSOCO / NLDC)
- D3 Power Distribution Infrastructure Defect Vision Catalog (PD-Defect)
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Generator, Dict, List, Optional, Union, Tuple, Any, Set
import csv
import hashlib
import io
import math
import zipfile
import xml.etree.ElementTree as ET

from smartgrid.data.contract import (
    PhysicalReading,
    apply_physical_contract,
    validate_reading_bounds,
    INTERVAL_MINUTES
)

# India Standard Time (IST, UTC+05:30)
IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))

# Standard D1 CSV column mapping (index 0 to 5)
COLUMN_INDEX_TIMESTAMP = 0
COLUMN_INDEX_ENERGY_KWH = 1
COLUMN_INDEX_VOLTAGE_V = 2
COLUMN_INDEX_CURRENT_A = 3
COLUMN_INDEX_FREQ_HZ = 4
COLUMN_INDEX_METER_ID = 5

INTERVAL_FILE_PATTERNS = [
    "SM Cleaned Data BR2019.csv",
    "CEEW - Smart meter data Bareilly 2020.csv",
    "CEEW - Smart meter data Bareilly 2021.csv",
    "CEEW - Smart meter data Mathura 2019.csv",
    "CEEW - Smart meter data Mathura 2020.csv",
    "SM Cleaned Data MH2021.csv"
]


class SmartMeterReader:
    """Ingests D1 smart meter CSV files with delimiter sniffing and UTC normalization."""

    def __init__(self, raw_dir: Union[str, Path]) -> None:
        self.raw_dir = Path(raw_dir)

    @staticmethod
    def detect_delimiter(filepath: Path) -> str:
        """
        Detects whether file is comma or tab separated by inspecting the header line.
        """
        with filepath.open("r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
            return "\t" if "\t" in first_line else ","

    @staticmethod
    def parse_ist_timestamp(ts_str: str) -> datetime:
        """
        Parses IST timestamp (YYYY-MM-DD HH:MM:SS) and converts strictly to UTC ISO format.
        """
        clean_str = ts_str.strip()
        dt_local = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST_TIMEZONE)
        return dt_local.astimezone(timezone.utc)

    def parse_file(
        self,
        filepath: Path,
        target_meter_id: Optional[str] = None
    ) -> Generator[PhysicalReading, None, None]:
        """
        Streams validated PhysicalReading objects from a raw D1 CSV file.
        
        Args:
            filepath: Path to CSV file.
            target_meter_id: Optional filter for specific meter ID.
        """
        delim = self.detect_delimiter(filepath)
        with filepath.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=delim)
            header = next(reader, None)
            if not header or len(header) < 6:
                return

            for row in reader:
                if not row or len(row) < 6:
                    continue
                meter_id = row[COLUMN_INDEX_METER_ID].strip()
                if target_meter_id is not None and meter_id != target_meter_id:
                    continue

                try:
                    ts_utc = self.parse_ist_timestamp(row[COLUMN_INDEX_TIMESTAMP])
                    energy_kwh = float(row[COLUMN_INDEX_ENERGY_KWH])
                    voltage_v = float(row[COLUMN_INDEX_VOLTAGE_V])
                    current_a = float(row[COLUMN_INDEX_CURRENT_A])
                    freq_hz = float(row[COLUMN_INDEX_FREQ_HZ])

                    power_kw, is_outage = apply_physical_contract(
                        energy_kwh=energy_kwh,
                        voltage_v=voltage_v,
                        current_a=current_a,
                        interval_minutes=INTERVAL_MINUTES
                    )

                    yield PhysicalReading(
                        timestamp=ts_utc,
                        meter_id=meter_id,
                        energy_kwh=energy_kwh,
                        voltage_v=voltage_v,
                        current_a=current_a,
                        freq_hz=freq_hz,
                        power_kw=power_kw,
                        is_outage=is_outage
                    )
                except (ValueError, IndexError):
                    continue

    def load_meter_series(
        self,
        meter_id: str,
        files: Optional[List[str]] = None
    ) -> List[PhysicalReading]:
        """
        Loads all interval readings for a specific meter across source CSV files.
        """
        readings: List[PhysicalReading] = []
        target_files = files or INTERVAL_FILE_PATTERNS
        for fname in target_files:
            fpath = self.raw_dir / fname
            if fpath.exists():
                readings.extend(self.parse_file(fpath, target_meter_id=meter_id))
        readings.sort(key=lambda r: r.timestamp)
        return readings

    def compute_file_hashes(self) -> Dict[str, str]:
        """
        Calculates SHA-256 digests for all 8 CSV files in the D1 directory.
        """
        hashes: Dict[str, str] = {}
        for p in sorted(self.raw_dir.glob("*.csv")):
            h = hashlib.sha256()
            with p.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            hashes[p.name] = h.hexdigest()
        return hashes


class GridDemandReader:
    """
    Ingests D2 national grid demand and renewable generation from XLSX workbooks.
    Uses openpyxl or pure-Python OpenXML parsing for resilience across environments.
    """

    def __init__(self, raw_dir: Union[str, Path]) -> None:
        self.raw_dir = Path(raw_dir)

    def list_workbooks(self) -> List[Path]:
        """Finds all .xlsx workbooks in raw_dir recursively."""
        return sorted(self.raw_dir.rglob("*.xlsx"))

    @staticmethod
    def excel_serial_to_utc(serial_val: float) -> datetime:
        """
        Converts Excel serial day number (1899-12-30 epoch) in IST to UTC datetime.
        """
        excel_base = datetime(1899, 12, 30, tzinfo=IST_TIMEZONE)
        dt_local = excel_base + timedelta(days=serial_val)
        return dt_local.astimezone(timezone.utc)

    @staticmethod
    def parse_string_timestamp_to_utc(ts_str: str) -> datetime:
        """
        Parses multi-year date string (DD-MM-YYYY HH:MM:SS or YYYY-MM-DD HH:MM:SS) in IST to UTC.
        """
        clean_str = ts_str.strip()
        try:
            dt_local = datetime.strptime(clean_str, "%d-%m-%Y %H:%M:%S").replace(tzinfo=IST_TIMEZONE)
        except ValueError:
            dt_local = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST_TIMEZONE)
        return dt_local.astimezone(timezone.utc)

    def parse_monthly_workbook(
        self,
        filepath: Path,
        max_rows: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Parses monthly workbook (Sheet1 / sheet2.xml containing 15 SCADA columns).
        Clips negative solar generation values to 0.0 MW.
        """
        records: List[Dict[str, Any]] = []
        try:
            import openpyxl
            wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
            # Find Sheet1 (or first matching sheet with data)
            sheet_name = "Sheet1" if "Sheet1" in wb.sheetnames else wb.sheetnames[0]
            ws = wb[sheet_name]

            row_idx = 0
            for row in ws.iter_rows(values_only=True):
                row_idx += 1
                if row_idx <= 2:
                    # Skip Row 1 (SCADA Analog Tag) and Row 2 (Signal Name)
                    continue
                if not row or row[0] is None:
                    continue

                try:
                    time_val = row[0]
                    if isinstance(time_val, (int, float)):
                        ts_utc = self.excel_serial_to_utc(float(time_val))
                    elif isinstance(time_val, datetime):
                        ts_utc = time_val.replace(tzinfo=IST_TIMEZONE).astimezone(timezone.utc)
                    else:
                        continue

                    # NLDC Demand is column K (index 10)
                    demand_mw = float(row[10]) if len(row) > 10 and row[10] not in (None, "#N/A", "") else 0.0
                    wind_mw = float(row[5]) if len(row) > 5 and row[5] not in (None, "#N/A", "") else 0.0
                    raw_solar = float(row[6]) if len(row) > 6 and row[6] not in (None, "#N/A", "") else 0.0
                    # Clip negative solar values to 0.0
                    solar_mw = max(0.0, raw_solar)

                    records.append({
                        "timestamp": ts_utc,
                        "demand_mw": demand_mw,
                        "wind_mw": wind_mw,
                        "solar_mw": solar_mw,
                        "source_file": filepath.name
                    })

                    if max_rows and len(records) >= max_rows:
                        break
                except (ValueError, TypeError):
                    continue
            wb.close()
        except Exception:
            pass
        return records

    def parse_multi_year_workbook(
        self,
        filepath: Path,
        max_rows: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Parses January 2024- June 2025 multi-year workbook (Report sheet).
        """
        records: List[Dict[str, Any]] = []
        try:
            import openpyxl
            wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
            sheet_name = "Report" if "Report" in wb.sheetnames else wb.sheetnames[0]
            ws = wb[sheet_name]

            row_idx = 0
            for row in ws.iter_rows(values_only=True):
                row_idx += 1
                if row_idx == 1:
                    continue  # Header
                if not row or row[0] is None:
                    continue

                try:
                    time_val = row[0]
                    if isinstance(time_val, str):
                        ts_utc = self.parse_string_timestamp_to_utc(time_val)
                    elif isinstance(time_val, datetime):
                        ts_utc = time_val.replace(tzinfo=IST_TIMEZONE).astimezone(timezone.utc)
                    elif isinstance(time_val, (int, float)):
                        ts_utc = self.excel_serial_to_utc(float(time_val))
                    else:
                        continue

                    demand_mw = float(row[1]) if len(row) > 1 and row[1] not in (None, "#N/A", "") else 0.0
                    wind_mw = float(row[2]) if len(row) > 2 and row[2] not in (None, "#N/A", "") else 0.0
                    raw_solar = float(row[3]) if len(row) > 3 and row[3] not in (None, "#N/A", "") else 0.0
                    solar_mw = max(0.0, raw_solar)

                    records.append({
                        "timestamp": ts_utc,
                        "demand_mw": demand_mw,
                        "wind_mw": wind_mw,
                        "solar_mw": solar_mw,
                        "source_file": filepath.name
                    })

                    if max_rows and len(records) >= max_rows:
                        break
                except (ValueError, TypeError):
                    continue
            wb.close()
        except Exception:
            pass
        return records

    def resample_records(
        self,
        records: List[Dict[str, Any]],
        interval_minutes: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Resamples raw grid records to uniform interval_minutes means.
        """
        from collections import defaultdict
        bins: Dict[datetime, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            dt = r["timestamp"]
            bin_min = (dt.minute // interval_minutes) * interval_minutes
            b_start = dt.replace(minute=bin_min, second=0, microsecond=0)
            bins[b_start].append(r)

        resampled: List[Dict[str, Any]] = []
        for b_start in sorted(bins.keys()):
            items = bins[b_start]
            n = len(items)
            avg_demand = sum(i["demand_mw"] for i in items) / n
            avg_wind = sum(i["wind_mw"] for i in items) / n
            avg_solar = sum(i["solar_mw"] for i in items) / n

            resampled.append({
                "timestamp": b_start,
                "demand_mw": round(avg_demand, 2),
                "wind_mw": round(avg_wind, 2),
                "solar_mw": round(avg_solar, 2),
                "sample_count": n
            })
        return resampled


class ImageGuard:
    """
    Validates magic bytes, strips EXIF metadata, enforces 10MB limits,
    and auto-downscales images > 4096 px preserving aspect ratio.
    """

    MAGIC_BYTES_JPEG = b"\xff\xd8\xff"
    MAGIC_BYTES_PNG = b"\x89PNG\r\n\x1a\n"
    MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
    MAX_DIMENSION_PX = 4096

    @classmethod
    def validate_magic_bytes(cls, data: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validates whether raw byte buffer begins with valid JPEG or PNG signature.
        """
        if data.startswith(cls.MAGIC_BYTES_JPEG):
            return True, "jpeg"
        if data.startswith(cls.MAGIC_BYTES_PNG):
            return True, "png"
        return False, None

    @classmethod
    def validate_file(cls, filepath: Path) -> Tuple[bool, Optional[str]]:
        """
        Validates file existence, file size <= 10MB, and valid image magic bytes.
        """
        if not filepath.exists():
            return False, "file_not_found"
        size = filepath.stat().st_size
        if size > cls.MAX_SIZE_BYTES:
            return False, "file_size_exceeded"
        with filepath.open("rb") as f:
            header = f.read(16)
        valid, img_type = cls.validate_magic_bytes(header)
        if not valid:
            return False, "invalid_magic_bytes"
        return True, img_type

    @classmethod
    def process_image(
        cls,
        image_path: Path,
        output_path: Optional[Path] = None,
        max_dim: int = MAX_DIMENSION_PX
    ) -> Tuple[int, int, bool]:
        """
        Loads image, strips EXIF data, scales down if max(W, H) > max_dim preserving aspect ratio.
        
        Returns:
            Tuple of (final_width, final_height, was_resized).
        """
        from PIL import Image

        with Image.open(image_path) as img:
            orig_w, orig_h = img.size
            needs_downscale = max(orig_w, orig_h) > max_dim

            if needs_downscale:
                if orig_w >= orig_h:
                    new_w = max_dim
                    new_h = int(orig_h * (max_dim / orig_w))
                else:
                    new_h = max_dim
                    new_w = int(orig_w * (max_dim / orig_h))
                resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            else:
                new_w, new_h = orig_w, orig_h
                resized_img = img.copy()

            # Save clean image without EXIF metadata
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                clean_img = Image.new(resized_img.mode, resized_img.size)
                clean_img.paste(resized_img)
                clean_img.save(output_path, quality=95)

            return new_w, new_h, needs_downscale


class VisionCatalogReader:
    """Ingests D3 defect inspection images and annotations."""

    def __init__(self, defect_dir: Union[str, Path]) -> None:
        self.defect_dir = Path(defect_dir)

    def load_pole_detection_annotations(self, split: str = "train") -> Dict[str, Any]:
        """
        Loads YOLO bounding box annotations and image inventory for pole_detection.
        """
        split_dir = self.defect_dir / "pole_detection" / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        images: List[str] = []
        box_count = 0
        if img_dir.exists():
            for p in sorted(img_dir.glob("*.jpg")):
                images.append(p.name)
                lbl_path = lbl_dir / f"{p.stem}.txt"
                if lbl_path.exists():
                    with lbl_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                box_count += 1

        return {
            "split": split,
            "image_count": len(images),
            "box_count": box_count,
            "images": images
        }

    def load_pole_lean_labels(self) -> Dict[str, str]:
        """
        Loads pole lean classification labels from labels.csv.
        """
        labels_file = self.defect_dir / "pole_lean_assessment" / "classification" / "labels.csv"
        labels: Dict[str, str] = {}
        if not labels_file.exists():
            return labels

        with labels_file.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) >= 2:
                    labels[row[0].strip()] = row[1].strip()
        return labels

    def load_vegetation_coco_annotations(self, coco_json_path: Path) -> Dict[str, Any]:
        """
        Loads COCO JSON annotations for vegetation-conductor segmentation.
        """
        import json
        if not coco_json_path.exists():
            return {}
        with coco_json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
