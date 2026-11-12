"""
SmartgridAI Chronological Splitter and Manifest Signer.
Partitions admitted 15-minute time series into strictly chronological
70/15/15 train/validation/test sets and signs split_manifest.json with SHA-256 digests.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union
import json

from smartgrid.data.contract import ResampledReading


@dataclass(frozen=True)
class SplitBoundary:
    """Timestamp boundaries and row count for a dataset partition."""
    start: str  # ISO-8601 UTC
    end: str    # ISO-8601 UTC
    rows: int

    def to_dict(self) -> Dict[str, Any]:
        """Converts boundary to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "rows": self.rows
        }


@dataclass(frozen=True)
class MeterSplits:
    """Train, validation, and test partitions for an admitted meter."""
    meter_id: str
    train: SplitBoundary
    valid: SplitBoundary
    test: SplitBoundary

    def to_dict(self) -> Dict[str, Any]:
        """Converts meter splits to dictionary."""
        return {
            "train": self.train.to_dict(),
            "valid": self.valid.to_dict(),
            "test": self.test.to_dict()
        }


class ChronologicalSplitter:
    """Partitions time series chronologically per meter without look-ahead leakage."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> None:
        if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio} + {valid_ratio} + {test_ratio}"
            )
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def split_meter_series(
        self,
        series: List[ResampledReading]
    ) -> Tuple[List[ResampledReading], List[ResampledReading], List[ResampledReading], MeterSplits]:
        """
        Splits a single meter's 15-minute time series chronologically into
        70% train, 15% validation, and 15% test slices.
        
        Args:
            series: List of ResampledReading objects for a single meter.
            
        Returns:
            Tuple of (train_slice, valid_slice, test_slice, MeterSplits).
            
        Raises:
            ValueError: If series contains fewer than 3 records.
        """
        m = len(series)
        if m < 3:
            raise ValueError(f"Series has {m} observations; minimum 3 rows required to split.")

        sorted_series = sorted(series, key=lambda r: r.timestamp)
        n_train = int(self.train_ratio * m)
        n_valid = int(self.valid_ratio * m)
        n_test = m - n_train - n_valid

        if n_train == 0 or n_valid == 0 or n_test == 0:
            raise ValueError(f"Series length {m} too small to allocate non-empty 70/15/15 partitions.")

        train_data = sorted_series[:n_train]
        valid_data = sorted_series[n_train: n_train + n_valid]
        test_data = sorted_series[n_train + n_valid:]

        # Verify temporal invariants: strict chronological non-overlap
        assert train_data[-1].timestamp < valid_data[0].timestamp, (
            f"Temporal leakage detected: Train end ({train_data[-1].timestamp}) "
            f">= Valid start ({valid_data[0].timestamp})"
        )
        assert valid_data[-1].timestamp < test_data[0].timestamp, (
            f"Temporal leakage detected: Valid end ({valid_data[-1].timestamp}) "
            f">= Test start ({test_data[0].timestamp})"
        )
        assert len(train_data) + len(valid_data) + len(test_data) == m, (
            f"Row count mismatch: {len(train_data)} + {len(valid_data)} + {len(test_data)} != {m}"
        )

        meter_id = sorted_series[0].meter_id

        splits = MeterSplits(
            meter_id=meter_id,
            train=SplitBoundary(
                start=train_data[0].timestamp.isoformat(),
                end=train_data[-1].timestamp.isoformat(),
                rows=len(train_data)
            ),
            valid=SplitBoundary(
                start=valid_data[0].timestamp.isoformat(),
                end=valid_data[-1].timestamp.isoformat(),
                rows=len(valid_data)
            ),
            test=SplitBoundary(
                start=test_data[0].timestamp.isoformat(),
                end=test_data[-1].timestamp.isoformat(),
                rows=len(test_data)
            )
        )

        return train_data, valid_data, test_data, splits

    def generate_manifest(
        self,
        meter_splits: Dict[str, MeterSplits],
        source_hashes: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Compiles the complete split_manifest.json dictionary matching the project specification.
        """
        splits_dict: Dict[str, Any] = {}
        for m_id in sorted(meter_splits.keys()):
            splits_dict[m_id] = meter_splits[m_id].to_dict()

        return {
            "manifest_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file_sha256": source_hashes,
            "split_policy": {
                "type": "chronological_per_meter",
                "train_ratio": self.train_ratio,
                "valid_ratio": self.valid_ratio,
                "test_ratio": self.test_ratio
            },
            "splits": splits_dict
        }
