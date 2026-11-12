"""
SmartgridAI Data Engineering and Preprocessing Pipeline.
Includes physical data contracts, multi-format telemetry readers,
5-gate quality admission engine, and chronological splitting.
"""

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
)
from smartgrid.data.ingestion import (
    SmartMeterReader,
    GridDemandReader,
    ImageGuard,
    VisionCatalogReader,
)
from smartgrid.data.quality_gates import (
    QualityEngine,
    GateResult,
    MeterEvaluation,
)
from smartgrid.data.splitter import (
    ChronologicalSplitter,
    SplitBoundary,
    MeterSplits,
)

__all__ = [
    "PhysicalReading",
    "ResampledReading",
    "compute_active_power",
    "is_outage_reading",
    "apply_physical_contract",
    "validate_reading_bounds",
    "INTERVAL_MINUTES",
    "CONVERSION_FACTOR",
    "VOLTAGE_OUTAGE_THRESHOLD_V",
    "CURRENT_OUTAGE_THRESHOLD_A",
    "MIN_SPAN_DAYS",
    "MIN_BIN_COVERAGE",
    "MAX_OUTAGE_FRACTION",
    "SmartMeterReader",
    "GridDemandReader",
    "ImageGuard",
    "VisionCatalogReader",
    "QualityEngine",
    "GateResult",
    "MeterEvaluation",
    "ChronologicalSplitter",
    "SplitBoundary",
    "MeterSplits",
]
