"""Pydantic v2 schemas package for SmartgridAI."""

from backend.app.schemas.common import (
    DataEnvelope,
    ErrorDetail,
    ErrorEnvelope,
    MetaEnvelope,
)
from backend.app.schemas.analytics import (
    OverviewData,
    SeriesData,
    TelemetryReading,
)
from backend.app.schemas.ann import (
    DecisionBoundary,
    PredictData,
    PredictRequest,
    XorData,
)
from backend.app.schemas.forecast import (
    ForecastData,
    ForecastRequest,
)
from backend.app.schemas.infrastructure import (
    ClassifyData,
    DetectData,
    Detection,
    Segment,
    SegmentData,
)
from backend.app.schemas.training import (
    CompareRequest,
    CompareResponseData,
    JobData,
)
from backend.app.schemas.generative import (
    AnomaliesData,
    AnomalyItem,
    GanData,
    GanRequest,
    VaeData,
    VaeRequest,
)
from backend.app.schemas.models import (
    MetricsData,
    ModelMetric,
)

__all__ = [
    "MetaEnvelope",
    "DataEnvelope",
    "ErrorDetail",
    "ErrorEnvelope",
    "OverviewData",
    "TelemetryReading",
    "SeriesData",
    "PredictRequest",
    "PredictData",
    "DecisionBoundary",
    "XorData",
    "ForecastRequest",
    "ForecastData",
    "ClassifyData",
    "Detection",
    "DetectData",
    "Segment",
    "SegmentData",
    "CompareRequest",
    "CompareResponseData",
    "JobData",
    "GanRequest",
    "GanData",
    "VaeRequest",
    "VaeData",
    "AnomalyItem",
    "AnomaliesData",
    "ModelMetric",
    "MetricsData",
]
