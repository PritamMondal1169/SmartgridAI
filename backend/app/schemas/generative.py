"""Pydantic v2 schemas for Generative & Anomaly (M5) endpoints."""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field, field_validator


class GanRequest(BaseModel):
    meter_id: str = Field(default="MTR_01", description="Conditioning meter identifier")
    n_samples: int = Field(default=5, gt=0, le=100, description="Number of synthetic 24h profiles (max 100)")


class GanData(BaseModel):
    meter_id: str
    samples: List[List[float]]
    eval_metrics: Dict[str, float]


class VaeRequest(BaseModel):
    meter_id: str = Field(default="MTR_01", description="Meter identifier")
    window: List[float] = Field(..., description="Daily telemetry window of exactly 96 15-minute readings")

    @field_validator("window")
    @classmethod
    def validate_window_length(cls, v: List[float]) -> List[float]:
        if len(v) != 96:
            raise ValueError(f"Window length must be 96 (got {len(v)})")
        return v


class VaeData(BaseModel):
    meter_id: str
    reconstruction: List[float]
    reconstruction_error: float
    p99_threshold: float
    is_anomaly: bool


class AnomalyItem(BaseModel):
    id: str
    meter_id: str
    score: float
    threshold: float
    severity: str
    timestamp: str


class AnomaliesData(BaseModel):
    items: List[AnomalyItem]
    total: int
    page: int
    pages: int
