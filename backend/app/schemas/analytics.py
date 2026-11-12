"""Pydantic v2 schemas for Grid Analytics endpoints."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class OverviewData(BaseModel):
    total_meters: int = Field(..., description="Total smart meters cataloged")
    admitted_meters: int = Field(..., description="Admitted compliant meters (passing 5 gates)")
    admission_rate: float = Field(..., description="Admission fraction between 0.0 and 1.0")
    total_consumption_kwh: float = Field(..., description="Total net consumption across admitted meters")
    active_anomalies_count: int = Field(..., description="Currently active unresolved anomaly alerts")


class TelemetryReading(BaseModel):
    timestamp: str
    power_kw: Optional[float] = None
    is_outage: bool
    voltage_v: float
    current_a: float


class SeriesData(BaseModel):
    meter_id: str
    series: List[TelemetryReading]
