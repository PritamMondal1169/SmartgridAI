"""Pydantic v2 schemas for Sequence Forecasting (M4) endpoints."""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    meter_id: str = Field(..., max_length=1000, description="Target smart meter identifier")
    history_h: int = Field(default=168, gt=0, description="History window in hours")
    horizon_h: int = Field(default=24, gt=0, le=168, description="Forecast horizon in hours (max 168)")


class ForecastData(BaseModel):
    meter_id: str
    arch: str
    forecast: List[float]
    baselines: Dict[str, List[float]]
    metrics: Dict[str, float]
