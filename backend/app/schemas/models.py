"""Pydantic v2 schemas for Model Registry and Metrics endpoints."""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel


class ModelMetric(BaseModel):
    model_name: str
    module: str
    architecture: str
    config_hash: str
    dataset_hash: str
    seed: int
    metrics: Dict[str, float]
    baseline: Dict[str, Any]


class MetricsData(BaseModel):
    metrics: List[ModelMetric]
    count: int
