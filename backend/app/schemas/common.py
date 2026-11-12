"""Standardized Response Envelopes and Base Models for SmartgridAI API."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class MetaEnvelope(BaseModel):
    """Execution and provenance metadata envelope."""
    model_version: str = "v1.0.0"
    dataset_hash: str = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
    latency_ms: float = 24.5


class DataEnvelope(BaseModel):
    """Standardized root envelope for all successful API responses."""
    data: Any
    meta: MetaEnvelope = Field(default_factory=MetaEnvelope)


class ErrorDetail(BaseModel):
    """Structured error payload details."""
    code: str
    message: str
    detail: Optional[Dict[str, Any]] = None


class ErrorEnvelope(BaseModel):
    """Standardized root envelope for all error API responses."""
    error: ErrorDetail
