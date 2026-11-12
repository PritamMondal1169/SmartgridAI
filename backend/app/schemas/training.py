"""Pydantic v2 schemas for Training Dynamics (M2) & Async Jobs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CompareRequest(BaseModel):
    optimizers: List[str] = Field(..., description="List of optimizers to evaluate")
    init: str = Field(..., description="Weight initialization method")
    regularisation: Optional[str] = Field(default="None", description="Regularization strategy")
    seeds: List[int] = Field(..., description="Random seeds for statistical replication")

    @field_validator("optimizers")
    @classmethod
    def validate_optimizers_non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("optimizer list cannot be empty")
        return v

    @field_validator("seeds")
    @classmethod
    def validate_seeds_non_empty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("seeds list must contain at least 1 seed")
        return v

    @field_validator("init")
    @classmethod
    def validate_init_name(cls, v: str) -> str:
        if v not in ("Random", "Xavier", "He"):
            raise ValueError("Invalid init; must be one of 'Random', 'Xavier', or 'He'")
        return v


class CompareResponseData(BaseModel):
    job_id: str
    state: str


class JobData(BaseModel):
    job_id: str
    state: str
    started_at: str
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
