"""Pydantic v2 schemas for ANN (M1) endpoints."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Numerical feature vector for demand inference")

    @field_validator("features")
    @classmethod
    def validate_features_non_empty(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Feature vector cannot be empty")
        return v


class PredictData(BaseModel):
    mlp_prediction: float
    ridge_prediction: float
    delta: float


class DecisionBoundary(BaseModel):
    x: List[float]
    y: List[float]
    surface: List[List[int]]


class XorData(BaseModel):
    seed: int
    hidden: bool
    converged: bool
    epochs: int
    final_loss: float
    decision_boundary: DecisionBoundary
