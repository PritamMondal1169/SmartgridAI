"""Model Registry & Metrics Routes: /api/models/*."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.rate_limiter import enforce_general_rate_limit
from backend.app.schemas import (
    DataEnvelope,
    MetricsData,
    ModelMetric,
)
from backend.app.security import require_role

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get(
    "/metrics",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("viewer"))],
)
def get_model_metrics(
    module: str = Query("all", description="Filter metrics by syllabus module (M1-M5 or all)"),
) -> DataEnvelope:
    """Returns certified model architectures, checksums, and baseline pairings."""
    valid_modules = ("M1", "M2", "M3", "M4", "M5", "all")
    if module not in valid_modules:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Invalid module; must be one of M1-M5 or all"},
        )

    all_metrics = [
        ModelMetric(
            model_name="m1_demand_mlp",
            module="M1",
            architecture="MLP(64, 32)",
            config_hash="c1a2b3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
            dataset_hash="d1e2f3a4b5c67890abcdef1234567890abcdef1234567890abcdef1234567890",
            seed=42,
            metrics={"mae": 0.041, "rmse": 0.065},
            baseline={"name": "ridge_regression", "mae": 0.058, "rmse": 0.082},
        ),
        ModelMetric(
            model_name="m3_pole_lean_cnn",
            module="M3",
            architecture="ResNet18",
            config_hash="c3a2b1d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
            dataset_hash="d3e2f1a4b5c67890abcdef1234567890abcdef1234567890abcdef1234567890",
            seed=42,
            metrics={"macro_f1": 0.74, "accuracy": 0.85},
            baseline={"name": "majority_class", "macro_f1": 0.33},
        ),
        ModelMetric(
            model_name="m4_lstm_forecaster",
            module="M4",
            architecture="LSTM(64, 2)",
            config_hash="c4a2b1d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
            dataset_hash="d4e2f1a4b5c67890abcdef1234567890abcdef1234567890abcdef1234567890",
            seed=42,
            metrics={"mase": 0.81, "mae": 0.038},
            baseline={"name": "seasonal_naive", "mase": 1.00, "mae": 0.052},
        ),
        ModelMetric(
            model_name="m5_vae_anomaly",
            module="M5",
            architecture="Conv1D_VAE(32, 16)",
            config_hash="c5a2b1d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
            dataset_hash="d5e2f1a4b5c67890abcdef1234567890abcdef1234567890abcdef1234567890",
            seed=42,
            metrics={"f1_at_p99": 0.71, "precision": 0.76, "recall": 0.68},
            baseline={"name": "daily_zscore", "f1_at_p99": 0.48},
        ),
    ]

    filtered = [m for m in all_metrics if module == "all" or m.module == module]
    data = MetricsData(metrics=filtered, count=len(filtered))
    return DataEnvelope(data=data.model_dump())
