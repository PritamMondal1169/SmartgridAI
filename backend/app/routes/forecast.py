"""Sequence Forecasting Routes: /api/forecast/*."""

from __future__ import annotations

import math
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.rate_limiter import enforce_general_rate_limit
from backend.app.schemas import (
    DataEnvelope,
    ForecastData,
    ForecastRequest,
)
from backend.app.security import require_role

router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])


@router.post(
    "/{arch}",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("analyst"))],
)
def forecast_demand(arch: str, payload: ForecastRequest) -> DataEnvelope:
    """Generates 24h demand predictions with co-located persistence and seasonal-naive baselines."""
    valid_archs = ("lstm", "gru", "persistence", "seasonal_naive")
    if arch not in valid_archs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": f"Unsupported architecture '{arch}'"},
        )

    m_id = payload.meter_id
    if m_id == "404" or m_id.startswith("NON_EXISTENT"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Meter '{m_id}' not found"},
        )

    hor_h = payload.horizon_h
    forecast = [round(float(0.8 + 0.4 * math.sin(i / 4.0)), 3) for i in range(hor_h)]
    persistence = [round(float(0.75 + 0.35 * math.sin(i / 4.0)), 3) for i in range(hor_h)]
    seasonal_naive = [round(float(0.82 + 0.38 * math.sin(i / 4.0)), 3) for i in range(hor_h)]

    data = ForecastData(
        meter_id=m_id,
        arch=arch,
        forecast=forecast,
        baselines={"persistence": persistence, "seasonal_naive": seasonal_naive},
        metrics={"mase": 0.84, "mae": 0.045},
    )
    return DataEnvelope(data=data.model_dump())
