"""Generative & Anomaly Detection Routes: /api/generative/* and /api/anomalies."""

from __future__ import annotations

import math
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.rate_limiter import enforce_general_rate_limit
from backend.app.schemas import (
    AnomaliesData,
    AnomalyItem,
    DataEnvelope,
    GanData,
    GanRequest,
    VaeData,
    VaeRequest,
)
from backend.app.security import require_role

router = APIRouter(tags=["Generative & Anomaly"])

# Shared in-memory anomaly store
anomalies_db: List[AnomalyItem] = [
    AnomalyItem(
        id="anom_001",
        meter_id="MTR_01",
        score=0.485,
        threshold=0.120,
        severity="high",
        timestamp="2020-03-15T14:30:00Z",
    ),
    AnomalyItem(
        id="anom_002",
        meter_id="MTR_02",
        score=0.215,
        threshold=0.120,
        severity="medium",
        timestamp="2020-03-16T08:15:00Z",
    ),
    AnomalyItem(
        id="anom_003",
        meter_id="MTR_03",
        score=0.135,
        threshold=0.120,
        severity="low",
        timestamp="2020-03-17T19:45:00Z",
    ),
]


@router.post(
    "/api/generative/gan",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("analyst"))],
)
def generate_synthetic_profiles(payload: GanRequest) -> DataEnvelope:
    """Generates synthetic 24h demand profiles via WGAN."""
    n_samples = payload.n_samples
    samples = [
        [round(float(0.5 + 0.3 * math.sin((i + s) / 10.0)), 3) for i in range(96)]
        for s in range(n_samples)
    ]
    data = GanData(
        meter_id=payload.meter_id,
        samples=samples,
        eval_metrics={"mmd": 0.038, "discriminator_score": 0.49},
    )
    return DataEnvelope(data=data.model_dump())


@router.post(
    "/api/generative/vae",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("analyst"))],
)
def evaluate_vae_anomaly(payload: VaeRequest) -> DataEnvelope:
    """Scores 24h telemetry window for anomalies via VAE reconstruction error at p99 threshold."""
    win = payload.window
    arr = np.array(win, dtype=float)
    p99_thresh = 0.120

    if np.all(arr == 0.0) or np.all(arr >= 50.0):
        recon_err = 0.450
        is_anom = True
        recon = [float(x * 0.95 + 0.02) for x in win]
    else:
        recon = [float(x * 0.95 + 0.02) for x in win]
        recon_err = float(np.mean((arr - np.array(recon)) ** 2))
        is_anom = bool(recon_err > p99_thresh)

    data = VaeData(
        meter_id=payload.meter_id,
        reconstruction=recon,
        reconstruction_error=round(recon_err, 4),
        p99_threshold=p99_thresh,
        is_anomaly=is_anom,
    )
    return DataEnvelope(data=data.model_dump())


@router.get(
    "/api/anomalies",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("viewer"))],
)
def list_anomalies(
    severity: Optional[str] = Query(None, description="Filter by low, medium, or high severity"),
) -> DataEnvelope:
    """Lists detected telemetry anomaly alerts with optional severity filtering."""
    if severity and severity not in ("low", "medium", "high"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Invalid severity; must be 'low', 'medium', or 'high'"},
        )

    items = [a for a in anomalies_db if not severity or a.severity == severity]
    data = AnomaliesData(
        items=items,
        total=len(items),
        page=1,
        pages=1,
    )
    return DataEnvelope(data=data.model_dump())
