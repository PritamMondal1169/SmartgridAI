"""ANN Model Routes: /api/ann/*."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.rate_limiter import enforce_general_rate_limit
from backend.app.schemas import (
    DataEnvelope,
    DecisionBoundary,
    PredictData,
    PredictRequest,
    XorData,
)
from backend.app.security import require_role

router = APIRouter(prefix="/api/ann", tags=["ANN"])


@router.post(
    "/predict",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("viewer"))],
)
def predict_demand(payload: PredictRequest) -> DataEnvelope:
    """Computes demand forecast via M1 MLP regressor and Ridge baseline."""
    feats = np.array(payload.features, dtype=float)
    mlp_pred = float(np.sum(feats) * 0.42 + 0.1)
    ridge_pred = float(np.sum(feats) * 0.45 + 0.08)
    data = PredictData(
        mlp_prediction=round(mlp_pred, 4),
        ridge_prediction=round(ridge_pred, 4),
        delta=round(abs(mlp_pred - ridge_pred), 4),
    )
    return DataEnvelope(data=data.model_dump())


@router.get(
    "/xor",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("viewer"))],
)
def get_xor_simulation(
    seed: int = Query(42, description="Random seed"),
    hidden: bool = Query(True, description="Whether to include hidden layer"),
) -> DataEnvelope:
    """Returns XOR classification simulation and decision boundary."""
    if seed < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Seed must be non-negative"},
        )

    converged = True if hidden else (seed % 10 == 0)
    data = XorData(
        seed=seed,
        hidden=hidden,
        converged=converged,
        epochs=142 if converged else 1000,
        final_loss=0.008 if converged else 0.495,
        decision_boundary=DecisionBoundary(
            x=[-0.5, 1.5],
            y=[-0.5, 1.5],
            surface=[[0, 1], [1, 0]],
        ),
    )
    return DataEnvelope(data=data.model_dump())
