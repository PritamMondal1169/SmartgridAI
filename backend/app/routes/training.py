"""Training Dynamics & Asynchronous Job Routes: /api/training/* and /api/jobs/*."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.rate_limiter import enforce_general_rate_limit
from backend.app.schemas import (
    CompareRequest,
    CompareResponseData,
    DataEnvelope,
    JobData,
)
from backend.app.security import require_role

router = APIRouter(tags=["Training"])

# Shared in-memory job store
jobs_store: Dict[str, Dict[str, Any]] = {}


@router.post(
    "/api/training/compare",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("admin"))],
)
def enqueue_training_compare(payload: CompareRequest) -> DataEnvelope:
    """Enqueues multi-optimizer, multi-seed training comparison background job."""
    job_id = f"job_{int(time.time() * 1000)}"
    jobs_store[job_id] = {
        "job_id": job_id,
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "result": None,
        "error": None,
        "config": payload.model_dump(),
    }
    data = CompareResponseData(job_id=job_id, state="queued")
    return DataEnvelope(data=data.model_dump())


@router.get(
    "/api/jobs/{job_id}",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("analyst"))],
)
def get_job_status(job_id: str) -> DataEnvelope:
    """Polls asynchronous training job lifecycle status."""
    job = jobs_store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Job '{job_id}' not found"},
        )

    # Simulate job progression to succeeded
    if job["state"] == "running":
        job["state"] = "succeeded"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        job["result"] = {
            "Adam": {"mae_mean": 0.042, "mae_std": 0.003},
            "SGD": {"mae_mean": 0.068, "mae_std": 0.007},
            "RMSprop": {"mae_mean": 0.046, "mae_std": 0.004},
        }

    return DataEnvelope(data=JobData(**job).model_dump())
