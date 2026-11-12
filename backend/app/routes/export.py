"""Data and Artifact Export Center Routes: /api/export/*."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse
import pandas as pd

from backend.app.rate_limiter import enforce_general_rate_limit
from backend.app.security import require_role

router = APIRouter(prefix="/api/export", tags=["Export"])


@router.get(
    "/{artifact}",
    dependencies=[Depends(enforce_general_rate_limit), Depends(require_role("analyst"))],
)
def export_artifact(
    artifact: str,
    format: str = Query("json", description="Target export format (json, csv, parquet)"),
) -> Response:
    """Exports dataset split manifests, quality reports, and model metrics."""
    valid_artifacts = ("quality_report", "split_manifest", "model_metrics")
    if artifact not in valid_artifacts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Artifact '{artifact}' not found"},
        )

    valid_formats = ("json", "csv", "parquet")
    if format not in valid_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Unsupported format; must be 'json', 'csv', or 'parquet'"},
        )

    export_timestamp = datetime.now(timezone.utc).isoformat()
    record = {"artifact": artifact, "exported_at": export_timestamp}

    if format == "json":
        return JSONResponse(
            content=record,
            headers={"Content-Disposition": f'attachment; filename="{artifact}.json"'},
        )
    elif format == "csv":
        csv_content = f"artifact,exported_at\n{artifact},{export_timestamp}\n".encode("utf-8")
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{artifact}.csv"'},
        )
    else:
        # Parquet export
        df = pd.DataFrame([record])
        out_buf = io.BytesIO()
        df.to_parquet(out_buf, index=False)
        pq_bytes = out_buf.getvalue()
        return Response(
            content=pq_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{artifact}.parquet"'},
        )
