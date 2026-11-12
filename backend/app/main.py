"""SmartgridAI Modular FastAPI Application Package Entrypoint."""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.routes import (
    analytics_router,
    ann_router,
    auth_router,
    export_router,
    forecast_router,
    generative_router,
    infrastructure_router,
    models_router,
    training_router,
    vision_router,
)
from backend.app.schemas import ErrorDetail, ErrorEnvelope

app = FastAPI(
    title="SmartgridAI Inference & Analytics API",
    description="Coursework demonstration backend service for grid analytics, machine learning inference, and inspection triage.",
    version="1.0.0",
)

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Standardizes all HTTPExceptions into ErrorEnvelope schema."""
    code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        413: "PAYLOAD_TOO_LARGE",
        422: "UNPROCESSABLE_ENTITY",
        429: "RATE_LIMIT_EXCEEDED",
    }
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        payload = ErrorEnvelope(
            error=ErrorDetail(
                code=exc.detail["code"],
                message=exc.detail.get("message", "An error occurred"),
                detail=exc.detail.get("detail"),
            )
        ).model_dump()
    else:
        payload = ErrorEnvelope(
            error=ErrorDetail(
                code=code_map.get(exc.status_code, "INTERNAL_ERROR"),
                message=str(exc.detail) if exc.detail else "An error occurred",
            )
        ).model_dump()

    return JSONResponse(
        status_code=exc.status_code,
        content=payload,
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Standardizes Pydantic RequestValidationErrors into 422 ErrorEnvelope."""
    err_msgs = [f"{e.get('loc', [])}: {e.get('msg', '')}" for e in exc.errors()]
    payload = ErrorEnvelope(
        error=ErrorDetail(
            code="UNPROCESSABLE_ENTITY",
            message=f"Request schema validation failed: {'; '.join(err_msgs[:3])}",
            detail={"errors": [str(e) for e in exc.errors()]},
        )
    ).model_dump()

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=payload,
    )


# Mount All Module Routers
app.include_router(auth_router)
app.include_router(analytics_router)
app.include_router(ann_router)
app.include_router(training_router)
app.include_router(infrastructure_router)
app.include_router(forecast_router)
app.include_router(generative_router)
app.include_router(models_router)
app.include_router(vision_router)
app.include_router(export_router)

# Mount Frontend Static SPA if built
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        target = frontend_dist / full_path
        if full_path and target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(frontend_dist / "index.html")
