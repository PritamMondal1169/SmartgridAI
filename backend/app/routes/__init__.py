"""FastAPI routers package for SmartgridAI."""

from backend.app.routes.analytics import router as analytics_router
from backend.app.routes.ann import router as ann_router
from backend.app.routes.auth import router as auth_router
from backend.app.routes.export import router as export_router
from backend.app.routes.forecast import router as forecast_router
from backend.app.routes.generative import router as generative_router
from backend.app.routes.infrastructure import router as infrastructure_router
from backend.app.routes.models import router as models_router
from backend.app.routes.training import router as training_router
from backend.app.routes.vision import router as vision_router

__all__ = [
    "analytics_router",
    "ann_router",
    "auth_router",
    "export_router",
    "forecast_router",
    "generative_router",
    "infrastructure_router",
    "models_router",
    "training_router",
    "vision_router",
]

