"""SmartgridAI Backend Application Configuration."""

import os

JWT_SECRET: str = os.getenv("JWT_SECRET", "test-secret-key-smartgridai-2026-opaque-box")
JWT_ALGORITHM: str = "HS256"

GENERAL_RATE_LIMIT: int = 60
VISION_RATE_LIMIT: int = 10
RATE_LIMIT_WINDOW_SECONDS: float = 60.0

MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION_PX: int = 4096
MIN_IMAGE_DIMENSION_PX: int = 2
