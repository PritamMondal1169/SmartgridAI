"""In-Memory Rate Limiting Tracker and Enforcer for SmartgridAI API."""

from __future__ import annotations

import time
from typing import Dict, List, Optional
from fastapi import Header, HTTPException, Request, status

from backend.app.config import (
    GENERAL_RATE_LIMIT,
    RATE_LIMIT_WINDOW_SECONDS,
    VISION_RATE_LIMIT,
)


class RateLimitTracker:
    """Sliding-window in-memory request counter for rate limiting."""

    def __init__(self) -> None:
        self.general_counts: Dict[str, List[float]] = {}
        self.vision_counts: Dict[str, List[float]] = {}

    def reset(self) -> None:
        """Clears all rate limit windows (used between test runs)."""
        self.general_counts.clear()
        self.vision_counts.clear()

    def check_general(
        self,
        key: str,
        limit: int = GENERAL_RATE_LIMIT,
        window: float = RATE_LIMIT_WINDOW_SECONDS,
    ) -> bool:
        """Evaluates whether key is within general request limit."""
        now = time.time()
        calls = self.general_counts.setdefault(key, [])
        self.general_counts[key] = [t for t in calls if now - t < window]
        if len(self.general_counts[key]) >= limit:
            return False
        self.general_counts[key].append(now)
        return True

    def check_vision(
        self,
        key: str,
        limit: int = VISION_RATE_LIMIT,
        window: float = RATE_LIMIT_WINDOW_SECONDS,
    ) -> bool:
        """Evaluates whether key is within vision upload request limit."""
        now = time.time()
        calls = self.vision_counts.setdefault(key, [])
        self.vision_counts[key] = [t for t in calls if now - t < window]
        if len(self.vision_counts[key]) >= limit:
            return False
        self.vision_counts[key].append(now)
        return True


rate_limiter = RateLimitTracker()


def enforce_general_rate_limit(
    authorization: Optional[str] = Header(None),
) -> None:
    """Enforces general 60 req/min rate limit."""
    client_key = authorization or "anonymous"
    if not rate_limiter.check_general(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "RATE_LIMIT_EXCEEDED", "message": "General API rate limit (60 req/min) exceeded"},
            headers={"Retry-After": "60"},
        )


def enforce_vision_rate_limit(
    authorization: Optional[str] = Header(None),
) -> None:
    """Enforces vision 10 uploads/min rate limit."""
    client_key = authorization or "anonymous"
    if not rate_limiter.check_vision(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "RATE_LIMIT_EXCEEDED", "message": "Vision upload rate limit (10 uploads/min) exceeded"},
            headers={"Retry-After": "60"},
        )
