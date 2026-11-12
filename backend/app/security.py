"""JWT Authentication and Role-Based Access Control (RBAC) Security Module."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, Header, HTTPException, status

from backend.app.config import JWT_SECRET


def b64_encode(data: bytes) -> str:
    """URL-safe base64 encoding without trailing padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def b64_decode(data: str) -> bytes:
    """URL-safe base64 decoding with restored padding."""
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_jwt_token(
    sub: str,
    role: str = "viewer",
    exp_delta: timedelta = timedelta(hours=24),
    secret: str = JWT_SECRET,
) -> str:
    """Generates standard HMAC-SHA256 JWT token."""
    header = {"alg": "HS256", "typ": "JWT"}
    exp_ts = int((datetime.now(timezone.utc) + exp_delta).timestamp())
    payload = {"sub": sub, "role": role, "exp": exp_ts}

    header_b64 = b64_encode(json.dumps(header).encode("utf-8"))
    payload_b64 = b64_encode(json.dumps(payload).encode("utf-8"))
    to_sign = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), to_sign, hashlib.sha256).digest()
    sig_b64 = b64_encode(sig)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_jwt_token(token: str, secret: str = JWT_SECRET) -> Dict[str, Any]:
    """Decodes and validates HMAC-SHA256 JWT token signature and expiration."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT token format")

    header_b64, payload_b64, sig_b64 = parts
    to_sign = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), to_sign, hashlib.sha256).digest()
    actual_sig = b64_decode(sig_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid JWT signature")

    payload = json.loads(b64_decode(payload_b64).decode("utf-8"))
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if payload.get("exp", 0) < now_ts:
        raise TimeoutError("Token has expired")

    return payload


def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """FastAPI dependency extracting and validating JWT Bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid authorization header"},
        )

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_jwt_token(token)
        return payload
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "TOKEN_EXPIRED", "message": "Token has expired"},
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Invalid JWT token or signature"},
        )


def require_role(min_role: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """FastAPI dependency factory enforcing strict RBAC hierarchy: viewer < analyst < admin."""
    role_weights = {"viewer": 1, "analyst": 2, "admin": 3}
    required_weight = role_weights.get(min_role.lower(), 1)

    def _role_checker(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = user.get("role", "viewer").lower()
        user_weight = role_weights.get(user_role, 0)
        if user_weight < required_weight:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "FORBIDDEN", "message": f"{min_role.capitalize()} privilege required"},
            )
        return user

    return _role_checker
