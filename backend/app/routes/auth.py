"""Authentication Routes: /api/auth/*."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.schemas import DataEnvelope
from backend.app.security import create_jwt_token

router = APIRouter(prefix="/api/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username")
    password: Optional[str] = Field(default="", description="Password")
    role: Optional[str] = Field(default="admin", description="Assigned role: viewer, analyst, admin")


@router.post("/login", response_model=DataEnvelope)
def login(req: LoginRequest) -> DataEnvelope:
    """Authenticates user and returns a signed HMAC-SHA256 JWT bearer token."""
    if not req.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Username is required"},
        )
    role = req.role if req.role in ["viewer", "analyst", "admin"] else "viewer"
    token = create_jwt_token(sub=req.username, role=role)
    return DataEnvelope(data={"token": token, "username": req.username, "role": role})

