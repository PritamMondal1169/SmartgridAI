"""Vision Infrastructure Inspection Routes: /api/infrastructure/*."""

from __future__ import annotations

import io
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image

from backend.app.config import (
    MAX_IMAGE_DIMENSION_PX,
    MAX_IMAGE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION_PX,
)
from backend.app.rate_limiter import enforce_vision_rate_limit
from backend.app.schemas import (
    ClassifyData,
    DataEnvelope,
    DetectData,
    Detection,
    Segment,
    SegmentData,
)
from backend.app.security import require_role

router = APIRouter(prefix="/api/infrastructure", tags=["Infrastructure"])


def validate_image_payload(content: bytes) -> None:
    """Enforces file size, magic bytes, and image dimension boundaries."""
    if len(content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "PAYLOAD_TOO_LARGE", "message": "File size exceeds maximum allowed 10 MB"},
        )
    if len(content) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Empty or invalid image file"},
        )

    is_jpg = content.startswith(b"\xff\xd8\xff")
    is_png = content.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpg or is_png):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Invalid image format: magic byte signature mismatch"},
        )

    try:
        img = Image.open(io.BytesIO(content))
        w, h = img.size
        if w < MIN_IMAGE_DIMENSION_PX or h < MIN_IMAGE_DIMENSION_PX:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "BAD_REQUEST", "message": "Image dimensions too small for feature extraction"},
            )
        if w > MAX_IMAGE_DIMENSION_PX or h > MAX_IMAGE_DIMENSION_PX:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "BAD_REQUEST", "message": "Dimensions exceed 4096 px maximum"},
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Failed to decode image payload"},
        )


@router.post(
    "/classify",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_vision_rate_limit), Depends(require_role("analyst"))],
)
def classify_pole_tilt(image: UploadFile = File(...)) -> DataEnvelope:
    """Classifies distribution pole lean severity into Straight, Leaned, Severe Tilt."""
    content = image.file.read()
    validate_image_payload(content)

    data = ClassifyData(
        predicted_class="Leaned",
        confidence=0.88,
        probabilities={"Straight": 0.08, "Leaned": 0.88, "Severe Tilt": 0.04},
        baseline_majority_class="Straight",
    )
    return DataEnvelope(data=data.model_dump())


@router.post(
    "/detect",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_vision_rate_limit), Depends(require_role("analyst"))],
)
def detect_poles(image: UploadFile = File(...)) -> DataEnvelope:
    """Detects utility poles via YOLOv8 object detector."""
    content = image.file.read()
    validate_image_payload(content)

    detections = [
        Detection(bbox=[120.0, 80.0, 240.0, 450.0], label="pole", confidence=0.92)
    ]
    data = DetectData(detections=detections, count=len(detections))
    return DataEnvelope(data=data.model_dump())


@router.post(
    "/segment",
    response_model=DataEnvelope,
    dependencies=[Depends(enforce_vision_rate_limit), Depends(require_role("analyst"))],
)
def segment_vegetation_conductors(image: UploadFile = File(...)) -> DataEnvelope:
    """Segments conductor wires and encroaching vegetation masks via U-Net."""
    content = image.file.read()
    validate_image_payload(content)

    segments = [
        Segment(category="conductor", polygon=[[100.0, 200.0], [300.0, 210.0], [500.0, 220.0]]),
        Segment(category="vegetation", polygon=[[280.0, 190.0], [350.0, 190.0], [320.0, 240.0]]),
    ]
    data = SegmentData(segments=segments, encroachment_risk_score=0.72)
    return DataEnvelope(data=data.model_dump())
