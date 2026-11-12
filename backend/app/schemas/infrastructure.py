"""Pydantic v2 schemas for Vision Infrastructure Inspection (M3) endpoints."""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel


class ClassifyData(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    baseline_majority_class: str


class Detection(BaseModel):
    bbox: List[float]
    label: str
    confidence: float


class DetectData(BaseModel):
    detections: List[Detection]
    count: int


class Segment(BaseModel):
    category: str
    polygon: List[List[float]]


class SegmentData(BaseModel):
    segments: List[Segment]
    encroachment_risk_score: float
