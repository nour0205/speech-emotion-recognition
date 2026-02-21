"""Pydantic schemas for API request/response models.

This module defines all the request and response schemas used by the
API endpoints, ensuring consistent serialization and validation.

Example:
    >>> from src.api.schemas import PredictResponse
    >>> response = PredictResponse(
    ...     emotion="happy",
    ...     confidence=0.85,
    ...     model_name="baseline",
    ...     duration_sec=2.5
    ... )
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Health Endpoint
# =============================================================================


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    
    status: str = Field(
        default="ok",
        description="Service status",
        examples=["ok"],
    )
    model_id: str = Field(
        description="Loaded model identifier",
        examples=["baseline"],
    )
    device: str = Field(
        description="Device used for inference",
        examples=["cpu", "cuda"],
    )


# =============================================================================
# Predict Endpoint
# =============================================================================


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint (single-clip emotion)."""
    
    emotion: str = Field(
        description="Predicted canonical emotion label",
        examples=["happy", "sad", "angry", "neutral"],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted emotion",
        examples=[0.85],
    )
    scores: dict[str, float] | None = Field(
        default=None,
        description="Per-label probability scores (if include_scores=true)",
        examples=[{"happy": 0.85, "sad": 0.05, "angry": 0.05, "neutral": 0.05}],
    )
    model_name: str = Field(
        description="Name of the model used for prediction",
        examples=["speechbrain-iemocap"],
    )
    duration_sec: float = Field(
        ge=0.0,
        description="Duration of the input audio in seconds",
        examples=[2.5],
    )


# =============================================================================
# Timeline Endpoint
# =============================================================================


class SegmentSchema(BaseModel):
    """Schema for a single emotion segment in the timeline."""
    
    start_sec: float = Field(
        ge=0.0,
        description="Segment start time in seconds",
        examples=[0.0],
    )
    end_sec: float = Field(
        ge=0.0,
        description="Segment end time in seconds",
        examples=[2.5],
    )
    emotion: str = Field(
        description="Predicted emotion for this segment",
        examples=["happy"],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Average confidence score for the segment",
        examples=[0.85],
    )
    scores: dict[str, float] | None = Field(
        default=None,
        description="Average per-label probability scores",
    )


class WindowSchema(BaseModel):
    """Schema for a single window prediction."""
    
    index: int = Field(
        ge=0,
        description="Window index (0-based)",
        examples=[0],
    )
    start_sec: float = Field(
        ge=0.0,
        description="Window start time in seconds",
        examples=[0.0],
    )
    end_sec: float = Field(
        ge=0.0,
        description="Window end time in seconds",
        examples=[2.0],
    )
    emotion: str = Field(
        description="Predicted emotion for this window",
        examples=["happy"],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the window",
        examples=[0.85],
    )
    scores: dict[str, float] | None = Field(
        default=None,
        description="Per-label probability scores",
    )


class SmoothingInfoSchema(BaseModel):
    """Schema for smoothing configuration applied."""
    
    method: str = Field(
        description="Smoothing method used",
        examples=["hysteresis"],
    )
    hysteresis_min_run: int | None = Field(
        default=None,
        description="Hysteresis min run parameter (if method=hysteresis)",
    )
    majority_window: int | None = Field(
        default=None,
        description="Majority window size (if method=majority)",
    )
    ema_alpha: float | None = Field(
        default=None,
        description="EMA alpha coefficient (if method=ema)",
    )


class TimelineResponse(BaseModel):
    """Response schema for /timeline endpoint."""
    
    model_name: str = Field(
        description="Name of the model used for prediction",
        examples=["speechbrain-iemocap"],
    )
    sample_rate: int = Field(
        description="Audio sample rate in Hz",
        examples=[16000],
    )
    duration_sec: float = Field(
        ge=0.0,
        description="Total audio duration in seconds",
        examples=[10.5],
    )
    window_sec: float = Field(
        gt=0.0,
        description="Window duration in seconds",
        examples=[2.0],
    )
    hop_sec: float = Field(
        gt=0.0,
        description="Hop/stride duration in seconds",
        examples=[0.5],
    )
    pad_mode: str = Field(
        description="Padding mode used",
        examples=["zero"],
    )
    smoothing: dict[str, Any] = Field(
        description="Smoothing configuration applied",
        examples=[{"method": "hysteresis", "hysteresis_min_run": 3}],
    )
    segments: list[SegmentSchema] = Field(
        description="List of emotion segments",
    )
    windows: list[WindowSchema] | None = Field(
        default=None,
        description="Per-window predictions (if include_windows=true)",
    )


# =============================================================================
# Error Response
# =============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    code: str = Field(
        description="Error code for programmatic handling",
        examples=["INVALID_AUDIO", "MODEL_LOAD_FAILED"],
    )
    message: str = Field(
        description="Human-readable error message",
        examples=["Failed to decode audio file"],
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error context",
    )


class ApiErrorResponse(BaseModel):
    """Standard error response wrapper."""
    
    error: ErrorDetail = Field(
        description="Error details",
    )
