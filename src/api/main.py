"""FastAPI application for Speech Emotion Recognition.

This module provides the main FastAPI application with endpoints for:
- GET /health: Service health check
- POST /predict: Single-clip emotion prediction
- POST /timeline: Emotion timeline generation

Example:
    Run with uvicorn:
    
    $ uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    
    Or use the run script:
    
    $ ./scripts/run_api.sh
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from audioio import AudioConfig
from model.infer import predict_clip
from timeline import (
    MergeConfig,
    SmoothingConfig,
    WindowingConfig,
    generate_timeline,
)

from .config import Settings, get_settings
from .deps import get_audio_config, get_model_manager, init_model_manager
from .errors import (
    InvalidInputError,
    InvalidWindowingError,
    register_exception_handlers,
)
from .logging import add_middleware, setup_logging
from .schemas import (
    HealthResponse,
    PredictResponse,
    SegmentSchema,
    TimelineResponse,
    WindowSchema,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events.
    
    Startup:
        - Initialize logging
        - Load model to avoid cold start on first request
        
    Shutdown:
        - Clean up resources
    """
    settings = get_settings()
    
    # Setup logging
    setup_logging(settings.log_level)
    logger.info(
        "Starting %s v%s",
        settings.app_name,
        settings.app_version,
    )
    
    # Initialize and preload model
    model_manager = init_model_manager(settings)
    logger.info(
        "Preloading model: model_id=%s device=%s",
        settings.model_id,
        settings.device,
    )
    model_manager.load()
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Application shutdown")


# =============================================================================
# Application Factory
# =============================================================================


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        settings: Application settings. If None, loads from environment.
        
    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Speech Emotion Recognition API - Predict emotions from audio clips and generate emotion timelines.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Add CORS middleware (allow all in development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware (request ID, timing)
    add_middleware(app)
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Register routes
    register_routes(app)
    
    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes on the application.
    
    Args:
        app: The FastAPI application.
    """
    
    # =========================================================================
    # Health Endpoint
    # =========================================================================
    
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
        description="Check service health and model status.",
    )
    async def health() -> HealthResponse:
        """Health check endpoint.
        
        Returns:
            HealthResponse with status and model info.
        """
        settings = get_settings()
        model_manager = get_model_manager()
        
        return HealthResponse(
            status="ok" if model_manager.is_loaded else "loading",
            model_id=settings.model_id,
            device=settings.device,
        )
    
    # =========================================================================
    # Predict Endpoint
    # =========================================================================
    
    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["Prediction"],
        summary="Predict emotion from audio",
        description="Predict the emotion of a single audio clip.",
    )
    async def predict(
        file: Annotated[UploadFile, File(description="WAV audio file")],
        include_scores: Annotated[
            bool,
            Form(description="Include per-label probability scores"),
        ] = None,
    ) -> PredictResponse:
        """Predict emotion from an uploaded audio file.
        
        Args:
            file: WAV audio file to analyze.
            include_scores: Whether to include per-label scores.
                If None, uses default from settings.
        
        Returns:
            PredictResponse with emotion prediction.
        """
        settings = get_settings()
        
        # Use default if not specified
        if include_scores is None:
            include_scores = settings.include_scores_default
        
        # Read file bytes
        audio_bytes = await file.read()
        
        if not audio_bytes:
            raise InvalidInputError(
                message="Empty file uploaded",
                details={"filename": file.filename},
            )
        
        # Get audio config
        audio_config = get_audio_config(settings)
        
        # Log timing
        start_time = time.perf_counter()
        
        # Run prediction
        result = predict_clip(
            path_or_bytes=audio_bytes,
            audio_config=audio_config,
            model_id=settings.model_id,
            device=settings.device,
        )
        
        inference_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Prediction complete: emotion=%s confidence=%.3f inference_ms=%.2f",
            result.emotion,
            result.confidence,
            inference_ms,
        )
        
        return PredictResponse(
            emotion=result.emotion,
            confidence=float(result.confidence),
            scores=dict(result.scores) if include_scores else None,
            model_name=result.model_name,
            duration_sec=float(result.duration_sec),
        )
    
    # =========================================================================
    # Timeline Endpoint
    # =========================================================================
    
    @app.post(
        "/timeline",
        response_model=TimelineResponse,
        tags=["Prediction"],
        summary="Generate emotion timeline",
        description="Generate a timeline of emotion predictions for an audio file.",
    )
    async def timeline(
        file: Annotated[UploadFile, File(description="WAV audio file")],
        window_sec: Annotated[
            float | None,
            Form(description="Window duration in seconds", gt=0),
        ] = None,
        hop_sec: Annotated[
            float | None,
            Form(description="Hop/stride duration in seconds", gt=0),
        ] = None,
        pad_mode: Annotated[
            str | None,
            Form(description="Padding mode: none, zero, or reflect"),
        ] = None,
        smoothing_method: Annotated[
            str | None,
            Form(description="Smoothing method: none, majority, hysteresis, or ema"),
        ] = None,
        hysteresis_min_run: Annotated[
            int | None,
            Form(description="Min consecutive windows for emotion switch (hysteresis)", ge=1),
        ] = None,
        majority_window: Annotated[
            int | None,
            Form(description="Window size for majority voting (must be odd)", ge=1),
        ] = None,
        ema_alpha: Annotated[
            float | None,
            Form(description="EMA alpha coefficient (0-1)", gt=0, le=1),
        ] = None,
        include_windows: Annotated[
            bool,
            Form(description="Include per-window predictions in response"),
        ] = None,
        include_scores: Annotated[
            bool,
            Form(description="Include per-label probability scores"),
        ] = None,
    ) -> TimelineResponse:
        """Generate emotion timeline from an uploaded audio file.
        
        Args:
            file: WAV audio file to analyze.
            window_sec: Window duration in seconds.
            hop_sec: Hop/stride duration in seconds.
            pad_mode: Padding mode for partial windows.
            smoothing_method: Smoothing algorithm to apply.
            hysteresis_min_run: Hysteresis parameter.
            majority_window: Majority voting window size.
            ema_alpha: EMA smoothing alpha.
            include_windows: Include per-window predictions.
            include_scores: Include per-label scores.
        
        Returns:
            TimelineResponse with emotion segments and metadata.
        """
        settings = get_settings()
        
        # Apply defaults from settings
        if window_sec is None:
            window_sec = settings.default_window_sec
        if hop_sec is None:
            hop_sec = settings.default_hop_sec
        if pad_mode is None:
            pad_mode = settings.default_pad_mode
        if smoothing_method is None:
            smoothing_method = settings.default_smoothing_method
        if hysteresis_min_run is None:
            hysteresis_min_run = settings.default_hysteresis_min_run
        if majority_window is None:
            majority_window = settings.default_majority_window
        if ema_alpha is None:
            ema_alpha = settings.default_ema_alpha
        if include_windows is None:
            include_windows = settings.include_windows_default
        if include_scores is None:
            include_scores = settings.include_scores_default
        
        # Validate hop_sec <= window_sec
        if hop_sec > window_sec:
            raise InvalidWindowingError(
                message=f"hop_sec ({hop_sec}) cannot exceed window_sec ({window_sec})",
                details={"hop_sec": hop_sec, "window_sec": window_sec},
            )
        
        # Validate pad_mode
        valid_pad_modes = {"none", "zero", "reflect"}
        if pad_mode not in valid_pad_modes:
            raise InvalidWindowingError(
                message=f"Invalid pad_mode: {pad_mode}. Must be one of {valid_pad_modes}",
                details={"pad_mode": pad_mode, "valid_modes": list(valid_pad_modes)},
            )
        
        # Validate smoothing_method
        valid_smoothing_methods = {"none", "majority", "hysteresis", "ema"}
        if smoothing_method not in valid_smoothing_methods:
            raise InvalidInputError(
                message=f"Invalid smoothing_method: {smoothing_method}. Must be one of {valid_smoothing_methods}",
                details={
                    "smoothing_method": smoothing_method,
                    "valid_methods": list(valid_smoothing_methods),
                },
            )
        
        # Read file bytes
        audio_bytes = await file.read()
        
        if not audio_bytes:
            raise InvalidInputError(
                message="Empty file uploaded",
                details={"filename": file.filename},
            )
        
        # Build configurations
        audio_config = get_audio_config(settings)
        
        windowing_config = WindowingConfig(
            window_sec=window_sec,
            hop_sec=hop_sec,
            pad_mode=pad_mode,
        )
        
        smoothing_config = SmoothingConfig(
            method=smoothing_method,
            hysteresis_min_run=hysteresis_min_run,
            majority_window=majority_window,
            ema_alpha=ema_alpha,
        )
        
        merge_config = MergeConfig()
        
        # Log timing breakdown
        start_time = time.perf_counter()
        
        # Generate timeline
        result = generate_timeline(
            path_or_bytes=audio_bytes,
            audio_config=audio_config,
            windowing_config=windowing_config,
            model_id=settings.model_id,
            device=settings.device,
            smoothing_config=smoothing_config,
            merge_config=merge_config,
            include_windows=include_windows,
            include_scores=include_scores,
        )
        
        timeline_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Timeline generated: segments=%d duration_sec=%.2f timeline_ms=%.2f",
            len(result.segments),
            result.duration_sec,
            timeline_ms,
        )
        
        # Build response
        segments = [
            SegmentSchema(
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                emotion=seg.emotion,
                confidence=seg.confidence,
                scores=seg.scores if include_scores else None,
            )
            for seg in result.segments
        ]
        
        windows = None
        if include_windows and result.windows:
            windows = [
                WindowSchema(
                    index=w.index,
                    start_sec=w.start_sec,
                    end_sec=w.end_sec,
                    emotion=w.emotion,
                    confidence=w.confidence,
                    scores=w.scores if include_scores else None,
                )
                for w in result.windows
            ]
        
        return TimelineResponse(
            model_name=result.model_name,
            sample_rate=result.sample_rate,
            duration_sec=result.duration_sec,
            window_sec=result.window_sec,
            hop_sec=result.hop_sec,
            pad_mode=result.pad_mode,
            smoothing=result.smoothing,
            segments=segments,
            windows=windows,
        )


# =============================================================================
# Application Instance
# =============================================================================


# Create the application instance
app = create_app()
