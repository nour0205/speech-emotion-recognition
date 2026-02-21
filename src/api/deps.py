"""FastAPI dependencies for the SER API.

This module provides dependency injection functions for:
- Settings access
- Model management
- Audio configuration

Example:
    >>> from fastapi import Depends
    >>> from src.api.deps import get_settings
    
    >>> @app.get("/")
    >>> async def endpoint(settings = Depends(get_settings)):
    ...     return {"model": settings.model_id}
"""

import logging
from functools import lru_cache
from typing import Any

from audioio import AudioConfig
from model.registry import get_model

from .config import Settings, get_settings as _get_settings


logger = logging.getLogger(__name__)


# Re-export get_settings for dependency injection
get_settings = _get_settings


# =============================================================================
# Model Management
# =============================================================================


class ModelManager:
    """Manages model loading and access for the API.
    
    This class handles:
    - Lazy model loading on first request
    - Caching loaded models
    - Providing model access to endpoints
    
    Attributes:
        model_id: The model identifier.
        device: The device for inference.
        _model: Cached model instance.
    """
    
    def __init__(self, model_id: str, device: str) -> None:
        """Initialize the model manager.
        
        Args:
            model_id: Model identifier to load.
            device: Device for inference ("cpu" or "cuda").
        """
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._loaded = False
    
    def load(self) -> None:
        """Preload the model.
        
        Call this at startup to avoid cold start latency.
        """
        if not self._loaded:
            logger.info(
                "Loading model: model_id=%s device=%s",
                self.model_id,
                self.device,
            )
            self._model = get_model(model_id=self.model_id, device=self.device)
            self._loaded = True
            logger.info("Model loaded successfully")
    
    @property
    def model(self) -> Any:
        """Get the loaded model, loading if necessary.
        
        Returns:
            The loaded SER model instance.
        """
        if not self._loaded:
            self.load()
        return self._model
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded


# Global model manager instance
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance.
    
    Returns:
        The ModelManager instance.
        
    Raises:
        RuntimeError: If model manager is not initialized.
    """
    if _model_manager is None:
        raise RuntimeError(
            "Model manager not initialized. Call init_model_manager() first."
        )
    return _model_manager


def init_model_manager(settings: Settings) -> ModelManager:
    """Initialize the global model manager.
    
    Args:
        settings: Application settings.
        
    Returns:
        The initialized ModelManager instance.
    """
    global _model_manager
    _model_manager = ModelManager(
        model_id=settings.model_id,
        device=settings.device,
    )
    return _model_manager


# =============================================================================
# Audio Configuration
# =============================================================================


def get_audio_config(settings: Settings | None = None) -> AudioConfig:
    """Get audio configuration from settings.
    
    Args:
        settings: Application settings. If None, uses get_settings().
        
    Returns:
        AudioConfig with settings applied.
    """
    if settings is None:
        settings = get_settings()
    
    return AudioConfig(
        target_sample_rate=settings.target_sample_rate,
        max_duration_sec=settings.max_duration_sec,
    )
