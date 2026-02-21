"""Configuration management for the SER API service.

This module provides centralized configuration using pydantic-settings,
loading values from environment variables with sensible defaults.

Example:
    >>> from src.api.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.app_name)
    SER Service
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables.
    Environment variables use uppercase names matching the attribute names.
    
    Attributes:
        app_name: Name of the application for OpenAPI docs.
        app_version: API version string.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        model_id: Model identifier for inference.
        device: Device for model inference ("cpu" or "cuda").
        target_sample_rate: Target sample rate for audio preprocessing.
        max_duration_sec: Maximum allowed audio duration in seconds.
        default_window_sec: Default window size for timeline generation.
        default_hop_sec: Default hop size for timeline generation.
        default_pad_mode: Default padding mode for windowing.
        default_smoothing_method: Default smoothing method.
        default_hysteresis_min_run: Default hysteresis min run parameter.
        include_windows_default: Whether to include windows by default.
        include_scores_default: Whether to include scores by default.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application settings
    app_name: str = "SER Service"
    app_version: str = "1.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Model settings
    model_id: str = "baseline"
    device: Literal["cpu", "cuda"] = "cpu"
    
    # Audio settings
    target_sample_rate: int = 16000
    max_duration_sec: float = 600.0
    
    # Timeline defaults
    default_window_sec: float = 2.0
    default_hop_sec: float = 0.5
    default_pad_mode: Literal["none", "zero", "reflect"] = "zero"
    
    # Smoothing defaults
    default_smoothing_method: Literal["none", "majority", "hysteresis", "ema"] = "hysteresis"
    default_hysteresis_min_run: int = 3
    default_majority_window: int = 5
    default_ema_alpha: float = 0.6
    
    # Response defaults
    include_windows_default: bool = False
    include_scores_default: bool = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings instance with values from environment.
    """
    return Settings()
