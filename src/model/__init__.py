"""Model module for Speech Emotion Recognition inference.

This module provides:
- Canonical emotion label definitions and mapping
- Model registry for loading pretrained SER models
- Inference functions for single clips and waveforms

Example:
    >>> from model import predict_clip, predict_waveform, CANONICAL_LABELS
    >>> result = predict_clip("speech.wav")
    >>> print(result["emotion"], result["confidence"])
    happy 0.92
    
    >>> import torch
    >>> waveform = torch.randn(1, 16000)  # 1 second at 16kHz
    >>> result = predict_waveform(waveform, sample_rate=16000)
"""

from .errors import InferenceError, ModelError, ModelLoadError
from .infer import predict_clip, predict_waveform
from .labels import CANONICAL_LABELS, map_raw_to_canonical
from .registry import get_model
from .types import PredictionResult

__all__ = [
    # Main inference functions
    "predict_clip",
    "predict_waveform",
    # Labels
    "CANONICAL_LABELS",
    "map_raw_to_canonical",
    # Registry
    "get_model",
    # Types
    "PredictionResult",
    # Errors
    "ModelError",
    "ModelLoadError",
    "InferenceError",
]
