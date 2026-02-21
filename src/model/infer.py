"""Inference functions for Speech Emotion Recognition.

This module provides high-level inference functions for emotion prediction
on audio files and waveforms.

Example:
    >>> from model.infer import predict_clip, predict_waveform
    >>> import torch
    
    # Predict from file
    >>> result = predict_clip("speech.wav")
    >>> print(result.emotion, result.confidence)
    
    # Predict from waveform
    >>> waveform = torch.randn(1, 16000)
    >>> result = predict_waveform(waveform, sample_rate=16000)
"""

from pathlib import Path
from typing import Union

import torch

from audioio import AudioConfig, load_validate_preprocess

from .errors import InferenceError
from .labels import map_raw_to_canonical
from .registry import get_model
from .types import PredictionResult


def predict_clip(
    path_or_bytes: Union[str, Path, bytes],
    audio_config: AudioConfig | None = None,
    model_id: str = "baseline",
    device: str = "cpu",
) -> PredictionResult:
    """Predict emotion from an audio file or bytes.
    
    This function:
    1. Loads and preprocesses the audio using audioio pipeline
    2. Runs inference using the specified model
    3. Maps raw model outputs to canonical labels
    4. Returns a structured prediction result
    
    Args:
        path_or_bytes: Path to audio file (str/Path) or raw WAV bytes.
        audio_config: Audio preprocessing configuration.
            If None, uses default AudioConfig().
        model_id: Identifier of the model to use. Default "baseline".
        device: Device to run inference on ("cpu" or "cuda").
        
    Returns:
        PredictionResult with predicted emotion, confidence, and scores.
        
    Raises:
        AudioDecodeError: If audio cannot be loaded.
        AudioValidationError: If audio fails validation.
        AudioPreprocessError: If preprocessing fails.
        ModelLoadError: If model cannot be loaded.
        InferenceError: If inference fails.
        
    Examples:
        >>> result = predict_clip("speech.wav")
        >>> print(f"Emotion: {result.emotion}, Confidence: {result.confidence:.2%}")
        
        >>> with open("speech.wav", "rb") as f:
        ...     result = predict_clip(f.read())
    """
    # Load, validate, and preprocess audio
    waveform, sample_rate = load_validate_preprocess(path_or_bytes, audio_config)
    
    # Compute duration
    duration_sec = waveform.shape[1] / sample_rate
    
    # Run inference
    result = predict_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        model_id=model_id,
        device=device,
    )
    
    # Update duration (predict_waveform computes it from waveform too,
    # but we can update to be precise)
    result.duration_sec = duration_sec
    
    return result


def predict_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    model_id: str = "baseline",
    device: str = "cpu",
) -> PredictionResult:
    """Predict emotion from a preprocessed waveform.
    
    This function assumes the waveform is already preprocessed
    (mono, float32, correct sample rate). Use this when you have
    already loaded and preprocessed audio, or for batch processing.
    
    Args:
        waveform: Audio waveform tensor with shape [1, T] (mono, float32).
        sample_rate: Sample rate of the waveform.
        model_id: Identifier of the model to use. Default "baseline".
        device: Device to run inference on ("cpu" or "cuda").
        
    Returns:
        PredictionResult with predicted emotion, confidence, and scores.
        
    Raises:
        ModelLoadError: If model cannot be loaded.
        InferenceError: If inference fails.
        
    Examples:
        >>> import torch
        >>> waveform = torch.randn(1, 16000)  # 1 second at 16kHz
        >>> result = predict_waveform(waveform, sample_rate=16000)
        >>> print(result.emotion)
    """
    # Validate waveform
    if not isinstance(waveform, torch.Tensor):
        raise InferenceError(
            message=f"Waveform must be torch.Tensor, got {type(waveform).__name__}",
            code="INVALID_INPUT",
            details={"type": type(waveform).__name__},
        )
    
    if waveform.dim() != 2 or waveform.shape[0] != 1:
        raise InferenceError(
            message=f"Waveform must have shape [1, T], got {list(waveform.shape)}",
            code="INVALID_INPUT",
            details={"shape": list(waveform.shape)},
        )
    
    if waveform.dtype != torch.float32:
        raise InferenceError(
            message=f"Waveform must be float32, got {waveform.dtype}",
            code="INVALID_INPUT",
            details={"dtype": str(waveform.dtype)},
        )
    
    # Get model from registry (cached)
    model = get_model(model_id=model_id, device=device)
    
    # Run inference to get raw scores
    raw_scores = model.predict(waveform, sample_rate)
    
    # Map to canonical labels
    canonical_scores = map_raw_to_canonical(raw_scores, normalize=True)
    
    # Find predicted emotion (argmax)
    predicted_emotion = max(canonical_scores.keys(), key=lambda k: canonical_scores[k])
    confidence = canonical_scores[predicted_emotion]
    
    # Find raw label (argmax of raw scores)
    raw_label = max(raw_scores.keys(), key=lambda k: raw_scores[k]) if raw_scores else None
    
    # Compute duration
    duration_sec = waveform.shape[1] / sample_rate
    
    return PredictionResult(
        emotion=predicted_emotion,
        confidence=confidence,
        scores=canonical_scores,
        model_name=model.name,
        raw_label=raw_label,
        raw_scores=raw_scores,
        duration_sec=duration_sec,
    )


def predict_batch_waveforms(
    waveforms: list[torch.Tensor],
    sample_rate: int,
    model_id: str = "baseline",
    device: str = "cpu",
) -> list[PredictionResult]:
    """Predict emotions for multiple waveforms.
    
    This is a helper function for batch processing. Note that this
    processes waveforms sequentially (not true batching) but reuses
    the loaded model. True batching may be added in future versions.
    
    Args:
        waveforms: List of waveform tensors, each with shape [1, T].
        sample_rate: Sample rate of all waveforms (must be the same).
        model_id: Identifier of the model to use.
        device: Device to run inference on.
        
    Returns:
        List of PredictionResult objects, one per input waveform.
        
    Note:
        This function is provided as a helper for Phase 4 timeline processing.
        It processes waveforms sequentially but benefits from model caching.
    """
    results = []
    for waveform in waveforms:
        result = predict_waveform(
            waveform=waveform,
            sample_rate=sample_rate,
            model_id=model_id,
            device=device,
        )
        results.append(result)
    return results
