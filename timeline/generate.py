"""Timeline generation orchestration for emotion recognition.

This module provides the main entry points for generating emotion timelines
from audio files or waveforms. It orchestrates the full pipeline:
1. Audio loading/preprocessing (Phase 1)
2. Audio segmentation into windows (Phase 2)
3. Per-window emotion inference (Phase 3)
4. Prediction smoothing (Phase 4)
5. Segment merging (Phase 4)

Example:
    >>> from timeline.generate import generate_timeline, generate_timeline_from_waveform
    >>> from timeline.smooth import SmoothingConfig
    >>> from timeline.merge import MergeConfig
    >>> from timeline.windowing import WindowingConfig
    
    # Generate timeline from file
    >>> result = generate_timeline("speech.wav")
    >>> print(result.to_dict())
    
    # Generate from waveform
    >>> import torch
    >>> waveform = torch.randn(1, 32000)
    >>> result = generate_timeline_from_waveform(waveform, sample_rate=16000)
"""

from pathlib import Path
from typing import Union

import torch

from audioio import AudioConfig, load_validate_preprocess
from model.infer import predict_waveform
from model.labels import CANONICAL_LABELS

from .merge import MergeConfig, merge_windows_to_segments
from .schema import Segment, TimelineResult, WindowPrediction
from .smooth import SmoothingConfig, smooth_windows
from .windowing import WindowingConfig, segment_audio


def generate_timeline_from_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    windowing_config: WindowingConfig | None = None,
    model_id: str = "baseline",
    device: str = "cpu",
    smoothing_config: SmoothingConfig | None = None,
    merge_config: MergeConfig | None = None,
    include_windows: bool = False,
    include_scores: bool = False,
) -> TimelineResult:
    """Generate emotion timeline from a preprocessed waveform.
    
    This function takes an already-loaded waveform (assumed to be
    preprocessed: mono, float32, correct sample rate) and generates
    a complete emotion timeline.
    
    Args:
        waveform: Audio waveform tensor with shape [1, T] (mono, float32).
        sample_rate: Sample rate of the waveform.
        windowing_config: Configuration for audio windowing.
            If None, uses default WindowingConfig().
        model_id: Identifier of the model to use for inference.
            Default "baseline".
        device: Device for model inference ("cpu" or "cuda").
        smoothing_config: Configuration for prediction smoothing.
            If None, uses default SmoothingConfig().
        merge_config: Configuration for segment merging.
            If None, uses default MergeConfig().
        include_windows: Whether to include per-window predictions
            in the result. Default False.
        include_scores: Whether to include per-label scores in
            windows and segments. Default False.
            
    Returns:
        TimelineResult with emotion segments and optional windows.
        
    Raises:
        InferenceError: If model inference fails.
        WindowingRuntimeError: If windowing fails.
        
    Example:
        >>> import torch
        >>> from timeline.generate import generate_timeline_from_waveform
        >>> waveform = torch.randn(1, 48000)  # 3 seconds at 16kHz
        >>> result = generate_timeline_from_waveform(waveform, 16000)
        >>> print(f"Found {len(result.segments)} emotion segments")
    """
    # Use default configs if not provided
    if windowing_config is None:
        windowing_config = WindowingConfig()
    if smoothing_config is None:
        smoothing_config = SmoothingConfig()
    if merge_config is None:
        merge_config = MergeConfig()
    
    # Compute duration
    num_samples = waveform.shape[1]
    duration_sec = num_samples / sample_rate
    
    # Step 1: Segment audio into windows (Phase 2)
    raw_windows = segment_audio(waveform, sample_rate, windowing_config)
    
    # Track if any window was padded
    is_padded = any(w.get("is_padded", False) for w in raw_windows)
    
    # Step 2: Run inference on each window (Phase 3)
    window_predictions = _predict_windows(
        raw_windows=raw_windows,
        sample_rate=sample_rate,
        model_id=model_id,
        device=device,
        include_scores=include_scores,
    )
    
    # Get model name from first prediction (all should be same model)
    model_name = model_id
    if window_predictions:
        # We'll get it from the prediction results
        pass
    
    # Step 3: Apply smoothing (Phase 4)
    smoothed_windows = smooth_windows(
        windows=window_predictions,
        config=smoothing_config,
        canonical_labels=list(CANONICAL_LABELS),
    )
    
    # Step 4: Merge into segments (Phase 4)
    segments = merge_windows_to_segments(
        windows=smoothed_windows,
        windowing_config=windowing_config,
        merge_config=merge_config,
        duration_sec=duration_sec,
    )
    
    # Build result
    result = TimelineResult(
        model_name=model_name,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        window_sec=windowing_config.window_sec,
        hop_sec=windowing_config.hop_sec,
        pad_mode=windowing_config.pad_mode,
        smoothing=smoothing_config.to_dict(),
        segments=segments,
        windows=smoothed_windows if include_windows else None,
        is_padded_timeline=is_padded,
        merge_config=merge_config.to_dict(),
    )
    
    return result


def generate_timeline(
    path_or_bytes: Union[str, Path, bytes],
    audio_config: AudioConfig | None = None,
    windowing_config: WindowingConfig | None = None,
    model_id: str = "baseline",
    device: str = "cpu",
    smoothing_config: SmoothingConfig | None = None,
    merge_config: MergeConfig | None = None,
    include_windows: bool = False,
    include_scores: bool = False,
) -> TimelineResult:
    """Generate emotion timeline from an audio file or bytes.
    
    This is the main entry point for timeline generation. It handles
    the complete pipeline from loading audio to generating the final
    timeline with emotion segments.
    
    Args:
        path_or_bytes: Path to audio file (str/Path) or raw WAV bytes.
        audio_config: Audio loading/preprocessing configuration.
            If None, uses default AudioConfig().
        windowing_config: Configuration for audio windowing.
            If None, uses default WindowingConfig().
        model_id: Identifier of the model to use for inference.
            Default "baseline".
        device: Device for model inference ("cpu" or "cuda").
        smoothing_config: Configuration for prediction smoothing.
            If None, uses default SmoothingConfig().
        merge_config: Configuration for segment merging.
            If None, uses default MergeConfig().
        include_windows: Whether to include per-window predictions
            in the result. Default False.
        include_scores: Whether to include per-label scores in
            windows and segments. Default False.
            
    Returns:
        TimelineResult with emotion segments and optional windows.
        
    Raises:
        AudioDecodeError: If audio cannot be loaded.
        AudioValidationError: If audio fails validation.
        AudioPreprocessError: If preprocessing fails.
        InferenceError: If model inference fails.
        
    Example:
        >>> from timeline.generate import generate_timeline
        >>> result = generate_timeline("speech.wav")
        >>> for segment in result.segments:
        ...     print(f"{segment.start_sec:.2f}s - {segment.end_sec:.2f}s: {segment.emotion}")
    """
    # Use default audio config if not provided
    if audio_config is None:
        audio_config = AudioConfig()
    
    # Step 1: Load, validate, preprocess audio (Phase 1)
    waveform, sample_rate = load_validate_preprocess(path_or_bytes, audio_config)
    
    # Generate timeline from waveform
    return generate_timeline_from_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        windowing_config=windowing_config,
        model_id=model_id,
        device=device,
        smoothing_config=smoothing_config,
        merge_config=merge_config,
        include_windows=include_windows,
        include_scores=include_scores,
    )


def _predict_windows(
    raw_windows: list[dict],
    sample_rate: int,
    model_id: str,
    device: str,
    include_scores: bool,
) -> list[WindowPrediction]:
    """Run inference on each window and build WindowPrediction list.
    
    Args:
        raw_windows: List of window dicts from segment_audio().
        sample_rate: Audio sample rate.
        model_id: Model identifier.
        device: Device for inference.
        include_scores: Whether to include per-label scores.
        
    Returns:
        List of WindowPrediction objects.
    """
    predictions: list[WindowPrediction] = []
    
    for window in raw_windows:
        # Get window waveform
        window_waveform = window["waveform"]
        
        # Run inference
        result = predict_waveform(
            waveform=window_waveform,
            sample_rate=sample_rate,
            model_id=model_id,
            device=device,
        )
        
        # Build WindowPrediction
        prediction = WindowPrediction(
            index=window["index"],
            start_sec=window["start_sec"],
            end_sec=window["end_sec"],
            emotion=result.emotion,
            confidence=result.confidence,
            scores=dict(result.scores) if include_scores else None,
            is_padded=window.get("is_padded", False),
        )
        predictions.append(prediction)
    
    return predictions
