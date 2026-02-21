"""Schema definitions for timeline emotion prediction output.

This module defines the data structures used for timeline-based emotion
recognition results, including window predictions, segments, and the
overall timeline result.

Example:
    >>> from timeline.schema import TimelineResult, Segment, WindowPrediction
    >>> segment = Segment(start_sec=0.0, end_sec=2.0, emotion="happy", confidence=0.85)
    >>> result = TimelineResult(
    ...     model_name="baseline",
    ...     sample_rate=16000,
    ...     duration_sec=10.0,
    ...     window_sec=2.0,
    ...     hop_sec=0.5,
    ...     pad_mode="zero",
    ...     smoothing={"method": "hysteresis", "hysteresis_min_run": 3},
    ...     segments=[segment],
    ... )
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WindowPrediction:
    """Prediction result for a single window in the timeline.
    
    Attributes:
        index: Window index (0-based).
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        emotion: Predicted canonical emotion label.
        confidence: Confidence score for the predicted emotion (0.0 to 1.0).
        scores: Optional probability scores for all canonical labels.
        is_padded: Whether this window was padded.
    """
    
    index: int
    start_sec: float
    end_sec: float
    emotion: str
    confidence: float
    scores: dict[str, float] | None = None
    is_padded: bool = False
    
    def to_dict(self, include_scores: bool = True) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Args:
            include_scores: Whether to include the scores dict.
            
        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "index": self.index,
            "start_sec": round(self.start_sec, 6),
            "end_sec": round(self.end_sec, 6),
            "emotion": self.emotion,
            "confidence": round(self.confidence, 6),
        }
        if include_scores and self.scores is not None:
            result["scores"] = {k: round(v, 6) for k, v in self.scores.items()}
        return result


@dataclass
class Segment:
    """A merged emotion segment in the timeline.
    
    Segments represent continuous time periods where the same emotion
    is predicted. Adjacent windows with the same emotion are merged
    into a single segment.
    
    Attributes:
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        emotion: Canonical emotion label for this segment.
        confidence: Average confidence score across constituent windows.
        scores: Optional average scores for all labels across windows.
        window_count: Number of windows merged into this segment.
    """
    
    start_sec: float
    end_sec: float
    emotion: str
    confidence: float
    scores: dict[str, float] | None = None
    window_count: int = 1
    
    def to_dict(self, include_scores: bool = True) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Args:
            include_scores: Whether to include the scores dict.
            
        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "start_sec": round(self.start_sec, 6),
            "end_sec": round(self.end_sec, 6),
            "emotion": self.emotion,
            "confidence": round(self.confidence, 6),
        }
        if include_scores and self.scores is not None:
            result["scores"] = {k: round(v, 6) for k, v in self.scores.items()}
        return result
    
    @property
    def duration_sec(self) -> float:
        """Return segment duration in seconds."""
        return self.end_sec - self.start_sec


@dataclass
class SmoothingInfo:
    """Information about smoothing applied to predictions.
    
    Attributes:
        method: Smoothing method name.
        params: Dictionary of smoothing parameters.
    """
    
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            **self.params,
        }


@dataclass
class TimelineResult:
    """Complete result of timeline-based emotion prediction.
    
    This class contains the full output of running emotion recognition
    on an audio file with windowing, smoothing, and segment merging.
    
    Attributes:
        model_name: Name/ID of the model used for prediction.
        sample_rate: Audio sample rate in Hz.
        duration_sec: Total audio duration in seconds.
        window_sec: Window duration in seconds.
        hop_sec: Hop/stride duration in seconds.
        pad_mode: Padding mode used ("none", "zero", "reflect").
        smoothing: Dictionary describing smoothing config applied.
        segments: List of merged emotion segments.
        windows: Optional list of per-window predictions.
        is_padded_timeline: Whether any window was padded.
        merge_config: Dictionary describing merge config applied.
    """
    
    model_name: str
    sample_rate: int
    duration_sec: float
    window_sec: float
    hop_sec: float
    pad_mode: str
    smoothing: dict[str, Any]
    segments: list[Segment]
    windows: list[WindowPrediction] | None = None
    is_padded_timeline: bool = False
    merge_config: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(
        self,
        include_windows: bool = False,
        include_scores: bool = False,
    ) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Args:
            include_windows: Whether to include the windows list.
            include_scores: Whether to include per-label scores.
            
        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result: dict[str, Any] = {
            "model_name": self.model_name,
            "sample_rate": self.sample_rate,
            "duration_sec": round(self.duration_sec, 6),
            "window_sec": self.window_sec,
            "hop_sec": self.hop_sec,
            "pad_mode": self.pad_mode,
            "smoothing": self.smoothing,
            "is_padded_timeline": self.is_padded_timeline,
            "segments": [seg.to_dict(include_scores=include_scores) for seg in self.segments],
        }
        
        if self.merge_config:
            result["merge_config"] = self.merge_config
        
        if include_windows and self.windows is not None:
            result["windows"] = [w.to_dict(include_scores=include_scores) for w in self.windows]
        
        return result
    
    @property
    def segment_count(self) -> int:
        """Return number of segments."""
        return len(self.segments)
    
    @property
    def window_count(self) -> int | None:
        """Return number of windows if available."""
        return len(self.windows) if self.windows is not None else None
