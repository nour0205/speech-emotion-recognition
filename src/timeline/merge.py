"""Segment merging logic for timeline emotion predictions.

This module provides functions to merge adjacent window predictions
with the same emotion into continuous segments, with optional handling
of short segments.

Example:
    >>> from timeline.merge import MergeConfig, merge_windows_to_segments
    >>> from timeline.schema import WindowPrediction
    >>> from timeline.windowing import WindowingConfig
    >>> config = MergeConfig(merge_adjacent=True, min_segment_sec=0.5)
    >>> segments = merge_windows_to_segments(windows, windowing_config, config)
"""

from dataclasses import dataclass
from typing import Literal

from .schema import Segment, WindowPrediction
from .windowing import WindowingConfig


ShortSegmentStrategy = Literal["merge_prev", "merge_next", "merge_best"]


@dataclass
class MergeConfig:
    """Configuration for merging window predictions into segments.
    
    Attributes:
        merge_adjacent: Whether to merge adjacent windows with the
            same emotion into single segments. Default True.
        min_segment_sec: Minimum segment duration in seconds.
            Segments shorter than this may be handled according to
            drop_short_segments and short_segment_strategy.
            Default 0.25.
        drop_short_segments: If True, short segments are merged into
            neighboring segments. Default False.
        short_segment_strategy: How to handle short segments when
            drop_short_segments is True:
            - "merge_prev": Merge into previous segment.
            - "merge_next": Merge into next segment.
            - "merge_best": Merge into neighbor with higher confidence.
            Default "merge_best".
        cap_to_duration: Whether to cap final segment end_sec to
            audio duration. Default True.
    """
    
    merge_adjacent: bool = True
    min_segment_sec: float = 0.25
    drop_short_segments: bool = False
    short_segment_strategy: ShortSegmentStrategy = "merge_best"
    cap_to_duration: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration on initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.min_segment_sec < 0:
            raise ValueError(
                f"min_segment_sec must be >= 0, got {self.min_segment_sec}"
            )
        
        valid_strategies = {"merge_prev", "merge_next", "merge_best"}
        if self.short_segment_strategy not in valid_strategies:
            raise ValueError(
                f"short_segment_strategy must be one of {valid_strategies}, "
                f"got '{self.short_segment_strategy}'"
            )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "merge_adjacent": self.merge_adjacent,
            "min_segment_sec": self.min_segment_sec,
            "drop_short_segments": self.drop_short_segments,
            "short_segment_strategy": self.short_segment_strategy,
            "cap_to_duration": self.cap_to_duration,
        }


def merge_windows_to_segments(
    windows: list[WindowPrediction],
    windowing_config: WindowingConfig,
    merge_config: MergeConfig,
    duration_sec: float | None = None,
) -> list[Segment]:
    """Merge window predictions into continuous emotion segments.
    
    This function takes a list of smoothed window predictions and
    merges adjacent windows with the same emotion into segments.
    
    Args:
        windows: List of (smoothed) WindowPrediction objects.
        windowing_config: Windowing configuration used to generate windows.
        merge_config: Merge configuration.
        duration_sec: Optional audio duration to cap segment boundaries.
        
    Returns:
        List of Segment objects, ordered by time and non-overlapping.
    """
    if not windows:
        return []
    
    if merge_config.merge_adjacent:
        segments = _merge_adjacent_windows(windows)
    else:
        # No merging - each window becomes its own segment
        segments = _windows_to_individual_segments(windows)
    
    # Handle short segments
    if merge_config.drop_short_segments:
        segments = _handle_short_segments(
            segments,
            merge_config.min_segment_sec,
            merge_config.short_segment_strategy,
        )
    
    # Cap to duration if specified
    if merge_config.cap_to_duration and duration_sec is not None:
        segments = _cap_segments_to_duration(segments, duration_sec)
    
    return segments


def _merge_adjacent_windows(windows: list[WindowPrediction]) -> list[Segment]:
    """Merge adjacent windows with the same emotion into segments.
    
    Args:
        windows: List of window predictions (assumed sorted by time).
        
    Returns:
        List of merged segments.
    """
    if not windows:
        return []
    
    segments: list[Segment] = []
    
    # Start first segment
    current_emotion = windows[0].emotion
    current_start = windows[0].start_sec
    current_end = windows[0].end_sec
    current_confidences = [windows[0].confidence]
    current_scores_list: list[dict[str, float] | None] = [windows[0].scores]
    window_count = 1
    
    for i in range(1, len(windows)):
        window = windows[i]
        
        if window.emotion == current_emotion:
            # Same emotion - extend segment
            current_end = window.end_sec
            current_confidences.append(window.confidence)
            current_scores_list.append(window.scores)
            window_count += 1
        else:
            # Different emotion - close current segment and start new one
            segments.append(_create_segment(
                start_sec=current_start,
                end_sec=current_end,
                emotion=current_emotion,
                confidences=current_confidences,
                scores_list=current_scores_list,
                window_count=window_count,
            ))
            
            # Start new segment
            current_emotion = window.emotion
            current_start = window.start_sec
            current_end = window.end_sec
            current_confidences = [window.confidence]
            current_scores_list = [window.scores]
            window_count = 1
    
    # Close final segment
    segments.append(_create_segment(
        start_sec=current_start,
        end_sec=current_end,
        emotion=current_emotion,
        confidences=current_confidences,
        scores_list=current_scores_list,
        window_count=window_count,
    ))
    
    return segments


def _windows_to_individual_segments(
    windows: list[WindowPrediction],
) -> list[Segment]:
    """Convert each window to an individual segment (no merging).
    
    Args:
        windows: List of window predictions.
        
    Returns:
        List of segments, one per window.
    """
    return [
        Segment(
            start_sec=w.start_sec,
            end_sec=w.end_sec,
            emotion=w.emotion,
            confidence=w.confidence,
            scores=dict(w.scores) if w.scores else None,
            window_count=1,
        )
        for w in windows
    ]


def _create_segment(
    start_sec: float,
    end_sec: float,
    emotion: str,
    confidences: list[float],
    scores_list: list[dict[str, float] | None],
    window_count: int,
) -> Segment:
    """Create a segment from merged window data.
    
    Args:
        start_sec: Segment start time.
        end_sec: Segment end time.
        emotion: Emotion label.
        confidences: List of confidence values from constituent windows.
        scores_list: List of score dicts from constituent windows.
        window_count: Number of merged windows.
        
    Returns:
        New Segment object.
    """
    # Average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Average scores (if available)
    avg_scores = _average_scores(scores_list)
    
    return Segment(
        start_sec=start_sec,
        end_sec=end_sec,
        emotion=emotion,
        confidence=avg_confidence,
        scores=avg_scores,
        window_count=window_count,
    )


def _average_scores(
    scores_list: list[dict[str, float] | None],
) -> dict[str, float] | None:
    """Average score dictionaries.
    
    Args:
        scores_list: List of score dictionaries (may contain None).
        
    Returns:
        Averaged scores dict, or None if no valid scores.
    """
    valid_scores = [s for s in scores_list if s is not None]
    
    if not valid_scores:
        return None
    
    # Get all labels
    all_labels = set()
    for scores in valid_scores:
        all_labels.update(scores.keys())
    
    # Average each label
    avg_scores: dict[str, float] = {}
    for label in all_labels:
        values = [s.get(label, 0.0) for s in valid_scores]
        avg_scores[label] = sum(values) / len(values)
    
    return avg_scores


def _handle_short_segments(
    segments: list[Segment],
    min_segment_sec: float,
    strategy: ShortSegmentStrategy,
) -> list[Segment]:
    """Merge short segments into their neighbors.
    
    Args:
        segments: List of segments to process.
        min_segment_sec: Minimum allowed segment duration.
        strategy: How to handle short segments.
        
    Returns:
        List of segments with short ones merged.
    """
    if not segments or min_segment_sec <= 0:
        return segments
    
    # Find short segments and merge them iteratively
    result = list(segments)
    changed = True
    
    while changed:
        changed = False
        new_result: list[Segment] = []
        skip_next = False
        
        for i, segment in enumerate(result):
            if skip_next:
                skip_next = False
                continue
            
            if segment.duration_sec >= min_segment_sec:
                new_result.append(segment)
                continue
            
            # Short segment - determine where to merge
            has_prev = len(new_result) > 0
            has_next = i + 1 < len(result)
            
            if not has_prev and not has_next:
                # Only segment - keep it
                new_result.append(segment)
                continue
            
            if not has_prev:
                # Merge with next
                merged = _merge_two_segments(segment, result[i + 1])
                new_result.append(merged)
                skip_next = True
                changed = True
                continue
            
            if not has_next:
                # Merge with previous
                merged = _merge_two_segments(new_result[-1], segment)
                new_result[-1] = merged
                changed = True
                continue
            
            # Both neighbors exist - use strategy
            prev_segment = new_result[-1]
            next_segment = result[i + 1]
            
            if strategy == "merge_prev":
                merged = _merge_two_segments(prev_segment, segment)
                new_result[-1] = merged
                changed = True
            elif strategy == "merge_next":
                merged = _merge_two_segments(segment, next_segment)
                new_result.append(merged)
                skip_next = True
                changed = True
            else:  # merge_best
                # Merge with neighbor that has higher confidence
                if prev_segment.confidence >= next_segment.confidence:
                    merged = _merge_two_segments(prev_segment, segment)
                    new_result[-1] = merged
                else:
                    merged = _merge_two_segments(segment, next_segment)
                    new_result.append(merged)
                    skip_next = True
                changed = True
        
        result = new_result
    
    return result


def _merge_two_segments(seg1: Segment, seg2: Segment) -> Segment:
    """Merge two adjacent segments into one.
    
    The merged segment takes the emotion of the longer segment,
    or the one with higher confidence if equal length.
    
    Args:
        seg1: First segment (earlier in time).
        seg2: Second segment (later in time).
        
    Returns:
        Merged segment.
    """
    # Determine dominant emotion
    if seg1.duration_sec > seg2.duration_sec:
        emotion = seg1.emotion
    elif seg2.duration_sec > seg1.duration_sec:
        emotion = seg2.emotion
    else:
        # Equal duration - use higher confidence
        emotion = seg1.emotion if seg1.confidence >= seg2.confidence else seg2.emotion
    
    # Weighted average confidence by duration
    total_duration = seg1.duration_sec + seg2.duration_sec
    if total_duration > 0:
        avg_confidence = (
            seg1.confidence * seg1.duration_sec +
            seg2.confidence * seg2.duration_sec
        ) / total_duration
    else:
        avg_confidence = (seg1.confidence + seg2.confidence) / 2
    
    # Merge scores
    scores_list = [seg1.scores, seg2.scores]
    avg_scores = _average_scores(scores_list)
    
    return Segment(
        start_sec=seg1.start_sec,
        end_sec=seg2.end_sec,
        emotion=emotion,
        confidence=avg_confidence,
        scores=avg_scores,
        window_count=seg1.window_count + seg2.window_count,
    )


def _cap_segments_to_duration(
    segments: list[Segment],
    duration_sec: float,
) -> list[Segment]:
    """Cap segment end times to audio duration.
    
    Args:
        segments: List of segments.
        duration_sec: Audio duration in seconds.
        
    Returns:
        List of segments with capped end times.
    """
    if not segments:
        return segments
    
    result = []
    for segment in segments:
        if segment.end_sec > duration_sec:
            # Cap the end time
            result.append(Segment(
                start_sec=segment.start_sec,
                end_sec=duration_sec,
                emotion=segment.emotion,
                confidence=segment.confidence,
                scores=segment.scores,
                window_count=segment.window_count,
            ))
        else:
            result.append(segment)
    
    return result
