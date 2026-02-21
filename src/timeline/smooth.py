"""Smoothing strategies for timeline emotion predictions.

This module provides different smoothing methods to reduce jitter in
window-by-window emotion predictions, resulting in more stable timelines.

Smoothing Methods:
    - none: No smoothing, keep original predictions.
    - majority: Replace each emotion with majority vote in a sliding window.
    - hysteresis: Only switch emotion when a new emotion persists for N windows.
    - ema: Exponential moving average applied to per-label scores.

Example:
    >>> from timeline.smooth import SmoothingConfig, smooth_windows
    >>> from timeline.schema import WindowPrediction
    >>> config = SmoothingConfig(method="hysteresis", hysteresis_min_run=3)
    >>> smoothed = smooth_windows(windows, config, canonical_labels)
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from .schema import WindowPrediction


SmoothingMethod = Literal["none", "majority", "hysteresis", "ema"]


@dataclass
class SmoothingConfig:
    """Configuration for smoothing window predictions.
    
    Attributes:
        method: Smoothing method to use. One of:
            - "none": No smoothing.
            - "majority": Majority vote in sliding window.
            - "hysteresis": Require N consecutive windows to switch.
            - "ema": Exponential moving average on scores.
        majority_window: Window size for majority voting (must be odd).
            Default 5.
        hysteresis_min_run: Minimum consecutive windows for emotion
            switch in hysteresis mode. Default 3.
        ema_alpha: Alpha coefficient for EMA (0 < alpha <= 1).
            Higher values give more weight to recent predictions.
            Default 0.6.
        min_confidence: Optional minimum confidence threshold.
            Predictions below this are not changed. Default 0.0.
    """
    
    method: SmoothingMethod = "hysteresis"
    majority_window: int = 5
    hysteresis_min_run: int = 3
    ema_alpha: float = 0.6
    min_confidence: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate configuration on initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        valid_methods = {"none", "majority", "hysteresis", "ema"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{self.method}'"
            )
        
        if self.majority_window < 1:
            raise ValueError(
                f"majority_window must be >= 1, got {self.majority_window}"
            )
        
        if self.majority_window % 2 == 0:
            raise ValueError(
                f"majority_window must be odd, got {self.majority_window}"
            )
        
        if self.hysteresis_min_run < 1:
            raise ValueError(
                f"hysteresis_min_run must be >= 1, got {self.hysteresis_min_run}"
            )
        
        if not 0 < self.ema_alpha <= 1:
            raise ValueError(
                f"ema_alpha must be in (0, 1], got {self.ema_alpha}"
            )
        
        if self.min_confidence < 0 or self.min_confidence > 1:
            raise ValueError(
                f"min_confidence must be in [0, 1], got {self.min_confidence}"
            )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {"method": self.method}
        if self.method == "majority":
            result["majority_window"] = self.majority_window
        elif self.method == "hysteresis":
            result["hysteresis_min_run"] = self.hysteresis_min_run
        elif self.method == "ema":
            result["ema_alpha"] = self.ema_alpha
        if self.min_confidence > 0:
            result["min_confidence"] = self.min_confidence
        return result


def smooth_windows(
    windows: list[WindowPrediction],
    config: SmoothingConfig,
    canonical_labels: list[str],
) -> list[WindowPrediction]:
    """Apply smoothing to a list of window predictions.
    
    This function applies the configured smoothing method to reduce
    jitter in emotion predictions across consecutive windows.
    
    Args:
        windows: List of WindowPrediction objects to smooth.
        config: Smoothing configuration.
        canonical_labels: List of canonical emotion labels.
        
    Returns:
        New list of WindowPrediction objects with smoothed emotions.
        Original objects are not modified.
        
    Raises:
        ValueError: If config is invalid or windows are empty.
    """
    if not windows:
        return []
    
    if config.method == "none":
        return _smooth_none(windows)
    elif config.method == "majority":
        return _smooth_majority(windows, config.majority_window)
    elif config.method == "hysteresis":
        return _smooth_hysteresis(windows, config.hysteresis_min_run)
    elif config.method == "ema":
        return _smooth_ema(windows, config.ema_alpha, canonical_labels)
    else:
        raise ValueError(f"Unknown smoothing method: {config.method}")


def _smooth_none(windows: list[WindowPrediction]) -> list[WindowPrediction]:
    """No smoothing - return deep copies of original windows."""
    return [deepcopy(w) for w in windows]


def _smooth_majority(
    windows: list[WindowPrediction],
    window_size: int,
) -> list[WindowPrediction]:
    """Apply majority vote smoothing.
    
    For each window, replace its emotion with the majority emotion
    in a centered sliding window. Ties are broken by higher average
    confidence for that emotion.
    
    Args:
        windows: List of window predictions.
        window_size: Size of the sliding window (must be odd).
        
    Returns:
        New list of smoothed window predictions.
    """
    n = len(windows)
    half_window = window_size // 2
    result = []
    
    for i, window in enumerate(windows):
        # Determine window range
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        
        # Get emotions and confidences in window
        window_emotions = []
        emotion_confidences: dict[str, list[float]] = {}
        
        for j in range(start_idx, end_idx):
            emotion = windows[j].emotion
            conf = windows[j].confidence
            window_emotions.append(emotion)
            if emotion not in emotion_confidences:
                emotion_confidences[emotion] = []
            emotion_confidences[emotion].append(conf)
        
        # Count votes
        counts = Counter(window_emotions)
        max_count = max(counts.values())
        
        # Find all emotions with max count
        candidates = [e for e, c in counts.items() if c == max_count]
        
        if len(candidates) == 1:
            smoothed_emotion = candidates[0]
        else:
            # Tie-breaker: highest average confidence
            avg_confidences = {
                e: sum(emotion_confidences[e]) / len(emotion_confidences[e])
                for e in candidates
            }
            smoothed_emotion = max(candidates, key=lambda e: avg_confidences[e])
        
        # Get confidence for smoothed emotion (average in window)
        smoothed_confidence = (
            sum(emotion_confidences[smoothed_emotion])
            / len(emotion_confidences[smoothed_emotion])
        )
        
        # Create new window prediction
        new_window = WindowPrediction(
            index=window.index,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            emotion=smoothed_emotion,
            confidence=smoothed_confidence,
            scores=deepcopy(window.scores) if window.scores else None,
            is_padded=window.is_padded,
        )
        result.append(new_window)
    
    return result


def _smooth_hysteresis(
    windows: list[WindowPrediction],
    min_run: int,
) -> list[WindowPrediction]:
    """Apply hysteresis smoothing.
    
    Only switch to a new emotion when it persists for at least
    `min_run` consecutive windows. When switching, the switch
    happens at the first window of the persistent run.
    
    Args:
        windows: List of window predictions.
        min_run: Minimum consecutive windows required to switch.
        
    Returns:
        New list of smoothed window predictions.
    """
    if not windows:
        return []
    
    n = len(windows)
    result_emotions: list[str] = []
    result_confidences: list[float] = []
    
    # Initialize with first window's emotion
    current_emotion = windows[0].emotion
    
    # First pass: identify runs and determine when to switch
    i = 0
    while i < n:
        # Find the length of the current run of the same emotion
        run_emotion = windows[i].emotion
        run_start = i
        run_confidences = [windows[i].confidence]
        
        j = i + 1
        while j < n and windows[j].emotion == run_emotion:
            run_confidences.append(windows[j].confidence)
            j += 1
        
        run_length = j - run_start
        
        # Check if we should switch
        if run_emotion != current_emotion and run_length >= min_run:
            # Switch to new emotion at start of this run
            current_emotion = run_emotion
        
        # Assign current_emotion to all windows in this run
        for k in range(run_start, j):
            result_emotions.append(current_emotion)
            result_confidences.append(windows[k].confidence)
        
        i = j
    
    # Build result windows
    result = []
    for i, window in enumerate(windows):
        new_window = WindowPrediction(
            index=window.index,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            emotion=result_emotions[i],
            confidence=result_confidences[i],
            scores=deepcopy(window.scores) if window.scores else None,
            is_padded=window.is_padded,
        )
        result.append(new_window)
    
    return result


def _smooth_ema(
    windows: list[WindowPrediction],
    alpha: float,
    canonical_labels: list[str],
) -> list[WindowPrediction]:
    """Apply exponential moving average smoothing on scores.
    
    Apply EMA to per-label scores, then determine emotion from
    argmax of smoothed scores.
    
    Args:
        windows: List of window predictions.
        alpha: EMA coefficient (0 < alpha <= 1).
        canonical_labels: List of canonical emotion labels.
        
    Returns:
        New list of smoothed window predictions.
    """
    if not windows:
        return []
    
    # Check if scores are available
    if all(w.scores is None for w in windows):
        # No scores available - fall back to no smoothing
        return _smooth_none(windows)
    
    result = []
    ema_scores: dict[str, float] | None = None
    
    for i, window in enumerate(windows):
        # Get current scores (use uniform if not available)
        if window.scores is not None:
            current_scores = window.scores
        else:
            # Create scores from emotion/confidence
            current_scores = {label: 0.0 for label in canonical_labels}
            if window.emotion in current_scores:
                current_scores[window.emotion] = window.confidence
                # Distribute remaining probability
                remaining = 1.0 - window.confidence
                other_labels = [l for l in canonical_labels if l != window.emotion]
                if other_labels:
                    per_label = remaining / len(other_labels)
                    for label in other_labels:
                        current_scores[label] = per_label
        
        # Initialize or update EMA
        if ema_scores is None:
            # First window - initialize EMA with current scores
            ema_scores = {label: current_scores.get(label, 0.0) for label in canonical_labels}
        else:
            # Update EMA: new_ema = alpha * current + (1 - alpha) * old_ema
            for label in canonical_labels:
                current_val = current_scores.get(label, 0.0)
                ema_scores[label] = alpha * current_val + (1 - alpha) * ema_scores[label]
        
        # Normalize EMA scores to sum to 1
        total = sum(ema_scores.values())
        if total > 0:
            normalized_scores = {k: v / total for k, v in ema_scores.items()}
        else:
            # Fallback to uniform
            normalized_scores = {k: 1.0 / len(canonical_labels) for k in canonical_labels}
        
        # Determine emotion from argmax of smoothed scores
        smoothed_emotion = max(normalized_scores.keys(), key=lambda k: normalized_scores[k])
        smoothed_confidence = normalized_scores[smoothed_emotion]
        
        new_window = WindowPrediction(
            index=window.index,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            emotion=smoothed_emotion,
            confidence=smoothed_confidence,
            scores=dict(normalized_scores),  # Store smoothed scores
            is_padded=window.is_padded,
        )
        result.append(new_window)
    
    return result
