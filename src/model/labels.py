"""Canonical emotion labels and mapping utilities.

This module defines the canonical emotion labels used throughout the project
and provides utilities for mapping model-specific labels to canonical labels.

Canonical Labels:
    - neutral: No strong emotional expression
    - happy: Positive emotion, joy
    - sad: Negative emotion, sorrow
    - angry: Negative emotion, anger/frustration
    - fear: Negative emotion, fear/anxiety
    - disgust: Negative emotion, revulsion
    - surprise: Can be positive or negative, unexpected

Model-Specific Mappings:
    The baseline model (SpeechBrain IEMOCAP) outputs:
    - "neu" -> "neutral"
    - "hap" -> "happy"
    - "sad" -> "sad"
    - "ang" -> "angry"
    
    Labels not supported by the baseline model (fear, disgust, surprise)
    will have zero probability in mapped outputs.
"""

from typing import Final


# Canonical emotion labels used by this project
CANONICAL_LABELS: Final[list[str]] = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
]

# Mapping from SpeechBrain IEMOCAP model labels to canonical labels
# The IEMOCAP model outputs: neu, hap, sad, ang
IEMOCAP_TO_CANONICAL: Final[dict[str, str]] = {
    "neu": "neutral",
    "hap": "happy",
    "sad": "sad",
    "ang": "angry",
}

# Labels supported by the baseline model
BASELINE_SUPPORTED_LABELS: Final[set[str]] = {"neutral", "happy", "sad", "angry"}

# Default mapping (IEMOCAP is the baseline)
DEFAULT_RAW_TO_CANONICAL: Final[dict[str, str]] = IEMOCAP_TO_CANONICAL


def map_raw_to_canonical(
    raw_scores: dict[str, float],
    mapping: dict[str, str] | None = None,
    normalize: bool = True,
) -> dict[str, float]:
    """Map raw model scores to canonical label scores.
    
    This function:
    1. Maps raw model labels to canonical labels
    2. Assigns 0.0 probability to unsupported canonical labels
    3. Optionally normalizes probabilities to sum to 1.0
    
    Args:
        raw_scores: Dictionary mapping raw labels to probability scores.
        mapping: Optional custom mapping from raw to canonical labels.
            If None, uses the default IEMOCAP mapping.
        normalize: Whether to normalize output scores to sum to 1.0.
            Default True.
    
    Returns:
        Dictionary mapping canonical labels to probability scores.
        All canonical labels will be present in the output.
        
    Examples:
        >>> raw_scores = {"neu": 0.1, "hap": 0.5, "sad": 0.2, "ang": 0.2}
        >>> canonical = map_raw_to_canonical(raw_scores)
        >>> canonical["happy"]
        0.5
        >>> canonical["fear"]  # Unsupported by baseline
        0.0
    """
    if mapping is None:
        mapping = DEFAULT_RAW_TO_CANONICAL
    
    # Initialize canonical scores with zeros
    canonical_scores: dict[str, float] = {label: 0.0 for label in CANONICAL_LABELS}
    
    # Map raw scores to canonical
    total = 0.0
    for raw_label, score in raw_scores.items():
        canonical_label = mapping.get(raw_label)
        if canonical_label and canonical_label in canonical_scores:
            canonical_scores[canonical_label] += score
            total += score
    
    # Normalize if requested and total > 0
    if normalize and total > 0:
        for label in canonical_scores:
            canonical_scores[label] /= total
    
    return canonical_scores


def get_canonical_label(raw_label: str, mapping: dict[str, str] | None = None) -> str | None:
    """Get the canonical label for a raw model label.
    
    Args:
        raw_label: Raw label from the model.
        mapping: Optional custom mapping. If None, uses default IEMOCAP mapping.
        
    Returns:
        Canonical label if mapping exists, None otherwise.
    """
    if mapping is None:
        mapping = DEFAULT_RAW_TO_CANONICAL
    return mapping.get(raw_label)


def is_label_supported(canonical_label: str) -> bool:
    """Check if a canonical label is supported by the baseline model.
    
    Args:
        canonical_label: A canonical emotion label.
        
    Returns:
        True if the baseline model can predict this emotion.
    """
    return canonical_label in BASELINE_SUPPORTED_LABELS
