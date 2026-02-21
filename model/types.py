"""Type definitions for model inference."""

from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class PredictionResult:
    """Result of emotion prediction on a single audio clip.
    
    Attributes:
        emotion: Canonical emotion label (e.g., "happy", "sad").
        confidence: Confidence score for the predicted emotion (0.0 to 1.0).
        scores: Probability scores for all canonical labels (sum to 1.0).
        model_name: Name/ID of the model used for prediction.
        raw_label: Original label from the model before mapping (if applicable).
        raw_scores: Original scores from the model before mapping (if applicable).
        duration_sec: Duration of the input audio in seconds.
    """
    
    emotion: str
    confidence: float
    scores: dict[str, float]
    model_name: str
    raw_label: str | None = None
    raw_scores: dict[str, float] | None = None
    duration_sec: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.
        
        All numeric values are converted to Python native types to ensure
        JSON serialization works correctly.
        """
        # Helper to convert numpy/tensor values to Python native types
        def to_native(val):
            if hasattr(val, 'item'):  # numpy/tensor scalar
                return val.item()
            return val
        
        def convert_scores(scores: dict | None) -> dict | None:
            if scores is None:
                return None
            return {k: float(to_native(v)) for k, v in scores.items()}
        
        return {
            "emotion": self.emotion,
            "confidence": float(to_native(self.confidence)),
            "scores": convert_scores(self.scores),
            "model_name": self.model_name,
            "raw_label": self.raw_label,
            "raw_scores": convert_scores(self.raw_scores),
            "duration_sec": float(to_native(self.duration_sec)),
        }


class BaseSERModel(Protocol):
    """Protocol defining the interface for SER models.
    
    All SER model implementations must conform to this interface.
    """
    
    @property
    def name(self) -> str:
        """Return the model name/identifier."""
        ...
    
    @property
    def raw_labels(self) -> list[str]:
        """Return the list of raw labels the model outputs."""
        ...
    
    def predict(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int,
    ) -> dict[str, float]:
        """Run inference on a waveform.
        
        Args:
            waveform: Audio waveform tensor with shape [1, T] (mono, float32).
            sample_rate: Sample rate of the waveform.
            
        Returns:
            Dictionary mapping raw label names to probability scores.
            Scores should sum to 1.0 (softmax outputs).
        """
        ...
