"""Pydantic models for emotion API."""

from enum import Enum
from pydantic import BaseModel, Field


class EmotionLabel(str, Enum):
    """Supported emotion labels from IEMOCAP."""
    ANGRY = "ang"
    HAPPY = "hap"
    SAD = "sad"
    NEUTRAL = "neu"


class EmotionResponse(BaseModel):
    """Response model for emotion prediction."""
    
    label: str = Field(
        ..., 
        description="Predicted emotion label",
        examples=["hap", "sad", "ang", "neu"]
    )
    emotion: str = Field(
        ...,
        description="Human-readable emotion name",
        examples=["Happy", "Sad", "Angry", "Neutral"]
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Prediction confidence score",
        examples=[0.85]
    )
    inference_time_sec: float = Field(
        ...,
        description="Time taken for inference in seconds",
        examples=[0.234]
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "label": "hap",
                "emotion": "Happy",
                "confidence": 0.92,
                "inference_time_sec": 0.156
            }
        }
    }


# Mapping from short labels to human-readable names
EMOTION_NAMES = {
    "ang": "Angry",
    "hap": "Happy", 
    "sad": "Sad",
    "neu": "Neutral",
}


def get_emotion_name(label: str) -> str:
    """Convert short label to human-readable name."""
    label_clean = label.lower().strip("[]'\"")
    for key, name in EMOTION_NAMES.items():
        if key in label_clean:
            return name
    return label
