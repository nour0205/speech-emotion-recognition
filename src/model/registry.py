"""Model registry for loading and caching SER models.

This module provides a centralized registry for loading pretrained SER models.
Models are cached to avoid redundant loading on repeated calls.

Example:
    >>> from model.registry import get_model
    >>> ser_model = get_model("baseline", device="cpu")
    >>> ser_model.predict(waveform, sample_rate=16000)
"""

import threading
from typing import Final

import torch
import torchaudio

from .errors import ModelLoadError
from .types import BaseSERModel


# Default model configuration
MODEL_SOURCE: Final[str] = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
DEFAULT_SAMPLE_RATE: Final[int] = 16000


class SpeechBrainSERModel:
    """SpeechBrain emotion recognition model wrapper.
    
    This class wraps the SpeechBrain wav2vec2-IEMOCAP model and provides
    a standardized interface for emotion prediction.
    
    Attributes:
        name: Model identifier.
        raw_labels: List of labels the model outputs.
    """
    
    def __init__(self, device: str = "cpu") -> None:
        """Initialize the model.
        
        Args:
            device: Device to run inference on ("cpu" or "cuda").
            
        Raises:
            ModelLoadError: If model fails to load.
        """
        self._name = "speechbrain-iemocap"
        self._raw_labels = ["ang", "hap", "neu", "sad"]
        self._device = device
        self._classifier = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the pretrained model from HuggingFace."""
        try:
            from speechbrain.inference.interfaces import foreign_class
            
            self._classifier = foreign_class(
                source=MODEL_SOURCE,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
            )
        except Exception as e:
            raise ModelLoadError(
                message=f"Failed to load SpeechBrain model: {e}",
                code="LOAD_FAILED",
                details={"model_source": MODEL_SOURCE, "error": str(e)},
            ) from e
    
    @property
    def name(self) -> str:
        """Return the model name/identifier."""
        return self._name
    
    @property
    def raw_labels(self) -> list[str]:
        """Return the list of raw labels the model outputs."""
        return self._raw_labels.copy()
    
    def _ensure_16k(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample to 16kHz if needed."""
        if sample_rate != DEFAULT_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, DEFAULT_SAMPLE_RATE)
            waveform = resampler(waveform)
        return waveform
    
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
            Scores sum to 1.0 (softmax outputs).
            
        Raises:
            InferenceError: If inference fails.
        """
        from .errors import InferenceError
        
        if self._classifier is None:
            raise InferenceError(
                message="Model not loaded",
                code="MODEL_NOT_LOADED",
                details={},
            )
        
        try:
            # Ensure correct sample rate
            waveform = self._ensure_16k(waveform, sample_rate)
            
            # SpeechBrain expects [batch, time]
            if waveform.dim() == 2 and waveform.shape[0] == 1:
                wavs = waveform.squeeze(0).unsqueeze(0).clone()
            elif waveform.dim() == 1:
                wavs = waveform.unsqueeze(0).clone()
            else:
                raise InferenceError(
                    message=f"Invalid waveform shape: {waveform.shape}. Expected [1, T] or [T]",
                    code="INVALID_INPUT",
                    details={"shape": list(waveform.shape)},
                )
            
            # Run inference
            out_prob, score, index, text_lab = self._classifier.classify_batch(wavs)
            
            # out_prob is [batch, num_classes] tensor with softmax probabilities
            # For single sample, extract probabilities
            probs = out_prob[0].detach().cpu().numpy()
            
            # Map to label names
            # IEMOCAP ordering: ang, hap, neu, sad (alphabetical)
            labels = self._raw_labels
            scores = {label: float(probs[i]) for i, label in enumerate(labels)}
            
            return scores
            
        except Exception as e:
            if isinstance(e, InferenceError):
                raise
            raise InferenceError(
                message=f"Inference failed: {e}",
                code="INFERENCE_FAILED",
                details={"error": str(e)},
            ) from e


# Thread-safe model cache
_model_cache: dict[str, BaseSERModel] = {}
_cache_lock = threading.Lock()

# Available model IDs and their implementations
AVAILABLE_MODELS: Final[dict[str, type]] = {
    "baseline": SpeechBrainSERModel,
    "speechbrain-iemocap": SpeechBrainSERModel,
}


def get_model(model_id: str = "baseline", device: str = "cpu") -> BaseSERModel:
    """Get or create a cached model instance.
    
    This function maintains a cache of loaded models. If a model with the
    given ID and device is already loaded, it returns the cached instance.
    Otherwise, it loads a new model and caches it.
    
    Args:
        model_id: Identifier of the model to load. Available:
            - "baseline": Default SpeechBrain IEMOCAP model
            - "speechbrain-iemocap": Same as baseline
        device: Device to run the model on ("cpu" or "cuda").
        
    Returns:
        A loaded SER model instance implementing BaseSERModel protocol.
        
    Raises:
        ModelLoadError: If model ID is unknown or loading fails.
        
    Examples:
        >>> model = get_model("baseline", device="cpu")
        >>> scores = model.predict(waveform, sample_rate=16000)
    """
    cache_key = f"{model_id}:{device}"
    
    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        
        if model_id not in AVAILABLE_MODELS:
            raise ModelLoadError(
                message=f"Unknown model ID: {model_id}",
                code="MODEL_NOT_FOUND",
                details={"model_id": model_id, "available": list(AVAILABLE_MODELS.keys())},
            )
        
        # Load model
        model_class = AVAILABLE_MODELS[model_id]
        model = model_class(device=device)
        
        # Cache and return
        _model_cache[cache_key] = model
        return model


def clear_cache() -> None:
    """Clear the model cache.
    
    This releases all cached model instances, freeing memory.
    Useful for testing or when switching between models.
    """
    global _model_cache
    with _cache_lock:
        _model_cache.clear()


def list_available_models() -> list[str]:
    """List available model IDs.
    
    Returns:
        List of model identifiers that can be passed to get_model().
    """
    return list(AVAILABLE_MODELS.keys())
