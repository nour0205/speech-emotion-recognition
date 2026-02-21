"""Smoke tests for model inference.

These tests validate basic functionality of the inference module.
Some tests require model download and are marked as integration tests.
"""

import os

import pytest
import torch

from model.types import PredictionResult
from model.errors import ModelError, ModelLoadError, InferenceError
from model.labels import CANONICAL_LABELS


# Skip integration tests unless explicitly enabled
SKIP_INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS", "0") != "1"


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""
    
    def test_create_prediction_result(self):
        """Should create PredictionResult with required fields."""
        result = PredictionResult(
            emotion="happy",
            confidence=0.85,
            scores={"happy": 0.85, "sad": 0.15, "neutral": 0.0, "angry": 0.0,
                    "fear": 0.0, "disgust": 0.0, "surprise": 0.0},
            model_name="test-model",
        )
        
        assert result.emotion == "happy"
        assert result.confidence == 0.85
        assert result.model_name == "test-model"
    
    def test_to_dict(self):
        """to_dict should return serializable dictionary."""
        result = PredictionResult(
            emotion="sad",
            confidence=0.75,
            scores={"sad": 0.75, "happy": 0.25, "neutral": 0.0, "angry": 0.0,
                    "fear": 0.0, "disgust": 0.0, "surprise": 0.0},
            model_name="test-model",
            raw_label="sad",
            raw_scores={"sad": 0.75, "hap": 0.25},
            duration_sec=2.5,
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d["emotion"] == "sad"
        assert d["confidence"] == 0.75
        assert d["duration_sec"] == 2.5
        assert d["raw_label"] == "sad"
    
    def test_optional_fields(self):
        """Optional fields should have sensible defaults."""
        result = PredictionResult(
            emotion="neutral",
            confidence=0.9,
            scores={"neutral": 0.9},
            model_name="test",
        )
        
        assert result.raw_label is None
        assert result.raw_scores is None
        assert result.duration_sec == 0.0


class TestModelErrors:
    """Tests for model error classes."""
    
    def test_model_error_attributes(self):
        """ModelError should have message, code, and details."""
        error = ModelError(
            message="Test error",
            code="TEST_CODE",
            details={"key": "value"},
        )
        
        assert error.message == "Test error"
        assert error.code == "TEST_CODE"
        assert error.details == {"key": "value"}
    
    def test_model_error_str(self):
        """ModelError str should include code and message."""
        error = ModelError(message="Test", code="CODE", details={})
        assert "CODE" in str(error)
        assert "Test" in str(error)
    
    def test_model_load_error(self):
        """ModelLoadError should be a ModelError."""
        error = ModelLoadError(message="Load failed", code="LOAD_FAILED")
        assert isinstance(error, ModelError)
    
    def test_inference_error(self):
        """InferenceError should be a ModelError."""
        error = InferenceError(message="Inference failed", code="INFERENCE_FAILED")
        assert isinstance(error, ModelError)


class TestPredictWaveformValidation:
    """Tests for predict_waveform input validation (no model required)."""
    
    def test_invalid_type(self):
        """Should raise InferenceError for non-tensor input."""
        from model.infer import predict_waveform
        
        with pytest.raises(InferenceError) as exc_info:
            predict_waveform([1, 2, 3], sample_rate=16000)  # type: ignore
        
        assert exc_info.value.code == "INVALID_INPUT"
    
    def test_invalid_shape_3d(self):
        """Should raise InferenceError for 3D tensor."""
        from model.infer import predict_waveform
        
        waveform = torch.randn(1, 1, 16000)  # 3D instead of 2D
        
        with pytest.raises(InferenceError) as exc_info:
            predict_waveform(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "INVALID_INPUT"
    
    def test_invalid_shape_stereo(self):
        """Should raise InferenceError for stereo input."""
        from model.infer import predict_waveform
        
        waveform = torch.randn(2, 16000)  # 2 channels
        
        with pytest.raises(InferenceError) as exc_info:
            predict_waveform(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "INVALID_INPUT"
    
    def test_invalid_dtype(self):
        """Should raise InferenceError for non-float32 tensor."""
        from model.infer import predict_waveform
        
        waveform = torch.randn(1, 16000).double()  # float64
        
        with pytest.raises(InferenceError) as exc_info:
            predict_waveform(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "INVALID_INPUT"


class TestModelRegistry:
    """Tests for model registry functionality."""
    
    def test_list_available_models(self):
        """Should list available model IDs."""
        from model.registry import list_available_models
        
        models = list_available_models()
        assert isinstance(models, list)
        assert "baseline" in models
    
    def test_unknown_model_raises_error(self):
        """Should raise ModelLoadError for unknown model ID."""
        from model.registry import get_model
        
        with pytest.raises(ModelLoadError) as exc_info:
            get_model("nonexistent-model-xyz")
        
        assert exc_info.value.code == "MODEL_NOT_FOUND"
        assert "nonexistent-model-xyz" in exc_info.value.message


@pytest.mark.integration
@pytest.mark.skipif(SKIP_INTEGRATION, reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to run.")
class TestModelInferenceIntegration:
    """Integration tests that require model download.
    
    These tests are skipped by default. To run them:
        RUN_INTEGRATION_TESTS=1 pytest tests/test_model_infer_smoke.py -v
    """
    
    def test_predict_waveform_sine(self):
        """Should predict emotion on a simple sine wave."""
        from model.infer import predict_waveform
        
        # Generate 2 seconds of sine wave at 440Hz
        sample_rate = 16000
        duration = 2.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).float()
        
        result = predict_waveform(waveform, sample_rate=sample_rate)
        
        assert isinstance(result, PredictionResult)
        assert result.emotion in CANONICAL_LABELS
        assert 0.0 <= result.confidence <= 1.0
        assert abs(sum(result.scores.values()) - 1.0) < 0.001
        assert result.model_name == "speechbrain-iemocap"
    
    def test_predict_waveform_noise(self):
        """Should predict emotion on random noise."""
        from model.infer import predict_waveform
        
        # Generate 1 second of random noise
        sample_rate = 16000
        waveform = torch.randn(1, sample_rate).float() * 0.5
        
        result = predict_waveform(waveform, sample_rate=sample_rate)
        
        assert isinstance(result, PredictionResult)
        assert result.emotion in CANONICAL_LABELS
    
    def test_model_caching(self):
        """Model should be cached between calls."""
        from model.registry import get_model
        
        model1 = get_model("baseline", device="cpu")
        model2 = get_model("baseline", device="cpu")
        
        # Should be the same instance
        assert model1 is model2
    
    def test_all_canonical_labels_in_scores(self):
        """Scores should contain all canonical labels."""
        from model.infer import predict_waveform
        
        sample_rate = 16000
        waveform = torch.randn(1, sample_rate * 2).float()
        
        result = predict_waveform(waveform, sample_rate=sample_rate)
        
        for label in CANONICAL_LABELS:
            assert label in result.scores
            assert isinstance(result.scores[label], float)
            assert result.scores[label] >= 0.0


@pytest.mark.integration
@pytest.mark.skipif(SKIP_INTEGRATION, reason="Integration tests disabled")
class TestPredictClipIntegration:
    """Integration tests for predict_clip function."""
    
    def test_predict_clip_from_bytes(self, tmp_path):
        """Should predict from WAV bytes."""
        import soundfile as sf
        import numpy as np
        from model.infer import predict_clip
        
        # Create a test WAV file
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        audio = np.random.randn(samples).astype(np.float32) * 0.5
        
        wav_path = tmp_path / "test.wav"
        sf.write(wav_path, audio, sample_rate)
        
        # Read as bytes
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        
        result = predict_clip(audio_bytes)
        
        assert isinstance(result, PredictionResult)
        assert result.emotion in CANONICAL_LABELS
        assert result.duration_sec == pytest.approx(duration, rel=0.1)
    
    def test_predict_clip_from_file(self, tmp_path):
        """Should predict from file path."""
        import soundfile as sf
        import numpy as np
        from model.infer import predict_clip
        
        # Create a test WAV file
        sample_rate = 16000
        duration = 1.5
        samples = int(sample_rate * duration)
        audio = np.random.randn(samples).astype(np.float32) * 0.3
        
        wav_path = tmp_path / "test.wav"
        sf.write(wav_path, audio, sample_rate)
        
        result = predict_clip(str(wav_path))
        
        assert isinstance(result, PredictionResult)
        assert result.duration_sec == pytest.approx(duration, rel=0.1)
