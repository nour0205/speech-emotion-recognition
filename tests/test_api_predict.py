"""Tests for the /predict API endpoint."""

import io
import os

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        app_name="SER Test Service",
        log_level="DEBUG",
        model_id="baseline",
        device="cpu",
        max_duration_sec=60.0,
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """Create a test client with the app."""
    app = create_app(test_settings)
    return TestClient(app)


def create_wav_bytes(
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> bytes:
    """Create in-memory WAV file bytes.
    
    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        frequency: Tone frequency in Hz.
        
    Returns:
        WAV file as bytes.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    waveform = (np.sin(2 * np.pi * frequency * t) * 0.5).astype(np.float32)
    
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def create_random_bytes(size: int = 1024) -> bytes:
    """Create random non-audio bytes."""
    return os.urandom(size)


class TestPredictInvalidInput:
    """Tests for /predict with invalid inputs."""
    
    def test_predict_invalid_audio_returns_400(self, client: TestClient) -> None:
        """Test that random bytes return 400 INVALID_AUDIO."""
        invalid_bytes = create_random_bytes(1024)
        
        response = client.post(
            "/predict",
            files={"file": ("test.wav", invalid_bytes, "audio/wav")},
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert data["error"]["code"] == "INVALID_AUDIO"
    
    def test_predict_empty_file_returns_422(self, client: TestClient) -> None:
        """Test that empty file returns 422 INVALID_INPUT."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", b"", "audio/wav")},
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_INPUT"
    
    def test_predict_missing_file_returns_422(self, client: TestClient) -> None:
        """Test that missing file returns 422."""
        response = client.post("/predict")
        
        # FastAPI returns 422 for missing required fields
        assert response.status_code == 422
    
    def test_predict_error_includes_request_id(self, client: TestClient) -> None:
        """Test that error responses include X-Request-ID."""
        invalid_bytes = create_random_bytes(1024)
        
        response = client.post(
            "/predict",
            files={"file": ("test.wav", invalid_bytes, "audio/wav")},
        )
        
        assert "X-Request-ID" in response.headers


class TestPredictSmoke:
    """Smoke tests for /predict endpoint (requires model).
    
    These tests are skipped unless RUN_INTEGRATION_TESTS=1.
    """
    
    @pytest.fixture(autouse=True)
    def skip_unless_integration(self) -> None:
        """Skip test unless integration tests are enabled."""
        if not os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1":
            pytest.skip("Integration tests disabled (set RUN_INTEGRATION_TESTS=1)")
    
    def test_predict_valid_audio_returns_200(self, client: TestClient) -> None:
        """Test prediction with valid audio."""
        wav_bytes = create_wav_bytes(duration_sec=1.0)
        
        response = client.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "emotion" in data
        assert "confidence" in data
        assert "model_name" in data
        assert "duration_sec" in data
        
        # Check types and ranges
        assert isinstance(data["emotion"], str)
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["duration_sec"] > 0
    
    def test_predict_with_include_scores(self, client: TestClient) -> None:
        """Test prediction with include_scores=true."""
        wav_bytes = create_wav_bytes(duration_sec=1.0)
        
        response = client.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"include_scores": "true"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Scores should be included
        assert "scores" in data
        assert data["scores"] is not None
        assert isinstance(data["scores"], dict)
        
        # Scores should sum to approximately 1.0
        total = sum(data["scores"].values())
        assert abs(total - 1.0) < 0.01
    
    def test_predict_without_include_scores(self, client: TestClient) -> None:
        """Test prediction with include_scores=false."""
        wav_bytes = create_wav_bytes(duration_sec=1.0)
        
        response = client.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"include_scores": "false"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Scores should not be included
        assert data.get("scores") is None
    
    def test_predict_response_time_header(self, client: TestClient) -> None:
        """Test that response includes timing header."""
        wav_bytes = create_wav_bytes(duration_sec=1.0)
        
        response = client.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        assert "X-Response-Time" in response.headers
