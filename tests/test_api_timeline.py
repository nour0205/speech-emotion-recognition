"""Tests for the /timeline API endpoint."""

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
        default_window_sec=2.0,
        default_hop_sec=0.5,
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """Create a test client with the app."""
    app = create_app(test_settings)
    return TestClient(app)


def create_wav_bytes(
    duration_sec: float = 3.0,
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


class TestTimelineInvalidParams:
    """Tests for /timeline with invalid parameters."""
    
    def test_timeline_hop_greater_than_window_returns_422(self, client: TestClient) -> None:
        """Test that hop_sec > window_sec returns 422 INVALID_WINDOWING."""
        wav_bytes = create_wav_bytes(duration_sec=3.0)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={
                "window_sec": "1.0",
                "hop_sec": "2.0",  # hop > window
            },
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_WINDOWING"
    
    def test_timeline_invalid_pad_mode_returns_422(self, client: TestClient) -> None:
        """Test that invalid pad_mode returns 422 INVALID_WINDOWING."""
        wav_bytes = create_wav_bytes(duration_sec=3.0)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={
                "pad_mode": "invalid_mode",
            },
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_WINDOWING"
    
    def test_timeline_invalid_smoothing_method_returns_422(self, client: TestClient) -> None:
        """Test that invalid smoothing_method returns 422 INVALID_INPUT."""
        wav_bytes = create_wav_bytes(duration_sec=3.0)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={
                "smoothing_method": "invalid_method",
            },
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_INPUT"
    
    def test_timeline_invalid_audio_returns_400(self, client: TestClient) -> None:
        """Test that invalid audio returns 400 INVALID_AUDIO."""
        invalid_bytes = create_random_bytes(1024)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", invalid_bytes, "audio/wav")},
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_AUDIO"
    
    def test_timeline_empty_file_returns_422(self, client: TestClient) -> None:
        """Test that empty file returns 422 INVALID_INPUT."""
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", b"", "audio/wav")},
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_INPUT"
    
    def test_timeline_error_includes_request_id(self, client: TestClient) -> None:
        """Test that error responses include X-Request-ID."""
        invalid_bytes = create_random_bytes(1024)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", invalid_bytes, "audio/wav")},
        )
        
        assert "X-Request-ID" in response.headers


class TestTimelineSmoke:
    """Smoke tests for /timeline endpoint (requires model).
    
    These tests are skipped unless RUN_INTEGRATION_TESTS=1.
    """
    
    @pytest.fixture(autouse=True)
    def skip_unless_integration(self) -> None:
        """Skip test unless integration tests are enabled."""
        if not os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1":
            pytest.skip("Integration tests disabled (set RUN_INTEGRATION_TESTS=1)")
    
    def test_timeline_valid_audio_returns_200(self, client: TestClient) -> None:
        """Test timeline generation with valid audio."""
        wav_bytes = create_wav_bytes(duration_sec=4.0)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "model_name" in data
        assert "sample_rate" in data
        assert "duration_sec" in data
        assert "window_sec" in data
        assert "hop_sec" in data
        assert "pad_mode" in data
        assert "smoothing" in data
        assert "segments" in data
        
        # Check types
        assert isinstance(data["segments"], list)
        assert data["duration_sec"] > 0
    
    def test_timeline_with_custom_windowing(self, client: TestClient) -> None:
        """Test timeline with custom windowing parameters."""
        wav_bytes = create_wav_bytes(duration_sec=5.0)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={
                "window_sec": "1.5",
                "hop_sec": "0.75",
                "pad_mode": "zero",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check windowing params are reflected
        assert data["window_sec"] == 1.5
        assert data["hop_sec"] == 0.75
        assert data["pad_mode"] == "zero"
    
    def test_timeline_with_include_windows(self, client: TestClient) -> None:
        """Test timeline with include_windows=true."""
        wav_bytes = create_wav_bytes(duration_sec=4.0)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"include_windows": "true"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Windows should be included
        assert "windows" in data
        assert data["windows"] is not None
        assert isinstance(data["windows"], list)
        assert len(data["windows"]) > 0
        
        # Check window structure
        window = data["windows"][0]
        assert "index" in window
        assert "start_sec" in window
        assert "end_sec" in window
        assert "emotion" in window
        assert "confidence" in window
    
    def test_timeline_with_include_scores(self, client: TestClient) -> None:
        """Test timeline with include_scores=true."""
        wav_bytes = create_wav_bytes(duration_sec=4.0)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={
                "include_windows": "true",
                "include_scores": "true",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check segments have scores
        if data["segments"]:
            segment = data["segments"][0]
            assert "scores" in segment
            assert segment["scores"] is not None
        
        # Check windows have scores
        if data["windows"]:
            window = data["windows"][0]
            assert "scores" in window
            assert window["scores"] is not None
    
    def test_timeline_segments_are_contiguous(self, client: TestClient) -> None:
        """Test that timeline segments cover the audio duration."""
        wav_bytes = create_wav_bytes(duration_sec=5.0)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        data = response.json()
        segments = data["segments"]
        
        if len(segments) > 1:
            # Check segments are contiguous
            for i in range(1, len(segments)):
                prev_end = segments[i - 1]["end_sec"]
                curr_start = segments[i]["start_sec"]
                # Allow small floating point tolerance
                assert abs(prev_end - curr_start) < 0.001
    
    def test_timeline_smoothing_methods(self, client: TestClient) -> None:
        """Test different smoothing methods."""
        wav_bytes = create_wav_bytes(duration_sec=4.0)
        
        for method in ["none", "majority", "hysteresis", "ema"]:
            response = client.post(
                "/timeline",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"smoothing_method": method},
            )
            
            assert response.status_code == 200, f"Failed for method: {method}"
            data = response.json()
            assert data["smoothing"]["method"] == method
    
    def test_timeline_response_time_header(self, client: TestClient) -> None:
        """Test that response includes timing header."""
        wav_bytes = create_wav_bytes(duration_sec=3.0)
        
        response = client.post(
            "/timeline",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        assert "X-Response-Time" in response.headers
