"""HTTP client for the Speech Emotion Recognition API."""

import httpx
from typing import Optional


class EmotionAPIClient:
    """Client for the emotion recognition backend API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(60.0)  # Model inference can take time
    
    def health_check(self) -> dict:
        """Check if the API is healthy."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
    
    def analyze_emotion(self, audio_bytes: bytes, filename: str = "audio.wav") -> dict:
        """Send audio to API for emotion analysis.
        
        Args:
            audio_bytes: Raw WAV audio bytes.
            filename: Filename to send with the request.
            
        Returns:
            Dictionary with label, emotion, confidence, and inference_time_sec.
        """
        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, audio_bytes, "audio/wav")}
            response = client.post(
                f"{self.base_url}/api/v1/emotion/analyze",
                files=files
            )
            response.raise_for_status()
            return response.json()
    
    def list_emotions(self) -> list[dict]:
        """Get list of supported emotions."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/api/v1/emotions")
            response.raise_for_status()
            return response.json()["emotions"]


# Default client instance
_client: Optional[EmotionAPIClient] = None


def get_client(base_url: str = "http://localhost:8000") -> EmotionAPIClient:
    """Get or create API client singleton."""
    global _client
    if _client is None or _client.base_url != base_url:
        _client = EmotionAPIClient(base_url)
    return _client
