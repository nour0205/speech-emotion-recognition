"""Tests for the /health API endpoint."""

import pytest
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
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """Create a test client with the app."""
    app = create_app(test_settings)
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""
    
    def test_health_returns_ok(self, client: TestClient) -> None:
        """Test that /health returns status ok."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "loading")
    
    def test_health_includes_model_id(self, client: TestClient, test_settings: Settings) -> None:
        """Test that /health includes model_id."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert data["model_id"] == test_settings.model_id
    
    def test_health_includes_device(self, client: TestClient, test_settings: Settings) -> None:
        """Test that /health includes device."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "device" in data
        assert data["device"] == test_settings.device
    
    def test_health_has_request_id_header(self, client: TestClient) -> None:
        """Test that response includes X-Request-ID header."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        # Should be a valid UUID or similar identifier
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 0
    
    def test_health_with_custom_request_id(self, client: TestClient) -> None:
        """Test that custom X-Request-ID is preserved."""
        custom_id = "test-request-123"
        response = client.get(
            "/health",
            headers={"X-Request-ID": custom_id},
        )
        
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id
