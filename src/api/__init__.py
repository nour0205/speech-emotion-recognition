"""FastAPI application for Speech Emotion Recognition.

This module provides a REST API with endpoints for:
- /health: Service health check
- /predict: Single-clip emotion prediction
- /timeline: Emotion timeline generation

Example:
    To run the API server:
    
    $ uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from .main import app

__all__ = ["app"]
