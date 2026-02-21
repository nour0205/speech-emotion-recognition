"""Pytest configuration and fixtures for the test suite."""

import os
import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_waveform() -> tuple[torch.Tensor, int]:
    """Generate a sample waveform for testing.
    
    Returns:
        Tuple of (waveform, sample_rate) where waveform is [1, 32000] (2 seconds).
    """
    sample_rate = 16000
    duration = 2.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Generate speech-like audio with multiple frequencies
    waveform = torch.zeros_like(t)
    for freq in [100, 200, 300, 400]:
        waveform += torch.sin(2 * torch.pi * freq * t) / 4
    waveform = waveform.unsqueeze(0).float()
    return waveform, sample_rate


@pytest.fixture
def short_waveform() -> tuple[torch.Tensor, int]:
    """Generate a short waveform (0.5 seconds).
    
    Returns:
        Tuple of (waveform, sample_rate).
    """
    sample_rate = 16000
    duration = 0.5
    samples = int(sample_rate * duration)
    waveform = torch.randn(1, samples).float() * 0.3
    return waveform, sample_rate


@pytest.fixture
def noise_waveform() -> tuple[torch.Tensor, int]:
    """Generate a noise waveform.
    
    Returns:
        Tuple of (waveform, sample_rate).
    """
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    waveform = torch.randn(1, samples).float() * 0.5
    return waveform, sample_rate


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests (require model download)"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow"
    )
