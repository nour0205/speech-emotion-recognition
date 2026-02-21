"""Test fixtures for audio tests.

This module provides utilities for generating in-memory WAV files for testing.
No binary files are committed - fixtures are generated programmatically.
"""

import io
import math

import numpy as np
import soundfile as sf


def generate_sine_wav_bytes(
    frequency: float = 440.0,
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
    channels: int = 1,
) -> bytes:
    """Generate a sine wave WAV file as bytes.
    
    Args:
        frequency: Sine wave frequency in Hz.
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Amplitude (0.0 to 1.0).
        channels: Number of channels (1=mono, 2=stereo).
        
    Returns:
        WAV file as bytes.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
    signal = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    
    if channels > 1:
        # Stack channels (create multi-channel)
        signal = np.column_stack([signal] * channels)
    
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV", subtype="FLOAT")
    buffer.seek(0)
    return buffer.read()


def generate_silence_wav_bytes(
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    """Generate a silent (all zeros) WAV file as bytes.
    
    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        channels: Number of channels.
        
    Returns:
        WAV file as bytes.
    """
    num_samples = int(sample_rate * duration_sec)
    signal = np.zeros(num_samples, dtype=np.float32)
    
    if channels > 1:
        signal = np.column_stack([signal] * channels)
    
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV", subtype="FLOAT")
    buffer.seek(0)
    return buffer.read()


def generate_noise_wav_bytes(
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
    channels: int = 1,
    seed: int = 42,
) -> bytes:
    """Generate white noise WAV file as bytes.
    
    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Noise amplitude.
        channels: Number of channels.
        seed: Random seed for reproducibility.
        
    Returns:
        WAV file as bytes.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(sample_rate * duration_sec)
    signal = (amplitude * rng.uniform(-1, 1, num_samples)).astype(np.float32)
    
    if channels > 1:
        signal = np.column_stack([signal] * channels)
    
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV", subtype="FLOAT")
    buffer.seek(0)
    return buffer.read()
