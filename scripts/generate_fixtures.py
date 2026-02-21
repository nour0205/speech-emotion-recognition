#!/usr/bin/env python3
"""Generate test audio fixtures.

Creates sample WAV files in tests/fixtures/ for testing purposes.

Usage:
    python scripts/generate_fixtures.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_sine_wave(
    frequency: float = 440.0,
    duration: float = 2.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def generate_white_noise(
    duration: float = 2.0,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate white noise."""
    samples = int(sample_rate * duration)
    return (amplitude * np.random.randn(samples)).astype(np.float32)


def generate_speech_like(
    duration: float = 2.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate speech-like audio (combination of frequencies)."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    
    # Mix of frequencies common in speech
    audio = np.zeros_like(t)
    frequencies = [85, 170, 255, 340, 425, 510]  # Typical male formants
    
    for i, freq in enumerate(frequencies):
        amplitude = 0.3 / (i + 1)  # Decreasing amplitude
        audio += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add some noise
    audio += 0.05 * np.random.randn(len(t)).astype(np.float32)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def main():
    """Generate test fixtures."""
    fixtures_dir = project_root / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating test fixtures in {fixtures_dir}")
    
    # Generate example.wav (speech-like)
    example_path = fixtures_dir / "example.wav"
    audio = generate_speech_like(duration=2.0, sample_rate=16000)
    sf.write(example_path, audio, 16000)
    print(f"  Created: {example_path}")
    
    # Generate sine wave
    sine_path = fixtures_dir / "sine_440hz.wav"
    audio = generate_sine_wave(frequency=440.0, duration=1.5, sample_rate=16000)
    sf.write(sine_path, audio, 16000)
    print(f"  Created: {sine_path}")
    
    # Generate noise
    noise_path = fixtures_dir / "white_noise.wav"
    audio = generate_white_noise(duration=1.0, sample_rate=16000)
    sf.write(noise_path, audio, 16000)
    print(f"  Created: {noise_path}")
    
    # Generate short audio (minimum duration)
    short_path = fixtures_dir / "short_500ms.wav"
    audio = generate_speech_like(duration=0.5, sample_rate=16000)
    sf.write(short_path, audio, 16000)
    print(f"  Created: {short_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
