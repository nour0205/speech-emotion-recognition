"""Tests for audioio.loader module."""

import io
import tempfile
from pathlib import Path

import pytest
import torch

from audioio import load_wav, load_wav_bytes
from audioio.errors import AudioDecodeError

from tests.fixtures import generate_sine_wav_bytes, generate_silence_wav_bytes


class TestLoadWavBytes:
    """Tests for load_wav_bytes function."""
    
    def test_load_mono_sine_wav(self):
        """Load a valid mono sine wave WAV."""
        wav_bytes = generate_sine_wav_bytes(
            frequency=440.0,
            duration_sec=1.0,
            sample_rate=16000,
            channels=1,
        )
        
        waveform, sr = load_wav_bytes(wav_bytes)
        
        assert isinstance(waveform, torch.Tensor)
        assert waveform.dtype == torch.float32
        assert waveform.shape[0] == 1  # mono
        assert waveform.shape[1] == 16000  # 1 second at 16kHz
        assert sr == 16000
    
    def test_load_stereo_wav(self):
        """Load a valid stereo WAV."""
        wav_bytes = generate_sine_wav_bytes(
            frequency=440.0,
            duration_sec=0.5,
            sample_rate=22050,
            channels=2,
        )
        
        waveform, sr = load_wav_bytes(wav_bytes)
        
        assert waveform.shape[0] == 2  # stereo
        assert sr == 22050
    
    def test_empty_bytes_raises_error(self):
        """Empty bytes should raise AudioDecodeError."""
        with pytest.raises(AudioDecodeError) as exc_info:
            load_wav_bytes(b"")
        
        assert exc_info.value.code == "EMPTY_FILE"
    
    def test_invalid_bytes_raises_error(self):
        """Invalid WAV bytes should raise AudioDecodeError."""
        with pytest.raises(AudioDecodeError) as exc_info:
            load_wav_bytes(b"not a wav file")
        
        assert exc_info.value.code == "INVALID_WAV"
        assert "bytes_length" in exc_info.value.details
    
    def test_random_garbage_raises_error(self):
        """Random garbage bytes should raise AudioDecodeError."""
        import os
        garbage = os.urandom(1024)
        
        with pytest.raises(AudioDecodeError) as exc_info:
            load_wav_bytes(garbage)
        
        assert exc_info.value.code == "INVALID_WAV"


class TestLoadWav:
    """Tests for load_wav function."""
    
    def test_load_from_file(self, tmp_path: Path):
        """Load a valid WAV from disk."""
        wav_bytes = generate_sine_wav_bytes(
            frequency=440.0,
            duration_sec=0.5,
            sample_rate=16000,
            channels=1,
        )
        
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(wav_bytes)
        
        waveform, sr = load_wav(wav_path)
        
        assert isinstance(waveform, torch.Tensor)
        assert waveform.dtype == torch.float32
        assert waveform.shape[0] == 1
        assert sr == 16000
    
    def test_load_from_string_path(self, tmp_path: Path):
        """Load using string path."""
        wav_bytes = generate_sine_wav_bytes(channels=1)
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(wav_bytes)
        
        waveform, sr = load_wav(str(wav_path))
        
        assert waveform.shape[0] == 1
    
    def test_file_not_found(self):
        """Non-existent file should raise AudioDecodeError."""
        with pytest.raises(AudioDecodeError) as exc_info:
            load_wav("/nonexistent/path/audio.wav")
        
        assert exc_info.value.code == "FILE_NOT_FOUND"
    
    def test_empty_file(self, tmp_path: Path):
        """Empty file should raise AudioDecodeError."""
        wav_path = tmp_path / "empty.wav"
        wav_path.write_bytes(b"")
        
        with pytest.raises(AudioDecodeError) as exc_info:
            load_wav(wav_path)
        
        assert exc_info.value.code == "EMPTY_FILE"
    
    def test_invalid_file(self, tmp_path: Path):
        """Invalid WAV file should raise AudioDecodeError."""
        wav_path = tmp_path / "invalid.wav"
        wav_path.write_bytes(b"not a wav file content")
        
        with pytest.raises(AudioDecodeError) as exc_info:
            load_wav(wav_path)
        
        assert exc_info.value.code == "INVALID_WAV"


class TestLoaderConsistency:
    """Tests for consistency between load_wav and load_wav_bytes."""
    
    def test_same_result_from_file_and_bytes(self, tmp_path: Path):
        """load_wav and load_wav_bytes should return identical results."""
        wav_bytes = generate_sine_wav_bytes(
            frequency=440.0,
            duration_sec=0.5,
            sample_rate=16000,
            channels=1,
        )
        
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(wav_bytes)
        
        waveform_file, sr_file = load_wav(wav_path)
        waveform_bytes, sr_bytes = load_wav_bytes(wav_bytes)
        
        assert sr_file == sr_bytes
        assert torch.allclose(waveform_file, waveform_bytes, atol=1e-6)
