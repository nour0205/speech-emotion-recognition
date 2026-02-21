"""Tests for audioio.validate module."""

import math

import pytest
import torch

from audioio import validate_wav
from audioio.errors import AudioValidationError


class TestValidateWavBasic:
    """Basic validation tests."""
    
    def test_valid_mono_audio(self):
        """Valid mono audio should pass validation."""
        waveform = torch.randn(1, 16000)  # 1 second at 16kHz
        
        # Should not raise
        validate_wav(waveform, sample_rate=16000)
    
    def test_valid_stereo_audio(self):
        """Valid stereo audio should pass with allow_stereo=True."""
        waveform = torch.randn(2, 16000)
        
        validate_wav(waveform, sample_rate=16000, allow_stereo=True)
    
    def test_short_audio_at_boundary(self):
        """Audio exactly at min duration should pass."""
        # 0.1 seconds at 16kHz = 1600 samples
        waveform = torch.randn(1, 1600)
        
        validate_wav(waveform, sample_rate=16000, min_duration_sec=0.1)


class TestValidateDtype:
    """Tests for dtype validation."""
    
    def test_non_tensor_raises_error(self):
        """Non-tensor input should raise INVALID_DTYPE."""
        import numpy as np
        waveform = np.random.randn(1, 16000)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "INVALID_DTYPE"
    
    def test_int_tensor_raises_error(self):
        """Integer tensor should raise INVALID_DTYPE."""
        waveform = torch.randint(-32768, 32767, (1, 16000), dtype=torch.int16)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "INVALID_DTYPE"
    
    def test_float64_passes(self):
        """Float64 tensor should pass."""
        waveform = torch.randn(1, 16000, dtype=torch.float64)
        
        validate_wav(waveform, sample_rate=16000)


class TestValidateEmpty:
    """Tests for empty audio validation."""
    
    def test_zero_samples_raises_error(self):
        """Empty waveform should raise EMPTY_AUDIO."""
        waveform = torch.empty(1, 0)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "EMPTY_AUDIO"
    
    def test_1d_tensor_raises_error(self):
        """1D tensor should raise error (need 2D)."""
        waveform = torch.randn(16000)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "INVALID_DTYPE"


class TestValidateNonFinite:
    """Tests for NaN/Inf detection."""
    
    def test_nan_values_raise_error(self):
        """Waveform with NaN should raise NON_FINITE."""
        waveform = torch.randn(1, 16000)
        waveform[0, 1000] = float("nan")
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "NON_FINITE"
        assert exc_info.value.details["nan_count"] == 1
    
    def test_inf_values_raise_error(self):
        """Waveform with Inf should raise NON_FINITE."""
        waveform = torch.randn(1, 16000)
        waveform[0, 500] = float("inf")
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "NON_FINITE"
        assert exc_info.value.details["inf_count"] == 1
    
    def test_negative_inf_raises_error(self):
        """Waveform with -Inf should raise NON_FINITE."""
        waveform = torch.randn(1, 16000)
        waveform[0, 0] = float("-inf")
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "NON_FINITE"


class TestValidateSampleRate:
    """Tests for sample rate validation."""
    
    def test_sample_rate_too_low(self):
        """Sample rate below 8kHz should raise INVALID_SAMPLE_RATE."""
        waveform = torch.randn(1, 1000)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=4000)
        
        assert exc_info.value.code == "INVALID_SAMPLE_RATE"
    
    def test_sample_rate_too_high(self):
        """Sample rate above 192kHz should raise INVALID_SAMPLE_RATE."""
        waveform = torch.randn(1, 10000)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=200000)
        
        assert exc_info.value.code == "INVALID_SAMPLE_RATE"
    
    def test_boundary_sample_rates_pass(self):
        """Boundary sample rates should pass."""
        waveform = torch.randn(1, 16000)
        
        # Minimum (8kHz)
        validate_wav(waveform, sample_rate=8000, min_duration_sec=0.01)
        
        # Maximum (192kHz)
        validate_wav(waveform, sample_rate=192000, min_duration_sec=0.01)


class TestValidateDuration:
    """Tests for duration validation."""
    
    def test_too_short_raises_error(self):
        """Audio shorter than min_duration should raise TOO_SHORT."""
        # 0.05 seconds at 16kHz = 800 samples
        waveform = torch.randn(1, 800)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000, min_duration_sec=0.1)
        
        assert exc_info.value.code == "TOO_SHORT"
        assert "duration_sec" in exc_info.value.details
    
    def test_too_long_raises_error(self):
        """Audio longer than max_duration should raise TOO_LONG."""
        # 2 seconds at 16kHz
        waveform = torch.randn(1, 32000)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000, max_duration_sec=1.0)
        
        assert exc_info.value.code == "TOO_LONG"
        assert exc_info.value.details["duration_sec"] == 2.0


class TestValidateChannels:
    """Tests for channel validation."""
    
    def test_stereo_rejected_when_not_allowed(self):
        """Stereo audio should be rejected when allow_stereo=False."""
        waveform = torch.randn(2, 16000)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000, allow_stereo=False)
        
        assert exc_info.value.code == "TOO_MANY_CHANNELS"
    
    def test_multi_channel_rejected_by_default(self):
        """Multi-channel (>2) audio should be rejected by default."""
        waveform = torch.randn(6, 16000)  # 5.1 surround
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "TOO_MANY_CHANNELS"
    
    def test_multi_channel_allowed_when_enabled(self):
        """Multi-channel audio should pass when allow_multi_channel=True."""
        waveform = torch.randn(6, 16000)
        
        validate_wav(waveform, sample_rate=16000, allow_multi_channel=True)


class TestValidateSilence:
    """Tests for silence detection."""
    
    def test_silence_rejected_by_default(self):
        """Near-silent audio should be rejected by default."""
        waveform = torch.zeros(1, 16000)
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "SILENCE"
        assert "rms" in exc_info.value.details
    
    def test_near_silence_rejected(self):
        """Very quiet audio should be rejected."""
        waveform = torch.ones(1, 16000) * 1e-6
        
        with pytest.raises(AudioValidationError) as exc_info:
            validate_wav(waveform, sample_rate=16000, silence_rms_threshold=1e-4)
        
        assert exc_info.value.code == "SILENCE"
    
    def test_silence_allowed_when_disabled(self):
        """Silent audio should pass when reject_silence=False."""
        waveform = torch.zeros(1, 16000)
        
        validate_wav(waveform, sample_rate=16000, reject_silence=False)
    
    def test_normal_audio_passes_silence_check(self):
        """Normal audio should pass silence detection."""
        waveform = torch.randn(1, 16000) * 0.1  # moderate amplitude
        
        validate_wav(waveform, sample_rate=16000, reject_silence=True)
