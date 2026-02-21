"""Tests for audioio.preprocess module."""

import pytest
import torch

from audioio import preprocess_audio, load_validate_preprocess, AudioConfig, load_wav_bytes
from audioio.errors import AudioPreprocessError

from tests.fixtures import generate_sine_wav_bytes


class TestPreprocessAudioBasic:
    """Basic preprocessing tests."""
    
    def test_mono_passthrough(self):
        """Mono audio should pass through with correct shape."""
        waveform = torch.randn(1, 16000)
        
        processed, sr = preprocess_audio(waveform, sample_rate=16000)
        
        assert processed.shape[0] == 1
        assert processed.dtype == torch.float32
        assert sr == 16000
    
    def test_stereo_to_mono(self):
        """Stereo audio should be averaged to mono."""
        # Create stereo with different channels
        left = torch.ones(1, 16000) * 0.5
        right = torch.ones(1, 16000) * 0.3
        waveform = torch.cat([left, right], dim=0)
        
        processed, sr = preprocess_audio(waveform, sample_rate=16000, normalize=False)
        
        assert processed.shape[0] == 1  # mono
        # Average of 0.5 and 0.3 = 0.4
        assert torch.allclose(processed, torch.ones(1, 16000) * 0.4)
    
    def test_output_shape_always_2d(self):
        """Output should always be [1, T] shape."""
        waveform = torch.randn(1, 32000)
        
        processed, sr = preprocess_audio(waveform, sample_rate=32000)
        
        assert processed.ndim == 2
        assert processed.shape[0] == 1


class TestPreprocessResampling:
    """Tests for resampling functionality."""
    
    def test_downsample_to_16k(self):
        """Audio should be downsampled to 16kHz."""
        # 1 second at 48kHz
        waveform = torch.randn(1, 48000)
        
        processed, sr = preprocess_audio(
            waveform, 
            sample_rate=48000, 
            target_sample_rate=16000,
        )
        
        assert sr == 16000
        # Should be approximately 16000 samples
        assert abs(processed.shape[1] - 16000) < 100  # Allow small variation
    
    def test_upsample_to_16k(self):
        """Audio should be upsampled to 16kHz."""
        # 1 second at 8kHz
        waveform = torch.randn(1, 8000)
        
        processed, sr = preprocess_audio(
            waveform,
            sample_rate=8000,
            target_sample_rate=16000,
        )
        
        assert sr == 16000
        assert abs(processed.shape[1] - 16000) < 100
    
    def test_no_resample_when_same_rate(self):
        """No resampling when rates match."""
        waveform = torch.randn(1, 16000)
        
        processed, sr = preprocess_audio(
            waveform,
            sample_rate=16000,
            target_sample_rate=16000,
            normalize=False,
        )
        
        assert sr == 16000
        assert processed.shape[1] == 16000


class TestPreprocessNormalization:
    """Tests for peak normalization."""
    
    def test_peak_normalization(self):
        """Audio should be peak normalized to target."""
        waveform = torch.tensor([[0.5, -0.25, 0.1, -0.3]])
        
        processed, sr = preprocess_audio(
            waveform,
            sample_rate=16000,
            normalize=True,
            peak_target=0.95,
        )
        
        # Peak should be approximately 0.95
        assert abs(processed.abs().max().item() - 0.95) < 0.01
    
    def test_quiet_audio_not_amplified(self):
        """Very quiet audio should not be amplified (eps check)."""
        waveform = torch.ones(1, 16000) * 1e-10
        
        processed, sr = preprocess_audio(
            waveform,
            sample_rate=16000,
            normalize=True,
            peak_target=0.95,
            eps=1e-8,
        )
        
        # Should NOT be amplified to 0.95
        assert processed.abs().max().item() < 0.01
    
    def test_normalization_disabled(self):
        """Normalization should be skippable."""
        waveform = torch.tensor([[0.5, -0.25, 0.1, -0.3]])
        
        processed, sr = preprocess_audio(
            waveform,
            sample_rate=16000,
            normalize=False,
        )
        
        # Peak should still be 0.5
        assert abs(processed.abs().max().item() - 0.5) < 0.01


class TestPreprocessMultiChannel:
    """Tests for multi-channel handling."""
    
    def test_multi_channel_raises_error(self):
        """More than 2 channels should raise error."""
        waveform = torch.randn(6, 16000)  # 5.1 surround
        
        with pytest.raises(AudioPreprocessError) as exc_info:
            preprocess_audio(waveform, sample_rate=16000)
        
        assert exc_info.value.code == "UNSUPPORTED_CHANNELS"


class TestPreprocessDeterminism:
    """Tests for deterministic behavior."""
    
    def test_same_input_same_output(self):
        """Same input should produce identical output."""
        waveform = torch.randn(1, 32000)
        
        processed1, sr1 = preprocess_audio(waveform.clone(), sample_rate=32000)
        processed2, sr2 = preprocess_audio(waveform.clone(), sample_rate=32000)
        
        assert sr1 == sr2
        assert torch.equal(processed1, processed2)
    
    def test_determinism_with_resampling(self):
        """Resampling should be deterministic."""
        waveform = torch.randn(1, 48000)
        
        processed1, _ = preprocess_audio(waveform.clone(), sample_rate=48000)
        processed2, _ = preprocess_audio(waveform.clone(), sample_rate=48000)
        
        assert torch.equal(processed1, processed2)


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_load_validate_preprocess_from_bytes(self):
        """Full pipeline from bytes works correctly."""
        wav_bytes = generate_sine_wav_bytes(
            frequency=440.0,
            duration_sec=1.0,
            sample_rate=44100,
            amplitude=0.5,
            channels=1,
        )
        
        config = AudioConfig(
            target_sample_rate=16000,
            normalize=True,
            peak_target=0.95,
        )
        
        waveform, sr = load_validate_preprocess(wav_bytes, config)
        
        assert waveform.shape[0] == 1  # mono
        assert sr == 16000
        assert waveform.dtype == torch.float32
        # Peak should be normalized
        assert abs(waveform.abs().max().item() - 0.95) < 0.01
    
    def test_stereo_to_mono_integration(self):
        """Stereo input becomes mono through pipeline."""
        wav_bytes = generate_sine_wav_bytes(
            frequency=440.0,
            duration_sec=0.5,
            sample_rate=16000,
            channels=2,
        )
        
        waveform, sr = load_validate_preprocess(wav_bytes, AudioConfig())
        
        assert waveform.shape[0] == 1  # mono output
    
    def test_end_to_end_determinism(self):
        """Full pipeline should be deterministic."""
        wav_bytes = generate_sine_wav_bytes(
            frequency=440.0,
            duration_sec=1.0,
            sample_rate=44100,
            channels=2,
        )
        
        config = AudioConfig()
        
        waveform1, sr1 = load_validate_preprocess(wav_bytes, config)
        waveform2, sr2 = load_validate_preprocess(wav_bytes, config)
        
        assert sr1 == sr2
        assert torch.allclose(waveform1, waveform2, atol=1e-7)
    
    def test_default_config(self):
        """Default config should work."""
        wav_bytes = generate_sine_wav_bytes(
            frequency=440.0,
            duration_sec=0.5,
            sample_rate=16000,
        )
        
        waveform, sr = load_validate_preprocess(wav_bytes)
        
        assert waveform.shape[0] == 1
        assert sr == 16000
