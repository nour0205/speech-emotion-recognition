"""Tests for timeline.windowing module."""

import pytest
import torch

from timeline import WindowingConfig, segment_audio
from timeline.errors import WindowingConfigError, WindowingRuntimeError
from timeline.utils import samples_to_seconds, seconds_to_samples, validate_waveform_shape


# Standard sample rate for all tests
SAMPLE_RATE = 16000


class TestSecondsToSamples:
    """Tests for seconds_to_samples utility."""
    
    def test_basic_conversion(self):
        """Test basic seconds to samples conversion."""
        assert seconds_to_samples(2.0, 16000) == 32000
        assert seconds_to_samples(0.5, 16000) == 8000
        assert seconds_to_samples(1.0, 16000) == 16000
    
    def test_rounding(self):
        """Test that rounding is applied correctly."""
        # 0.1 * 16000 = 1600 exactly
        assert seconds_to_samples(0.1, 16000) == 1600
        # Small fractional values: 0.0000625 * 16000 = 1.0 rounds to 1
        assert seconds_to_samples(0.0000625, 16000) == 1


class TestSamplesToSeconds:
    """Tests for samples_to_seconds utility."""
    
    def test_basic_conversion(self):
        """Test basic samples to seconds conversion."""
        assert samples_to_seconds(32000, 16000) == 2.0
        assert samples_to_seconds(8000, 16000) == 0.5
        assert samples_to_seconds(16000, 16000) == 1.0


class TestValidateWaveformShape:
    """Tests for validate_waveform_shape utility."""
    
    def test_valid_shape(self):
        """Test valid [1, T] shape."""
        waveform = torch.zeros(1, 16000)
        channels, samples = validate_waveform_shape(waveform)
        assert channels == 1
        assert samples == 16000
    
    def test_1d_tensor_raises(self):
        """Test that 1D tensor raises ValueError."""
        waveform = torch.zeros(16000)
        with pytest.raises(ValueError, match="2 dimensions"):
            validate_waveform_shape(waveform)
    
    def test_3d_tensor_raises(self):
        """Test that 3D tensor raises ValueError."""
        waveform = torch.zeros(1, 1, 16000)
        with pytest.raises(ValueError, match="2 dimensions"):
            validate_waveform_shape(waveform)
    
    def test_stereo_raises(self):
        """Test that stereo (2 channels) raises ValueError."""
        waveform = torch.zeros(2, 16000)
        with pytest.raises(ValueError, match="1 channel"):
            validate_waveform_shape(waveform)


class TestWindowingConfig:
    """Tests for WindowingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WindowingConfig()
        assert config.window_sec == 2.0
        assert config.hop_sec == 0.5
        assert config.pad_mode == "zero"
        assert config.include_partial_last_window is True
        assert config.min_window_sec == 0.25
        assert config.max_window_sec == 10.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = WindowingConfig(
            window_sec=3.0,
            hop_sec=1.0,
            pad_mode="reflect",
            include_partial_last_window=False,
        )
        assert config.window_sec == 3.0
        assert config.hop_sec == 1.0
        assert config.pad_mode == "reflect"
        assert config.include_partial_last_window is False
    
    def test_window_sec_zero_raises(self):
        """Test that window_sec <= 0 raises error."""
        with pytest.raises(WindowingConfigError) as exc_info:
            WindowingConfig(window_sec=0)
        assert exc_info.value.code == "INVALID_CONFIG"
        assert "window_sec" in exc_info.value.details.get("parameter", "")
    
    def test_window_sec_negative_raises(self):
        """Test that negative window_sec raises error."""
        with pytest.raises(WindowingConfigError) as exc_info:
            WindowingConfig(window_sec=-1.0)
        assert exc_info.value.code == "INVALID_CONFIG"
    
    def test_hop_sec_zero_raises(self):
        """Test that hop_sec <= 0 raises error."""
        with pytest.raises(WindowingConfigError) as exc_info:
            WindowingConfig(hop_sec=0)
        assert exc_info.value.code == "INVALID_CONFIG"
    
    def test_hop_sec_greater_than_window_raises(self):
        """Test that hop_sec > window_sec raises error."""
        with pytest.raises(WindowingConfigError) as exc_info:
            WindowingConfig(window_sec=1.0, hop_sec=2.0)
        assert exc_info.value.code == "INVALID_CONFIG"
        assert exc_info.value.details.get("hop_sec") == 2.0
        assert exc_info.value.details.get("window_sec") == 1.0
    
    def test_window_below_min_raises(self):
        """Test that window_sec < min_window_sec raises error."""
        with pytest.raises(WindowingConfigError) as exc_info:
            WindowingConfig(window_sec=0.1, min_window_sec=0.25)
        assert exc_info.value.code == "INVALID_CONFIG"
    
    def test_window_above_max_raises(self):
        """Test that window_sec > max_window_sec raises error."""
        with pytest.raises(WindowingConfigError) as exc_info:
            WindowingConfig(window_sec=15.0, max_window_sec=10.0)
        assert exc_info.value.code == "INVALID_CONFIG"
    
    def test_invalid_pad_mode_raises(self):
        """Test that invalid pad_mode raises error."""
        with pytest.raises(WindowingConfigError) as exc_info:
            WindowingConfig(pad_mode="invalid")
        assert exc_info.value.code == "INVALID_CONFIG"
        assert "pad_mode" in exc_info.value.details.get("parameter", "")


class TestSegmentAudioBasic:
    """Basic tests for segment_audio function."""
    
    def test_single_exact_window(self):
        """Test with exactly one window (input == window size)."""
        waveform = torch.randn(1, 32000)  # 2.0 seconds
        config = WindowingConfig(window_sec=2.0, hop_sec=0.5)
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # With hop=0.5s, starts: 0, 8000, 16000, 24000
        # Only first fits completely, rest need padding
        assert len(windows) >= 1
        assert windows[0]["index"] == 0
        assert windows[0]["start_sec"] == 0.0
        assert windows[0]["end_sec"] == 2.0
        assert windows[0]["start_sample"] == 0
        assert windows[0]["end_sample"] == 32000
        assert windows[0]["waveform"].shape == (1, 32000)
        assert windows[0]["is_padded"] is False
    
    def test_multiple_windows_no_padding(self):
        """Test with multiple complete windows that fit exactly."""
        # 4 seconds at 16kHz = 64000 samples
        # window=2s=32000, hop=1s=16000
        # starts: 0, 16000, 32000 -> 3 complete windows
        waveform = torch.randn(1, 64000)
        config = WindowingConfig(window_sec=2.0, hop_sec=1.0, include_partial_last_window=False)
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        assert len(windows) == 3
        
        assert windows[0]["start_sample"] == 0
        assert windows[0]["end_sample"] == 32000
        
        assert windows[1]["start_sample"] == 16000
        assert windows[1]["end_sample"] == 48000
        
        assert windows[2]["start_sample"] == 32000
        assert windows[2]["end_sample"] == 64000
        
        for w in windows:
            assert w["is_padded"] is False


class TestOverlapCorrectness:
    """Tests verifying overlap calculations."""
    
    def test_hop_sample_spacing(self):
        """Verify window spacing equals hop_samples."""
        waveform = torch.randn(1, 80000)  # 5 seconds
        config = WindowingConfig(window_sec=2.0, hop_sec=0.5)
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        hop_samples = seconds_to_samples(0.5, SAMPLE_RATE)  # 8000
        
        for i in range(len(windows) - 1):
            expected_diff = hop_samples
            actual_diff = windows[i + 1]["start_sample"] - windows[i]["start_sample"]
            assert actual_diff == expected_diff, f"Window {i} to {i+1}: expected {expected_diff}, got {actual_diff}"
    
    def test_windows_overlap_correctly(self):
        """Test that consecutive windows have correct overlap region."""
        waveform = torch.randn(1, 48000)  # 3 seconds
        config = WindowingConfig(window_sec=1.0, hop_sec=0.25)
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # Each window should share 0.75s (12000 samples) with the next
        window_samples = 16000
        hop_samples = 4000
        overlap_samples = window_samples - hop_samples  # 12000
        
        for i in range(len(windows) - 1):
            if not windows[i]["is_padded"] and not windows[i + 1]["is_padded"]:
                w1_end = windows[i]["end_sample"]
                w2_start = windows[i + 1]["start_sample"]
                actual_overlap = w1_end - w2_start
                assert actual_overlap == overlap_samples


class TestPaddingBehavior:
    """Tests for different padding modes."""
    
    def test_zero_padding_last_window(self):
        """Test zero padding of partial last window."""
        # 2.5 seconds = 40000 samples
        # window=2s=32000, hop=1s=16000
        # starts: 0, 16000, 32000
        # ends: 32000, 48000, 64000
        # Window 0: 0-32000 fits
        # Window 1: 16000-48000 needs padding (24000 actual + 8000 pad)
        # Window 2: 32000-64000 needs padding (8000 actual + 24000 pad)
        waveform = torch.randn(1, 40000)
        config = WindowingConfig(window_sec=2.0, hop_sec=1.0, pad_mode="zero")
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        assert len(windows) == 3
        
        # First window - no padding
        assert windows[0]["is_padded"] is False
        assert windows[0]["waveform"].shape == (1, 32000)
        
        # Second window - padded
        assert windows[1]["is_padded"] is True
        assert windows[1]["waveform"].shape == (1, 32000)
        assert windows[1]["actual_end_sample"] == 40000
        
        # Verify last 8000 samples are zeros (window 1 has 24000 actual, 8000 pad)
        padded_region = windows[1]["waveform"][:, 24000:]
        assert torch.all(padded_region == 0), "Padded region should be zeros"
        
        # Third window - heavily padded (8000 actual + 24000 pad)
        assert windows[2]["is_padded"] is True
        assert windows[2]["waveform"].shape == (1, 32000)
    
    def test_reflect_padding_last_window(self):
        """Test reflect padding of partial last window."""
        # Create deterministic waveform for reflect test
        waveform = torch.arange(40000, dtype=torch.float32).unsqueeze(0)  # [1, 40000]
        config = WindowingConfig(window_sec=2.0, hop_sec=1.0, pad_mode="reflect")
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # starts: 0, 16000, 32000 -> 3 windows
        assert len(windows) == 3
        assert windows[1]["is_padded"] is True
        assert windows[1]["waveform"].shape == (1, 32000)
        
        # Window 1 starts at 16000, has 24000 actual samples (16000-40000), 8000 padded
        last_window = windows[1]["waveform"]
        actual_samples = last_window[:, :24000]  # Actual data
        padded_samples = last_window[:, 24000:]  # Reflected padding (8000 samples)
        
        # Reflected region should be flip of last 8000 samples
        expected_reflected = torch.flip(actual_samples[:, -8000:], dims=[1])
        assert torch.equal(padded_samples, expected_reflected), "Reflect padding should be flipped tail"
    
    def test_no_padding_partial_window(self):
        """Test pad_mode='none' results in shorter last window."""
        waveform = torch.randn(1, 40000)  # 2.5 seconds
        config = WindowingConfig(window_sec=2.0, hop_sec=1.0, pad_mode="none")
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # starts: 0, 16000, 32000 -> 3 windows
        assert len(windows) == 3
        
        # First window - full size
        assert windows[0]["waveform"].shape == (1, 32000)
        assert windows[0]["is_padded"] is False
        
        # Second window - shorter (24000 actual samples: 16000-40000)
        assert windows[1]["waveform"].shape == (1, 24000)
        assert windows[1]["is_padded"] is False
        assert windows[1]["end_sample"] == 40000
        assert windows[1]["end_sec"] == 2.5
        
        # Third window - shortest (8000 actual samples: 32000-40000)
        assert windows[2]["waveform"].shape == (1, 8000)
        assert windows[2]["is_padded"] is False
        assert windows[2]["end_sample"] == 40000
    
    def test_exclude_partial_last_window(self):
        """Test include_partial_last_window=False drops partial window."""
        waveform = torch.randn(1, 40000)  # 2.5 seconds
        config = WindowingConfig(
            window_sec=2.0,
            hop_sec=1.0,
            include_partial_last_window=False,
        )
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # Only first window fits completely
        assert len(windows) == 1
        assert windows[0]["start_sample"] == 0
        assert windows[0]["end_sample"] == 32000


class TestInvalidInputs:
    """Tests for invalid input handling."""
    
    def test_1d_waveform_raises(self):
        """Test that 1D waveform raises WindowingRuntimeError."""
        waveform = torch.randn(16000)  # [T] instead of [1, T]
        config = WindowingConfig()
        
        with pytest.raises(WindowingRuntimeError) as exc_info:
            segment_audio(waveform, SAMPLE_RATE, config)
        
        assert exc_info.value.code == "INVALID_SHAPE"
    
    def test_stereo_waveform_raises(self):
        """Test that stereo waveform raises WindowingRuntimeError."""
        waveform = torch.randn(2, 16000)  # [2, T] instead of [1, T]
        config = WindowingConfig()
        
        with pytest.raises(WindowingRuntimeError) as exc_info:
            segment_audio(waveform, SAMPLE_RATE, config)
        
        assert exc_info.value.code == "INVALID_SHAPE"
    
    def test_empty_waveform_raises(self):
        """Test that empty waveform raises WindowingRuntimeError."""
        waveform = torch.randn(1, 0)  # [1, 0]
        config = WindowingConfig()
        
        with pytest.raises(WindowingRuntimeError) as exc_info:
            segment_audio(waveform, SAMPLE_RATE, config)
        
        assert exc_info.value.code == "EMPTY_INPUT"


class TestDeterminism:
    """Tests for deterministic behavior."""
    
    def test_same_input_same_output(self):
        """Running segment_audio twice produces identical results."""
        torch.manual_seed(42)
        waveform = torch.randn(1, 48000)
        config = WindowingConfig(window_sec=2.0, hop_sec=0.5, pad_mode="zero")
        
        windows1 = segment_audio(waveform, SAMPLE_RATE, config)
        windows2 = segment_audio(waveform, SAMPLE_RATE, config)
        
        assert len(windows1) == len(windows2)
        
        for w1, w2 in zip(windows1, windows2):
            assert w1["index"] == w2["index"]
            assert w1["start_sec"] == w2["start_sec"]
            assert w1["end_sec"] == w2["end_sec"]
            assert w1["start_sample"] == w2["start_sample"]
            assert w1["end_sample"] == w2["end_sample"]
            assert w1["is_padded"] == w2["is_padded"]
            assert torch.equal(w1["waveform"], w2["waveform"])
    
    def test_reflect_padding_deterministic(self):
        """Reflect padding should be deterministic."""
        waveform = torch.arange(40000, dtype=torch.float32).unsqueeze(0)
        config = WindowingConfig(window_sec=2.0, hop_sec=1.0, pad_mode="reflect")
        
        windows1 = segment_audio(waveform, SAMPLE_RATE, config)
        windows2 = segment_audio(waveform, SAMPLE_RATE, config)
        
        # Compare padded windows
        last1 = windows1[-1]["waveform"]
        last2 = windows2[-1]["waveform"]
        
        assert torch.equal(last1, last2), "Reflect padding should be deterministic"


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_very_short_audio(self):
        """Test with audio shorter than window size."""
        waveform = torch.randn(1, 8000)  # 0.5 seconds
        config = WindowingConfig(window_sec=2.0, hop_sec=0.5, pad_mode="zero")
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # Should produce one padded window
        assert len(windows) == 1
        assert windows[0]["is_padded"] is True
        assert windows[0]["waveform"].shape == (1, 32000)  # Full window size
    
    def test_exact_multiple_of_hop(self):
        """Test when audio length is exact multiple of hop."""
        # 64000 samples = 4 seconds
        # window=1s=16000, hop=0.5s=8000
        # starts: 0,8000,16000,24000,32000,40000,48000 -> 7 starts
        # window ending: 16000,24000,32000,40000,48000,56000,64000
        # All fit! 7 complete windows
        waveform = torch.randn(1, 64000)
        config = WindowingConfig(window_sec=1.0, hop_sec=0.5, include_partial_last_window=False)
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        assert len(windows) == 7
        for w in windows:
            assert w["is_padded"] is False
    
    def test_window_equals_hop(self):
        """Test when window_sec equals hop_sec (no overlap)."""
        waveform = torch.randn(1, 48000)  # 3 seconds
        config = WindowingConfig(window_sec=1.0, hop_sec=1.0)
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        assert len(windows) == 3
        
        # No overlap between consecutive windows
        assert windows[0]["end_sample"] == windows[1]["start_sample"]
        assert windows[1]["end_sample"] == windows[2]["start_sample"]
    
    def test_small_hop_many_windows(self):
        """Test with small hop producing many overlapping windows."""
        waveform = torch.randn(1, 32000)  # 2 seconds
        config = WindowingConfig(window_sec=1.0, hop_sec=0.25)  # 75% overlap
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # hop=4000, window=16000
        # starts: 0, 4000, 8000, 12000, 16000 (5 complete, ending at 32000)
        # + partial: 20000, 24000, 28000 (3 partial, padded)
        # Total: 8 windows with default include_partial_last_window=True
        assert len(windows) == 8
        
        # First 5 should not be padded
        for i in range(5):
            assert windows[i]["is_padded"] is False
        
        # Last 3 should be padded
        for i in range(5, 8):
            assert windows[i]["is_padded"] is True
    
    def test_timestamps_precision(self):
        """Test timestamp precision."""
        waveform = torch.randn(1, 48000)  # 3 seconds
        config = WindowingConfig(window_sec=1.0, hop_sec=0.5, include_partial_last_window=False)
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # hop=8000, window=16000
        # starts: 0, 8000, 16000, 24000, 32000 -> 5 complete windows
        expected_starts = [0.0, 0.5, 1.0, 1.5, 2.0]
        expected_ends = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        assert len(windows) == 5
        for i, w in enumerate(windows):
            assert abs(w["start_sec"] - expected_starts[i]) < 1e-9
            assert abs(w["end_sec"] - expected_ends[i]) < 1e-9


class TestIntegrationWithExactFit:
    """Integration tests matching spec examples."""
    
    def test_spec_example_exact_fit(self):
        """
        Test spec example: T=32000 (2.0s), window=2.0s, hop=0.5s
        
        starts: 0, 8000, 16000, 24000
        ends: 32000, 40000, 48000, 56000
        
        Only first fits (end=32000 <= T=32000), rest need padding.
        """
        waveform = torch.randn(1, 32000)
        config = WindowingConfig(window_sec=2.0, hop_sec=0.5, pad_mode="zero")
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # First window: 0-32000 (fits)
        assert windows[0]["start_sample"] == 0
        assert windows[0]["end_sample"] == 32000
        assert windows[0]["is_padded"] is False
        
        # Second window: 8000-40000 (needs pad)
        assert windows[1]["start_sample"] == 8000
        assert windows[1]["end_sample"] == 40000  # virtual end
        assert windows[1]["is_padded"] is True
        
        # Third window: 16000-48000 (needs pad)
        assert windows[2]["start_sample"] == 16000
        assert windows[2]["end_sample"] == 48000
        assert windows[2]["is_padded"] is True
        
        # Fourth window: 24000-56000 (needs pad)
        assert windows[3]["start_sample"] == 24000
        assert windows[3]["end_sample"] == 56000
        assert windows[3]["is_padded"] is True
        
        assert len(windows) == 4
    
    def test_spec_example_exclude_partial(self):
        """Same as above but with include_partial_last_window=False."""
        waveform = torch.randn(1, 32000)
        config = WindowingConfig(
            window_sec=2.0,
            hop_sec=0.5,
            include_partial_last_window=False,
        )
        
        windows = segment_audio(waveform, SAMPLE_RATE, config)
        
        # Only first window fits
        assert len(windows) == 1
        assert windows[0]["start_sample"] == 0
        assert windows[0]["end_sample"] == 32000
