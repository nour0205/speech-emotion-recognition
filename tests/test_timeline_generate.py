"""Tests for timeline.generate module (timeline generation orchestration).

This module contains unit tests for the schema classes and optional
integration tests (behind RUN_INTEGRATION_TESTS env var) for the full
generate_timeline pipeline.
"""

import os

import pytest

from timeline.merge import MergeConfig
from timeline.schema import Segment, SmoothingInfo, TimelineResult, WindowPrediction
from timeline.smooth import SmoothingConfig
from timeline.windowing import WindowingConfig


class TestWindowPrediction:
    """Tests for WindowPrediction dataclass."""
    
    def test_basic_creation(self):
        """Test basic WindowPrediction creation."""
        wp = WindowPrediction(
            index=0,
            start_sec=0.0,
            end_sec=2.0,
            emotion="happy",
            confidence=0.85,
        )
        
        assert wp.index == 0
        assert wp.start_sec == 0.0
        assert wp.end_sec == 2.0
        assert wp.emotion == "happy"
        assert wp.confidence == 0.85
        assert wp.scores is None
        assert wp.is_padded is False
    
    def test_with_scores(self):
        """Test WindowPrediction with scores."""
        scores = {"happy": 0.85, "sad": 0.1, "neutral": 0.05}
        wp = WindowPrediction(
            index=0,
            start_sec=0.0,
            end_sec=2.0,
            emotion="happy",
            confidence=0.85,
            scores=scores,
        )
        
        assert wp.scores == scores
    
    def test_to_dict_with_scores(self):
        """Test to_dict method including scores."""
        wp = WindowPrediction(
            index=0,
            start_sec=0.0,
            end_sec=2.0,
            emotion="happy",
            confidence=0.85,
            scores={"happy": 0.85, "sad": 0.15},
        )
        
        d = wp.to_dict(include_scores=True)
        
        assert d["index"] == 0
        assert d["emotion"] == "happy"
        assert "scores" in d
        assert d["scores"]["happy"] == 0.85
    
    def test_to_dict_without_scores(self):
        """Test to_dict method excluding scores."""
        wp = WindowPrediction(
            index=0,
            start_sec=0.0,
            end_sec=2.0,
            emotion="happy",
            confidence=0.85,
            scores={"happy": 0.85},
        )
        
        d = wp.to_dict(include_scores=False)
        
        assert "scores" not in d


class TestSegment:
    """Tests for Segment dataclass."""
    
    def test_basic_creation(self):
        """Test basic Segment creation."""
        seg = Segment(
            start_sec=0.0,
            end_sec=3.5,
            emotion="happy",
            confidence=0.82,
        )
        
        assert seg.start_sec == 0.0
        assert seg.end_sec == 3.5
        assert seg.emotion == "happy"
        assert seg.confidence == 0.82
        assert seg.window_count == 1
    
    def test_duration_property(self):
        """Test duration_sec property."""
        seg = Segment(
            start_sec=1.0,
            end_sec=5.5,
            emotion="sad",
            confidence=0.75,
        )
        
        assert abs(seg.duration_sec - 4.5) < 1e-6
    
    def test_to_dict(self):
        """Test to_dict method."""
        seg = Segment(
            start_sec=0.0,
            end_sec=2.0,
            emotion="neutral",
            confidence=0.9,
            scores={"neutral": 0.9, "happy": 0.1},
        )
        
        d = seg.to_dict(include_scores=True)
        assert d["emotion"] == "neutral"
        assert "scores" in d
        
        d_no_scores = seg.to_dict(include_scores=False)
        assert "scores" not in d_no_scores


class TestSmoothingInfo:
    """Tests for SmoothingInfo dataclass."""
    
    def test_basic_creation(self):
        """Test basic SmoothingInfo creation."""
        info = SmoothingInfo(method="hysteresis")
        assert info.method == "hysteresis"
        assert info.params == {}
    
    def test_with_params(self):
        """Test SmoothingInfo with parameters."""
        info = SmoothingInfo(
            method="hysteresis",
            params={"hysteresis_min_run": 3},
        )
        
        d = info.to_dict()
        assert d["method"] == "hysteresis"
        assert d["hysteresis_min_run"] == 3


class TestTimelineResult:
    """Tests for TimelineResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic TimelineResult creation."""
        segment = Segment(
            start_sec=0.0,
            end_sec=2.0,
            emotion="happy",
            confidence=0.85,
        )
        
        result = TimelineResult(
            model_name="baseline",
            sample_rate=16000,
            duration_sec=10.5,
            window_sec=2.0,
            hop_sec=0.5,
            pad_mode="zero",
            smoothing={"method": "hysteresis"},
            segments=[segment],
        )
        
        assert result.model_name == "baseline"
        assert result.sample_rate == 16000
        assert result.duration_sec == 10.5
        assert len(result.segments) == 1
        assert result.windows is None
        assert result.is_padded_timeline is False
    
    def test_segment_count(self):
        """Test segment_count property."""
        result = TimelineResult(
            model_name="baseline",
            sample_rate=16000,
            duration_sec=10.0,
            window_sec=2.0,
            hop_sec=0.5,
            pad_mode="zero",
            smoothing={},
            segments=[
                Segment(0.0, 3.0, "happy", 0.8),
                Segment(3.0, 7.0, "sad", 0.7),
                Segment(7.0, 10.0, "neutral", 0.9),
            ],
        )
        
        assert result.segment_count == 3
    
    def test_window_count_none(self):
        """Test window_count property when windows not included."""
        result = TimelineResult(
            model_name="baseline",
            sample_rate=16000,
            duration_sec=5.0,
            window_sec=2.0,
            hop_sec=0.5,
            pad_mode="zero",
            smoothing={},
            segments=[],
            windows=None,
        )
        
        assert result.window_count is None
    
    def test_window_count_with_windows(self):
        """Test window_count property when windows included."""
        windows = [
            WindowPrediction(i, i * 0.5, i * 0.5 + 2.0, "happy", 0.8)
            for i in range(5)
        ]
        
        result = TimelineResult(
            model_name="baseline",
            sample_rate=16000,
            duration_sec=5.0,
            window_sec=2.0,
            hop_sec=0.5,
            pad_mode="zero",
            smoothing={},
            segments=[],
            windows=windows,
        )
        
        assert result.window_count == 5
    
    def test_to_dict_minimal(self):
        """Test to_dict with minimal options."""
        result = TimelineResult(
            model_name="baseline",
            sample_rate=16000,
            duration_sec=5.0,
            window_sec=2.0,
            hop_sec=0.5,
            pad_mode="zero",
            smoothing={"method": "none"},
            segments=[Segment(0.0, 5.0, "neutral", 0.9)],
        )
        
        d = result.to_dict(include_windows=False, include_scores=False)
        
        assert d["model_name"] == "baseline"
        assert d["sample_rate"] == 16000
        assert "windows" not in d
        assert len(d["segments"]) == 1
        assert "scores" not in d["segments"][0]
    
    def test_to_dict_with_windows(self):
        """Test to_dict including windows."""
        windows = [
            WindowPrediction(0, 0.0, 2.0, "happy", 0.85, scores={"happy": 0.85}),
        ]
        
        result = TimelineResult(
            model_name="baseline",
            sample_rate=16000,
            duration_sec=5.0,
            window_sec=2.0,
            hop_sec=0.5,
            pad_mode="zero",
            smoothing={},
            segments=[Segment(0.0, 5.0, "happy", 0.85)],
            windows=windows,
        )
        
        d = result.to_dict(include_windows=True, include_scores=True)
        
        assert "windows" in d
        assert len(d["windows"]) == 1
        assert d["windows"][0]["emotion"] == "happy"


class TestConfigCombinations:
    """Tests for various configuration combinations."""
    
    def test_windowing_config_defaults(self):
        """Test WindowingConfig default values."""
        config = WindowingConfig()
        assert config.window_sec == 2.0
        assert config.hop_sec == 0.5
        assert config.pad_mode == "zero"
    
    def test_smoothing_config_variants(self):
        """Test various SmoothingConfig settings."""
        configs = [
            SmoothingConfig(method="none"),
            SmoothingConfig(method="majority", majority_window=5),
            SmoothingConfig(method="hysteresis", hysteresis_min_run=3),
            SmoothingConfig(method="ema", ema_alpha=0.7),
        ]
        
        for config in configs:
            assert config.method in {"none", "majority", "hysteresis", "ema"}
    
    def test_merge_config_variants(self):
        """Test various MergeConfig settings."""
        configs = [
            MergeConfig(merge_adjacent=True),
            MergeConfig(merge_adjacent=False),
            MergeConfig(drop_short_segments=True, short_segment_strategy="merge_prev"),
            MergeConfig(drop_short_segments=True, short_segment_strategy="merge_best"),
        ]
        
        for config in configs:
            assert isinstance(config.merge_adjacent, bool)


# Integration tests - only run when RUN_INTEGRATION_TESTS=1
@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.",
)
class TestGenerateTimelineIntegration:
    """Integration tests for full timeline generation pipeline.
    
    These tests require the model to be available and may download
    model weights. They are skipped by default.
    """
    
    def test_generate_timeline_synthetic_audio(self):
        """Test generate_timeline_from_waveform with synthetic audio."""
        import torch
        
        from timeline.generate import generate_timeline_from_waveform
        
        # Create synthetic audio (3 seconds at 16kHz)
        sample_rate = 16000
        duration_sec = 3.0
        num_samples = int(sample_rate * duration_sec)
        
        # Generate sine wave with noise (speech-like)
        t = torch.linspace(0, duration_sec, num_samples)
        waveform = torch.sin(2 * torch.pi * 440 * t)  # 440Hz sine
        waveform += 0.1 * torch.randn(num_samples)  # Add noise
        waveform = waveform.unsqueeze(0).float()  # [1, T]
        waveform = waveform / waveform.abs().max()  # Normalize
        
        # Configure for short windows to get multiple
        windowing_config = WindowingConfig(
            window_sec=1.0,
            hop_sec=0.5,
            pad_mode="zero",
        )
        
        smoothing_config = SmoothingConfig(method="hysteresis", hysteresis_min_run=2)
        merge_config = MergeConfig()
        
        # Generate timeline
        result = generate_timeline_from_waveform(
            waveform=waveform,
            sample_rate=sample_rate,
            windowing_config=windowing_config,
            model_id="baseline",
            device="cpu",
            smoothing_config=smoothing_config,
            merge_config=merge_config,
            include_windows=True,
            include_scores=True,
        )
        
        # Validate result structure
        assert result.model_name == "baseline"
        assert result.sample_rate == sample_rate
        assert abs(result.duration_sec - duration_sec) < 0.1
        assert result.window_sec == 1.0
        assert result.hop_sec == 0.5
        
        # Should have segments
        assert len(result.segments) > 0
        for seg in result.segments:
            assert seg.start_sec >= 0
            assert seg.end_sec <= duration_sec + 0.5  # Allow for padding
            assert seg.emotion in {"neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"}
            assert 0 <= seg.confidence <= 1.0
        
        # Should have windows
        assert result.windows is not None
        assert len(result.windows) > 0
        for w in result.windows:
            assert w.scores is not None  # include_scores=True
    
    def test_generate_timeline_json_serializable(self):
        """Test that TimelineResult can be serialized to JSON."""
        import json
        
        import torch
        
        from timeline.generate import generate_timeline_from_waveform
        
        # Minimal synthetic audio
        waveform = torch.randn(1, 32000).float()  # 2 seconds
        
        result = generate_timeline_from_waveform(
            waveform=waveform,
            sample_rate=16000,
            include_windows=True,
            include_scores=True,
        )
        
        # Should be JSON serializable
        d = result.to_dict(include_windows=True, include_scores=True)
        json_str = json.dumps(d)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model_name"] == "baseline"
        assert "segments" in parsed
    
    def test_generate_timeline_deterministic(self):
        """Test that timeline generation is deterministic."""
        import torch
        
        from timeline.generate import generate_timeline_from_waveform
        
        # Fixed seed for reproducibility
        torch.manual_seed(42)
        waveform = torch.randn(1, 32000).float()
        
        config = WindowingConfig(window_sec=1.0, hop_sec=0.5)
        
        # Generate twice
        result1 = generate_timeline_from_waveform(
            waveform=waveform.clone(),
            sample_rate=16000,
            windowing_config=config,
        )
        
        result2 = generate_timeline_from_waveform(
            waveform=waveform.clone(),
            sample_rate=16000,
            windowing_config=config,
        )
        
        # Results should be identical
        assert len(result1.segments) == len(result2.segments)
        for s1, s2 in zip(result1.segments, result2.segments):
            assert s1.emotion == s2.emotion
            assert abs(s1.confidence - s2.confidence) < 1e-6


class TestTimelineResultValidation:
    """Tests for timeline result validation rules."""
    
    def test_segments_ordered_and_non_overlapping(self):
        """Test that segments are ordered and non-overlapping."""
        # Create properly ordered, non-overlapping segments
        segments = [
            Segment(0.0, 2.0, "happy", 0.8),
            Segment(2.0, 4.0, "sad", 0.7),
            Segment(4.0, 6.0, "neutral", 0.9),
        ]
        
        for i in range(1, len(segments)):
            # Each segment should start at or after previous end
            assert segments[i].start_sec >= segments[i-1].end_sec
        
        for seg in segments:
            # end_sec should be >= start_sec
            assert seg.end_sec >= seg.start_sec
    
    def test_final_segment_end_near_duration(self):
        """Test that final segment end is close to audio duration."""
        duration_sec = 10.5
        
        result = TimelineResult(
            model_name="test",
            sample_rate=16000,
            duration_sec=duration_sec,
            window_sec=2.0,
            hop_sec=0.5,
            pad_mode="zero",
            smoothing={},
            segments=[
                Segment(0.0, 5.0, "happy", 0.8),
                Segment(5.0, 10.5, "neutral", 0.9),
            ],
        )
        
        # Last segment end should be close to duration
        assert abs(result.segments[-1].end_sec - duration_sec) < 1.0
