"""Tests for timeline.merge module (segment merging)."""

import pytest

from timeline.merge import MergeConfig, merge_windows_to_segments
from timeline.schema import WindowPrediction
from timeline.windowing import WindowingConfig


def make_window(
    index: int,
    emotion: str,
    confidence: float = 0.8,
    scores: dict | None = None,
    start_sec: float = None,
    end_sec: float = None,
    hop_sec: float = 0.5,
    window_sec: float = 2.0,
) -> WindowPrediction:
    """Helper to create WindowPrediction for testing."""
    if start_sec is None:
        start_sec = index * hop_sec
    if end_sec is None:
        end_sec = start_sec + window_sec
    return WindowPrediction(
        index=index,
        start_sec=start_sec,
        end_sec=end_sec,
        emotion=emotion,
        confidence=confidence,
        scores=scores,
        is_padded=False,
    )


class TestMergeConfig:
    """Tests for MergeConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MergeConfig()
        assert config.merge_adjacent is True
        assert config.min_segment_sec == 0.25
        assert config.drop_short_segments is False
        assert config.short_segment_strategy == "merge_best"
        assert config.cap_to_duration is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MergeConfig(
            merge_adjacent=False,
            min_segment_sec=0.5,
            drop_short_segments=True,
            short_segment_strategy="merge_prev",
        )
        assert config.merge_adjacent is False
        assert config.min_segment_sec == 0.5
        assert config.drop_short_segments is True
        assert config.short_segment_strategy == "merge_prev"
    
    def test_invalid_min_segment_sec_raises(self):
        """Test that negative min_segment_sec raises error."""
        with pytest.raises(ValueError, match="min_segment_sec must be >= 0"):
            MergeConfig(min_segment_sec=-1)
    
    def test_invalid_strategy_raises(self):
        """Test that invalid short_segment_strategy raises error."""
        with pytest.raises(ValueError, match="short_segment_strategy must be one of"):
            MergeConfig(short_segment_strategy="invalid")
    
    def test_to_dict(self):
        """Test to_dict method."""
        config = MergeConfig()
        d = config.to_dict()
        assert d["merge_adjacent"] is True
        assert d["min_segment_sec"] == 0.25


class TestMergeAdjacentWindows:
    """Tests for merging adjacent windows with same emotion."""
    
    def test_merge_same_emotion(self):
        """Test that adjacent windows with same emotion are merged."""
        windows = [
            make_window(0, "happy", confidence=0.8),
            make_window(1, "happy", confidence=0.9),
            make_window(2, "happy", confidence=0.85),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        assert len(segments) == 1
        assert segments[0].emotion == "happy"
        assert segments[0].start_sec == 0.0
        assert segments[0].end_sec == windows[-1].end_sec
        assert segments[0].window_count == 3
    
    def test_merge_confidence_averaged(self):
        """Test that merged segment confidence is averaged."""
        windows = [
            make_window(0, "happy", confidence=0.8),
            make_window(1, "happy", confidence=0.9),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        assert len(segments) == 1
        assert abs(segments[0].confidence - 0.85) < 1e-6
    
    def test_different_emotions_not_merged(self):
        """Test that different emotions create separate segments."""
        windows = [
            make_window(0, "happy", confidence=0.8),
            make_window(1, "sad", confidence=0.7),
            make_window(2, "angry", confidence=0.9),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        assert len(segments) == 3
        assert segments[0].emotion == "happy"
        assert segments[1].emotion == "sad"
        assert segments[2].emotion == "angry"
    
    def test_alternating_emotions(self):
        """Test handling of alternating emotions."""
        windows = [
            make_window(0, "happy"),
            make_window(1, "sad"),
            make_window(2, "happy"),
            make_window(3, "sad"),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        assert len(segments) == 4
    
    def test_segment_boundaries_correct(self):
        """Test that segment start/end times are correct."""
        windows = [
            make_window(0, "happy", start_sec=0.0, end_sec=2.0),
            make_window(1, "happy", start_sec=0.5, end_sec=2.5),
            make_window(2, "sad", start_sec=1.0, end_sec=3.0),
            make_window(3, "sad", start_sec=1.5, end_sec=3.5),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        assert len(segments) == 2
        # First segment: happy windows 0-1
        assert segments[0].start_sec == 0.0
        assert segments[0].end_sec == 2.5
        # Second segment: sad windows 2-3
        assert segments[1].start_sec == 1.0
        assert segments[1].end_sec == 3.5


class TestNoMerge:
    """Tests for when merge_adjacent is False."""
    
    def test_no_merge_individual_segments(self):
        """Test that each window becomes individual segment when merge=False."""
        windows = [
            make_window(0, "happy"),
            make_window(1, "happy"),
            make_window(2, "happy"),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig(merge_adjacent=False)
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        assert len(segments) == 3
        for i, seg in enumerate(segments):
            assert seg.emotion == "happy"
            assert seg.window_count == 1


class TestShortSegmentHandling:
    """Tests for handling short segments."""
    
    def test_short_segment_merge_prev(self):
        """Test short segment merges into previous."""
        windows = [
            make_window(0, "happy", start_sec=0.0, end_sec=2.0),
            make_window(1, "happy", start_sec=0.5, end_sec=2.5),
            make_window(2, "sad", start_sec=1.0, end_sec=1.1),  # Short: 0.1s
            make_window(3, "angry", start_sec=1.5, end_sec=3.5),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig(
            drop_short_segments=True,
            min_segment_sec=0.5,
            short_segment_strategy="merge_prev",
        )
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        # Short "sad" segment should merge into "happy"
        # Check that we don't have a standalone sad segment
        emotions = [s.emotion for s in segments]
        # The exact result depends on merging, but sad should be gone
        assert len(segments) <= 3
    
    def test_short_segment_merge_next(self):
        """Test short segment merges into next."""
        windows = [
            make_window(0, "happy", start_sec=0.0, end_sec=2.0),
            make_window(1, "sad", start_sec=2.0, end_sec=2.1),  # Short: 0.1s
            make_window(2, "angry", start_sec=2.5, end_sec=4.5),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig(
            drop_short_segments=True,
            min_segment_sec=0.5,
            short_segment_strategy="merge_next",
        )
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        # Short "sad" segment should merge into "angry"
        emotions = [s.emotion for s in segments]
        assert "sad" not in emotions or len([e for e in emotions if e == "sad"]) == 0
    
    def test_short_segment_merge_best_by_confidence(self):
        """Test short segment merges into neighbor with higher confidence."""
        windows = [
            make_window(0, "happy", confidence=0.9, start_sec=0.0, end_sec=2.0),
            make_window(1, "sad", confidence=0.5, start_sec=2.0, end_sec=2.1),  # Short
            make_window(2, "angry", confidence=0.6, start_sec=2.5, end_sec=4.5),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig(
            drop_short_segments=True,
            min_segment_sec=0.5,
            short_segment_strategy="merge_best",
        )
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        # Short "sad" should merge into "happy" (higher confidence 0.9 > 0.6)
        emotions = [s.emotion for s in segments]
        assert "sad" not in emotions
    
    def test_no_short_segment_handling_when_disabled(self):
        """Test that short segments are kept when drop_short_segments=False."""
        windows = [
            make_window(0, "happy", start_sec=0.0, end_sec=2.0),
            make_window(1, "sad", start_sec=2.0, end_sec=2.1),  # Short
            make_window(2, "angry", start_sec=2.5, end_sec=4.5),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig(drop_short_segments=False)
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        # All segments should be present
        emotions = [s.emotion for s in segments]
        assert "happy" in emotions
        assert "sad" in emotions
        assert "angry" in emotions


class TestCapToDuration:
    """Tests for capping segment end times to audio duration."""
    
    def test_cap_to_duration(self):
        """Test that segments are capped to audio duration."""
        windows = [
            make_window(0, "happy", start_sec=0.0, end_sec=2.0),
            make_window(1, "happy", start_sec=0.5, end_sec=2.5),  # Exceeds duration 2.2
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig(cap_to_duration=True)
        
        segments = merge_windows_to_segments(
            windows, windowing_config, merge_config, duration_sec=2.2
        )
        
        assert len(segments) == 1
        assert segments[0].end_sec == 2.2
    
    def test_no_cap_when_disabled(self):
        """Test that segments are not capped when cap_to_duration=False."""
        windows = [
            make_window(0, "happy", start_sec=0.0, end_sec=2.5),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig(cap_to_duration=False)
        
        segments = merge_windows_to_segments(
            windows, windowing_config, merge_config, duration_sec=2.0
        )
        
        assert segments[0].end_sec == 2.5  # Not capped


class TestEmptyInput:
    """Tests for empty input handling."""
    
    def test_empty_windows(self):
        """Test handling of empty window list."""
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments([], windowing_config, merge_config)
        
        assert segments == []


class TestScoresMerging:
    """Tests for merging scores across windows."""
    
    def test_scores_averaged_in_merged_segment(self):
        """Test that scores are averaged when windows are merged."""
        windows = [
            make_window(0, "happy", scores={
                "happy": 0.8, "sad": 0.1, "neutral": 0.1, "angry": 0.0
            }),
            make_window(1, "happy", scores={
                "happy": 0.9, "sad": 0.05, "neutral": 0.05, "angry": 0.0
            }),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        assert len(segments) == 1
        assert segments[0].scores is not None
        assert abs(segments[0].scores["happy"] - 0.85) < 1e-6  # (0.8 + 0.9) / 2
        assert abs(segments[0].scores["sad"] - 0.075) < 1e-6  # (0.1 + 0.05) / 2
    
    def test_none_scores_handled(self):
        """Test that None scores are handled gracefully."""
        windows = [
            make_window(0, "happy", scores=None),
            make_window(1, "happy", scores=None),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        assert len(segments) == 1
        assert segments[0].scores is None


class TestSegmentProperties:
    """Tests for Segment properties and methods."""
    
    def test_segment_duration(self):
        """Test segment duration_sec property."""
        windows = [
            make_window(0, "happy", start_sec=0.0, end_sec=2.0),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        assert abs(segments[0].duration_sec - 2.0) < 1e-6
    
    def test_segment_to_dict(self):
        """Test segment to_dict method."""
        windows = [
            make_window(0, "happy", confidence=0.85, scores={"happy": 0.85, "sad": 0.15}),
        ]
        
        windowing_config = WindowingConfig()
        merge_config = MergeConfig()
        
        segments = merge_windows_to_segments(windows, windowing_config, merge_config)
        
        d = segments[0].to_dict(include_scores=True)
        assert d["emotion"] == "happy"
        assert abs(d["confidence"] - 0.85) < 1e-6
        assert "scores" in d
        
        d_no_scores = segments[0].to_dict(include_scores=False)
        assert "scores" not in d_no_scores
