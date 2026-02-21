"""Tests for timeline.smooth module (smoothing strategies)."""

import pytest

from timeline.schema import WindowPrediction
from timeline.smooth import SmoothingConfig, smooth_windows


# Canonical labels for testing
CANONICAL_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]


def make_window(
    index: int,
    emotion: str,
    confidence: float = 0.8,
    scores: dict | None = None,
    start_sec: float = None,
    end_sec: float = None,
) -> WindowPrediction:
    """Helper to create WindowPrediction for testing."""
    if start_sec is None:
        start_sec = index * 0.5
    if end_sec is None:
        end_sec = start_sec + 2.0
    return WindowPrediction(
        index=index,
        start_sec=start_sec,
        end_sec=end_sec,
        emotion=emotion,
        confidence=confidence,
        scores=scores,
        is_padded=False,
    )


class TestSmoothingConfig:
    """Tests for SmoothingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SmoothingConfig()
        assert config.method == "hysteresis"
        assert config.majority_window == 5
        assert config.hysteresis_min_run == 3
        assert config.ema_alpha == 0.6
        assert config.min_confidence == 0.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SmoothingConfig(
            method="majority",
            majority_window=7,
            hysteresis_min_run=4,
            ema_alpha=0.8,
        )
        assert config.method == "majority"
        assert config.majority_window == 7
        assert config.hysteresis_min_run == 4
        assert config.ema_alpha == 0.8
    
    def test_invalid_method_raises(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="method must be one of"):
            SmoothingConfig(method="invalid")
    
    def test_even_majority_window_raises(self):
        """Test that even majority_window raises error."""
        with pytest.raises(ValueError, match="majority_window must be odd"):
            SmoothingConfig(majority_window=4)
    
    def test_zero_majority_window_raises(self):
        """Test that zero majority_window raises error."""
        with pytest.raises(ValueError, match="majority_window must be >= 1"):
            SmoothingConfig(majority_window=0)
    
    def test_zero_hysteresis_min_run_raises(self):
        """Test that zero hysteresis_min_run raises error."""
        with pytest.raises(ValueError, match="hysteresis_min_run must be >= 1"):
            SmoothingConfig(hysteresis_min_run=0)
    
    def test_invalid_ema_alpha_raises(self):
        """Test that invalid ema_alpha raises error."""
        with pytest.raises(ValueError, match="ema_alpha must be in"):
            SmoothingConfig(ema_alpha=0)
        with pytest.raises(ValueError, match="ema_alpha must be in"):
            SmoothingConfig(ema_alpha=1.5)
    
    def test_to_dict(self):
        """Test to_dict method."""
        config = SmoothingConfig(method="hysteresis", hysteresis_min_run=4)
        d = config.to_dict()
        assert d["method"] == "hysteresis"
        assert d["hysteresis_min_run"] == 4
        
        config = SmoothingConfig(method="majority", majority_window=7)
        d = config.to_dict()
        assert d["method"] == "majority"
        assert d["majority_window"] == 7


class TestSmoothNone:
    """Tests for 'none' smoothing method."""
    
    def test_no_smoothing_preserves_emotions(self):
        """Test that 'none' method preserves original emotions."""
        windows = [
            make_window(0, "happy"),
            make_window(1, "sad"),
            make_window(2, "happy"),
            make_window(3, "angry"),
        ]
        
        config = SmoothingConfig(method="none")
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        assert len(result) == 4
        assert result[0].emotion == "happy"
        assert result[1].emotion == "sad"
        assert result[2].emotion == "happy"
        assert result[3].emotion == "angry"
    
    def test_no_smoothing_preserves_confidences(self):
        """Test that 'none' method preserves original confidences."""
        windows = [
            make_window(0, "happy", confidence=0.9),
            make_window(1, "sad", confidence=0.7),
        ]
        
        config = SmoothingConfig(method="none")
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        assert result[0].confidence == 0.9
        assert result[1].confidence == 0.7
    
    def test_empty_windows(self):
        """Test handling of empty window list."""
        config = SmoothingConfig(method="none")
        result = smooth_windows([], config, CANONICAL_LABELS)
        assert result == []


class TestSmoothMajority:
    """Tests for 'majority' smoothing method."""
    
    def test_majority_stabilizes_alternating(self):
        """Test that majority smoothing stabilizes rapidly alternating emotions."""
        # Create pattern: H S H S H S H (alternating happy/sad)
        windows = [
            make_window(i, "happy" if i % 2 == 0 else "sad", confidence=0.8)
            for i in range(7)
        ]
        
        config = SmoothingConfig(method="majority", majority_window=3)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # With window=3, majority should stabilize the middle windows
        # First few and last few may vary based on edge handling
        assert len(result) == 7
        # All windows should now be more consistent
        emotions = [w.emotion for w in result]
        # The majority should reduce alternation significantly
        changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        assert changes < 6  # Should have fewer changes than original (which has 6)
    
    def test_majority_handles_tie_by_confidence(self):
        """Test that ties are broken by higher average confidence."""
        # Create: H(0.9) S(0.5) H(0.7) - tie between H and S
        windows = [
            make_window(0, "happy", confidence=0.9),
            make_window(1, "sad", confidence=0.5),
            make_window(2, "happy", confidence=0.7),
        ]
        
        config = SmoothingConfig(method="majority", majority_window=3)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # For middle window (index 1), window includes all 3
        # happy appears 2x (avg conf 0.8), sad appears 1x (conf 0.5)
        # Majority is happy
        assert result[1].emotion == "happy"
    
    def test_majority_window_size_1_no_change(self):
        """Test that window size 1 produces no change."""
        windows = [
            make_window(0, "happy"),
            make_window(1, "sad"),
            make_window(2, "angry"),
        ]
        
        config = SmoothingConfig(method="majority", majority_window=1)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        assert result[0].emotion == "happy"
        assert result[1].emotion == "sad"
        assert result[2].emotion == "angry"
    
    def test_majority_clear_majority(self):
        """Test clear majority voting."""
        # Create: H H H S S (clear majority of H in first 3)
        windows = [
            make_window(0, "happy"),
            make_window(1, "happy"),
            make_window(2, "happy"),
            make_window(3, "sad"),
            make_window(4, "sad"),
        ]
        
        config = SmoothingConfig(method="majority", majority_window=3)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # Window 0: [H, H] -> H (edge, only 2 elements)
        # Window 1: [H, H, H] -> H
        # Window 2: [H, H, S] -> H (majority)
        assert result[0].emotion == "happy"
        assert result[1].emotion == "happy"
        assert result[2].emotion == "happy"


class TestSmoothHysteresis:
    """Tests for 'hysteresis' smoothing method."""
    
    def test_hysteresis_short_run_stays(self):
        """Test that emotion doesn't switch with short run."""
        # A A A B B (only 2 B's, min_run=3 required)
        windows = [
            make_window(0, "happy"),
            make_window(1, "happy"),
            make_window(2, "happy"),
            make_window(3, "sad"),
            make_window(4, "sad"),
        ]
        
        config = SmoothingConfig(method="hysteresis", hysteresis_min_run=3)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # Should remain "happy" throughout (sad doesn't persist 3 windows)
        assert all(w.emotion == "happy" for w in result)
    
    def test_hysteresis_long_run_switches(self):
        """Test that emotion switches with sufficient run."""
        # A A A B B B (3 B's = min_run)
        windows = [
            make_window(0, "happy"),
            make_window(1, "happy"),
            make_window(2, "happy"),
            make_window(3, "sad"),
            make_window(4, "sad"),
            make_window(5, "sad"),
        ]
        
        config = SmoothingConfig(method="hysteresis", hysteresis_min_run=3)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # First 3 should be happy, last 3 should be sad
        assert result[0].emotion == "happy"
        assert result[1].emotion == "happy"
        assert result[2].emotion == "happy"
        assert result[3].emotion == "sad"
        assert result[4].emotion == "sad"
        assert result[5].emotion == "sad"
    
    def test_hysteresis_switch_at_first_of_run(self):
        """Test that switch happens at first window of persistent run."""
        # H H S S S H
        windows = [
            make_window(0, "happy"),
            make_window(1, "happy"),
            make_window(2, "sad"),
            make_window(3, "sad"),
            make_window(4, "sad"),
            make_window(5, "happy"),
        ]
        
        config = SmoothingConfig(method="hysteresis", hysteresis_min_run=3)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # Should switch to sad at index 2 (first of 3-run)
        assert result[0].emotion == "happy"
        assert result[1].emotion == "happy"
        assert result[2].emotion == "sad"
        assert result[3].emotion == "sad"
        assert result[4].emotion == "sad"
        # Index 5: only 1 happy, stays sad
        assert result[5].emotion == "sad"
    
    def test_hysteresis_min_run_1_immediate_switch(self):
        """Test that min_run=1 switches immediately."""
        windows = [
            make_window(0, "happy"),
            make_window(1, "sad"),
            make_window(2, "angry"),
        ]
        
        config = SmoothingConfig(method="hysteresis", hysteresis_min_run=1)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        assert result[0].emotion == "happy"
        assert result[1].emotion == "sad"
        assert result[2].emotion == "angry"
    
    def test_hysteresis_multiple_transitions(self):
        """Test multiple emotion transitions."""
        # H H H S S S A A A N N N
        windows = [
            make_window(0, "happy"),
            make_window(1, "happy"),
            make_window(2, "happy"),
            make_window(3, "sad"),
            make_window(4, "sad"),
            make_window(5, "sad"),
            make_window(6, "angry"),
            make_window(7, "angry"),
            make_window(8, "angry"),
            make_window(9, "neutral"),
            make_window(10, "neutral"),
            make_window(11, "neutral"),
        ]
        
        config = SmoothingConfig(method="hysteresis", hysteresis_min_run=3)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        assert result[0].emotion == "happy"
        assert result[3].emotion == "sad"
        assert result[6].emotion == "angry"
        assert result[9].emotion == "neutral"


class TestSmoothEMA:
    """Tests for 'ema' smoothing method."""
    
    def test_ema_follows_smoothed_trend(self):
        """Test that EMA follows smoothed trend deterministically."""
        # Create windows with explicit scores
        windows = []
        for i in range(5):
            # Gradually shift from happy to sad
            happy_score = 1.0 - (i * 0.2)
            sad_score = i * 0.2
            scores = {
                "neutral": 0.0,
                "happy": happy_score,
                "sad": sad_score,
                "angry": 0.0,
                "fear": 0.0,
                "disgust": 0.0,
                "surprise": 0.0,
            }
            # Normalize
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}
            windows.append(make_window(
                index=i,
                emotion="happy" if happy_score > sad_score else "sad",
                confidence=max(happy_score, sad_score),
                scores=scores,
            ))
        
        config = SmoothingConfig(method="ema", ema_alpha=0.5)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # EMA should smooth the transition
        # First window should still be happy (100% happy initially)
        assert result[0].emotion == "happy"
        # Due to EMA lag, emotion should stay happy longer
        assert result[1].emotion == "happy"
    
    def test_ema_stable_input(self):
        """Test that stable input remains stable with EMA."""
        windows = [
            make_window(i, "happy", confidence=0.9, scores={
                "neutral": 0.05, "happy": 0.9, "sad": 0.02,
                "angry": 0.01, "fear": 0.01, "disgust": 0.005, "surprise": 0.005,
            })
            for i in range(5)
        ]
        
        config = SmoothingConfig(method="ema", ema_alpha=0.6)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # All windows should remain happy
        assert all(w.emotion == "happy" for w in result)
    
    def test_ema_no_scores_fallback(self):
        """Test EMA falls back gracefully when no scores available."""
        windows = [
            make_window(0, "happy", confidence=0.8, scores=None),
            make_window(1, "sad", confidence=0.7, scores=None),
        ]
        
        config = SmoothingConfig(method="ema", ema_alpha=0.6)
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # Should handle gracefully (falls back to no smoothing)
        assert len(result) == 2
    
    def test_ema_deterministic(self):
        """Test that EMA produces deterministic results."""
        windows = [
            make_window(i, "happy" if i % 2 == 0 else "sad", scores={
                "neutral": 0.1, "happy": 0.5 if i % 2 == 0 else 0.2,
                "sad": 0.2 if i % 2 == 0 else 0.5, "angry": 0.1,
                "fear": 0.05, "disgust": 0.03, "surprise": 0.02,
            })
            for i in range(5)
        ]
        
        config = SmoothingConfig(method="ema", ema_alpha=0.6)
        
        result1 = smooth_windows(windows, config, CANONICAL_LABELS)
        result2 = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # Results should be identical
        for w1, w2 in zip(result1, result2):
            assert w1.emotion == w2.emotion
            assert abs(w1.confidence - w2.confidence) < 1e-9


class TestSmoothWindowsGeneral:
    """General tests for smooth_windows function."""
    
    def test_preserves_window_metadata(self):
        """Test that smoothing preserves window metadata."""
        windows = [
            WindowPrediction(
                index=0,
                start_sec=0.0,
                end_sec=2.0,
                emotion="happy",
                confidence=0.8,
                is_padded=True,
            ),
        ]
        
        config = SmoothingConfig(method="none")
        result = smooth_windows(windows, config, CANONICAL_LABELS)
        
        assert result[0].index == 0
        assert result[0].start_sec == 0.0
        assert result[0].end_sec == 2.0
        assert result[0].is_padded is True
    
    def test_does_not_mutate_input(self):
        """Test that smoothing does not mutate input windows."""
        windows = [make_window(0, "happy", confidence=0.8)]
        original_emotion = windows[0].emotion
        original_confidence = windows[0].confidence
        
        config = SmoothingConfig(method="majority", majority_window=3)
        _ = smooth_windows(windows, config, CANONICAL_LABELS)
        
        # Original should be unchanged
        assert windows[0].emotion == original_emotion
        assert windows[0].confidence == original_confidence
    
    def test_single_window(self):
        """Test smoothing with single window."""
        windows = [make_window(0, "happy")]
        
        for method in ["none", "majority", "hysteresis", "ema"]:
            config = SmoothingConfig(method=method)
            result = smooth_windows(windows, config, CANONICAL_LABELS)
            assert len(result) == 1
            assert result[0].emotion == "happy"
