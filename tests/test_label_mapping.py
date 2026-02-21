"""Tests for label mapping functionality."""

import pytest

from model.labels import (
    CANONICAL_LABELS,
    IEMOCAP_TO_CANONICAL,
    BASELINE_SUPPORTED_LABELS,
    map_raw_to_canonical,
    get_canonical_label,
    is_label_supported,
)


class TestCanonicalLabels:
    """Tests for canonical label definitions."""
    
    def test_canonical_labels_is_list(self):
        """Canonical labels should be a list."""
        assert isinstance(CANONICAL_LABELS, list)
    
    def test_canonical_labels_has_expected_emotions(self):
        """Canonical labels should include core emotions."""
        expected = {"neutral", "happy", "sad", "angry"}
        assert expected.issubset(set(CANONICAL_LABELS))
    
    def test_canonical_labels_are_lowercase(self):
        """All canonical labels should be lowercase strings."""
        for label in CANONICAL_LABELS:
            assert isinstance(label, str)
            assert label == label.lower()
    
    def test_canonical_labels_are_unique(self):
        """Canonical labels should have no duplicates."""
        assert len(CANONICAL_LABELS) == len(set(CANONICAL_LABELS))


class TestIEMOCAPMapping:
    """Tests for IEMOCAP to canonical mapping."""
    
    def test_iemocap_mapping_covers_all_raw_labels(self):
        """IEMOCAP mapping should cover all expected raw labels."""
        expected_raw = {"neu", "hap", "sad", "ang"}
        assert set(IEMOCAP_TO_CANONICAL.keys()) == expected_raw
    
    def test_iemocap_mapping_targets_are_canonical(self):
        """IEMOCAP mapping targets should all be canonical labels."""
        for target in IEMOCAP_TO_CANONICAL.values():
            assert target in CANONICAL_LABELS, f"{target} is not a canonical label"
    
    def test_iemocap_mapping_is_correct(self):
        """IEMOCAP mapping should map to correct canonical labels."""
        assert IEMOCAP_TO_CANONICAL["neu"] == "neutral"
        assert IEMOCAP_TO_CANONICAL["hap"] == "happy"
        assert IEMOCAP_TO_CANONICAL["sad"] == "sad"
        assert IEMOCAP_TO_CANONICAL["ang"] == "angry"


class TestMapRawToCanonical:
    """Tests for map_raw_to_canonical function."""
    
    def test_basic_mapping(self):
        """Basic mapping should work correctly."""
        raw_scores = {"neu": 0.1, "hap": 0.5, "sad": 0.2, "ang": 0.2}
        canonical = map_raw_to_canonical(raw_scores)
        
        # Check mapped values (normalized to sum to 1)
        assert canonical["neutral"] == pytest.approx(0.1)
        assert canonical["happy"] == pytest.approx(0.5)
        assert canonical["sad"] == pytest.approx(0.2)
        assert canonical["angry"] == pytest.approx(0.2)
    
    def test_all_canonical_labels_present(self):
        """All canonical labels should be present in output."""
        raw_scores = {"neu": 0.5, "hap": 0.5}
        canonical = map_raw_to_canonical(raw_scores)
        
        for label in CANONICAL_LABELS:
            assert label in canonical
    
    def test_unsupported_labels_are_zero(self):
        """Unsupported canonical labels should have zero probability."""
        raw_scores = {"neu": 0.25, "hap": 0.25, "sad": 0.25, "ang": 0.25}
        canonical = map_raw_to_canonical(raw_scores)
        
        # These are not in IEMOCAP, should be 0
        assert canonical["fear"] == 0.0
        assert canonical["disgust"] == 0.0
        assert canonical["surprise"] == 0.0
    
    def test_normalization(self):
        """With normalize=True, output should sum to 1."""
        raw_scores = {"neu": 0.1, "hap": 0.3, "sad": 0.2, "ang": 0.1}
        canonical = map_raw_to_canonical(raw_scores, normalize=True)
        
        total = sum(canonical.values())
        assert total == pytest.approx(1.0)
    
    def test_no_normalization(self):
        """With normalize=False, original values should be preserved."""
        raw_scores = {"neu": 0.1, "hap": 0.3, "sad": 0.2, "ang": 0.1}
        canonical = map_raw_to_canonical(raw_scores, normalize=False)
        
        # Values should match raw scores
        assert canonical["neutral"] == 0.1
        assert canonical["happy"] == 0.3
        assert canonical["sad"] == 0.2
        assert canonical["angry"] == 0.1
    
    def test_empty_raw_scores(self):
        """Empty raw scores should give all zeros."""
        canonical = map_raw_to_canonical({})
        
        for label in CANONICAL_LABELS:
            assert canonical[label] == 0.0
    
    def test_unknown_raw_labels_ignored(self):
        """Unknown raw labels should be ignored."""
        raw_scores = {"neu": 0.5, "hap": 0.5, "unknown_label": 0.5}
        canonical = map_raw_to_canonical(raw_scores)
        
        # Should normalize over known labels only
        total = sum(canonical.values())
        assert total == pytest.approx(1.0)
    
    def test_custom_mapping(self):
        """Custom mapping dict should work."""
        custom_mapping = {"a": "happy", "b": "sad"}
        raw_scores = {"a": 0.6, "b": 0.4}
        
        canonical = map_raw_to_canonical(raw_scores, mapping=custom_mapping)
        
        assert canonical["happy"] == pytest.approx(0.6)
        assert canonical["sad"] == pytest.approx(0.4)


class TestGetCanonicalLabel:
    """Tests for get_canonical_label function."""
    
    def test_known_label(self):
        """Should return canonical label for known raw label."""
        assert get_canonical_label("neu") == "neutral"
        assert get_canonical_label("hap") == "happy"
        assert get_canonical_label("sad") == "sad"
        assert get_canonical_label("ang") == "angry"
    
    def test_unknown_label(self):
        """Should return None for unknown raw label."""
        assert get_canonical_label("unknown") is None
        assert get_canonical_label("foo") is None
    
    def test_custom_mapping(self):
        """Should use custom mapping when provided."""
        custom = {"x": "happy"}
        assert get_canonical_label("x", mapping=custom) == "happy"
        assert get_canonical_label("neu", mapping=custom) is None


class TestIsLabelSupported:
    """Tests for is_label_supported function."""
    
    def test_supported_labels(self):
        """Supported labels should return True."""
        for label in BASELINE_SUPPORTED_LABELS:
            assert is_label_supported(label) is True
    
    def test_unsupported_labels(self):
        """Unsupported labels should return False."""
        unsupported = set(CANONICAL_LABELS) - BASELINE_SUPPORTED_LABELS
        for label in unsupported:
            assert is_label_supported(label) is False
    
    def test_invalid_label(self):
        """Invalid labels should return False."""
        assert is_label_supported("invalid") is False
        assert is_label_supported("") is False
