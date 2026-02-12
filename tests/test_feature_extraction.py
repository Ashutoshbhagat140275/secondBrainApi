"""
Unit tests for the feature extraction pipeline.

Run from the api/ directory:
    python -m pytest tests/test_feature_extraction.py -v
"""

import sys
import pathlib
import numpy as np
import pytest

# Ensure api/ is on sys.path
_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.feature_config import FEATURE_DIM, SAMPLE_RATE
from app.services.feature_extractor import _compute_functionals


class TestComputeFunctionals:
    """Tests for the _compute_functionals helper."""

    def test_normal_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_functionals(x)
        assert len(result) == 8
        assert result[0] == pytest.approx(3.0, abs=1e-6)  # mean
        assert all(isinstance(v, float) for v in result)

    def test_empty_after_nan_removal(self):
        x = np.array([np.nan, np.nan])
        result = _compute_functionals(x)
        assert result == [0.0] * 8

    def test_single_value(self):
        x = np.array([42.0])
        result = _compute_functionals(x)
        assert len(result) == 8
        assert result[0] == pytest.approx(42.0)  # mean
        assert result[1] == pytest.approx(0.0)  # std

    def test_with_nans(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = _compute_functionals(x)
        assert len(result) == 8
        assert result[0] == pytest.approx(3.0, abs=1e-6)  # mean of [1,3,5]


class TestFeatureConfig:
    """Tests for feature_config constants."""

    def test_feature_dim_value(self):
        assert FEATURE_DIM == 419

    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000


class TestExtractFullFeatures:
    """
    Integration test for extract_full_features.
    Requires a real audio file — generate a synthetic one.
    """

    @pytest.fixture
    def synthetic_wav(self, tmp_path):
        """Create a short synthetic WAV file for testing."""
        import soundfile as sf

        duration = 1.0  # seconds
        sr = SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Simple sine wave at 440 Hz
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, sr)
        return str(wav_path)

    def test_output_length(self, synthetic_wav):
        from app.services.feature_extractor import extract_full_features

        features = extract_full_features(synthetic_wav)
        assert len(features) == FEATURE_DIM

    def test_no_nans(self, synthetic_wav):
        from app.services.feature_extractor import extract_full_features

        features = extract_full_features(synthetic_wav)
        assert not any(np.isnan(v) for v in features)

    def test_all_floats(self, synthetic_wav):
        from app.services.feature_extractor import extract_full_features

        features = extract_full_features(synthetic_wav)
        assert all(isinstance(v, float) for v in features)
