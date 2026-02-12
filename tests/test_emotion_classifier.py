"""
Unit tests for the emotion classifier service.

Run from the api/ directory:
    python -m pytest tests/test_emotion_classifier.py -v
"""

import sys
import pathlib
import numpy as np
import pytest

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.feature_config import FEATURE_DIM, EMOTION_LABELS


class TestClassifyEmotionFallback:
    """
    Test classify_emotion when no trained model is available.
    It should return a safe fallback without crashing.
    """

    def test_fallback_returns_tuple(self):
        from app.services.emotion_analyzer import classify_emotion

        result = classify_emotion([0.0] * FEATURE_DIM)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fallback_label_is_string(self):
        from app.services.emotion_analyzer import classify_emotion

        label, _ = classify_emotion([0.0] * FEATURE_DIM)
        assert isinstance(label, str)

    def test_fallback_confidence_range(self):
        from app.services.emotion_analyzer import classify_emotion

        _, conf = classify_emotion([0.0] * FEATURE_DIM)
        assert 0.0 <= conf <= 1.0

    def test_fallback_returns_neutral(self):
        """Without a model, fallback should be 'neutral'."""
        from app.services.emotion_analyzer import classify_emotion

        label, conf = classify_emotion([0.0] * FEATURE_DIM)
        assert label == "neutral"
        assert conf == 0.5


class TestClassifyEmotionWithModel:
    """
    Test classify_emotion when a real model checkpoint exists.
    These tests are skipped if the model hasn't been trained yet.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        from app.services.feature_config import MODEL_PATH, SCALER_PATH

        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            pytest.skip("Trained model not available — skipping model tests")

    def test_returns_valid_label(self):
        from app.services.emotion_analyzer import classify_emotion

        features = list(np.random.randn(FEATURE_DIM).astype(float))
        label, conf = classify_emotion(features)
        assert label in EMOTION_LABELS

    def test_confidence_in_range(self):
        from app.services.emotion_analyzer import classify_emotion

        features = list(np.random.randn(FEATURE_DIM).astype(float))
        _, conf = classify_emotion(features)
        assert 0.0 <= conf <= 1.0

    def test_deterministic(self):
        """Same input should give same output."""
        from app.services.emotion_analyzer import classify_emotion

        np.random.seed(0)
        features = list(np.random.randn(FEATURE_DIM).astype(float))
        r1 = classify_emotion(features)
        r2 = classify_emotion(features)
        assert r1 == r2
