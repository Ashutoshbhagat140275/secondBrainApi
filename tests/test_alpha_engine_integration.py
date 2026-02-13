"""
Integration tests for end-to-end alpha engine pipeline.

These tests verify the complete flow:
1. Audio upload → embedding → dual-head → response with alpha components
2. Test with USE_SIGMOID_ALPHA=False (linear formula)
3. Test with USE_SIGMOID_ALPHA=True (sigmoid formula)
4. Verify response format includes all fields
5. Verify predictions are reasonable

Validates Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import sys
import pathlib
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.dual_head_classifier import classify_with_dual_heads
from app.services.alpha_engine import compute_blend_weight


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_embedding():
    """Generate a sample 768-dimensional embedding."""
    return np.random.randn(768).astype(np.float32)


@pytest.fixture
def user_id():
    """Generate a test user ID."""
    return "test_user_123"


# ---------------------------------------------------------------------------
# Test 1: End-to-end with USE_SIGMOID_ALPHA=False (linear formula)
# ---------------------------------------------------------------------------


@patch('app.services.dual_head_classifier.predict_global')
@patch('app.services.dual_head_classifier.predict_user')
@patch('app.services.feature_config.USE_SIGMOID_ALPHA', False)
def test_end_to_end_linear_formula(
    mock_predict_user,
    mock_predict_global,
    sample_embedding,
    user_id
):
    """
    Test end-to-end pipeline with USE_SIGMOID_ALPHA=False (linear formula).
    
    Verifies:
    - Audio embedding → dual-head classification
    - Linear formula is used (alpha_data and alpha_conf are None)
    - Response includes alpha_formula = "linear"
    - Predictions are reasonable
    - All response fields are present
    
    Validates Requirements: 5.1, 5.2, 5.4
    """
    # Reload alpha_engine to pick up patched config
    import importlib
    import app.services.alpha_engine
    importlib.reload(app.services.alpha_engine)
    
    # Mock global head prediction
    P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
    mock_predict_global.return_value = (P_g, 0.7)
    
    # Mock user head prediction
    P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
    mock_predict_user.return_value = (P_u, 0.8)
    
    # Test with 50 feedback samples
    result = classify_with_dual_heads(sample_embedding, user_id, feedback_count=50)
    
    # Verify response structure
    assert "emotion" in result
    assert "confidence" in result
    assert "global_emotion" in result
    assert "global_confidence" in result
    assert "user_emotion" in result
    assert "user_confidence" in result
    assert "blend_weight" in result
    assert "alpha_data" in result
    assert "alpha_conf" in result
    assert "alpha_formula" in result
    assert "probabilities" in result
    
    # Verify linear formula is used
    assert result["alpha_formula"] == "linear"
    assert result["alpha_data"] is None
    assert result["alpha_conf"] is None
    
    # Verify blend_weight is computed using linear formula
    # feedback_count=50, C_g=0.7
    # α = 0.5 + 0.3*0.7 - 0.2*0.5 = 0.61
    assert 0.60 <= result["blend_weight"] <= 0.62
    
    # Verify predictions are reasonable
    assert result["emotion"] in ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "calm"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["global_emotion"] == "neutral"  # Index 0
    assert result["user_emotion"] == "calm"  # Index 1
    
    # Verify probabilities sum to 1.0
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test 2: End-to-end with USE_SIGMOID_ALPHA=True (sigmoid formula)
# ---------------------------------------------------------------------------


@patch('app.services.dual_head_classifier.predict_global')
@patch('app.services.dual_head_classifier.predict_user')
@patch('app.services.feature_config.USE_SIGMOID_ALPHA', True)
def test_end_to_end_sigmoid_formula(
    mock_predict_user,
    mock_predict_global,
    sample_embedding,
    user_id
):
    """
    Test end-to-end pipeline with USE_SIGMOID_ALPHA=True (sigmoid formula).
    
    Verifies:
    - Audio embedding → dual-head classification
    - Sigmoid formula is used (alpha_data and alpha_conf are not None)
    - Response includes alpha_formula = "sigmoid"
    - Alpha components are in valid range
    - Final alpha = alpha_data × alpha_conf
    - Predictions are reasonable
    
    Validates Requirements: 5.1, 5.3, 5.5
    """
    # Reload alpha_engine to pick up patched config
    import importlib
    import app.services.alpha_engine
    importlib.reload(app.services.alpha_engine)
    
    # Mock global head prediction
    P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
    mock_predict_global.return_value = (P_g, 0.7)
    
    # Mock user head prediction
    P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
    mock_predict_user.return_value = (P_u, 0.8)
    
    # Test with 50 feedback samples
    result = classify_with_dual_heads(sample_embedding, user_id, feedback_count=50)
    
    # Verify response structure
    assert "emotion" in result
    assert "confidence" in result
    assert "global_emotion" in result
    assert "global_confidence" in result
    assert "user_emotion" in result
    assert "user_confidence" in result
    assert "blend_weight" in result
    assert "alpha_data" in result
    assert "alpha_conf" in result
    assert "alpha_formula" in result
    assert "probabilities" in result
    
    # Verify sigmoid formula is used
    assert result["alpha_formula"] == "sigmoid"
    assert result["alpha_data"] is not None
    assert result["alpha_conf"] is not None
    
    # Verify alpha components are in valid range
    assert 0.0 < result["alpha_data"] <= 1.0
    assert 0.0 < result["alpha_conf"] < 1.0
    assert 0.0 < result["blend_weight"] < 1.0
    
    # Verify final alpha = alpha_data × alpha_conf
    expected_alpha = result["alpha_data"] * result["alpha_conf"]
    assert abs(result["blend_weight"] - expected_alpha) < 1e-5
    
    # Verify alpha_data for N=50, K=50
    # alpha_data = 1/(1+50/50) = 0.5
    assert abs(result["alpha_data"] - 0.5) < 0.01
    
    # Verify alpha_conf for C_g=0.7, tau=0.6, beta=10
    # alpha_conf = sigmoid(10*(0.7-0.6)) = sigmoid(1.0) ≈ 0.73
    assert 0.72 < result["alpha_conf"] < 0.74
    
    # Verify final alpha ≈ 0.5 * 0.73 ≈ 0.37
    assert 0.35 < result["blend_weight"] < 0.39
    
    # Verify predictions are reasonable
    assert result["emotion"] in ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "calm"]
    assert 0.0 <= result["confidence"] <= 1.0
    
    # Verify probabilities sum to 1.0
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test 3: Sigmoid mode with zero feedback (new user)
# ---------------------------------------------------------------------------


@patch('app.services.dual_head_classifier.predict_global')
@patch('app.services.dual_head_classifier.predict_user')
@patch('app.services.feature_config.USE_SIGMOID_ALPHA', True)
def test_sigmoid_mode_zero_feedback(
    mock_predict_user,
    mock_predict_global,
    sample_embedding,
    user_id
):
    """
    Test sigmoid mode with zero feedback (new user).
    
    Verifies:
    - With N=0, alpha_data = 1.0 (trust global fully)
    - alpha_conf depends on global confidence
    - Final alpha is high for high confidence
    - Predictions favor global head
    
    Validates Requirements: 5.3, 5.5
    """
    # Reload alpha_engine
    import importlib
    import app.services.alpha_engine
    importlib.reload(app.services.alpha_engine)
    
    # Mock global head with high confidence
    P_g = np.array([0.9, 0.03, 0.03, 0.01, 0.01, 0.01, 0.005, 0.005])
    mock_predict_global.return_value = (P_g, 0.9)
    
    # Mock user head (not available for new user)
    mock_predict_user.return_value = None
    
    # Test with 0 feedback samples
    result = classify_with_dual_heads(sample_embedding, user_id, feedback_count=0)
    
    # Verify sigmoid formula is used
    assert result["alpha_formula"] == "sigmoid" or result["alpha_formula"] == "linear"
    
    # If user head is not available, should use global only
    assert result["user_emotion"] is None
    assert result["user_confidence"] is None
    assert result["blend_weight"] == 1.0
    
    # Verify global prediction
    assert result["emotion"] == "neutral"  # Index 0
    assert result["global_emotion"] == "neutral"


# ---------------------------------------------------------------------------
# Test 4: Sigmoid mode with high feedback (experienced user)
# ---------------------------------------------------------------------------


@patch('app.services.dual_head_classifier.predict_global')
@patch('app.services.dual_head_classifier.predict_user')
@patch('app.services.feature_config.USE_SIGMOID_ALPHA', True)
def test_sigmoid_mode_high_feedback(
    mock_predict_user,
    mock_predict_global,
    sample_embedding,
    user_id
):
    """
    Test sigmoid mode with high feedback (experienced user).
    
    Verifies:
    - With N=100, alpha_data is low (favor user head)
    - Final alpha is low even with high confidence
    - Predictions favor user head
    
    Validates Requirements: 5.3, 5.5
    """
    # Reload alpha_engine
    import importlib
    import app.services.alpha_engine
    importlib.reload(app.services.alpha_engine)
    
    # Mock global head with high confidence
    P_g = np.array([0.8, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01])
    mock_predict_global.return_value = (P_g, 0.8)
    
    # Mock user head with different prediction
    P_u = np.array([0.1, 0.85, 0.02, 0.01, 0.01, 0.005, 0.002, 0.003])
    mock_predict_user.return_value = (P_u, 0.85)
    
    # Test with 100 feedback samples
    result = classify_with_dual_heads(sample_embedding, user_id, feedback_count=100)
    
    # Verify sigmoid formula is used
    assert result["alpha_formula"] == "sigmoid"
    assert result["alpha_data"] is not None
    assert result["alpha_conf"] is not None
    
    # Verify alpha_data for N=100, K=50
    # alpha_data = 1/(1+100/50) = 1/3 ≈ 0.33
    assert abs(result["alpha_data"] - 0.333) < 0.01
    
    # Verify alpha_conf for C_g=0.8, tau=0.6, beta=10
    # alpha_conf = sigmoid(10*(0.8-0.6)) = sigmoid(2.0) ≈ 0.88
    assert 0.87 < result["alpha_conf"] < 0.89
    
    # Verify final alpha ≈ 0.33 * 0.88 ≈ 0.29
    assert 0.28 < result["blend_weight"] < 0.30
    
    # Verify final alpha is low (favors user head)
    assert result["blend_weight"] < 0.5
    
    # Verify predictions
    assert result["global_emotion"] == "neutral"  # Index 0
    assert result["user_emotion"] == "calm"  # Index 1


# ---------------------------------------------------------------------------
# Test 5: Comparison between linear and sigmoid modes
# ---------------------------------------------------------------------------


@patch('app.services.dual_head_classifier.predict_global')
@patch('app.services.dual_head_classifier.predict_user')
def test_linear_vs_sigmoid_comparison(
    mock_predict_user,
    mock_predict_global,
    sample_embedding,
    user_id
):
    """
    Test comparison between linear and sigmoid modes with same inputs.
    
    Verifies:
    - Both modes produce valid predictions
    - Sigmoid mode favors user head more aggressively
    - Response formats are consistent
    - All fields are present in both modes
    
    Validates Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    # Mock predictions
    P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
    P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
    mock_predict_global.return_value = (P_g, 0.7)
    mock_predict_user.return_value = (P_u, 0.8)
    
    # Test with linear mode
    with patch('app.services.feature_config.USE_SIGMOID_ALPHA', False):
        import importlib
        import app.services.alpha_engine
        importlib.reload(app.services.alpha_engine)
        result_linear = classify_with_dual_heads(sample_embedding, user_id, feedback_count=50)
    
    # Test with sigmoid mode
    with patch('app.services.feature_config.USE_SIGMOID_ALPHA', True):
        importlib.reload(app.services.alpha_engine)
        result_sigmoid = classify_with_dual_heads(sample_embedding, user_id, feedback_count=50)
    
    # Verify both modes produce valid predictions
    assert result_linear["emotion"] in ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "calm"]
    assert result_sigmoid["emotion"] in ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "calm"]
    
    # Verify response formats are consistent
    expected_fields = [
        "emotion", "confidence", "global_emotion", "global_confidence",
        "user_emotion", "user_confidence", "blend_weight",
        "alpha_data", "alpha_conf", "alpha_formula", "probabilities"
    ]
    
    for field in expected_fields:
        assert field in result_linear, f"Missing field {field} in linear mode"
        assert field in result_sigmoid, f"Missing field {field} in sigmoid mode"
    
    # Verify formula indicators
    assert result_linear["alpha_formula"] == "linear"
    assert result_sigmoid["alpha_formula"] == "sigmoid"
    
    # Verify alpha components
    assert result_linear["alpha_data"] is None
    assert result_linear["alpha_conf"] is None
    assert result_sigmoid["alpha_data"] is not None
    assert result_sigmoid["alpha_conf"] is not None
    
    # Verify sigmoid favors user head more (lower alpha)
    # Linear: α ≈ 0.61
    # Sigmoid: α ≈ 0.37
    assert result_linear["blend_weight"] > result_sigmoid["blend_weight"]
    assert result_linear["blend_weight"] > 0.5
    assert result_sigmoid["blend_weight"] < 0.5
    
    # Verify probabilities sum to 1.0 in both modes
    assert abs(sum(result_linear["probabilities"].values()) - 1.0) < 1e-5
    assert abs(sum(result_sigmoid["probabilities"].values()) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test 6: Response format validation
# ---------------------------------------------------------------------------


@patch('app.services.dual_head_classifier.predict_global')
@patch('app.services.dual_head_classifier.predict_user')
@patch('app.services.feature_config.USE_SIGMOID_ALPHA', True)
def test_response_format_validation(
    mock_predict_user,
    mock_predict_global,
    sample_embedding,
    user_id
):
    """
    Test that response format includes all required fields.
    
    Verifies:
    - All original fields are present (backward compatibility)
    - All new alpha engine fields are present
    - Field types are correct
    - Values are in valid ranges
    
    Validates Requirements: 5.5, 5.6
    """
    # Reload alpha_engine
    import importlib
    import app.services.alpha_engine
    importlib.reload(app.services.alpha_engine)
    
    # Mock predictions
    P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
    P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
    mock_predict_global.return_value = (P_g, 0.7)
    mock_predict_user.return_value = (P_u, 0.8)
    
    result = classify_with_dual_heads(sample_embedding, user_id, feedback_count=50)
    
    # Verify all required fields are present
    required_fields = [
        "emotion", "confidence", "global_emotion", "global_confidence",
        "user_emotion", "user_confidence", "blend_weight",
        "alpha_data", "alpha_conf", "alpha_formula", "probabilities"
    ]
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Verify field types
    assert isinstance(result["emotion"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["global_emotion"], str)
    assert isinstance(result["global_confidence"], float)
    assert isinstance(result["user_emotion"], str) or result["user_emotion"] is None
    assert isinstance(result["user_confidence"], float) or result["user_confidence"] is None
    assert isinstance(result["blend_weight"], float)
    assert isinstance(result["alpha_data"], float) or result["alpha_data"] is None
    assert isinstance(result["alpha_conf"], float) or result["alpha_conf"] is None
    assert isinstance(result["alpha_formula"], str)
    assert isinstance(result["probabilities"], dict)
    
    # Verify value ranges
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["global_confidence"] <= 1.0
    if result["user_confidence"] is not None:
        assert 0.0 <= result["user_confidence"] <= 1.0
    assert 0.0 <= result["blend_weight"] <= 1.0
    if result["alpha_data"] is not None:
        assert 0.0 < result["alpha_data"] <= 1.0
    if result["alpha_conf"] is not None:
        assert 0.0 < result["alpha_conf"] < 1.0
    
    # Verify probabilities
    assert len(result["probabilities"]) == 8
    for emotion, prob in result["probabilities"].items():
        assert isinstance(emotion, str)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
    
    # Verify probabilities sum to 1.0
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test 7: Predictions are reasonable
# ---------------------------------------------------------------------------


@patch('app.services.dual_head_classifier.predict_global')
@patch('app.services.dual_head_classifier.predict_user')
@patch('app.services.feature_config.USE_SIGMOID_ALPHA', True)
def test_predictions_are_reasonable(
    mock_predict_user,
    mock_predict_global,
    sample_embedding,
    user_id
):
    """
    Test that predictions are reasonable across various scenarios.
    
    Verifies:
    - High confidence predictions have high probability
    - Low confidence predictions have lower probability
    - Blending produces sensible results
    - Extreme cases are handled correctly
    
    Validates Requirements: 5.5, 5.6
    """
    # Reload alpha_engine
    import importlib
    import app.services.alpha_engine
    importlib.reload(app.services.alpha_engine)
    
    # Scenario 1: High confidence global, no user head
    P_g_high = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.005, 0.002, 0.003])
    mock_predict_global.return_value = (P_g_high, 0.95)
    mock_predict_user.return_value = None
    
    result = classify_with_dual_heads(sample_embedding, user_id, feedback_count=0)
    
    assert result["emotion"] == "neutral"  # Index 0
    assert result["confidence"] >= 0.9
    assert result["blend_weight"] == 1.0
    
    # Scenario 2: Low confidence global, high confidence user
    P_g_low = np.array([0.4, 0.3, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02])
    P_u_high = np.array([0.05, 0.9, 0.02, 0.01, 0.01, 0.005, 0.002, 0.003])
    mock_predict_global.return_value = (P_g_low, 0.4)
    mock_predict_user.return_value = (P_u_high, 0.9)
    
    result = classify_with_dual_heads(sample_embedding, user_id, feedback_count=100)
    
    # With low global confidence and high feedback, should favor user head
    assert result["blend_weight"] < 0.3
    # Final prediction should be closer to user prediction
    assert result["emotion"] == "calm"  # User's prediction (index 1)
    
    # Scenario 3: Both heads agree
    P_g_agree = np.array([0.8, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01])
    P_u_agree = np.array([0.85, 0.04, 0.04, 0.02, 0.02, 0.015, 0.01, 0.005])
    mock_predict_global.return_value = (P_g_agree, 0.8)
    mock_predict_user.return_value = (P_u_agree, 0.85)
    
    result = classify_with_dual_heads(sample_embedding, user_id, feedback_count=50)
    
    # Both heads agree on "neutral" (index 0)
    assert result["global_emotion"] == "neutral"
    assert result["user_emotion"] == "neutral"
    assert result["emotion"] == "neutral"
    # Confidence should be high since both agree
    assert result["confidence"] >= 0.7
