"""
Unit tests for the dual-head classifier module.

Tests cover:
- compute_blend_weight() with various feedback counts and confidences
- blend_predictions() probability distribution properties
- classify_with_dual_heads() with global only and both heads
- Alpha clamping and edge cases
- Numerical stability and error handling

Validates Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.dual_head_classifier import (
    compute_blend_weight,
    blend_predictions,
    classify_with_dual_heads,
)


class TestComputeBlendWeight:
    """Test the compute_blend_weight function (Requirement 4.1)."""
    
    def test_insufficient_feedback_returns_global_only(self):
        """Test that feedback_count < 20 returns alpha = 1.0 (global only)."""
        # Requirement 4.2: When user has 0-19 feedback samples, use only Global_Head
        assert compute_blend_weight(0.5, 0) == 1.0
        assert compute_blend_weight(0.7, 10) == 1.0
        assert compute_blend_weight(0.9, 19) == 1.0
    
    def test_medium_feedback_blending(self):
        """Test blending with 20-99 feedback samples."""
        # Requirement 4.3: When user has 20-99 feedback samples, blend with increasing user weight
        
        # feedback_count = 20, C_g = 0.7
        # α = 0.5 + 0.3*0.7 - 0.2*min(20/100, 1.0) = 0.5 + 0.21 - 0.04 = 0.67
        alpha = compute_blend_weight(0.7, 20)
        assert np.isclose(alpha, 0.67, atol=0.01)
        
        # feedback_count = 50, C_g = 0.7
        # α = 0.5 + 0.3*0.7 - 0.2*min(50/100, 1.0) = 0.5 + 0.21 - 0.1 = 0.61
        alpha = compute_blend_weight(0.7, 50)
        assert np.isclose(alpha, 0.61, atol=0.01)
        
        # feedback_count = 99, C_g = 0.7
        # α = 0.5 + 0.3*0.7 - 0.2*min(99/100, 1.0) = 0.5 + 0.21 - 0.198 = 0.512
        alpha = compute_blend_weight(0.7, 99)
        assert np.isclose(alpha, 0.512, atol=0.01)
    
    def test_high_feedback_low_confidence_favors_user(self):
        """Test that 100+ feedback with low confidence favors user head."""
        # Requirement 4.4: When user has 100+ feedback AND C_g < 0.6, favor User_Head (α ≈ 0.3)
        
        # feedback_count = 100, C_g = 0.5
        # α = 0.5 + 0.3*0.5 - 0.2*min(100/100, 1.0) = 0.5 + 0.15 - 0.2 = 0.45
        alpha = compute_blend_weight(0.5, 100)
        assert np.isclose(alpha, 0.45, atol=0.01)
        
        # feedback_count = 100, C_g = 0.3
        # α = 0.5 + 0.3*0.3 - 0.2*1.0 = 0.5 + 0.09 - 0.2 = 0.39
        alpha = compute_blend_weight(0.3, 100)
        assert np.isclose(alpha, 0.39, atol=0.01)
        
        # feedback_count = 200, C_g = 0.4 (should still clamp at 0.3)
        # α = 0.5 + 0.3*0.4 - 0.2*1.0 = 0.5 + 0.12 - 0.2 = 0.42
        alpha = compute_blend_weight(0.4, 200)
        assert np.isclose(alpha, 0.42, atol=0.01)
    
    def test_high_feedback_high_confidence_maintains_global_weight(self):
        """Test that high global confidence maintains higher global weight."""
        # Requirement 4.5: When C_g > 0.9, maintain higher global weight regardless of feedback
        
        # feedback_count = 100, C_g = 0.9
        # α = 0.5 + 0.3*0.9 - 0.2*1.0 = 0.5 + 0.27 - 0.2 = 0.57
        alpha = compute_blend_weight(0.9, 100)
        assert np.isclose(alpha, 0.57, atol=0.01)
        
        # feedback_count = 200, C_g = 0.95
        # α = 0.5 + 0.3*0.95 - 0.2*1.0 = 0.5 + 0.285 - 0.2 = 0.585
        alpha = compute_blend_weight(0.95, 200)
        assert np.isclose(alpha, 0.585, atol=0.01)
    
    def test_alpha_clamping_lower_bound(self):
        """Test that alpha is clamped to minimum 0.3."""
        # Even with very low confidence and high feedback, alpha >= 0.3
        
        # feedback_count = 1000, C_g = 0.0
        # α = 0.5 + 0.3*0.0 - 0.2*1.0 = 0.3 (at lower bound)
        alpha = compute_blend_weight(0.0, 1000)
        assert alpha == 0.3
        
        # feedback_count = 500, C_g = 0.1
        # α = 0.5 + 0.3*0.1 - 0.2*1.0 = 0.33
        alpha = compute_blend_weight(0.1, 500)
        assert np.isclose(alpha, 0.33, atol=0.01)
    
    def test_alpha_clamping_upper_bound(self):
        """Test that alpha is clamped to maximum 1.0."""
        # With very high confidence and low feedback, alpha <= 1.0
        
        # feedback_count = 20, C_g = 1.0
        # α = 0.5 + 0.3*1.0 - 0.2*0.2 = 0.5 + 0.3 - 0.04 = 0.76 (< 1.0, no clamping)
        alpha = compute_blend_weight(1.0, 20)
        assert np.isclose(alpha, 0.76, atol=0.01)
        assert alpha <= 1.0
    
    def test_alpha_range_always_valid(self):
        """Test that alpha is always in [0.3, 1.0] for any valid inputs."""
        # Test a wide range of inputs
        for feedback_count in [0, 10, 20, 50, 100, 200, 500]:
            for confidence in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
                alpha = compute_blend_weight(confidence, feedback_count)
                assert 0.3 <= alpha <= 1.0, \
                    f"Alpha {alpha} out of range for feedback={feedback_count}, conf={confidence}"
    
    def test_feedback_count_saturation(self):
        """Test that feedback_count saturates at 100 in the formula."""
        # feedback_count > 100 should have same effect as feedback_count = 100
        
        alpha_100 = compute_blend_weight(0.7, 100)
        alpha_200 = compute_blend_weight(0.7, 200)
        alpha_1000 = compute_blend_weight(0.7, 1000)
        
        assert np.isclose(alpha_100, alpha_200)
        assert np.isclose(alpha_100, alpha_1000)


class TestBlendPredictions:
    """Test the blend_predictions function."""
    
    def test_blend_with_alpha_1_returns_global(self):
        """Test that alpha=1.0 returns only global predictions."""
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.7, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        
        P_f = blend_predictions(P_g, P_u, alpha=1.0)
        
        assert np.allclose(P_f, P_g, atol=1e-5)
    
    def test_blend_with_alpha_0_returns_user(self):
        """Test that alpha=0.0 returns only user predictions."""
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.7, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        
        P_f = blend_predictions(P_g, P_u, alpha=0.0)
        
        assert np.allclose(P_f, P_u, atol=1e-5)
    
    def test_blend_with_alpha_0_5_equal_weight(self):
        """Test that alpha=0.5 gives equal weight to both predictions."""
        P_g = np.array([0.8, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
        P_u = np.array([0.2, 0.6, 0.1, 0.05, 0.02, 0.01, 0.005, 0.005])
        
        P_f = blend_predictions(P_g, P_u, alpha=0.5)
        
        # Expected: 0.5 * P_g + 0.5 * P_u
        expected = 0.5 * P_g + 0.5 * P_u
        expected = expected / expected.sum()  # Normalize
        
        assert np.allclose(P_f, expected, atol=1e-5)
    
    def test_blended_distribution_sums_to_one(self):
        """Test that blended distribution always sums to 1.0."""
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.7, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            P_f = blend_predictions(P_g, P_u, alpha)
            assert np.isclose(P_f.sum(), 1.0, atol=1e-5)
    
    def test_blended_distribution_non_negative(self):
        """Test that all probabilities in blended distribution are non-negative."""
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.7, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            P_f = blend_predictions(P_g, P_u, alpha)
            assert np.all(P_f >= 0.0)
    
    def test_blended_distribution_bounded(self):
        """Test that all probabilities are in [0, 1]."""
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.7, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            P_f = blend_predictions(P_g, P_u, alpha)
            assert np.all(P_f <= 1.0)
    
    def test_blend_with_different_distributions(self):
        """Test blending with various probability distributions."""
        # Uniform distributions
        P_uniform = np.ones(8) / 8
        P_peaked = np.array([0.9, 0.02, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005])
        
        P_f = blend_predictions(P_uniform, P_peaked, alpha=0.5)
        
        assert np.isclose(P_f.sum(), 1.0, atol=1e-5)
        assert np.all(P_f >= 0.0)
        assert np.all(P_f <= 1.0)
    
    def test_blend_numerical_stability(self):
        """Test that blending handles numerical precision issues."""
        # Create distributions that don't sum exactly to 1.0 due to floating point
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_g = P_g + 1e-10  # Add small noise
        
        P_u = np.array([0.1, 0.7, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = P_u - 1e-10  # Subtract small noise
        
        P_f = blend_predictions(P_g, P_u, alpha=0.5)
        
        # Should still sum to 1.0 after normalization
        assert np.isclose(P_f.sum(), 1.0, atol=1e-5)


class TestClassifyWithDualHeads:
    """Test the classify_with_dual_heads function."""
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_global_only_with_insufficient_feedback(self, mock_predict_user, mock_predict_global):
        """Test classification with feedback_count < 20 (global only)."""
        # Requirement 4.2: Use only Global_Head when feedback_count < 20
        
        # Setup mocks
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        mock_predict_global.return_value = (P_g, 0.7)
        mock_predict_user.return_value = None
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=10)
        
        # Should use global only
        assert result["emotion"] == "neutral"  # Index 0
        assert np.isclose(result["confidence"], 0.7)
        assert result["global_emotion"] == "neutral"
        assert np.isclose(result["global_confidence"], 0.7)
        assert result["user_emotion"] is None
        assert result["user_confidence"] is None
        assert result["blend_weight"] == 1.0
        
        # Verify predict_user was called but returned None
        mock_predict_user.assert_called_once()
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_global_only_when_user_head_unavailable(self, mock_predict_user, mock_predict_global):
        """Test classification when user head is not available."""
        # Setup mocks
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        mock_predict_global.return_value = (P_g, 0.7)
        mock_predict_user.return_value = None  # User head not available
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=50)
        
        # Should fall back to global only
        assert result["emotion"] == "neutral"
        assert result["user_emotion"] is None
        assert result["blend_weight"] == 1.0
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_dual_head_with_both_available(self, mock_predict_user, mock_predict_global):
        """Test classification with both global and user heads available."""
        # Requirement 3.4: Compute blended prediction when both heads are active
        
        # Setup mocks
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
        mock_predict_global.return_value = (P_g, 0.7)
        mock_predict_user.return_value = (P_u, 0.8)
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=50)
        
        # Should blend predictions
        assert result["global_emotion"] == "neutral"  # Index 0
        assert result["user_emotion"] == "calm"  # Index 1
        assert result["user_confidence"] is not None
        assert 0.3 <= result["blend_weight"] <= 1.0
        
        # Final emotion should be from blended distribution
        assert result["emotion"] in ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "calm"]
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Probabilities should sum to 1.0
        prob_sum = sum(result["probabilities"].values())
        assert np.isclose(prob_sum, 1.0, atol=1e-5)
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_dual_head_blending_weight_calculation(self, mock_predict_user, mock_predict_global):
        """Test that blending weight is calculated correctly."""
        # Requirement 4.1: Blending weight computed dynamically
        
        # Setup mocks
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
        mock_predict_global.return_value = (P_g, 0.7)
        mock_predict_user.return_value = (P_u, 0.8)
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=50)
        
        # Expected alpha for feedback_count=50, C_g=0.7
        # α = 0.5 + 0.3*0.7 - 0.2*0.5 = 0.61
        expected_alpha = 0.61
        assert np.isclose(result["blend_weight"], expected_alpha, atol=0.01)
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_dual_head_with_high_feedback_low_confidence(self, mock_predict_user, mock_predict_global):
        """Test that user head is favored with high feedback and low global confidence."""
        # Requirement 4.4: Favor User_Head when feedback >= 100 and C_g < 0.6
        
        # Setup mocks
        P_g = np.array([0.5, 0.15, 0.15, 0.1, 0.05, 0.02, 0.015, 0.015])
        P_u = np.array([0.1, 0.85, 0.02, 0.01, 0.01, 0.005, 0.002, 0.003])
        mock_predict_global.return_value = (P_g, 0.5)  # Low confidence
        mock_predict_user.return_value = (P_u, 0.85)  # High confidence
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=100)
        
        # Expected alpha for feedback_count=100, C_g=0.5
        # α = 0.5 + 0.3*0.5 - 0.2*1.0 = 0.45
        expected_alpha = 0.45
        assert np.isclose(result["blend_weight"], expected_alpha, atol=0.01)
        
        # User head should have more influence (alpha < 0.5)
        assert result["blend_weight"] < 0.5
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_dual_head_with_high_confidence_maintains_global(self, mock_predict_user, mock_predict_global):
        """Test that global head maintains weight with high confidence."""
        # Requirement 4.5: Maintain higher global weight when C_g > 0.9
        
        # Setup mocks
        P_g = np.array([0.9, 0.03, 0.03, 0.01, 0.01, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
        mock_predict_global.return_value = (P_g, 0.9)  # High confidence
        mock_predict_user.return_value = (P_u, 0.8)
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=100)
        
        # Expected alpha for feedback_count=100, C_g=0.9
        # α = 0.5 + 0.3*0.9 - 0.2*1.0 = 0.57
        expected_alpha = 0.57
        assert np.isclose(result["blend_weight"], expected_alpha, atol=0.01)
        
        # Global head should maintain significant influence (alpha > 0.5)
        assert result["blend_weight"] > 0.5
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_probabilities_dict_format(self, mock_predict_user, mock_predict_global):
        """Test that probabilities dictionary has correct format."""
        # Setup mocks
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        mock_predict_global.return_value = (P_g, 0.7)
        mock_predict_user.return_value = None
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=10)
        
        # Check probabilities dict
        assert "probabilities" in result
        assert len(result["probabilities"]) == 8
        
        # All emotion labels should be present
        expected_labels = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "calm"]
        for label in expected_labels:
            assert label in result["probabilities"]
            assert 0.0 <= result["probabilities"][label] <= 1.0
        
        # Should sum to 1.0
        prob_sum = sum(result["probabilities"].values())
        assert np.isclose(prob_sum, 1.0, atol=1e-5)
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_edge_case_zero_feedback(self, mock_predict_user, mock_predict_global):
        """Test classification with zero feedback."""
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        mock_predict_global.return_value = (P_g, 0.7)
        mock_predict_user.return_value = None
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=0)
        
        assert result["blend_weight"] == 1.0
        assert result["user_emotion"] is None
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_edge_case_exactly_20_feedback(self, mock_predict_user, mock_predict_global):
        """Test classification with exactly 20 feedback samples (threshold)."""
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
        mock_predict_global.return_value = (P_g, 0.7)
        mock_predict_user.return_value = (P_u, 0.8)
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=20)
        
        # Should start blending at 20 feedback samples
        assert result["blend_weight"] < 1.0
        assert result["user_emotion"] is not None
    
    @patch('app.services.dual_head_classifier.predict_global')
    @patch('app.services.dual_head_classifier.predict_user')
    def test_edge_case_very_high_feedback(self, mock_predict_user, mock_predict_global):
        """Test classification with very high feedback count."""
        P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
        mock_predict_global.return_value = (P_g, 0.7)
        mock_predict_user.return_value = (P_u, 0.8)
        
        embedding = np.random.randn(768)
        result = classify_with_dual_heads(embedding, "user_123", feedback_count=1000)
        
        # Feedback count should saturate at 100 in formula
        assert 0.3 <= result["blend_weight"] <= 1.0
