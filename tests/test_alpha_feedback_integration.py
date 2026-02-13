"""
Integration Tests for Alpha Engine and Feedback Loop

This module tests the integration between the feedback loop and alpha engine,
verifying that feedback_count correctly influences alpha_data computation.

Feature: feedback-loop-personalization
Task: 4. CONSOLIDATED: Implement Async Training Engine and Model Persistence

Tests:
- Alpha engine uses feedback_count from User model
- Feedback count increases affect alpha_data (inverse relationship)
- Integration with dual-head classifier blending
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.alpha_engine import compute_alpha_data, compute_blend_weight
from app.services.feature_config import ALPHA_FEEDBACK_SCALE_K


class TestAlphaFeedbackIntegration:
    """Integration tests for alpha engine and feedback loop."""
    
    def test_alpha_data_uses_feedback_count(self):
        """
        Test that alpha_data computation uses feedback_count from User model.
        
        **Validates: Requirements 14.1, 14.5**
        """
        # Test various feedback counts
        test_cases = [
            (0, 1.0),  # No feedback → trust global fully
            (50, 0.5),  # K feedback → equal weight (K=50 default)
            (100, 0.333),  # 2K feedback → favor user head
            (200, 0.2),  # 4K feedback → strongly favor user head
        ]
        
        for feedback_count, expected_alpha_data in test_cases:
            alpha_data = compute_alpha_data(feedback_count, K=50.0)
            
            # Verify alpha_data matches expected value (with tolerance)
            assert abs(alpha_data - expected_alpha_data) < 0.01, (
                f"For feedback_count={feedback_count}, expected alpha_data≈{expected_alpha_data}, "
                f"got {alpha_data:.3f}"
            )
    
    def test_feedback_count_inverse_relationship(self):
        """
        Test that alpha_data decreases as feedback_count increases.
        
        **Validates: Requirements 14.2**
        """
        K = ALPHA_FEEDBACK_SCALE_K
        
        # Test monotonic decrease
        feedback_counts = [0, 10, 20, 30, 50, 100, 200, 500]
        alpha_values = [compute_alpha_data(count, K) for count in feedback_counts]
        
        # Verify monotonic decrease
        for i in range(len(alpha_values) - 1):
            assert alpha_values[i] > alpha_values[i + 1], (
                f"Alpha data should decrease: "
                f"alpha_data({feedback_counts[i]})={alpha_values[i]:.4f} should be > "
                f"alpha_data({feedback_counts[i+1]})={alpha_values[i+1]:.4f}"
            )
    
    def test_feedback_count_zero_special_case(self):
        """
        Test that feedback_count=0 results in alpha_data=1.0 (trust global fully).
        
        **Validates: Requirements 14.3**
        """
        alpha_data = compute_alpha_data(0, K=50.0)
        assert alpha_data == 1.0, (
            f"With 0 feedback, should trust global fully (alpha_data=1.0), got {alpha_data}"
        )
    
    def test_feedback_count_equals_k_special_case(self):
        """
        Test that feedback_count=K results in alpha_data=0.5 (equal weight).
        
        **Validates: Requirements 14.4**
        """
        K = 50.0
        alpha_data = compute_alpha_data(int(K), K)
        assert abs(alpha_data - 0.5) < 1e-10, (
            f"With feedback_count=K, should have equal weight (alpha_data=0.5), got {alpha_data}"
        )
    
    def test_alpha_formula_correctness(self):
        """
        Test that alpha_data formula is correctly implemented: 1 / (1 + N/K).
        
        **Validates: Requirements 14.5**
        """
        test_cases = [
            (0, 50.0, 1.0),
            (25, 50.0, 0.6667),
            (50, 50.0, 0.5),
            (100, 50.0, 0.3333),
            (150, 50.0, 0.25),
        ]
        
        for feedback_count, K, expected in test_cases:
            alpha_data = compute_alpha_data(feedback_count, K)
            expected_formula = 1.0 / (1.0 + feedback_count / K)
            
            # Verify formula matches
            assert abs(alpha_data - expected_formula) < 1e-10, (
                f"Formula mismatch for N={feedback_count}, K={K}: "
                f"got {alpha_data:.10f}, expected {expected_formula:.10f}"
            )
            
            # Verify against expected value
            assert abs(alpha_data - expected) < 0.01, (
                f"For N={feedback_count}, K={K}, expected ≈{expected}, got {alpha_data:.4f}"
            )
    
    def test_blend_weight_uses_feedback_count(self):
        """
        Test that compute_blend_weight correctly uses feedback_count.
        
        **Validates: Requirements 14.1, 14.5**
        """
        global_confidence = 0.8
        
        # Test with different feedback counts
        result_0 = compute_blend_weight(global_confidence, feedback_count=0)
        result_50 = compute_blend_weight(global_confidence, feedback_count=50)
        result_100 = compute_blend_weight(global_confidence, feedback_count=100)
        
        # Extract alpha values
        alpha_0 = result_0["alpha"]
        alpha_50 = result_50["alpha"]
        alpha_100 = result_100["alpha"]
        
        # Verify monotonic decrease (more feedback → lower alpha)
        assert alpha_0 > alpha_50 > alpha_100, (
            f"Alpha should decrease with feedback: "
            f"alpha(0)={alpha_0:.4f} > alpha(50)={alpha_50:.4f} > alpha(100)={alpha_100:.4f}"
        )
    
    def test_integration_with_user_model(self):
        """
        Test integration: User model feedback_count → alpha_data computation.
        
        This simulates the full flow:
        1. User submits feedback → feedback_count increments
        2. Prediction request → alpha engine reads feedback_count
        3. Alpha engine computes alpha_data using feedback_count
        4. Blending uses alpha_data to combine predictions
        
        **Validates: Requirements 14.1, 14.2, 14.5**
        """
        # Simulate user with increasing feedback
        user_feedback_counts = [0, 20, 30, 50, 100]
        global_confidence = 0.75
        
        alpha_values = []
        for feedback_count in user_feedback_counts:
            result = compute_blend_weight(global_confidence, feedback_count)
            alpha_values.append(result["alpha"])
        
        # Verify alpha decreases as feedback increases
        for i in range(len(alpha_values) - 1):
            assert alpha_values[i] > alpha_values[i + 1], (
                f"Alpha should decrease with more feedback: "
                f"alpha({user_feedback_counts[i]})={alpha_values[i]:.4f} should be > "
                f"alpha({user_feedback_counts[i+1]})={alpha_values[i+1]:.4f}"
            )
        
        # Verify first value (no feedback) has highest alpha
        assert alpha_values[0] > 0.5, (
            f"With no feedback, alpha should be high (> 0.5), got {alpha_values[0]:.4f}"
        )
        
        # Verify last value (lots of feedback) has lower alpha
        assert alpha_values[-1] < alpha_values[0], (
            f"With lots of feedback, alpha should be lower than initial: "
            f"{alpha_values[-1]:.4f} < {alpha_values[0]:.4f}"
        )
    
    def test_alpha_data_component_in_blend_result(self):
        """
        Test that blend_weight result includes alpha_data component.
        
        **Validates: Requirements 14.1**
        """
        result = compute_blend_weight(
            global_confidence=0.8,
            feedback_count=25,
            user_id="test_user"
        )
        
        # Verify result structure
        assert "alpha" in result
        assert "formula" in result
        
        # If using sigmoid formula, should have alpha_data component
        if result["formula"] == "sigmoid":
            assert "alpha_data" in result
            assert "alpha_conf" in result
            
            # Verify alpha_data is computed correctly
            expected_alpha_data = compute_alpha_data(25, K=ALPHA_FEEDBACK_SCALE_K)
            assert abs(result["alpha_data"] - expected_alpha_data) < 1e-10, (
                f"alpha_data mismatch: got {result['alpha_data']:.10f}, "
                f"expected {expected_alpha_data:.10f}"
            )


class TestAlphaEngineEdgeCases:
    """Test edge cases in alpha engine integration."""
    
    def test_very_large_feedback_count(self):
        """Test alpha_data with very large feedback counts."""
        large_counts = [1000, 5000, 10000]
        K = 50.0
        
        for count in large_counts:
            alpha_data = compute_alpha_data(count, K)
            
            # Should approach 0 but never be exactly 0
            assert 0.0 < alpha_data < 0.1, (
                f"For large feedback_count={count}, alpha_data should be close to 0, "
                f"got {alpha_data:.6f}"
            )
    
    def test_different_k_values(self):
        """Test alpha_data with different K scaling constants."""
        feedback_count = 50
        k_values = [10.0, 50.0, 100.0, 200.0]
        
        alpha_values = [compute_alpha_data(feedback_count, K) for K in k_values]
        
        # Verify: larger K → higher alpha_data (slower decay)
        for i in range(len(alpha_values) - 1):
            assert alpha_values[i] < alpha_values[i + 1], (
                f"Larger K should give higher alpha_data: "
                f"alpha(K={k_values[i]})={alpha_values[i]:.4f} should be < "
                f"alpha(K={k_values[i+1]})={alpha_values[i+1]:.4f}"
            )
    
    def test_feedback_count_boundary_values(self):
        """Test alpha_data at boundary values."""
        K = 50.0
        
        # Test at 0
        alpha_0 = compute_alpha_data(0, K)
        assert alpha_0 == 1.0
        
        # Test at 1
        alpha_1 = compute_alpha_data(1, K)
        assert 0.98 < alpha_1 < 1.0
        
        # Test at K-1, K, K+1
        alpha_k_minus_1 = compute_alpha_data(49, K)
        alpha_k = compute_alpha_data(50, K)
        alpha_k_plus_1 = compute_alpha_data(51, K)
        
        assert alpha_k_minus_1 > alpha_k > alpha_k_plus_1
        assert abs(alpha_k - 0.5) < 1e-10


class TestAlphaFeedbackScenarios:
    """Test realistic scenarios of feedback accumulation."""
    
    def test_new_user_journey(self):
        """
        Test alpha evolution as a new user provides feedback.
        
        Simulates: User starts with 0 feedback, provides corrections over time.
        """
        K = 50.0
        
        # User journey: 0 → 5 → 10 → 20 → 30 → 50 → 100 feedback
        journey = [0, 5, 10, 20, 30, 50, 100]
        
        alpha_journey = [compute_alpha_data(count, K) for count in journey]
        
        # Verify smooth decrease
        for i in range(len(alpha_journey) - 1):
            assert alpha_journey[i] > alpha_journey[i + 1]
        
        # Verify key milestones
        assert alpha_journey[0] == 1.0  # Start: trust global fully
        assert abs(alpha_journey[5] - 0.5) < 1e-10  # At K: equal weight
        assert alpha_journey[6] < 0.35  # At 2K: favor user head
    
    def test_active_user_scenario(self):
        """
        Test alpha for very active user with lots of feedback.
        
        Active users should have low alpha (trust user head more).
        """
        K = 50.0
        active_user_feedback = 200  # Lots of feedback
        
        alpha_data = compute_alpha_data(active_user_feedback, K)
        
        # Should strongly favor user head
        assert alpha_data < 0.25, (
            f"Active user (N={active_user_feedback}) should have low alpha, got {alpha_data:.4f}"
        )
        
        # But never exactly 0
        assert alpha_data > 0.0
    
    def test_casual_user_scenario(self):
        """
        Test alpha for casual user with minimal feedback.
        
        Casual users should have high alpha (trust global more).
        """
        K = 50.0
        casual_user_feedback = 5  # Minimal feedback
        
        alpha_data = compute_alpha_data(casual_user_feedback, K)
        
        # Should mostly trust global head
        assert alpha_data > 0.9, (
            f"Casual user (N={casual_user_feedback}) should have high alpha, got {alpha_data:.4f}"
        )
        
        # But not exactly 1.0 (unless N=0)
        assert alpha_data < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
