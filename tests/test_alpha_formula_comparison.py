"""
Comparison tests between sigmoid and linear alpha formulas.

This module compares the behavior of the new sigmoid-based alpha formula
against the legacy linear formula to document behavioral differences.

Key comparisons:
- New user (N=0) with varying confidence
- Medium feedback (N=50) with varying confidence
- High feedback (N=100) with varying confidence
- Verification that sigmoid favors user head more aggressively

Validates Requirement: 6.5
"""

import pytest
import numpy as np
from typing import List, Tuple

from app.services.alpha_engine import (
    compute_alpha_sigmoid,
    compute_alpha_linear
)


class TestFormulaComparison:
    """Compare sigmoid and linear alpha formulas."""
    
    def test_new_user_low_confidence(self):
        """Compare formulas for new user (N=0) with low confidence (C_g=0.5)."""
        N = 0
        C_g = 0.5
        
        # Linear formula
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        
        # Sigmoid formula
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Document behavior
        # Linear: N < 20 → alpha = 1.0 (hardcoded)
        # Sigmoid: alpha_data = 1.0, alpha_conf ≈ 0.27, alpha ≈ 0.27
        
        assert alpha_linear == 1.0, "Linear should return 1.0 for N < 20"
        assert alpha_data == 1.0, "Sigmoid alpha_data should be 1.0 for N=0"
        assert alpha_conf < 0.5, "Sigmoid alpha_conf should be < 0.5 for C_g < tau"
        assert alpha_sigmoid < 0.5, "Sigmoid alpha should be < 0.5"
        
        # Key difference: Linear ignores confidence for new users
        # Sigmoid responds to confidence even for new users
        assert alpha_linear > alpha_sigmoid, \
            "Linear favors global more for new users with low confidence"

    def test_new_user_medium_confidence(self):
        """Compare formulas for new user (N=0) with medium confidence (C_g=0.7)."""
        N = 0
        C_g = 0.7
        
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Linear: N < 20 → alpha = 1.0
        # Sigmoid: alpha_data = 1.0, alpha_conf ≈ 0.73, alpha ≈ 0.73
        
        assert alpha_linear == 1.0
        assert alpha_data == 1.0
        assert alpha_conf > 0.5, "Sigmoid alpha_conf should be > 0.5 for C_g > tau"
        assert alpha_sigmoid > 0.5
        
        # Linear still ignores confidence
        assert alpha_linear > alpha_sigmoid
    
    def test_new_user_high_confidence(self):
        """Compare formulas for new user (N=0) with high confidence (C_g=0.9)."""
        N = 0
        C_g = 0.9
        
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Linear: N < 20 → alpha = 1.0
        # Sigmoid: alpha_data = 1.0, alpha_conf ≈ 0.95, alpha ≈ 0.95
        
        assert alpha_linear == 1.0
        assert alpha_data == 1.0
        assert alpha_conf > 0.9, "Sigmoid alpha_conf should be very high for C_g=0.9"
        assert alpha_sigmoid > 0.9
        
        # Both formulas favor global head strongly
        # Sigmoid is slightly lower but still high
        assert abs(alpha_linear - alpha_sigmoid) < 0.1

    def test_medium_feedback_low_confidence(self):
        """Compare formulas for medium feedback (N=50) with low confidence (C_g=0.5)."""
        N = 50
        C_g = 0.5
        
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Linear: α = 0.5 + 0.3·0.5 - 0.2·0.5 = 0.5 + 0.15 - 0.1 = 0.55
        # Sigmoid: alpha_data = 0.5, alpha_conf ≈ 0.27, alpha ≈ 0.14
        
        assert np.isclose(alpha_linear, 0.55, atol=0.01)
        assert np.isclose(alpha_data, 0.5, atol=0.01)
        assert alpha_conf < 0.5
        assert alpha_sigmoid < 0.2
        
        # Key difference: Sigmoid favors user head much more aggressively
        assert alpha_linear > alpha_sigmoid
        assert alpha_linear - alpha_sigmoid > 0.3, \
            "Sigmoid should favor user head significantly more"
    
    def test_medium_feedback_medium_confidence(self):
        """Compare formulas for medium feedback (N=50) with medium confidence (C_g=0.7)."""
        N = 50
        C_g = 0.7
        
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Linear: α = 0.5 + 0.3·0.7 - 0.2·0.5 = 0.5 + 0.21 - 0.1 = 0.61
        # Sigmoid: alpha_data = 0.5, alpha_conf ≈ 0.73, alpha ≈ 0.37
        
        assert np.isclose(alpha_linear, 0.61, atol=0.01)
        assert np.isclose(alpha_data, 0.5, atol=0.01)
        assert alpha_conf > 0.5
        assert alpha_sigmoid < 0.4
        
        # Sigmoid still favors user head more
        assert alpha_linear > alpha_sigmoid
        assert alpha_linear - alpha_sigmoid > 0.2

    def test_medium_feedback_high_confidence(self):
        """Compare formulas for medium feedback (N=50) with high confidence (C_g=0.9)."""
        N = 50
        C_g = 0.9
        
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Linear: α = 0.5 + 0.3·0.9 - 0.2·0.5 = 0.5 + 0.27 - 0.1 = 0.67
        # Sigmoid: alpha_data = 0.5, alpha_conf ≈ 0.95, alpha ≈ 0.48
        
        assert np.isclose(alpha_linear, 0.67, atol=0.01)
        assert np.isclose(alpha_data, 0.5, atol=0.01)
        assert alpha_conf > 0.9
        assert alpha_sigmoid < 0.5
        
        # Even with high confidence, sigmoid favors user head more
        assert alpha_linear > alpha_sigmoid
        assert alpha_linear - alpha_sigmoid > 0.15
    
    def test_high_feedback_low_confidence(self):
        """Compare formulas for high feedback (N=100) with low confidence (C_g=0.5)."""
        N = 100
        C_g = 0.5
        
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Linear: α = 0.5 + 0.3·0.5 - 0.2·1.0 = 0.5 + 0.15 - 0.2 = 0.45
        # Sigmoid: alpha_data ≈ 0.33, alpha_conf ≈ 0.27, alpha ≈ 0.09
        
        assert np.isclose(alpha_linear, 0.45, atol=0.01)
        assert np.isclose(alpha_data, 0.333, atol=0.01)
        assert alpha_conf < 0.5
        assert alpha_sigmoid < 0.15
        
        # Sigmoid strongly favors user head
        assert alpha_linear > alpha_sigmoid
        assert alpha_linear - alpha_sigmoid > 0.3

    def test_high_feedback_medium_confidence(self):
        """Compare formulas for high feedback (N=100) with medium confidence (C_g=0.7)."""
        N = 100
        C_g = 0.7
        
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Linear: α = 0.5 + 0.3·0.7 - 0.2·1.0 = 0.5 + 0.21 - 0.2 = 0.51
        # Sigmoid: alpha_data ≈ 0.33, alpha_conf ≈ 0.73, alpha ≈ 0.24
        
        assert np.isclose(alpha_linear, 0.51, atol=0.01)
        assert np.isclose(alpha_data, 0.333, atol=0.01)
        assert alpha_conf > 0.5
        assert alpha_sigmoid < 0.3
        
        # Sigmoid favors user head significantly more
        assert alpha_linear > alpha_sigmoid
        assert alpha_linear - alpha_sigmoid > 0.2
    
    def test_high_feedback_high_confidence(self):
        """Compare formulas for high feedback (N=100) with high confidence (C_g=0.9)."""
        N = 100
        C_g = 0.9
        
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Linear: α = 0.5 + 0.3·0.9 - 0.2·1.0 = 0.5 + 0.27 - 0.2 = 0.57
        # Sigmoid: alpha_data ≈ 0.33, alpha_conf ≈ 0.95, alpha ≈ 0.31
        
        assert np.isclose(alpha_linear, 0.57, atol=0.01)
        assert np.isclose(alpha_data, 0.333, atol=0.01)
        assert alpha_conf > 0.9
        assert alpha_sigmoid < 0.35
        
        # Even with high confidence, sigmoid favors user head more
        assert alpha_linear > alpha_sigmoid
        assert alpha_linear - alpha_sigmoid > 0.2

    def test_sigmoid_favors_user_head_more_aggressively(self):
        """Verify that sigmoid favors user head more aggressively than linear."""
        # Test across a range of feedback counts and confidence values
        feedback_counts = [25, 50, 75, 100, 150, 200]
        confidence_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for N in feedback_counts:
            for C_g in confidence_values:
                alpha_linear = compute_alpha_linear(
                    global_confidence=C_g,
                    feedback_count=N
                )
                _, _, alpha_sigmoid = compute_alpha_sigmoid(
                    global_confidence=C_g,
                    feedback_count=N
                )
                
                # For N >= 20, sigmoid should always favor user head more
                # (i.e., lower alpha = less weight on global head)
                assert alpha_sigmoid <= alpha_linear, \
                    f"N={N}, C_g={C_g}: sigmoid ({alpha_sigmoid:.3f}) should be " \
                    f"<= linear ({alpha_linear:.3f})"
    
    def test_comparison_table_generation(self):
        """Generate comparison table for documentation purposes."""
        # This test generates a comparison table showing behavioral differences
        
        scenarios = [
            # (description, N, C_g)
            ("New user, low conf", 0, 0.5),
            ("New user, med conf", 0, 0.7),
            ("New user, high conf", 0, 0.9),
            ("Medium feedback, low conf", 50, 0.5),
            ("Medium feedback, med conf", 50, 0.7),
            ("Medium feedback, high conf", 50, 0.9),
            ("High feedback, low conf", 100, 0.5),
            ("High feedback, med conf", 100, 0.7),
            ("High feedback, high conf", 100, 0.9),
        ]
        
        print("\n" + "="*80)
        print("ALPHA FORMULA COMPARISON TABLE")
        print("="*80)
        print(f"{'Scenario':<30} {'N':>5} {'C_g':>5} {'Linear':>8} {'Sigmoid':>8} {'Diff':>8}")
        print("-"*80)
        
        for description, N, C_g in scenarios:
            alpha_linear = compute_alpha_linear(
                global_confidence=C_g,
                feedback_count=N
            )
            alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
                global_confidence=C_g,
                feedback_count=N
            )
            diff = alpha_linear - alpha_sigmoid
            
            print(f"{description:<30} {N:>5} {C_g:>5.1f} "
                  f"{alpha_linear:>8.3f} {alpha_sigmoid:>8.3f} {diff:>8.3f}")
        
        print("="*80)
        print("Key observations:")
        print("- Linear ignores confidence for N < 20 (always returns 1.0)")
        print("- Sigmoid responds to confidence even for new users")
        print("- Sigmoid favors user head more aggressively at all feedback levels")
        print("- Difference is most pronounced at medium feedback levels")
        print("="*80 + "\n")
        
        # This test always passes - it's for documentation
        assert True

    def test_linear_hardcoded_threshold_behavior(self):
        """Test linear formula's hardcoded behavior for N < 20."""
        # Linear formula has special case: N < 20 → alpha = 1.0
        
        for N in range(0, 20):
            for C_g in [0.3, 0.5, 0.7, 0.9]:
                alpha_linear = compute_alpha_linear(
                    global_confidence=C_g,
                    feedback_count=N
                )
                
                # Should always be 1.0 regardless of confidence
                assert alpha_linear == 1.0, \
                    f"Linear should return 1.0 for N={N}, got {alpha_linear}"
        
        # At N=20, linear formula starts using the actual formula
        alpha_20 = compute_alpha_linear(global_confidence=0.7, feedback_count=20)
        assert alpha_20 < 1.0, "Linear should use formula for N >= 20"
    
    def test_sigmoid_smooth_transition_vs_linear_discontinuity(self):
        """Test that sigmoid has smooth transition while linear has discontinuity at N=20."""
        C_g = 0.7
        
        # Linear has discontinuity at N=20
        alpha_linear_19 = compute_alpha_linear(global_confidence=C_g, feedback_count=19)
        alpha_linear_20 = compute_alpha_linear(global_confidence=C_g, feedback_count=20)
        
        # Should have a jump from 1.0 to ~0.71
        assert alpha_linear_19 == 1.0
        assert alpha_linear_20 < 0.75
        linear_jump = alpha_linear_19 - alpha_linear_20
        assert linear_jump > 0.25, "Linear has discontinuity at N=20"
        
        # Sigmoid has smooth transition
        _, _, alpha_sigmoid_19 = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=19
        )
        _, _, alpha_sigmoid_20 = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=20
        )
        
        sigmoid_diff = abs(alpha_sigmoid_19 - alpha_sigmoid_20)
        assert sigmoid_diff < 0.05, "Sigmoid should have smooth transition"
        
        # Sigmoid transition is much smoother than linear
        assert sigmoid_diff < linear_jump

    def test_multiplicative_vs_additive_behavior(self):
        """Test multiplicative (sigmoid) vs additive (linear) combination logic."""
        # Scenario: Low confidence (C_g=0.3) with medium feedback (N=50)
        N = 50
        C_g = 0.3
        
        alpha_linear = compute_alpha_linear(global_confidence=C_g, feedback_count=N)
        alpha_data, alpha_conf, alpha_sigmoid = compute_alpha_sigmoid(
            global_confidence=C_g,
            feedback_count=N
        )
        
        # Linear uses additive logic: α = 0.5 + 0.3·C_g - 0.2·feedback_factor
        # Components can partially cancel each other out
        # α = 0.5 + 0.09 - 0.1 = 0.49
        
        # Sigmoid uses multiplicative logic: α = alpha_data × alpha_conf
        # Both components must be high for high alpha
        # α = 0.5 × 0.05 = 0.025
        
        assert np.isclose(alpha_linear, 0.49, atol=0.01)
        assert alpha_sigmoid < 0.1
        
        # Multiplicative logic is more conservative
        # When confidence is low, sigmoid strongly favors user head
        assert alpha_sigmoid < alpha_linear / 4, \
            "Multiplicative logic should be much more conservative with low confidence"
    
    def test_confidence_sensitivity_comparison(self):
        """Compare how sensitive each formula is to confidence changes."""
        N = 50  # Medium feedback
        
        # Test confidence range from 0.3 to 0.9
        confidence_values = [0.3, 0.5, 0.7, 0.9]
        
        linear_alphas = []
        sigmoid_alphas = []
        
        for C_g in confidence_values:
            alpha_linear = compute_alpha_linear(
                global_confidence=C_g,
                feedback_count=N
            )
            _, _, alpha_sigmoid = compute_alpha_sigmoid(
                global_confidence=C_g,
                feedback_count=N
            )
            
            linear_alphas.append(alpha_linear)
            sigmoid_alphas.append(alpha_sigmoid)
        
        # Calculate ranges (max - min)
        linear_range = max(linear_alphas) - min(linear_alphas)
        sigmoid_range = max(sigmoid_alphas) - min(sigmoid_alphas)
        
        # Sigmoid should have wider range (more sensitive to confidence)
        # Linear: ~0.49 to ~0.67 (range ~0.18)
        # Sigmoid: ~0.025 to ~0.48 (range ~0.45)
        
        assert sigmoid_range > linear_range, \
            "Sigmoid should be more sensitive to confidence changes"
        
        print(f"\nConfidence sensitivity at N={N}:")
        print(f"Linear range: {linear_range:.3f}")
        print(f"Sigmoid range: {sigmoid_range:.3f}")
        print(f"Sigmoid is {sigmoid_range/linear_range:.1f}x more sensitive\n")

    def test_feedback_sensitivity_comparison(self):
        """Compare how sensitive each formula is to feedback count changes."""
        C_g = 0.7  # Medium confidence
        
        # Test feedback range from 0 to 200
        feedback_counts = [0, 25, 50, 100, 200]
        
        linear_alphas = []
        sigmoid_alphas = []
        
        for N in feedback_counts:
            alpha_linear = compute_alpha_linear(
                global_confidence=C_g,
                feedback_count=N
            )
            _, _, alpha_sigmoid = compute_alpha_sigmoid(
                global_confidence=C_g,
                feedback_count=N
            )
            
            linear_alphas.append(alpha_linear)
            sigmoid_alphas.append(alpha_sigmoid)
        
        # Calculate ranges
        linear_range = max(linear_alphas) - min(linear_alphas)
        sigmoid_range = max(sigmoid_alphas) - min(sigmoid_alphas)
        
        # Both should decrease with feedback, but sigmoid more aggressively
        # Linear: 1.0 to ~0.51 (range ~0.49)
        # Sigmoid: ~0.73 to ~0.14 (range ~0.59)
        
        assert sigmoid_range > linear_range * 0.8, \
            "Sigmoid should have comparable or greater feedback sensitivity"
        
        print(f"\nFeedback sensitivity at C_g={C_g}:")
        print(f"Linear range: {linear_range:.3f}")
        print(f"Sigmoid range: {sigmoid_range:.3f}\n")
    
    def test_extreme_scenarios(self):
        """Test extreme scenarios to highlight behavioral differences."""
        extreme_cases = [
            # (description, N, C_g, expected_behavior)
            ("New user, very low conf", 0, 0.2, "sigmoid_much_lower"),
            ("New user, very high conf", 0, 0.95, "similar"),
            ("Lots of feedback, very low conf", 200, 0.2, "sigmoid_much_lower"),
            ("Lots of feedback, very high conf", 200, 0.95, "sigmoid_lower"),
        ]
        
        print("\n" + "="*80)
        print("EXTREME SCENARIO COMPARISON")
        print("="*80)
        
        for description, N, C_g, expected_behavior in extreme_cases:
            alpha_linear = compute_alpha_linear(
                global_confidence=C_g,
                feedback_count=N
            )
            _, _, alpha_sigmoid = compute_alpha_sigmoid(
                global_confidence=C_g,
                feedback_count=N
            )
            
            diff = alpha_linear - alpha_sigmoid
            ratio = alpha_linear / alpha_sigmoid if alpha_sigmoid > 0.01 else float('inf')
            
            print(f"\n{description}:")
            print(f"  N={N}, C_g={C_g}")
            print(f"  Linear:  {alpha_linear:.4f}")
            print(f"  Sigmoid: {alpha_sigmoid:.4f}")
            print(f"  Diff:    {diff:.4f}")
            print(f"  Ratio:   {ratio:.2f}x" if ratio != float('inf') else "  Ratio:   >100x")
            
            if expected_behavior == "sigmoid_much_lower":
                assert alpha_sigmoid < alpha_linear / 2, \
                    f"{description}: sigmoid should be much lower"
            elif expected_behavior == "sigmoid_lower":
                assert alpha_sigmoid < alpha_linear, \
                    f"{description}: sigmoid should be lower"
            elif expected_behavior == "similar":
                assert abs(alpha_linear - alpha_sigmoid) < 0.2, \
                    f"{description}: should be similar"
        
        print("="*80 + "\n")
