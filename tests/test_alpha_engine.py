"""
Unit tests for the alpha engine module (sigmoid-based blending).

Tests cover:
- compute_alpha_data() with various feedback counts and K values
- compute_alpha_conf() with various confidence values and tau/beta parameters
- Edge cases: N=0, N=K, N=2K, N→∞, C_g=0, C_g=1
- Monotonic properties
- Different hyperparameter values
- Parameter validation

Validates Requirements: 1.1-1.6, 2.1-2.7, 6.1, 6.2
"""

import pytest
import numpy as np

from app.services.alpha_engine import (
    compute_alpha_data,
    compute_alpha_conf,
    compute_alpha_sigmoid
)


class TestComputeAlphaData:
    """Test the compute_alpha_data function (Requirement 1)."""
    
    def test_zero_feedback_returns_one(self):
        """Test that N=0 returns alpha_data=1.0 (Requirement 1.3)."""
        # With no feedback, should trust global head fully
        alpha_data = compute_alpha_data(feedback_count=0)
        assert alpha_data == 1.0
    
    def test_feedback_equals_k_returns_half(self):
        """Test that N=K returns alpha_data=0.5 (Requirement 1.4)."""
        # At K feedback samples, should give equal weight
        K = 50
        alpha_data = compute_alpha_data(feedback_count=K, K=K)
        assert alpha_data == 0.5
        
        # Test with different K values
        for K in [25, 50, 100]:
            alpha_data = compute_alpha_data(feedback_count=K, K=K)
            assert np.isclose(alpha_data, 0.5, atol=1e-10)
    
    def test_feedback_equals_2k_returns_one_third(self):
        """Test that N=2K returns alpha_data≈0.33 (Requirement 1.5)."""
        # At 2K feedback samples, should favor user head more
        K = 50
        alpha_data = compute_alpha_data(feedback_count=2*K, K=K)
        expected = 1.0 / (1.0 + 2.0)  # 1/3 ≈ 0.333...
        assert np.isclose(alpha_data, expected, atol=1e-10)
        assert np.isclose(alpha_data, 0.333333, atol=0.001)
    
    def test_high_feedback_approaches_zero(self):
        """Test that N→∞ approaches alpha_data→0.0 (Requirement 1.5)."""
        # With very high feedback, should trust user head almost exclusively
        K = 50
        
        # Test progressively higher feedback counts
        alpha_1000 = compute_alpha_data(feedback_count=1000, K=K)
        alpha_10000 = compute_alpha_data(feedback_count=10000, K=K)
        alpha_100000 = compute_alpha_data(feedback_count=100000, K=K)
        
        # Should approach zero
        assert alpha_1000 < 0.05
        assert alpha_10000 < 0.01
        assert alpha_100000 < 0.001
        
        # Should be monotonically decreasing
        assert alpha_1000 > alpha_10000 > alpha_100000
    
    def test_monotonic_decrease(self):
        """Test that alpha_data decreases as N increases (Requirement 1.6)."""
        # More feedback should always result in lower alpha_data
        K = 50
        
        feedback_counts = [0, 10, 25, 50, 75, 100, 150, 200, 500]
        alpha_values = [compute_alpha_data(n, K=K) for n in feedback_counts]
        
        # Verify monotonic decrease
        for i in range(len(alpha_values) - 1):
            assert alpha_values[i] > alpha_values[i + 1], \
                f"Not monotonic: alpha[{feedback_counts[i]}]={alpha_values[i]} " \
                f"<= alpha[{feedback_counts[i+1]}]={alpha_values[i+1]}"
    
    def test_different_k_values(self):
        """Test alpha_data with different K values (25, 50, 100)."""
        # K controls the decay rate
        # Smaller K → faster decay
        # Larger K → slower decay
        
        N = 50
        
        alpha_k25 = compute_alpha_data(feedback_count=N, K=25)
        alpha_k50 = compute_alpha_data(feedback_count=N, K=50)
        alpha_k100 = compute_alpha_data(feedback_count=N, K=100)
        
        # At N=50:
        # K=25: alpha = 1/(1+50/25) = 1/3 ≈ 0.33
        # K=50: alpha = 1/(1+50/50) = 1/2 = 0.5
        # K=100: alpha = 1/(1+50/100) = 1/1.5 ≈ 0.67
        
        assert np.isclose(alpha_k25, 0.333333, atol=0.001)
        assert np.isclose(alpha_k50, 0.5, atol=0.001)
        assert np.isclose(alpha_k100, 0.666667, atol=0.001)
        
        # Larger K should give higher alpha (slower decay)
        assert alpha_k25 < alpha_k50 < alpha_k100
    
    def test_k25_decay_rate(self):
        """Test alpha_data decay with K=25."""
        K = 25
        
        # Test key points
        assert compute_alpha_data(0, K=K) == 1.0
        assert compute_alpha_data(K, K=K) == 0.5
        assert np.isclose(compute_alpha_data(2*K, K=K), 0.333333, atol=0.001)
        
        # At N=25, should be at half-life
        alpha_25 = compute_alpha_data(25, K=K)
        assert alpha_25 == 0.5
    
    def test_k100_decay_rate(self):
        """Test alpha_data decay with K=100."""
        K = 100
        
        # Test key points
        assert compute_alpha_data(0, K=K) == 1.0
        assert compute_alpha_data(K, K=K) == 0.5
        assert np.isclose(compute_alpha_data(2*K, K=K), 0.333333, atol=0.001)
        
        # At N=100, should be at half-life
        alpha_100 = compute_alpha_data(100, K=K)
        assert alpha_100 == 0.5
    
    def test_default_k_value(self):
        """Test that default K=50 is used when not specified."""
        # Should use ALPHA_FEEDBACK_SCALE_K from config (default 50)
        alpha_with_default = compute_alpha_data(feedback_count=50)
        alpha_with_explicit = compute_alpha_data(feedback_count=50, K=50)
        
        assert alpha_with_default == alpha_with_explicit
        assert alpha_with_default == 0.5
    
    def test_edge_case_n_equals_zero(self):
        """Test edge case: N=0 (no feedback)."""
        for K in [25, 50, 100]:
            alpha_data = compute_alpha_data(feedback_count=0, K=K)
            assert alpha_data == 1.0
    
    def test_edge_case_very_large_n(self):
        """Test edge case: N→∞ (infinite feedback)."""
        K = 50
        
        # Test with extremely large N
        alpha_data = compute_alpha_data(feedback_count=1_000_000, K=K)
        
        # Should be very close to zero
        assert alpha_data < 0.0001
        assert alpha_data > 0.0  # But never exactly zero
    
    def test_formula_correctness(self):
        """Test that formula alpha_data = 1/(1+N/K) is implemented correctly."""
        K = 50
        
        test_cases = [
            (0, 1.0),
            (25, 2/3),
            (50, 0.5),
            (100, 1/3),
            (150, 0.25),
        ]
        
        for N, expected in test_cases:
            alpha_data = compute_alpha_data(feedback_count=N, K=K)
            assert np.isclose(alpha_data, expected, atol=1e-10), \
                f"N={N}: expected {expected}, got {alpha_data}"
    
    def test_bounds_always_valid(self):
        """Test that alpha_data is always in range (0, 1]."""
        K = 50
        
        # Test a wide range of feedback counts
        for N in [0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 10000]:
            alpha_data = compute_alpha_data(feedback_count=N, K=K)
            assert 0.0 < alpha_data <= 1.0, \
                f"alpha_data={alpha_data} out of range for N={N}"
    
    def test_invalid_k_raises_error(self):
        """Test that K <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="K must be > 0"):
            compute_alpha_data(feedback_count=50, K=0)
        
        with pytest.raises(ValueError, match="K must be > 0"):
            compute_alpha_data(feedback_count=50, K=-10)
    
    def test_numerical_stability(self):
        """Test numerical stability with edge values."""
        K = 50
        
        # Very small N
        alpha_small = compute_alpha_data(feedback_count=1, K=K)
        assert 0.0 < alpha_small <= 1.0
        assert np.isclose(alpha_small, 50/51, atol=1e-10)
        
        # Very large N
        alpha_large = compute_alpha_data(feedback_count=1_000_000, K=K)
        assert 0.0 < alpha_large <= 1.0
        assert alpha_large < 0.0001
    
    def test_comparison_across_k_values(self):
        """Test that smaller K leads to faster decay."""
        N = 100
        
        alpha_k25 = compute_alpha_data(feedback_count=N, K=25)
        alpha_k50 = compute_alpha_data(feedback_count=N, K=50)
        alpha_k100 = compute_alpha_data(feedback_count=N, K=100)
        
        # At same N, larger K should give higher alpha
        assert alpha_k25 < alpha_k50 < alpha_k100
        
        # Verify specific values
        # K=25: 1/(1+100/25) = 1/5 = 0.2
        # K=50: 1/(1+100/50) = 1/3 ≈ 0.333
        # K=100: 1/(1+100/100) = 1/2 = 0.5
        assert np.isclose(alpha_k25, 0.2, atol=1e-10)
        assert np.isclose(alpha_k50, 0.333333, atol=0.001)
        assert np.isclose(alpha_k100, 0.5, atol=1e-10)



class TestComputeAlphaConf:
    """Test the compute_alpha_conf function (Requirement 2)."""
    
    def test_confidence_equals_tau_returns_half(self):
        """Test that C_g=τ returns alpha_conf=0.5 (Requirement 2.5)."""
        # At the threshold, sigmoid should return exactly 0.5
        tau = 0.6
        alpha_conf = compute_alpha_conf(global_confidence=tau, tau=tau, beta=10)
        assert np.isclose(alpha_conf, 0.5, atol=1e-10)
        
        # Test with different tau values
        for tau in [0.5, 0.6, 0.7]:
            alpha_conf = compute_alpha_conf(global_confidence=tau, tau=tau, beta=10)
            assert np.isclose(alpha_conf, 0.5, atol=1e-10), \
                f"tau={tau}: expected 0.5, got {alpha_conf}"
    
    def test_confidence_below_tau_returns_less_than_half(self):
        """Test that C_g<τ returns alpha_conf<0.5 (Requirement 2.4)."""
        # Below threshold should favor user head
        tau = 0.6
        beta = 10
        
        test_cases = [0.3, 0.4, 0.5, 0.55, 0.59]
        
        for C_g in test_cases:
            alpha_conf = compute_alpha_conf(global_confidence=C_g, tau=tau, beta=beta)
            assert alpha_conf < 0.5, \
                f"C_g={C_g} < tau={tau}: expected alpha_conf < 0.5, got {alpha_conf}"
    
    def test_confidence_above_tau_returns_greater_than_half(self):
        """Test that C_g>τ returns alpha_conf>0.5 (Requirement 2.6)."""
        # Above threshold should favor global head
        tau = 0.6
        beta = 10
        
        test_cases = [0.61, 0.65, 0.7, 0.8, 0.9]
        
        for C_g in test_cases:
            alpha_conf = compute_alpha_conf(global_confidence=C_g, tau=tau, beta=beta)
            assert alpha_conf > 0.5, \
                f"C_g={C_g} > tau={tau}: expected alpha_conf > 0.5, got {alpha_conf}"
    
    def test_monotonic_increase(self):
        """Test that alpha_conf increases as C_g increases (Requirement 2.7)."""
        # Higher confidence should always result in higher alpha_conf
        tau = 0.6
        beta = 10
        
        confidence_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        alpha_values = [
            compute_alpha_conf(C_g, tau=tau, beta=beta) 
            for C_g in confidence_values
        ]
        
        # Verify monotonic increase
        for i in range(len(alpha_values) - 1):
            assert alpha_values[i] < alpha_values[i + 1], \
                f"Not monotonic: alpha[C_g={confidence_values[i]}]={alpha_values[i]} " \
                f">= alpha[C_g={confidence_values[i+1]}]={alpha_values[i+1]}"
    
    def test_different_tau_values(self):
        """Test alpha_conf with different tau values (0.5, 0.6, 0.7)."""
        # tau controls the center point of the sigmoid
        beta = 10
        C_g = 0.7
        
        alpha_tau50 = compute_alpha_conf(global_confidence=C_g, tau=0.5, beta=beta)
        alpha_tau60 = compute_alpha_conf(global_confidence=C_g, tau=0.6, beta=beta)
        alpha_tau70 = compute_alpha_conf(global_confidence=C_g, tau=0.7, beta=beta)
        
        # At C_g=0.7:
        # tau=0.5: C_g > tau by 0.2 → high alpha_conf
        # tau=0.6: C_g > tau by 0.1 → medium-high alpha_conf
        # tau=0.7: C_g = tau → alpha_conf = 0.5
        
        assert alpha_tau50 > alpha_tau60 > alpha_tau70
        assert np.isclose(alpha_tau70, 0.5, atol=1e-10)
        assert alpha_tau50 > 0.8  # Should be quite high
        assert alpha_tau60 > 0.7  # Should be high
    
    def test_tau_50(self):
        """Test alpha_conf with tau=0.5."""
        tau = 0.5
        beta = 10
        
        # At tau=0.5, sigmoid center is at C_g=0.5
        assert np.isclose(compute_alpha_conf(0.5, tau=tau, beta=beta), 0.5, atol=1e-10)
        
        # Below tau
        assert compute_alpha_conf(0.3, tau=tau, beta=beta) < 0.5
        assert compute_alpha_conf(0.4, tau=tau, beta=beta) < 0.5
        
        # Above tau
        assert compute_alpha_conf(0.6, tau=tau, beta=beta) > 0.5
        assert compute_alpha_conf(0.7, tau=tau, beta=beta) > 0.5
    
    def test_tau_60(self):
        """Test alpha_conf with tau=0.6 (default)."""
        tau = 0.6
        beta = 10
        
        # At tau=0.6, sigmoid center is at C_g=0.6
        assert np.isclose(compute_alpha_conf(0.6, tau=tau, beta=beta), 0.5, atol=1e-10)
        
        # Below tau
        assert compute_alpha_conf(0.4, tau=tau, beta=beta) < 0.5
        assert compute_alpha_conf(0.5, tau=tau, beta=beta) < 0.5
        
        # Above tau
        assert compute_alpha_conf(0.7, tau=tau, beta=beta) > 0.5
        assert compute_alpha_conf(0.8, tau=tau, beta=beta) > 0.5
    
    def test_tau_70(self):
        """Test alpha_conf with tau=0.7."""
        tau = 0.7
        beta = 10
        
        # At tau=0.7, sigmoid center is at C_g=0.7
        assert np.isclose(compute_alpha_conf(0.7, tau=tau, beta=beta), 0.5, atol=1e-10)
        
        # Below tau
        assert compute_alpha_conf(0.5, tau=tau, beta=beta) < 0.5
        assert compute_alpha_conf(0.6, tau=tau, beta=beta) < 0.5
        
        # Above tau
        assert compute_alpha_conf(0.8, tau=tau, beta=beta) > 0.5
        assert compute_alpha_conf(0.9, tau=tau, beta=beta) > 0.5
    
    def test_different_beta_values(self):
        """Test alpha_conf with different beta values (5, 10, 20)."""
        # beta controls the steepness of the sigmoid transition
        tau = 0.6
        C_g = 0.7
        
        alpha_beta5 = compute_alpha_conf(global_confidence=C_g, tau=tau, beta=5)
        alpha_beta10 = compute_alpha_conf(global_confidence=C_g, tau=tau, beta=10)
        alpha_beta20 = compute_alpha_conf(global_confidence=C_g, tau=tau, beta=20)
        
        # All should be > 0.5 since C_g > tau
        assert alpha_beta5 > 0.5
        assert alpha_beta10 > 0.5
        assert alpha_beta20 > 0.5
        
        # Higher beta → steeper transition → more extreme values
        # At C_g=0.7 (0.1 above tau), higher beta should give higher alpha_conf
        assert alpha_beta5 < alpha_beta10 < alpha_beta20
    
    def test_beta_5_sharpness(self):
        """Test alpha_conf with beta=5 (gentle transition)."""
        tau = 0.6
        beta = 5
        
        # Gentle transition means values change more gradually
        alpha_55 = compute_alpha_conf(0.55, tau=tau, beta=beta)
        alpha_60 = compute_alpha_conf(0.60, tau=tau, beta=beta)
        alpha_65 = compute_alpha_conf(0.65, tau=tau, beta=beta)
        
        # Should be monotonically increasing
        assert alpha_55 < alpha_60 < alpha_65
        
        # Center should still be 0.5
        assert np.isclose(alpha_60, 0.5, atol=1e-10)
        
        # Transitions should be gentler (closer to 0.5)
        assert 0.3 < alpha_55 < 0.5
        assert 0.5 < alpha_65 < 0.7
    
    def test_beta_10_sharpness(self):
        """Test alpha_conf with beta=10 (medium transition, default)."""
        tau = 0.6
        beta = 10
        
        # Medium transition
        alpha_55 = compute_alpha_conf(0.55, tau=tau, beta=beta)
        alpha_60 = compute_alpha_conf(0.60, tau=tau, beta=beta)
        alpha_65 = compute_alpha_conf(0.65, tau=tau, beta=beta)
        
        # Should be monotonically increasing
        assert alpha_55 < alpha_60 < alpha_65
        
        # Center should still be 0.5
        assert np.isclose(alpha_60, 0.5, atol=1e-10)
        
        # Transitions should be medium (more spread than beta=20)
        assert 0.2 < alpha_55 < 0.5
        assert 0.5 < alpha_65 < 0.8
    
    def test_beta_20_sharpness(self):
        """Test alpha_conf with beta=20 (sharp transition)."""
        tau = 0.6
        beta = 20
        
        # Sharp transition means values change more quickly
        alpha_55 = compute_alpha_conf(0.55, tau=tau, beta=beta)
        alpha_60 = compute_alpha_conf(0.60, tau=tau, beta=beta)
        alpha_65 = compute_alpha_conf(0.65, tau=tau, beta=beta)
        
        # Should be monotonically increasing
        assert alpha_55 < alpha_60 < alpha_65
        
        # Center should still be 0.5
        assert np.isclose(alpha_60, 0.5, atol=1e-10)
        
        # Transitions should be sharper (more extreme values)
        assert 0.1 < alpha_55 < 0.5
        assert 0.5 < alpha_65 < 0.9
    
    def test_edge_case_confidence_zero(self):
        """Test edge case: C_g=0 (minimum confidence)."""
        tau = 0.6
        
        for beta in [5, 10, 20]:
            alpha_conf = compute_alpha_conf(global_confidence=0.0, tau=tau, beta=beta)
            
            # Should be very low (close to 0) but not exactly 0
            assert 0.0 < alpha_conf < 0.1, \
                f"beta={beta}: expected alpha_conf near 0, got {alpha_conf}"
            
            # Higher beta should give lower alpha_conf at C_g=0
            if beta == 20:
                assert alpha_conf < 0.01
    
    def test_edge_case_confidence_one(self):
        """Test edge case: C_g=1 (maximum confidence)."""
        tau = 0.6
        
        for beta in [5, 10, 20]:
            alpha_conf = compute_alpha_conf(global_confidence=1.0, tau=tau, beta=beta)
            
            # Should be very high (close to 1) but not exactly 1
            # Note: beta=5 gives ~0.88, beta=10 gives ~0.98, beta=20 gives ~0.999
            assert 0.8 < alpha_conf < 1.0, \
                f"beta={beta}: expected alpha_conf near 1, got {alpha_conf}"
            
            # Higher beta should give higher alpha_conf at C_g=1
            if beta == 20:
                assert alpha_conf > 0.99
    
    def test_sigmoid_formula_correctness(self):
        """Test that sigmoid formula is implemented correctly (Requirement 2.1)."""
        tau = 0.6
        beta = 10
        C_g = 0.7
        
        # Manual calculation: alpha_conf = 1 / (1 + exp(-β(C_g - τ)))
        z = beta * (C_g - tau)  # 10 * (0.7 - 0.6) = 1.0
        expected = 1.0 / (1.0 + np.exp(-z))  # 1 / (1 + exp(-1)) ≈ 0.731
        
        alpha_conf = compute_alpha_conf(global_confidence=C_g, tau=tau, beta=beta)
        
        assert np.isclose(alpha_conf, expected, atol=1e-10)
        assert np.isclose(alpha_conf, 0.7310585786, atol=1e-6)
    
    def test_smooth_transition_no_discontinuities(self):
        """Test that sigmoid transition is smooth (Requirement 2.7)."""
        tau = 0.6
        beta = 10
        
        # Test very small steps around tau
        confidence_values = np.linspace(0.55, 0.65, 100)
        alpha_values = [
            compute_alpha_conf(C_g, tau=tau, beta=beta) 
            for C_g in confidence_values
        ]
        
        # Check that differences are small (no jumps)
        for i in range(len(alpha_values) - 1):
            diff = abs(alpha_values[i + 1] - alpha_values[i])
            assert diff < 0.05, \
                f"Large jump detected: {diff} between C_g={confidence_values[i]} and {confidence_values[i+1]}"
        
        # Verify monotonic increase
        for i in range(len(alpha_values) - 1):
            assert alpha_values[i] < alpha_values[i + 1]
    
    def test_bounds_always_valid(self):
        """Test that alpha_conf is always in range (0, 1)."""
        tau = 0.6
        beta = 10
        
        # Test a wide range of confidence values
        for C_g in np.linspace(0.0, 1.0, 50):
            alpha_conf = compute_alpha_conf(global_confidence=C_g, tau=tau, beta=beta)
            assert 0.0 < alpha_conf < 1.0, \
                f"alpha_conf={alpha_conf} out of range for C_g={C_g}"
    
    def test_default_tau_and_beta_values(self):
        """Test that default tau=0.6 and beta=10 are used when not specified."""
        C_g = 0.7
        
        # Should use defaults from config
        alpha_with_defaults = compute_alpha_conf(global_confidence=C_g)
        alpha_with_explicit = compute_alpha_conf(global_confidence=C_g, tau=0.6, beta=10)
        
        assert np.isclose(alpha_with_defaults, alpha_with_explicit, atol=1e-10)
    
    def test_invalid_tau_raises_error(self):
        """Test that tau not in (0, 1) raises ValueError."""
        # tau = 0 (boundary)
        with pytest.raises(ValueError, match="tau must be in \\(0, 1\\)"):
            compute_alpha_conf(global_confidence=0.5, tau=0.0, beta=10)
        
        # tau = 1 (boundary)
        with pytest.raises(ValueError, match="tau must be in \\(0, 1\\)"):
            compute_alpha_conf(global_confidence=0.5, tau=1.0, beta=10)
        
        # tau < 0
        with pytest.raises(ValueError, match="tau must be in \\(0, 1\\)"):
            compute_alpha_conf(global_confidence=0.5, tau=-0.1, beta=10)
        
        # tau > 1
        with pytest.raises(ValueError, match="tau must be in \\(0, 1\\)"):
            compute_alpha_conf(global_confidence=0.5, tau=1.5, beta=10)
    
    def test_invalid_beta_raises_error(self):
        """Test that beta <= 0 raises ValueError."""
        # beta = 0
        with pytest.raises(ValueError, match="beta must be > 0"):
            compute_alpha_conf(global_confidence=0.5, tau=0.6, beta=0)
        
        # beta < 0
        with pytest.raises(ValueError, match="beta must be > 0"):
            compute_alpha_conf(global_confidence=0.5, tau=0.6, beta=-5)
    
    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme beta and confidence values."""
        # Very high beta with extreme confidence
        tau = 0.6
        beta = 100
        
        # Should not overflow or underflow
        alpha_low = compute_alpha_conf(global_confidence=0.0, tau=tau, beta=beta)
        alpha_high = compute_alpha_conf(global_confidence=1.0, tau=tau, beta=beta)
        
        assert 0.0 < alpha_low < 1.0
        assert 0.0 < alpha_high <= 1.0  # Can be exactly 1.0 due to floating point
        assert alpha_low < 0.001  # Very close to 0
        assert alpha_high > 0.999  # Very close to 1
    
    def test_symmetry_around_tau(self):
        """Test that sigmoid is symmetric around tau."""
        tau = 0.6
        beta = 10
        delta = 0.1
        
        # Points equidistant from tau
        alpha_below = compute_alpha_conf(global_confidence=tau - delta, tau=tau, beta=beta)
        alpha_above = compute_alpha_conf(global_confidence=tau + delta, tau=tau, beta=beta)
        
        # Should be symmetric: alpha_below + alpha_above ≈ 1.0
        assert np.isclose(alpha_below + alpha_above, 1.0, atol=1e-10)
        
        # Distance from 0.5 should be equal
        assert np.isclose(0.5 - alpha_below, alpha_above - 0.5, atol=1e-10)
    
    def test_comparison_across_beta_values_at_threshold(self):
        """Test that all beta values give 0.5 at tau."""
        tau = 0.6
        
        for beta in [5, 10, 20, 50, 100]:
            alpha_conf = compute_alpha_conf(global_confidence=tau, tau=tau, beta=beta)
            assert np.isclose(alpha_conf, 0.5, atol=1e-10), \
                f"beta={beta}: expected 0.5 at tau, got {alpha_conf}"
    
    def test_beta_effect_on_transition_width(self):
        """Test that higher beta creates narrower transition region."""
        tau = 0.6
        
        # Measure width of transition region (0.25 to 0.75)
        def find_confidence_for_alpha(target_alpha, beta):
            # Binary search to find C_g that gives target alpha
            low, high = 0.0, 1.0
            for _ in range(50):  # Sufficient iterations for convergence
                mid = (low + high) / 2
                alpha = compute_alpha_conf(mid, tau=tau, beta=beta)
                if alpha < target_alpha:
                    low = mid
                else:
                    high = mid
            return (low + high) / 2
        
        # Find confidence values that give alpha=0.25 and alpha=0.75
        for beta in [5, 10, 20]:
            C_g_25 = find_confidence_for_alpha(0.25, beta)
            C_g_75 = find_confidence_for_alpha(0.75, beta)
            width = C_g_75 - C_g_25
            
            # Higher beta should give narrower width
            if beta == 5:
                width_beta5 = width
            elif beta == 10:
                width_beta10 = width
            elif beta == 20:
                width_beta20 = width
        
        # Verify narrowing with higher beta
        assert width_beta5 > width_beta10 > width_beta20


class TestComputeAlphaSigmoid:
    """Test the compute_alpha_sigmoid function (Requirement 3)."""
    
    def test_multiplicative_property(self):
        """Test that alpha = alpha_data × alpha_conf (Requirement 3.1)."""
        # Test various combinations to verify multiplicative formula
        test_cases = [
            # (feedback_count, global_confidence, K, tau, beta)
            (0, 0.8, 50, 0.6, 10),
            (25, 0.7, 50, 0.6, 10),
            (50, 0.6, 50, 0.6, 10),
            (100, 0.5, 50, 0.6, 10),
            (200, 0.9, 50, 0.6, 10),
        ]
        
        for N, C_g, K, tau, beta in test_cases:
            alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
                global_confidence=C_g,
                feedback_count=N,
                K=K,
                tau=tau,
                beta=beta
            )
            
            # Verify multiplicative property
            expected_alpha = alpha_data * alpha_conf
            assert np.isclose(alpha_final, expected_alpha, atol=1e-10), \
                f"N={N}, C_g={C_g}: expected {expected_alpha}, got {alpha_final}"
    
    def test_bounds_always_valid(self):
        """Test that alpha ∈ [0, 1] for all valid inputs (Requirement 3.4)."""
        # Test a wide range of inputs
        feedback_counts = [0, 10, 25, 50, 100, 200, 500, 1000]
        confidence_values = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        
        for N in feedback_counts:
            for C_g in confidence_values:
                alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
                    global_confidence=C_g,
                    feedback_count=N
                )
                
                # Verify all three components are in valid range
                assert 0.0 < alpha_data <= 1.0, \
                    f"alpha_data={alpha_data} out of range for N={N}, C_g={C_g}"
                assert 0.0 < alpha_conf < 1.0, \
                    f"alpha_conf={alpha_conf} out of range for N={N}, C_g={C_g}"
                assert 0.0 < alpha_final < 1.0, \
                    f"alpha_final={alpha_final} out of range for N={N}, C_g={C_g}"
    
    def test_both_components_high_gives_high_alpha(self):
        """Test that high alpha_data AND high alpha_conf → high alpha (Requirement 3.3)."""
        # Scenario: New user (N=0) with high confidence (C_g=0.9)
        # Expected: alpha_data ≈ 1.0, alpha_conf ≈ 0.95, alpha ≈ 0.95
        
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=0.9,
            feedback_count=0,
            K=50,
            tau=0.6,
            beta=10
        )
        
        # Both components should be high
        assert alpha_data > 0.9, f"Expected high alpha_data, got {alpha_data}"
        assert alpha_conf > 0.9, f"Expected high alpha_conf, got {alpha_conf}"
        
        # Final alpha should be high
        assert alpha_final > 0.9, f"Expected high alpha, got {alpha_final}"
        
        # Verify multiplicative property
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_both_components_low_gives_low_alpha(self):
        """Test that low alpha_data AND low alpha_conf → low alpha (Requirement 3.2)."""
        # Scenario: Lots of feedback (N=200) with low confidence (C_g=0.3)
        # Expected: alpha_data ≈ 0.2, alpha_conf ≈ 0.05, alpha ≈ 0.01
        
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=0.3,
            feedback_count=200,
            K=50,
            tau=0.6,
            beta=10
        )
        
        # Both components should be low
        assert alpha_data < 0.3, f"Expected low alpha_data, got {alpha_data}"
        assert alpha_conf < 0.3, f"Expected low alpha_conf, got {alpha_conf}"
        
        # Final alpha should be very low
        assert alpha_final < 0.1, f"Expected low alpha, got {alpha_final}"
        
        # Verify multiplicative property
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_low_alpha_data_high_alpha_conf_gives_low_alpha(self):
        """Test that low alpha_data OR high alpha_conf → low alpha (Requirement 3.2)."""
        # Scenario: Lots of feedback (N=200) with high confidence (C_g=0.9)
        # Expected: alpha_data ≈ 0.2, alpha_conf ≈ 0.95, alpha ≈ 0.19
        
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=0.9,
            feedback_count=200,
            K=50,
            tau=0.6,
            beta=10
        )
        
        # alpha_data should be low, alpha_conf should be high
        assert alpha_data < 0.3, f"Expected low alpha_data, got {alpha_data}"
        assert alpha_conf > 0.9, f"Expected high alpha_conf, got {alpha_conf}"
        
        # Final alpha should be low (limited by low alpha_data)
        assert alpha_final < 0.3, f"Expected low alpha, got {alpha_final}"
        
        # Verify multiplicative property
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_high_alpha_data_low_alpha_conf_gives_low_alpha(self):
        """Test that high alpha_data OR low alpha_conf → low alpha (Requirement 3.2)."""
        # Scenario: No feedback (N=0) with low confidence (C_g=0.3)
        # Expected: alpha_data = 1.0, alpha_conf ≈ 0.05, alpha ≈ 0.05
        
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=0.3,
            feedback_count=0,
            K=50,
            tau=0.6,
            beta=10
        )
        
        # alpha_data should be high, alpha_conf should be low
        assert alpha_data > 0.9, f"Expected high alpha_data, got {alpha_data}"
        assert alpha_conf < 0.3, f"Expected low alpha_conf, got {alpha_conf}"
        
        # Final alpha should be low (limited by low alpha_conf)
        assert alpha_final < 0.3, f"Expected low alpha, got {alpha_final}"
        
        # Verify multiplicative property
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_edge_case_zero_feedback_low_confidence(self):
        """Test edge case: N=0, C_g=0.0 (new user, minimum confidence)."""
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=0.0,
            feedback_count=0,
            K=50,
            tau=0.6,
            beta=10
        )
        
        # alpha_data should be 1.0 (no feedback)
        assert alpha_data == 1.0
        
        # alpha_conf should be very low (C_g << tau)
        assert alpha_conf < 0.1
        
        # Final alpha should be low (limited by alpha_conf)
        assert alpha_final < 0.1
        
        # Verify multiplicative property
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_edge_case_zero_feedback_high_confidence(self):
        """Test edge case: N=0, C_g=1.0 (new user, maximum confidence)."""
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=1.0,
            feedback_count=0,
            K=50,
            tau=0.6,
            beta=10
        )
        
        # alpha_data should be 1.0 (no feedback)
        assert alpha_data == 1.0
        
        # alpha_conf should be very high (C_g >> tau)
        assert alpha_conf > 0.9
        
        # Final alpha should be high (both components high)
        assert alpha_final > 0.9
        
        # Verify multiplicative property
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_edge_case_high_feedback_low_confidence(self):
        """Test edge case: N=1000, C_g=0.0 (lots of feedback, minimum confidence)."""
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=0.0,
            feedback_count=1000,
            K=50,
            tau=0.6,
            beta=10
        )
        
        # alpha_data should be very low (lots of feedback)
        assert alpha_data < 0.1
        
        # alpha_conf should be very low (C_g << tau)
        assert alpha_conf < 0.1
        
        # Final alpha should be very low (both components low)
        assert alpha_final < 0.01
        
        # Verify multiplicative property
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_edge_case_high_feedback_high_confidence(self):
        """Test edge case: N=1000, C_g=1.0 (lots of feedback, maximum confidence)."""
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=1.0,
            feedback_count=1000,
            K=50,
            tau=0.6,
            beta=10
        )
        
        # alpha_data should be very low (lots of feedback)
        assert alpha_data < 0.1
        
        # alpha_conf should be very high (C_g >> tau)
        assert alpha_conf > 0.9
        
        # Final alpha should be low (limited by low alpha_data)
        assert alpha_final < 0.1
        
        # Verify multiplicative property
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_edge_case_medium_feedback_threshold_confidence(self):
        """Test edge case: N=K, C_g=τ (both at threshold)."""
        K = 50
        tau = 0.6
        
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=tau,
            feedback_count=K,
            K=K,
            tau=tau,
            beta=10
        )
        
        # Both components should be exactly 0.5
        assert np.isclose(alpha_data, 0.5, atol=1e-10)
        assert np.isclose(alpha_conf, 0.5, atol=1e-10)
        
        # Final alpha should be 0.25 (0.5 × 0.5)
        assert np.isclose(alpha_final, 0.25, atol=1e-10)
        
        # Verify multiplicative property
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_returns_all_three_components(self):
        """Test that function returns tuple of (alpha_data, alpha_conf, alpha_final)."""
        result = compute_alpha_sigmoid(
            global_confidence=0.7,
            feedback_count=50
        )
        
        # Should return a tuple of 3 elements
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        alpha_data, alpha_conf, alpha_final = result
        
        # All should be floats
        assert isinstance(alpha_data, (float, np.floating))
        assert isinstance(alpha_conf, (float, np.floating))
        assert isinstance(alpha_final, (float, np.floating))
        
        # All should be in valid range
        assert 0.0 < alpha_data <= 1.0
        assert 0.0 < alpha_conf < 1.0
        assert 0.0 < alpha_final < 1.0
    
    def test_uses_default_hyperparameters(self):
        """Test that default K, tau, beta are used when not specified."""
        # Call without hyperparameters
        alpha_data1, alpha_conf1, alpha_final1 = compute_alpha_sigmoid(
            global_confidence=0.7,
            feedback_count=50
        )
        
        # Call with explicit defaults (K=50, tau=0.6, beta=10)
        alpha_data2, alpha_conf2, alpha_final2 = compute_alpha_sigmoid(
            global_confidence=0.7,
            feedback_count=50,
            K=50,
            tau=0.6,
            beta=10
        )
        
        # Should be identical
        assert np.isclose(alpha_data1, alpha_data2, atol=1e-10)
        assert np.isclose(alpha_conf1, alpha_conf2, atol=1e-10)
        assert np.isclose(alpha_final1, alpha_final2, atol=1e-10)
    
    def test_different_hyperparameters(self):
        """Test alpha combination with different hyperparameter values."""
        # Test with K=25, tau=0.5, beta=5
        alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
            global_confidence=0.7,
            feedback_count=25,
            K=25,
            tau=0.5,
            beta=5
        )
        
        # At N=K=25, alpha_data should be 0.5
        assert np.isclose(alpha_data, 0.5, atol=1e-10)
        
        # At C_g=0.7 > tau=0.5, alpha_conf should be > 0.5
        assert alpha_conf > 0.5
        
        # Final alpha should be product
        assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_and_logic_both_must_be_high(self):
        """Test AND logic: both components must be high for high alpha."""
        # Test all four combinations of high/low
        test_cases = [
            # (N, C_g, expected_alpha_range)
            (0, 0.9, "high"),      # high data, high conf → high alpha
            (0, 0.3, "low"),       # high data, low conf → low alpha
            (200, 0.9, "low"),     # low data, high conf → low alpha
            (200, 0.3, "low"),     # low data, low conf → low alpha
        ]
        
        for N, C_g, expected_range in test_cases:
            alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
                global_confidence=C_g,
                feedback_count=N,
                K=50,
                tau=0.6,
                beta=10
            )
            
            if expected_range == "high":
                assert alpha_final > 0.8, \
                    f"N={N}, C_g={C_g}: expected high alpha, got {alpha_final}"
            else:  # "low"
                assert alpha_final < 0.3, \
                    f"N={N}, C_g={C_g}: expected low alpha, got {alpha_final}"
    
    def test_or_logic_either_low_gives_low(self):
        """Test OR logic: if either component is low, alpha is low."""
        # Test cases where at least one component is low
        test_cases = [
            # (N, C_g, description)
            (0, 0.3, "high data, low conf"),
            (200, 0.9, "low data, high conf"),
            (200, 0.3, "low data, low conf"),
        ]
        
        for N, C_g, description in test_cases:
            alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
                global_confidence=C_g,
                feedback_count=N,
                K=50,
                tau=0.6,
                beta=10
            )
            
            # If either component is low, final alpha should be low
            if alpha_data < 0.3 or alpha_conf < 0.3:
                assert alpha_final < 0.3, \
                    f"{description}: expected low alpha, got {alpha_final}"
    
    def test_no_explicit_clamping_needed(self):
        """Test that alpha ∈ [0, 1] naturally without explicit clamping (Requirement 3.4)."""
        # Test extreme cases that might overflow/underflow
        extreme_cases = [
            (0, 0.0),      # Minimum both
            (0, 1.0),      # Maximum conf, minimum feedback
            (10000, 0.0),  # Maximum feedback, minimum conf
            (10000, 1.0),  # Maximum both
        ]
        
        for N, C_g in extreme_cases:
            alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
                global_confidence=C_g,
                feedback_count=N,
                K=50,
                tau=0.6,
                beta=10
            )
            
            # All values should be in valid range without explicit clamping
            assert 0.0 < alpha_data <= 1.0
            assert 0.0 < alpha_conf < 1.0
            assert 0.0 < alpha_final < 1.0
            
            # Verify multiplicative property holds
            assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10)
    
    def test_realistic_scenarios(self):
        """Test realistic user scenarios."""
        scenarios = [
            # (description, N, C_g, expected_behavior)
            ("New user, confident prediction", 0, 0.85, "high"),
            ("New user, uncertain prediction", 0, 0.45, "low"),
            ("Some feedback, confident prediction", 25, 0.85, "medium-high"),
            ("Some feedback, uncertain prediction", 25, 0.45, "low"),
            ("Lots of feedback, confident prediction", 100, 0.85, "low-medium"),
            ("Lots of feedback, uncertain prediction", 100, 0.45, "low"),
        ]
        
        for description, N, C_g, expected_behavior in scenarios:
            alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
                global_confidence=C_g,
                feedback_count=N,
                K=50,
                tau=0.6,
                beta=10
            )
            
            # Verify multiplicative property
            assert np.isclose(alpha_final, alpha_data * alpha_conf, atol=1e-10), \
                f"{description}: multiplicative property failed"
            
            # Verify bounds
            assert 0.0 < alpha_final < 1.0, \
                f"{description}: alpha out of bounds"
