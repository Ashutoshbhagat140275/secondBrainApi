"""
Property-based tests for the alpha engine module.

Uses Hypothesis to generate random inputs and verify mathematical properties
that should hold universally across the input space.

Tests cover:
- Property 1: Alpha Data Monotonicity - alpha_data(N) decreases as N increases
- Property 2: Alpha Conf Monotonicity - alpha_conf(C_g) increases as C_g increases
- Property 3: Alpha Bounds - 0 < alpha < 1 for all valid inputs
- Property 4: Smoothness - Alpha changes smoothly with small perturbations

**Validates: Requirements 6.4, 6.5, 6.6**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings

from app.services.alpha_engine import (
    compute_alpha_data,
    compute_alpha_conf,
    compute_alpha_sigmoid
)


# ============================================================================
# Property 1: Alpha Data Monotonicity
# ============================================================================

@given(
    n1=st.integers(min_value=0, max_value=10000),
    n2=st.integers(min_value=0, max_value=10000),
    K=st.floats(min_value=1.0, max_value=200.0)
)
@settings(max_examples=200)
def test_property_alpha_data_monotonicity(n1, n2, K):
    """
    Property 1: Alpha Data Monotonicity
    
    For any K > 0, alpha_data(N) is monotonically decreasing.
    If N1 < N2, then alpha_data(N1) > alpha_data(N2).
    
    **Validates: Requirements 6.4**
    """
    # Ensure n1 < n2 for monotonicity test
    if n1 >= n2:
        n1, n2 = n2, n1
    
    # Skip if they're equal (no monotonicity to test)
    assume(n1 < n2)
    
    # Compute alpha_data for both feedback counts
    alpha_data_n1 = compute_alpha_data(feedback_count=n1, K=K)
    alpha_data_n2 = compute_alpha_data(feedback_count=n2, K=K)
    
    # Verify monotonic decrease: more feedback → lower alpha_data
    assert alpha_data_n1 > alpha_data_n2, (
        f"Monotonicity violated: alpha_data({n1})={alpha_data_n1:.6f} "
        f"<= alpha_data({n2})={alpha_data_n2:.6f} with K={K:.2f}"
    )


@given(
    n=st.integers(min_value=0, max_value=10000),
    K=st.floats(min_value=1.0, max_value=200.0)
)
@settings(max_examples=200)
def test_property_alpha_data_bounds(n, K):
    """
    Property: Alpha Data Bounds
    
    For any N >= 0 and K > 0, alpha_data should be in range (0, 1].
    
    **Validates: Requirements 6.4**
    """
    alpha_data = compute_alpha_data(feedback_count=n, K=K)
    
    # Verify bounds
    assert 0.0 < alpha_data <= 1.0, (
        f"alpha_data out of bounds: {alpha_data} for N={n}, K={K:.2f}"
    )


# ============================================================================
# Property 2: Alpha Conf Monotonicity
# ============================================================================

@given(
    c_g1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    c_g2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    tau=st.floats(min_value=0.01, max_value=0.99),
    beta=st.floats(min_value=1.0, max_value=50.0)
)
@settings(max_examples=200)
def test_property_alpha_conf_monotonicity(c_g1, c_g2, tau, beta):
    """
    Property 2: Alpha Conf Monotonicity
    
    For any tau in (0, 1) and beta > 0, alpha_conf(C_g) is monotonically increasing.
    If C_g1 < C_g2, then alpha_conf(C_g1) < alpha_conf(C_g2).
    
    **Validates: Requirements 6.5**
    """
    # Ensure c_g1 < c_g2 for monotonicity test
    if c_g1 >= c_g2:
        c_g1, c_g2 = c_g2, c_g1
    
    # Skip if they're equal or too close (floating point precision)
    assume(c_g2 - c_g1 > 1e-6)
    
    # Compute alpha_conf for both confidence values
    alpha_conf_c1 = compute_alpha_conf(global_confidence=c_g1, tau=tau, beta=beta)
    alpha_conf_c2 = compute_alpha_conf(global_confidence=c_g2, tau=tau, beta=beta)
    
    # Verify monotonic increase: higher confidence → higher alpha_conf
    # Use <= to handle floating point precision at extremes
    assert alpha_conf_c1 <= alpha_conf_c2, (
        f"Monotonicity violated: alpha_conf({c_g1:.4f})={alpha_conf_c1:.6f} "
        f"> alpha_conf({c_g2:.4f})={alpha_conf_c2:.6f} with tau={tau:.2f}, beta={beta:.2f}"
    )


@given(
    c_g=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    tau=st.floats(min_value=0.01, max_value=0.99),
    beta=st.floats(min_value=1.0, max_value=50.0)
)
@settings(max_examples=200)
def test_property_alpha_conf_bounds(c_g, tau, beta):
    """
    Property: Alpha Conf Bounds
    
    For any C_g in [0, 1], tau in (0, 1), and beta > 0,
    alpha_conf should be in range (0, 1].
    
    Note: Due to floating point precision, sigmoid can reach exactly 1.0
    at extreme values (C_g=1.0 with high beta).
    
    **Validates: Requirements 6.5**
    """
    alpha_conf = compute_alpha_conf(global_confidence=c_g, tau=tau, beta=beta)
    
    # Verify bounds (sigmoid approaches but rarely reaches exactly 0 or 1)
    assert 0.0 < alpha_conf <= 1.0, (
        f"alpha_conf out of bounds: {alpha_conf} for C_g={c_g:.4f}, tau={tau:.2f}, beta={beta:.2f}"
    )


# ============================================================================
# Property 3: Alpha Bounds
# ============================================================================

@given(
    n=st.integers(min_value=0, max_value=10000),
    c_g=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    K=st.floats(min_value=1.0, max_value=200.0),
    tau=st.floats(min_value=0.01, max_value=0.99),
    beta=st.floats(min_value=1.0, max_value=50.0)
)
@settings(max_examples=300)
def test_property_alpha_bounds(n, c_g, K, tau, beta):
    """
    Property 3: Alpha Bounds
    
    For any valid inputs (N >= 0, C_g in [0,1], K > 0, tau in (0,1), beta > 0),
    the final alpha should be in range (0, 1].
    
    This tests the multiplicative combination: alpha = alpha_data × alpha_conf
    
    Note: Due to floating point precision, alpha_conf can reach exactly 1.0
    at extreme values, making alpha_final potentially equal to alpha_data.
    
    **Validates: Requirements 6.6**
    """
    alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
        global_confidence=c_g,
        feedback_count=n,
        K=K,
        tau=tau,
        beta=beta
    )
    
    # Verify all three components are in valid range
    assert 0.0 < alpha_data <= 1.0, (
        f"alpha_data out of bounds: {alpha_data}"
    )
    assert 0.0 < alpha_conf <= 1.0, (
        f"alpha_conf out of bounds: {alpha_conf}"
    )
    assert 0.0 < alpha_final <= 1.0, (
        f"alpha_final out of bounds: {alpha_final} for N={n}, C_g={c_g:.4f}"
    )
    
    # Verify multiplicative property
    expected_alpha = alpha_data * alpha_conf
    assert np.isclose(alpha_final, expected_alpha, atol=1e-10), (
        f"Multiplicative property violated: {alpha_final} != {alpha_data} × {alpha_conf}"
    )


# ============================================================================
# Property 4: Smoothness (No Discontinuities)
# ============================================================================

@given(
    n=st.integers(min_value=0, max_value=5000),
    K=st.floats(min_value=10.0, max_value=200.0)
)
@settings(max_examples=200)
def test_property_alpha_data_smoothness(n, K):
    """
    Property 4a: Alpha Data Smoothness
    
    Alpha_data changes smoothly with small perturbations in N.
    For small delta_N, the change in alpha_data should be small.
    
    Tests that there are no discontinuities or jumps in the alpha_data function.
    
    **Validates: Requirements 6.6**
    """
    # Compute alpha_data at N and N+1 (small perturbation)
    alpha_data_n = compute_alpha_data(feedback_count=n, K=K)
    alpha_data_n_plus_1 = compute_alpha_data(feedback_count=n + 1, K=K)
    
    # Compute the change
    delta_alpha = abs(alpha_data_n - alpha_data_n_plus_1)
    
    # The change should be small (no jumps)
    # For the formula 1/(1+N/K), the derivative is -K/(K+N)^2
    # Maximum change for delta_N=1 is approximately K/(K+N)^2
    max_expected_change = K / ((K + n) ** 2) + 0.001  # Small tolerance
    
    assert delta_alpha <= max_expected_change, (
        f"Discontinuity detected: alpha_data changed by {delta_alpha:.6f} "
        f"from N={n} to N={n+1} (expected <= {max_expected_change:.6f})"
    )


@given(
    c_g=st.floats(min_value=0.0, max_value=0.999),  # Leave room for +0.001
    tau=st.floats(min_value=0.01, max_value=0.99),
    beta=st.floats(min_value=1.0, max_value=50.0)
)
@settings(max_examples=200)
def test_property_alpha_conf_smoothness(c_g, tau, beta):
    """
    Property 4b: Alpha Conf Smoothness
    
    Alpha_conf changes smoothly with small perturbations in C_g.
    For small delta_C_g, the change in alpha_conf should be small.
    
    Tests that there are no discontinuities or jumps in the alpha_conf function.
    
    **Validates: Requirements 6.6**
    """
    # Small perturbation in confidence
    delta_c_g = 0.001
    c_g_perturbed = min(c_g + delta_c_g, 1.0)
    
    # Compute alpha_conf at both points
    alpha_conf_1 = compute_alpha_conf(global_confidence=c_g, tau=tau, beta=beta)
    alpha_conf_2 = compute_alpha_conf(global_confidence=c_g_perturbed, tau=tau, beta=beta)
    
    # Compute the change
    delta_alpha = abs(alpha_conf_2 - alpha_conf_1)
    
    # The change should be small (no jumps)
    # For sigmoid, the derivative is β·exp(-β(C_g - τ)) / (1 + exp(-β(C_g - τ)))^2
    # Maximum derivative is β/4 (at C_g = τ)
    # So for delta_C_g = 0.001, max change is approximately β/4 * 0.001
    max_expected_change = (beta / 4.0) * delta_c_g + 0.001  # Small tolerance
    
    assert delta_alpha <= max_expected_change, (
        f"Discontinuity detected: alpha_conf changed by {delta_alpha:.6f} "
        f"from C_g={c_g:.4f} to C_g={c_g_perturbed:.4f} "
        f"(expected <= {max_expected_change:.6f})"
    )


@given(
    n=st.integers(min_value=0, max_value=5000),
    c_g=st.floats(min_value=0.0, max_value=0.999),
    K=st.floats(min_value=10.0, max_value=200.0),
    tau=st.floats(min_value=0.01, max_value=0.99),
    beta=st.floats(min_value=1.0, max_value=50.0)
)
@settings(max_examples=200)
def test_property_alpha_final_smoothness(n, c_g, K, tau, beta):
    """
    Property 4c: Final Alpha Smoothness
    
    Final alpha changes smoothly with small perturbations in both N and C_g.
    Tests the combined smoothness of the multiplicative formula.
    
    **Validates: Requirements 6.6**
    """
    # Compute alpha at base point
    _, _, alpha_base = compute_alpha_sigmoid(
        global_confidence=c_g,
        feedback_count=n,
        K=K,
        tau=tau,
        beta=beta
    )
    
    # Perturb N by 1
    _, _, alpha_n_perturbed = compute_alpha_sigmoid(
        global_confidence=c_g,
        feedback_count=n + 1,
        K=K,
        tau=tau,
        beta=beta
    )
    
    # Perturb C_g by 0.001
    c_g_perturbed = min(c_g + 0.001, 1.0)
    _, _, alpha_c_perturbed = compute_alpha_sigmoid(
        global_confidence=c_g_perturbed,
        feedback_count=n,
        K=K,
        tau=tau,
        beta=beta
    )
    
    # Changes should be small
    delta_alpha_n = abs(alpha_n_perturbed - alpha_base)
    delta_alpha_c = abs(alpha_c_perturbed - alpha_base)
    
    # Since alpha = alpha_data × alpha_conf, and both are smooth,
    # the product should also be smooth
    # We use a generous bound since it's a product of two functions
    max_expected_change = 0.1  # Generous bound for smoothness
    
    assert delta_alpha_n <= max_expected_change, (
        f"Discontinuity in N: alpha changed by {delta_alpha_n:.6f} "
        f"from N={n} to N={n+1}"
    )
    
    assert delta_alpha_c <= max_expected_change, (
        f"Discontinuity in C_g: alpha changed by {delta_alpha_c:.6f} "
        f"from C_g={c_g:.4f} to C_g={c_g_perturbed:.4f}"
    )


# ============================================================================
# Additional Property Tests
# ============================================================================

@given(
    n=st.integers(min_value=0, max_value=10000),
    c_g=st.floats(min_value=0.0, max_value=1.0),
    K=st.floats(min_value=1.0, max_value=200.0),
    tau=st.floats(min_value=0.01, max_value=0.99),
    beta=st.floats(min_value=1.0, max_value=50.0)
)
@settings(max_examples=200)
def test_property_multiplicative_combination(n, c_g, K, tau, beta):
    """
    Property: Multiplicative Combination
    
    The final alpha should always equal alpha_data × alpha_conf.
    This is a fundamental property of the sigmoid-based formula.
    
    **Validates: Requirements 6.4, 6.5, 6.6**
    """
    alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
        global_confidence=c_g,
        feedback_count=n,
        K=K,
        tau=tau,
        beta=beta
    )
    
    expected_alpha = alpha_data * alpha_conf
    
    assert np.isclose(alpha_final, expected_alpha, atol=1e-10), (
        f"Multiplicative property violated: {alpha_final:.10f} != "
        f"{alpha_data:.10f} × {alpha_conf:.10f} = {expected_alpha:.10f}"
    )


@given(
    n=st.integers(min_value=0, max_value=10000),
    c_g=st.floats(min_value=0.0, max_value=1.0),
    K=st.floats(min_value=1.0, max_value=200.0),
    tau=st.floats(min_value=0.01, max_value=0.99),
    beta=st.floats(min_value=1.0, max_value=50.0)
)
@settings(max_examples=200)
def test_property_and_logic(n, c_g, K, tau, beta):
    """
    Property: AND Logic
    
    If either alpha_data OR alpha_conf is low (< 0.3), then final alpha should be low.
    This tests the AND logic of the multiplicative combination.
    
    **Validates: Requirements 6.6**
    """
    alpha_data, alpha_conf, alpha_final = compute_alpha_sigmoid(
        global_confidence=c_g,
        feedback_count=n,
        K=K,
        tau=tau,
        beta=beta
    )
    
    # If either component is low, final alpha should be low
    if alpha_data < 0.3 or alpha_conf < 0.3:
        # Final alpha should be less than max(alpha_data, alpha_conf)
        # In fact, it should be less than min(alpha_data, alpha_conf)
        assert alpha_final < max(alpha_data, alpha_conf), (
            f"AND logic violated: alpha_final={alpha_final:.4f} should be < "
            f"max({alpha_data:.4f}, {alpha_conf:.4f}) when one is low"
        )
        
        # More specifically, it should be the product (which is less than both)
        assert alpha_final <= min(alpha_data, alpha_conf), (
            f"AND logic violated: alpha_final={alpha_final:.4f} should be <= "
            f"min({alpha_data:.4f}, {alpha_conf:.4f})"
        )
