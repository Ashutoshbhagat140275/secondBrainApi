"""
Alpha Engine for Sigmoid-Based Blending

This module implements the refined alpha computation strategy that separates
data availability (feedback count) from confidence (prediction certainty).

The new sigmoid-based formula provides:
- Smooth exponential decay for feedback count (alpha_data)
- Smooth sigmoid transition for confidence (alpha_conf)
- Multiplicative combination (both must agree to trust global head)
- No hard clamping needed (natural bounds from sigmoid)

Backward compatibility is maintained via USE_SIGMOID_ALPHA flag.
"""

import numpy as np
import logging
from typing import Tuple, Dict

from app.services import feature_config
from app.services.feature_config import (
    ALPHA_FEEDBACK_SCALE_K,
    ALPHA_CONFIDENCE_THRESHOLD_TAU,
    ALPHA_SIGMOID_SHARPNESS_BETA,
)

logger = logging.getLogger(__name__)


# Log alpha engine configuration at module initialization
def _log_alpha_config():
    """Log alpha engine configuration at startup."""
    formula = "sigmoid" if feature_config.USE_SIGMOID_ALPHA else "linear"
    logger.info(
        f"Alpha engine initialized: formula={formula}"
    )
    
    if feature_config.USE_SIGMOID_ALPHA:
        logger.info(
            f"Alpha engine hyperparameters: "
            f"K={ALPHA_FEEDBACK_SCALE_K}, "
            f"tau={ALPHA_CONFIDENCE_THRESHOLD_TAU}, "
            f"beta={ALPHA_SIGMOID_SHARPNESS_BETA}"
        )


# Initialize logging when module is imported
_log_alpha_config()


def compute_alpha_data(feedback_count: int, K: float = None) -> float:
    """
    Compute data-driven alpha component based on feedback count.
    
    Formula: alpha_data = 1 / (1 + N/K)
    
    This provides exponential decay as users provide more feedback:
    - N = 0: alpha_data = 1.0 (no feedback → trust global fully)
    - N = K: alpha_data = 0.5 (K feedback samples → equal weight)
    - N → ∞: alpha_data → 0.0 (infinite feedback → user only)
    
    Parameters
    ----------
    feedback_count : int
        Number of feedback samples user has provided
    K : float, optional
        Scaling constant (default from config). Must be > 0.
    
    Returns
    -------
    float
        Alpha data component in range (0, 1]
    
    Raises
    ------
    ValueError
        If K <= 0
    """
    if K is None:
        K = ALPHA_FEEDBACK_SCALE_K
    
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    
    alpha_data = 1.0 / (1.0 + feedback_count / K)
    return alpha_data


def compute_alpha_conf(
    global_confidence: float,
    tau: float = None,
    beta: float = None
) -> float:
    """
    Compute confidence-driven alpha component using sigmoid.
    
    Formula: alpha_conf = 1 / (1 + exp(-β(C_g - τ)))
    
    This provides smooth S-curve transition based on confidence:
    - C_g < τ: alpha_conf < 0.5 (low confidence → favor user head)
    - C_g = τ: alpha_conf = 0.5 (threshold → equal weight)
    - C_g > τ: alpha_conf > 0.5 (high confidence → favor global head)
    
    Parameters
    ----------
    global_confidence : float
        Confidence from global head, range [0, 1]
    tau : float, optional
        Confidence threshold for sigmoid center (default from config).
        Must be in range (0, 1).
    beta : float, optional
        Sigmoid sharpness parameter (default from config).
        Must be > 0. Higher values = steeper transition.
    
    Returns
    -------
    float
        Alpha confidence component in range (0, 1)
    
    Raises
    ------
    ValueError
        If tau not in (0, 1) or beta <= 0
    """
    if tau is None:
        tau = ALPHA_CONFIDENCE_THRESHOLD_TAU
    if beta is None:
        beta = ALPHA_SIGMOID_SHARPNESS_BETA
    
    if not (0 < tau < 1):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    
    # Sigmoid function: 1 / (1 + exp(-β(C_g - τ)))
    z = beta * (global_confidence - tau)
    alpha_conf = 1.0 / (1.0 + np.exp(-z))
    
    return alpha_conf


def compute_alpha_sigmoid(
    global_confidence: float,
    feedback_count: int,
    K: float = None,
    tau: float = None,
    beta: float = None,
    user_id: str = None
) -> Tuple[float, float, float]:
    """
    Compute final alpha using sigmoid-based formula.
    
    Returns all three components for debugging and monitoring:
    - alpha_data: Data availability component (feedback decay)
    - alpha_conf: Confidence component (sigmoid)
    - alpha_final: Multiplicative combination
    
    The multiplicative design ensures both components must agree to trust
    the global head. If either is low, final alpha is low.
    
    Parameters
    ----------
    global_confidence : float
        Confidence from global head, range [0, 1]
    feedback_count : int
        Number of feedback samples user has provided
    K : float, optional
        Feedback scaling constant (default from config)
    tau : float, optional
        Confidence threshold (default from config)
    beta : float, optional
        Sigmoid sharpness (default from config)
    user_id : str, optional
        User identifier for logging context
    
    Returns
    -------
    tuple
        (alpha_data, alpha_conf, alpha_final)
        All values in range [0, 1]
    """
    alpha_data = compute_alpha_data(feedback_count, K)
    alpha_conf = compute_alpha_conf(global_confidence, tau, beta)
    alpha_final = alpha_data * alpha_conf
    
    user_context = f"user_id={user_id}, " if user_id else ""
    logger.debug(
        f"Alpha computation (sigmoid): {user_context}"
        f"feedback_count={feedback_count}, "
        f"global_confidence={global_confidence:.3f}, "
        f"alpha_data={alpha_data:.3f}, "
        f"alpha_conf={alpha_conf:.3f}, "
        f"alpha_final={alpha_final:.3f}"
    )
    
    return alpha_data, alpha_conf, alpha_final


def compute_alpha_linear(
    global_confidence: float,
    feedback_count: int,
    user_id: str = None
) -> float:
    """
    Compute alpha using legacy linear formula (for backward compatibility).
    
    Formula: α = 0.5 + 0.3·C_g - 0.2·min(feedback_count/100, 1.0)
    Clamped to [0.3, 1.0]
    
    Special case: For feedback_count < 20, returns 1.0 (trust global fully)
    
    Parameters
    ----------
    global_confidence : float
        Confidence from global head, range [0, 1]
    feedback_count : int
        Number of feedback samples user has provided
    user_id : str, optional
        User identifier for logging context
    
    Returns
    -------
    float
        Alpha in range [0.3, 1.0]
    """
    # Special case: very few feedback samples → trust global fully
    if feedback_count < 20:
        alpha = 1.0
    else:
        # Linear formula with feedback decay
        feedback_factor = min(feedback_count / 100.0, 1.0)
        alpha = 0.5 + 0.3 * global_confidence - 0.2 * feedback_factor
        
        # Clamp to valid range
        alpha = max(0.3, min(1.0, alpha))
    
    user_context = f"user_id={user_id}, " if user_id else ""
    logger.debug(
        f"Alpha computation (linear): {user_context}"
        f"feedback_count={feedback_count}, "
        f"global_confidence={global_confidence:.3f}, "
        f"alpha_final={alpha:.3f}"
    )
    
    return alpha


def compute_blend_weight(
    global_confidence: float,
    feedback_count: int,
    user_id: str = None
) -> Dict[str, any]:
    """
    Compute blending weight using configured formula (sigmoid or linear).
    
    This is the main entry point that respects the USE_SIGMOID_ALPHA flag.
    It routes to either the new sigmoid-based formula or the legacy linear
    formula based on configuration.
    
    Parameters
    ----------
    global_confidence : float
        Confidence from global head, range [0, 1]
    feedback_count : int
        Number of feedback samples user has provided
    user_id : str, optional
        User identifier for logging context
    
    Returns
    -------
    dict
        {
            "alpha": float,  # Final blending weight
            "alpha_data": float | None,  # Data component (sigmoid only)
            "alpha_conf": float | None,  # Confidence component (sigmoid only)
            "formula": str  # "sigmoid" or "linear"
        }
    
    Examples
    --------
    >>> # Sigmoid mode (USE_SIGMOID_ALPHA = True)
    >>> result = compute_blend_weight(0.8, 25)
    >>> result
    {'alpha': 0.56, 'alpha_data': 0.67, 'alpha_conf': 0.83, 'formula': 'sigmoid'}
    
    >>> # Linear mode (USE_SIGMOID_ALPHA = False)
    >>> result = compute_blend_weight(0.8, 25)
    >>> result
    {'alpha': 0.74, 'alpha_data': None, 'alpha_conf': None, 'formula': 'linear'}
    """
    if feature_config.USE_SIGMOID_ALPHA:
        alpha_data, alpha_conf, alpha = compute_alpha_sigmoid(
            global_confidence, feedback_count, user_id=user_id
        )
        return {
            "alpha": alpha,
            "alpha_data": alpha_data,
            "alpha_conf": alpha_conf,
            "formula": "sigmoid"
        }
    else:
        alpha = compute_alpha_linear(global_confidence, feedback_count, user_id=user_id)
        return {
            "alpha": alpha,
            "alpha_data": None,
            "alpha_conf": None,
            "formula": "linear"
        }
