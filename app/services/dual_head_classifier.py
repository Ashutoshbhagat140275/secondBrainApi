"""
Dual-head emotion classifier with adaptive blending.

This module orchestrates the dual-head inference pipeline, combining predictions
from the global emotion head (trained on public datasets) and user-specific heads
(trained on individual feedback) using an adaptive blending strategy.

The blending weight α dynamically adjusts based on:
- Global head confidence (higher confidence → more global weight)
- User feedback count (more feedback → more user weight)

This enables graceful transition from robust baseline predictions to personalized
predictions as users provide feedback.
"""

import numpy as np
import logging
from typing import Dict, Optional

from app.services.global_emotion_head import predict_global
from app.services.user_emotion_head import predict_user
from app.services.feature_config import EMOTION_LABELS
from app.services.alpha_engine import compute_blend_weight

logger = logging.getLogger(__name__)




def blend_predictions(
    P_g: np.ndarray,
    P_u: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Blend global and user probability distributions.
    
    Computes the final probability distribution as a weighted combination:
        P_f = α·P_g + (1-α)·P_u
    
    The result is normalized to ensure it sums to 1.0 for numerical stability.
    
    Parameters
    ----------
    P_g : np.ndarray
        Global head probability distribution, shape (8,)
    P_u : np.ndarray
        User head probability distribution, shape (8,)
    alpha : float
        Blending weight in range [0, 1]
        - α = 1.0: Use only P_g
        - α = 0.5: Equal blend
        - α = 0.0: Use only P_u
    
    Returns
    -------
    np.ndarray
        Blended probability distribution P_f, shape (8,), sums to 1.0
    
    Examples
    --------
    >>> P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
    >>> P_u = np.array([0.1, 0.7, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
    >>> P_f = blend_predictions(P_g, P_u, alpha=0.5)
    >>> np.isclose(P_f.sum(), 1.0)
    True
    >>> P_f[0]  # Blend of 0.7 and 0.1 with equal weight
    0.4
    
    Notes
    -----
    Normalization ensures numerical stability even if input distributions
    have small floating-point errors that cause them to not sum exactly to 1.0.
    """
    # Weighted combination
    P_f = alpha * P_g + (1.0 - alpha) * P_u
    
    # Normalize to ensure sum = 1.0 (numerical stability)
    P_f = P_f / np.sum(P_f)
    
    return P_f


def classify_with_dual_heads(
    embedding: np.ndarray,
    user_id: str,
    feedback_count: int
) -> Dict:
    """
    Classify emotion using global and user heads with adaptive blending.
    
    This is the main entry point for dual-head inference. It:
    1. Gets prediction from global head (always available)
    2. Gets prediction from user head (if trained)
    3. Computes adaptive blending weight
    4. Blends predictions if user head exists
    5. Returns comprehensive prediction metadata
    
    Parameters
    ----------
    embedding : np.ndarray
        768-dimensional Wav2Vec2 embedding, shape (768,)
    user_id : str
        Unique identifier for the user
    feedback_count : int
        Number of feedback samples the user has provided
    
    Returns
    -------
    dict
        Prediction result with the following fields:
        - emotion (str): Final blended emotion label
        - confidence (float): Final confidence score
        - global_emotion (str): Global head prediction
        - global_confidence (float): Global head confidence
        - user_emotion (str | None): User head prediction (None if not available)
        - user_confidence (float | None): User head confidence (None if not available)
        - blend_weight (float): Alpha value used for blending
        - probabilities (dict): Full probability distribution {emotion: prob}
    
    Examples
    --------
    >>> embedding = np.random.randn(768)
    >>> result = classify_with_dual_heads(embedding, "user_123", feedback_count=50)
    >>> result["emotion"]
    'happy'
    >>> result["confidence"]
    0.78
    >>> result["blend_weight"]
    0.55
    
    Notes
    -----
    - If user has < 20 feedback samples, only global head is used
    - If user head is not available, falls back to global only
    - All predictions include full metadata for transparency
    """
    # Step 1: Get global head prediction (always available)
    P_g, C_g = predict_global(embedding)
    global_idx = int(np.argmax(P_g))
    global_emotion = EMOTION_LABELS[global_idx]
    
    logger.debug(
        f"Global prediction for user {user_id}: {global_emotion} "
        f"(confidence: {C_g:.3f})"
    )
    
    # Step 2: Try to get user head prediction
    user_result = predict_user(embedding, user_id)
    
    # Step 3: Compute blending weight using alpha engine
    alpha_result = compute_blend_weight(C_g, feedback_count, user_id=user_id)
    alpha = alpha_result["alpha"]
    
    # Step 4: Blend predictions if user head is available
    if user_result is not None:
        P_u, C_u = user_result
        user_idx = int(np.argmax(P_u))
        user_emotion = EMOTION_LABELS[user_idx]
        
        # Blend probability distributions
        P_f = blend_predictions(P_g, P_u, alpha)
        final_idx = int(np.argmax(P_f))
        final_emotion = EMOTION_LABELS[final_idx]
        final_confidence = float(np.max(P_f))
        
        logger.info(
            f"Dual-head prediction for user {user_id}: {final_emotion} "
            f"(confidence: {final_confidence:.3f}, alpha: {alpha:.3f}, "
            f"global: {global_emotion}, user: {user_emotion})"
        )
        
        # Build full probability distribution
        probabilities = {label: float(prob) for label, prob in zip(EMOTION_LABELS, P_f)}
        
        return {
            "emotion": final_emotion,
            "confidence": final_confidence,
            "global_emotion": global_emotion,
            "global_confidence": float(C_g),
            "user_emotion": user_emotion,
            "user_confidence": float(C_u),
            "blend_weight": alpha,
            "alpha_data": alpha_result.get("alpha_data"),
            "alpha_conf": alpha_result.get("alpha_conf"),
            "alpha_formula": alpha_result["formula"],
            "probabilities": probabilities,
        }
    
    else:
        # User head not available — use global only
        logger.info(
            f"Global-only prediction for user {user_id}: {global_emotion} "
            f"(confidence: {C_g:.3f}, feedback_count: {feedback_count})"
        )
        
        # Build full probability distribution
        probabilities = {label: float(prob) for label, prob in zip(EMOTION_LABELS, P_g)}
        
        return {
            "emotion": global_emotion,
            "confidence": float(C_g),
            "global_emotion": global_emotion,
            "global_confidence": float(C_g),
            "user_emotion": None,
            "user_confidence": None,
            "blend_weight": 1.0,  # Global only
            "alpha_data": None,
            "alpha_conf": None,
            "alpha_formula": "linear",
            "probabilities": probabilities,
        }
