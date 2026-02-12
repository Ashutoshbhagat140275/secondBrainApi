"""
Global emotion head for dual-head emotion recognition system.

This module provides a lightweight linear classifier that maps 768-dimensional
Wav2Vec2 embeddings to 8 emotion logits. The global head is trained once on
public datasets (RAVDESS + CREMA-D) and serves as a robust baseline for all users.

The model is loaded once at startup and cached in memory for efficient inference.
If the model artifact is not available, a fallback prediction is returned.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional
from pathlib import Path

from app.services.feature_config import (
    EMBEDDING_DIM,
    NUM_CLASSES,
    EMOTION_LABELS,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)

# ── Module-level singleton (loaded once) ──────────────────────────────────
_global_head_model: Optional[torch.nn.Module] = None

# Global head model path
GLOBAL_HEAD_PATH = MODEL_DIR / "global_emotion_head.pt"


class GlobalEmotionHead(nn.Module):
    """
    Lightweight linear classifier for global emotion recognition.
    
    Architecture:
        Input(768) → Linear(8) → Logits
    
    No hidden layers needed since Wav2Vec2 embeddings are already high-level
    representations. This simplicity enables fast inference and small model size.
    
    Parameters
    ----------
    embedding_dim : int, default=768
        Dimensionality of input embeddings (Wav2Vec2-base output)
    num_classes : int, default=8
        Number of emotion classes to predict
    """
    
    def __init__(self, embedding_dim: int = 768, num_classes: int = 8):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear classifier.
        
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (batch_size, embedding_dim)
        
        Returns
        -------
        torch.Tensor
            Raw logits of shape (batch_size, num_classes)
        """
        return self.linear(x)


def load_global_head() -> bool:
    """
    Load the trained global emotion head from disk.
    
    This function is called once at app startup or lazily on first inference.
    The loaded model is cached in the module-level ``_global_head_model`` variable.
    
    Returns
    -------
    bool
        True if the model was loaded successfully, False otherwise.
    
    Notes
    -----
    If the model file does not exist, a warning is logged and the function
    returns False. The ``predict_global()`` function will use a fallback
    prediction in this case.
    """
    global _global_head_model
    
    # Return early if model is already loaded (singleton pattern)
    if _global_head_model is not None:
        logger.debug("Global emotion head already loaded (using cached model)")
        return True
    
    if not GLOBAL_HEAD_PATH.exists():
        logger.warning(
            f"Global emotion head not found at {GLOBAL_HEAD_PATH}. "
            "predict_global() will use fallback predictions. "
            "Train the global head first using train_wav2vec2.py --save-as-global-head"
        )
        return False
    
    try:
        model = GlobalEmotionHead(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
        state = torch.load(str(GLOBAL_HEAD_PATH), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _global_head_model = model
        
        logger.info(f"Global emotion head loaded successfully from {GLOBAL_HEAD_PATH}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to load global emotion head: {e}", exc_info=True)
        _global_head_model = None
        return False


def predict_global(embedding: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Predict emotion probabilities using the global head.
    
    This function computes the probability distribution P_g over 8 emotions
    and the confidence score C_g = max(P_g) for a given embedding.
    
    Parameters
    ----------
    embedding : np.ndarray
        768-dimensional Wav2Vec2 embedding of shape (768,)
    
    Returns
    -------
    (P_g, C_g) : Tuple[np.ndarray, float]
        - P_g: Probability distribution over 8 emotions, shape (8,)
        - C_g: Confidence score (maximum probability)
    
    Notes
    -----
    If the global head is not loaded, returns a uniform distribution with
    low confidence as a fallback.
    
    Examples
    --------
    >>> embedding = np.random.randn(768)
    >>> P_g, C_g = predict_global(embedding)
    >>> P_g.shape
    (8,)
    >>> 0.0 <= C_g <= 1.0
    True
    """
    global _global_head_model
    
    # Lazy-load on first call if not yet initialized
    if _global_head_model is None:
        load_global_head()
    
    # Fallback if model is still unavailable
    if _global_head_model is None:
        logger.warning("Global head unavailable — returning uniform fallback distribution.")
        # Return uniform distribution with low confidence
        P_g = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        C_g = float(1.0 / NUM_CLASSES)
        return (P_g, C_g)
    
    try:
        # Validate input shape
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Expected embedding shape ({EMBEDDING_DIM},), got {embedding.shape}"
            )
        
        # Convert to tensor and add batch dimension
        x = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0)  # (1, 768)
        
        with torch.no_grad():
            logits = _global_head_model(x)  # (1, NUM_CLASSES)
            probs = F.softmax(logits, dim=1)[0]  # (NUM_CLASSES,)
        
        # Convert to numpy
        P_g = probs.numpy()
        C_g = float(np.max(P_g))
        
        # Log prediction
        predicted_idx = int(np.argmax(P_g))
        predicted_label = EMOTION_LABELS[predicted_idx]
        logger.debug(
            f"Global head prediction: {predicted_label} "
            f"(confidence: {C_g:.3f})"
        )
        
        return (P_g, C_g)
    
    except Exception as e:
        logger.error(f"Global head prediction failed: {e}", exc_info=True)
        # Return uniform distribution as fallback
        P_g = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        C_g = float(1.0 / NUM_CLASSES)
        return (P_g, C_g)
