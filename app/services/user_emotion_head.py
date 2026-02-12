"""
User-specific emotion head for personalized emotion recognition.

This module provides per-user linear classifiers that adapt to individual
expression patterns through feedback. Each user has their own UserEmotionHead
with identical architecture to the GlobalEmotionHead but independent weights
trained on user-specific feedback data.

User models are stored at models/user_heads/{user_id}.pt and loaded lazily
with LRU caching (max 100 models in memory) for efficient inference.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional
from pathlib import Path
from functools import lru_cache

from app.services.feature_config import (
    EMBEDDING_DIM,
    NUM_CLASSES,
    EMOTION_LABELS,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)

# ── User head storage configuration ───────────────────────────────────────
USER_HEADS_DIR = MODEL_DIR / "user_heads"
USER_HEAD_CACHE_SIZE = 100  # Max number of user models to keep in memory


class UserEmotionHead(nn.Module):
    """
    Per-user linear classifier for personalized emotion recognition.
    
    Architecture:
        Input(768) → Linear(8) → Logits
    
    Identical architecture to GlobalEmotionHead but trained on user-specific
    feedback data. Each user has their own independent weights that adapt to
    their unique emotional expression patterns.
    
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


@lru_cache(maxsize=USER_HEAD_CACHE_SIZE)
def load_user_head(user_id: str) -> Optional[nn.Module]:
    """
    Load user-specific emotion head from disk with LRU caching.
    
    This function loads a trained user head from models/user_heads/{user_id}.pt
    and caches it in memory. The LRU cache automatically evicts least-recently-used
    models when the cache size exceeds USER_HEAD_CACHE_SIZE (100 models).
    
    Parameters
    ----------
    user_id : str
        Unique identifier for the user
    
    Returns
    -------
    Optional[nn.Module]
        Loaded UserEmotionHead model in eval mode, or None if the user has
        no trained model yet.
    
    Notes
    -----
    - Returns None if the model file does not exist (user has not provided
      sufficient feedback for training yet)
    - The LRU cache keeps up to 100 models in memory for fast inference
    - Models are automatically evicted when cache is full
    
    Examples
    --------
    >>> model = load_user_head("user_123")
    >>> if model is not None:
    ...     # User has a trained model
    ...     pass
    """
    # Ensure user_heads directory exists
    USER_HEADS_DIR.mkdir(parents=True, exist_ok=True)
    
    user_model_path = USER_HEADS_DIR / f"{user_id}.pt"
    
    if not user_model_path.exists():
        logger.debug(f"No trained model found for user {user_id} at {user_model_path}")
        return None
    
    try:
        model = UserEmotionHead(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
        state = torch.load(str(user_model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        
        logger.info(f"User emotion head loaded for user {user_id} from {user_model_path}")
        return model
    
    except Exception as e:
        logger.error(
            f"Failed to load user emotion head for user {user_id}: {e}",
            exc_info=True
        )
        return None


def predict_user(embedding: np.ndarray, user_id: str) -> Optional[Tuple[np.ndarray, float]]:
    """
    Predict emotion probabilities using the user-specific head.
    
    This function computes the probability distribution P_u over 8 emotions
    and the confidence score C_u = max(P_u) for a given embedding using the
    user's personalized model.
    
    Parameters
    ----------
    embedding : np.ndarray
        768-dimensional Wav2Vec2 embedding of shape (768,)
    user_id : str
        Unique identifier for the user
    
    Returns
    -------
    Optional[Tuple[np.ndarray, float]]
        - If user has a trained model: (P_u, C_u) where
          - P_u: Probability distribution over 8 emotions, shape (8,)
          - C_u: Confidence score (maximum probability)
        - If user has no trained model: None
    
    Notes
    -----
    Returns None if the user has not provided sufficient feedback for training
    (< 20 samples). In this case, the dual-head classifier will use only the
    global head for prediction.
    
    Examples
    --------
    >>> embedding = np.random.randn(768)
    >>> result = predict_user(embedding, "user_123")
    >>> if result is not None:
    ...     P_u, C_u = result
    ...     print(f"User confidence: {C_u:.3f}")
    """
    # Load user model (uses LRU cache)
    user_model = load_user_head(user_id)
    
    if user_model is None:
        logger.debug(f"No user model available for user {user_id} — returning None")
        return None
    
    try:
        # Validate input shape
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Expected embedding shape ({EMBEDDING_DIM},), got {embedding.shape}"
            )
        
        # Convert to tensor and add batch dimension
        x = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0)  # (1, 768)
        
        with torch.no_grad():
            logits = user_model(x)  # (1, NUM_CLASSES)
            probs = F.softmax(logits, dim=1)[0]  # (NUM_CLASSES,)
        
        # Convert to numpy
        P_u = probs.numpy()
        C_u = float(np.max(P_u))
        
        # Log prediction
        predicted_idx = int(np.argmax(P_u))
        predicted_label = EMOTION_LABELS[predicted_idx]
        logger.debug(
            f"User head prediction for {user_id}: {predicted_label} "
            f"(confidence: {C_u:.3f})"
        )
        
        return (P_u, C_u)
    
    except Exception as e:
        logger.error(
            f"User head prediction failed for user {user_id}: {e}",
            exc_info=True
        )
        return None


def create_fresh_user_head() -> nn.Module:
    """
    Create a new untrained user emotion head.
    
    This function initializes a fresh UserEmotionHead with random weights
    for training on user-specific feedback data. The model is returned in
    training mode (not eval mode).
    
    Returns
    -------
    nn.Module
        New UserEmotionHead model with randomly initialized weights
    
    Notes
    -----
    This function is used by the training pipeline when a user reaches the
    minimum feedback threshold (20 samples) and does not yet have a trained
    model. The model is trained and then saved to models/user_heads/{user_id}.pt.
    
    Examples
    --------
    >>> model = create_fresh_user_head()
    >>> model.training
    True
    >>> # Train the model on user feedback data
    >>> # Save to models/user_heads/{user_id}.pt
    """
    model = UserEmotionHead(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
    logger.debug("Created fresh user emotion head with random initialization")
    return model
