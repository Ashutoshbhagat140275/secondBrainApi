"""
Emotion classification service for Wav2Vec2 neural embeddings.

Loads a pre-trained PyTorch classifier head (``EmbeddingClassifier``) and a
fitted ``StandardScaler`` at first use, then runs inference on 768-dimensional
embeddings produced by the Wav2Vec2 encoder.

If the model artifacts have not been trained yet, a warning is logged
and a safe fallback ("neutral", 0.5) is returned so the application
does not crash.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import logging
from typing import Tuple, Optional
from pathlib import Path

from app.services.feature_config import (
    EMBEDDING_DIM,
    NUM_CLASSES,
    EMOTION_LABELS,
    EMBEDDING_CLASSIFIER_PATH,
    EMBEDDING_SCALER_PATH,
)

logger = logging.getLogger(__name__)

# ── module-level singletons (loaded once) ─────────────────────────────────
_classifier_model: Optional[torch.nn.Module] = None
_embedding_scaler = None  # sklearn StandardScaler


class EmbeddingClassifier(nn.Module):
    """
    Classifier head for Wav2Vec2 embeddings.
    
    Architecture:
        Input(768) → Linear(128) → BN → ReLU → Dropout(0.3)
                  → Linear(64) → ReLU → Dropout(0.2)
                  → Linear(8)  (logits)
    
    Simpler than legacy MLP since Wav2Vec2 already learned rich representations.
    """
    
    def __init__(self, embedding_dim: int = 768, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_emotion_classifier() -> bool:
    """
    Load the trained classifier head and feature scaler from disk.

    Called once — either explicitly at app startup or lazily on first
    ``classify_emotion_from_embedding()`` call.

    Returns True if artifacts were loaded successfully, False otherwise.
    """
    global _classifier_model, _embedding_scaler

    model_path = Path(EMBEDDING_CLASSIFIER_PATH)
    scaler_path = Path(EMBEDDING_SCALER_PATH)

    if not model_path.exists():
        logger.warning(
            f"Embedding classifier not found at {model_path}. "
            "classify_emotion_from_embedding() will use fallback predictions. "
            "Run the training pipeline first."
        )
        return False

    if not scaler_path.exists():
        logger.warning(
            f"Embedding scaler not found at {scaler_path}. "
            "classify_emotion_from_embedding() will use fallback predictions."
        )
        return False

    try:
        model = EmbeddingClassifier(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _classifier_model = model

        _embedding_scaler = joblib.load(str(scaler_path))

        logger.info("Embedding classifier and scaler loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load embedding classifier: {e}", exc_info=True)
        _classifier_model = None
        _embedding_scaler = None
        return False


def classify_emotion_from_embedding(embedding: np.ndarray) -> Tuple[str, float]:
    """
    Predict the most likely emotion label from a 768-dimensional embedding.

    Parameters
    ----------
    embedding : np.ndarray
        768-dimensional neural embedding produced by Wav2Vec2 encoder.
        Expected shape: (768,)

    Returns
    -------
    (emotion_label, confidence) : Tuple[str, float]
        The predicted emotion string and its softmax probability.
    """
    global _classifier_model, _embedding_scaler

    # Lazy-load on first call if not yet initialised
    if _classifier_model is None:
        load_emotion_classifier()

    # Fallback if model is still unavailable
    if _classifier_model is None or _embedding_scaler is None:
        logger.warning("Embedding classifier unavailable — returning fallback prediction.")
        return ("neutral", 0.5)

    try:
        # Validate input shape
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Expected embedding shape ({EMBEDDING_DIM},), got {embedding.shape}"
            )

        # Normalise with the same scaler used during training
        x = embedding.reshape(1, -1)
        x = _embedding_scaler.transform(x)
        tensor = torch.from_numpy(x.astype(np.float32))

        with torch.no_grad():
            logits = _classifier_model(tensor)  # (1, NUM_CLASSES)
            probs = F.softmax(logits, dim=1)[0]  # (NUM_CLASSES,)

        idx = int(torch.argmax(probs).item())
        label = EMOTION_LABELS[idx]
        confidence = float(probs[idx].item())

        logger.info(f"Emotion classified: {label} (confidence: {confidence:.3f})")
        return (label, confidence)

    except Exception as e:
        logger.error(f"Emotion classification failed: {e}", exc_info=True)
        return ("neutral", 0.5)
