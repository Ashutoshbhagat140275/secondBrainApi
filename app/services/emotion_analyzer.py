"""
Emotion classification service — DNN-based inference.

Loads a pre-trained PyTorch MLP (``EmotionMLP``) and a fitted
``StandardScaler`` at first use, then runs inference on the 419-dim
feature vector produced by ``feature_extractor.extract_full_features()``.

If the model artifacts have not been trained yet, a warning is logged
and a safe fallback ("neutral", 0.5) is returned so the application
does not crash.
"""

import numpy as np
import torch
import torch.nn.functional as F
import joblib
import logging
from typing import Tuple, List, Optional
from pathlib import Path

from app.services.feature_config import (
    FEATURE_DIM,
    NUM_CLASSES,
    EMOTION_LABELS,
    MODEL_PATH,
    SCALER_PATH,
)

logger = logging.getLogger(__name__)

# ── module-level singletons (loaded once) ─────────────────────────────────
_emotion_model: Optional[torch.nn.Module] = None
_feature_scaler = None  # sklearn StandardScaler


def load_emotion_model() -> bool:
    """
    Load the trained MLP and feature scaler from disk.

    Called once — either explicitly at app startup or lazily on first
    ``classify_emotion()`` call.

    Returns True if artifacts were loaded successfully, False otherwise.
    """
    global _emotion_model, _feature_scaler

    model_path = Path(MODEL_PATH)
    scaler_path = Path(SCALER_PATH)

    if not model_path.exists():
        logger.warning(
            f"Emotion model not found at {model_path}. "
            "classify_emotion() will use fallback predictions. "
            "Run the training pipeline first."
        )
        return False

    if not scaler_path.exists():
        logger.warning(
            f"Feature scaler not found at {scaler_path}. "
            "classify_emotion() will use fallback predictions."
        )
        return False

    try:
        # Lazy import to avoid circular dependency at module level
        from training.model import EmotionMLP

        model = EmotionMLP(input_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _emotion_model = model

        _feature_scaler = joblib.load(str(scaler_path))

        logger.info("Emotion model and scaler loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load emotion model: {e}", exc_info=True)
        _emotion_model = None
        _feature_scaler = None
        return False


def classify_emotion(features: List[float]) -> Tuple[str, float]:
    """
    Predict the most likely emotion label from a feature vector.

    Parameters
    ----------
    features : List[float]
        Full 419-dimensional feature vector produced by
        ``feature_extractor.extract_full_features()``.

    Returns
    -------
    (emotion_label, confidence) : Tuple[str, float]
        The predicted emotion string and its softmax probability.
    """
    global _emotion_model, _feature_scaler

    # Lazy-load on first call if not yet initialised
    if _emotion_model is None:
        load_emotion_model()

    # Fallback if model is still unavailable
    if _emotion_model is None or _feature_scaler is None:
        logger.warning("Emotion model unavailable — returning fallback prediction.")
        return ("neutral", 0.5)

    try:
        # Normalise with the same scaler used during training
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        x = _feature_scaler.transform(x)
        tensor = torch.from_numpy(x.astype(np.float32))

        with torch.no_grad():
            logits = _emotion_model(tensor)  # (1, NUM_CLASSES)
            probs = F.softmax(logits, dim=1)[0]  # (NUM_CLASSES,)

        idx = int(torch.argmax(probs).item())
        label = EMOTION_LABELS[idx]
        confidence = float(probs[idx].item())

        logger.info(f"Emotion classified: {label} (confidence: {confidence:.3f})")
        return (label, confidence)

    except Exception as e:
        logger.error(f"Emotion classification failed: {e}", exc_info=True)
        return ("neutral", 0.5)
