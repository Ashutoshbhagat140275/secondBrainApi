"""
Emotion classification service — DNN-based inference using Wav2Vec2 neural embeddings.

Uses Wav2Vec2 pretrained model to extract 768-dimensional neural embeddings
and classifies emotions using a trained classifier head.

If the model artifacts have not been trained yet, a warning is logged
and a safe fallback ("neutral", 0.5) is returned so the application
does not crash.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def load_emotion_model() -> bool:
    """
    Load the trained Wav2Vec2 emotion classifier and scaler from disk.

    Called once — either explicitly at app startup or lazily on first
    ``classify_emotion_from_audio()`` call.

    Returns True if artifacts were loaded successfully, False otherwise.
    """
    try:
        from app.services.emotion_classifier import load_emotion_classifier
        
        success = load_emotion_classifier()
        if success:
            logger.info(
                "Neural embedding model loaded successfully. "
                "Feature extraction: Wav2Vec2 (768-dim)"
            )
        else:
            logger.warning(
                "No emotion model artifacts found. "
                "Classification will use fallback predictions. "
                "Run the training pipeline to generate model artifacts."
            )
        return success
        
    except Exception as e:
        logger.error(
            f"Failed to load neural embedding model: {e}", 
            exc_info=True
        )
        return False


def classify_emotion_from_audio(audio_path: str) -> Tuple[str, float]:
    """
    Predict emotion directly from audio file path using Wav2Vec2 embeddings.
    
    Extracts neural embeddings using Wav2Vec2 encoder and classifies
    using the trained emotion classifier head.

    Parameters
    ----------
    audio_path : str
        Path to preprocessed audio file (16kHz, mono).

    Returns
    -------
    (emotion_label, confidence) : Tuple[str, float]
        The predicted emotion string and its softmax probability.
    """
    try:
        from app.services.wav2vec2_encoder import extract_wav2vec2_embedding
        from app.services.emotion_classifier import classify_emotion_from_embedding
        
        embedding = extract_wav2vec2_embedding(audio_path)
        return classify_emotion_from_embedding(embedding)
        
    except Exception as e:
        logger.error(
            f"Emotion classification from audio failed: {e}", 
            exc_info=True
        )
        return ("neutral", 0.5)
