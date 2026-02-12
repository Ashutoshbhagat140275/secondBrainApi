"""
Wav2Vec2 neural embedding extraction service.

Loads Facebook's pretrained Wav2Vec2-base model from HuggingFace and
extracts 768-dimensional learned representations from audio waveforms.

Replaces manual feature extraction (MFCCs, pitch, jitter/shimmer) with
robust neural embeddings that capture prosody, rhythm, and speaking style.

If the model fails to load, a warning is logged and extraction calls
will raise RuntimeError to signal unavailability.
"""

import numpy as np
import torch
import librosa
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# ── module-level singletons (loaded once) ─────────────────────────────────
_wav2vec2_model: Optional[torch.nn.Module] = None
_wav2vec2_processor = None  # transformers.Wav2Vec2Processor

# Constants
EMBEDDING_DIM = 768  # Wav2Vec2-base output dimension
MIN_AUDIO_LENGTH = 4800  # 0.3 seconds at 16kHz
TARGET_SAMPLE_RATE = 16000  # Hz


def load_wav2vec2_model() -> bool:
    """
    Load the pretrained Wav2Vec2-base model from HuggingFace.

    Called once — either explicitly at app startup or lazily on first
    ``extract_wav2vec2_embedding()`` call.

    Returns
    -------
    bool
        True if model loaded successfully, False otherwise.
    """
    global _wav2vec2_model, _wav2vec2_processor

    # Already loaded
    if _wav2vec2_model is not None and _wav2vec2_processor is not None:
        return True

    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        logger.info("Loading Wav2Vec2-base model from HuggingFace...")

        _wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        _wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        _wav2vec2_model.eval()  # Set to inference mode

        logger.info("Wav2Vec2 model loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"Failed to load Wav2Vec2 model: {e}", exc_info=True)
        _wav2vec2_model = None
        _wav2vec2_processor = None
        return False


def extract_wav2vec2_embedding(audio_path: str) -> np.ndarray:
    """
    Extract 768-dimensional neural embedding from audio file.

    Loads audio at 16kHz, processes through Wav2Vec2 encoder, and applies
    mean pooling across the time dimension to produce a fixed-size vector.

    Parameters
    ----------
    audio_path : str
        Path to preprocessed audio file (16kHz, mono).

    Returns
    -------
    np.ndarray
        Shape (768,) embedding vector.

    Raises
    ------
    RuntimeError
        If Wav2Vec2 model is not loaded or inference fails.
    ValueError
        If audio file is invalid or cannot be loaded.
    """
    global _wav2vec2_model, _wav2vec2_processor

    # Lazy-load on first call if not yet initialized
    if _wav2vec2_model is None or _wav2vec2_processor is None:
        load_wav2vec2_model()

    # Check if model is available
    if _wav2vec2_model is None or _wav2vec2_processor is None:
        raise RuntimeError(
            "Wav2Vec2 model not available. Check logs for loading errors."
        )

    try:
        # Load audio at 16kHz (matches preprocessing)
        waveform, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)

        # Validate waveform
        if len(waveform) == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        # Check for NaN values in waveform
        if np.isnan(waveform).any():
            logger.warning(
                f"Audio contains NaN values, replacing with zeros: {audio_path}"
            )
            waveform = np.nan_to_num(waveform, nan=0.0)

        # Handle short audio (< 0.3 seconds)
        if len(waveform) < MIN_AUDIO_LENGTH:
            logger.warning(
                f"Audio is very short ({len(waveform)} samples, "
                f"{len(waveform)/TARGET_SAMPLE_RATE:.2f}s), padding to minimum length."
            )
            # Pad with zeros to minimum length
            waveform = np.pad(
                waveform, (0, MIN_AUDIO_LENGTH - len(waveform)), mode="constant"
            )

        # Process through Wav2Vec2
        inputs = _wav2vec2_processor(
            waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = _wav2vec2_model(**inputs)
            # outputs.last_hidden_state shape: (1, T, 768)
            embeddings = outputs.last_hidden_state

        # Mean pooling across time dimension
        pooled = embeddings.mean(dim=1).squeeze(0)  # (768,)

        # Convert to numpy
        embedding = pooled.numpy()

        # Final validation: check for NaN in embedding
        if np.isnan(embedding).any():
            logger.error(
                f"Embedding contains NaN values after extraction: {audio_path}"
            )
            # Replace NaN with zeros as fallback
            embedding = np.nan_to_num(embedding, nan=0.0)

        logger.debug(
            f"Extracted embedding: shape={embedding.shape}, "
            f"mean={embedding.mean():.4f}, std={embedding.std():.4f}"
        )

        return embedding

    except Exception as e:
        logger.error(
            f"Failed to extract Wav2Vec2 embedding from {audio_path}: {e}",
            exc_info=True,
        )
        raise RuntimeError(f"Wav2Vec2 embedding extraction failed: {e}") from e
