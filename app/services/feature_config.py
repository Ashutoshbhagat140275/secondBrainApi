"""
Centralized configuration for the emotion classification feature pipeline.

This module is the single source of truth for:
- Neural embedding dimensionality (Wav2Vec2)
- Emotion class labels and mappings
- Audio processing constants
- Model artifact paths

Shared by both the training scripts and the inference (FastAPI) code
to guarantee feature parity between training and serving.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Audio processing constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000  # Hz — must match preprocess_audio() output

# ---------------------------------------------------------------------------
# Neural embedding constants (Wav2Vec2)
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 768  # Wav2Vec2-base output dimension
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"

# ---------------------------------------------------------------------------
# Emotion labels
# ---------------------------------------------------------------------------
# Union of RAVDESS (8 classes) — used as the canonical label set.
# CREMA-D's 6 classes are a subset (no "calm" or "surprised").
EMOTION_LABELS = [
    "neutral",  # 0
    "calm",  # 1
    "happy",  # 2
    "sad",  # 3
    "angry",  # 4
    "fearful",  # 5
    "disgusted",  # 6
    "surprised",  # 7
]
NUM_CLASSES = len(EMOTION_LABELS)  # 8

# Mapping from RAVDESS emotion code (1-indexed) to label index
RAVDESS_CODE_TO_IDX = {
    1: 0,  # neutral
    2: 1,  # calm
    3: 2,  # happy
    4: 3,  # sad
    5: 4,  # angry
    6: 5,  # fearful
    7: 6,  # disgusted (RAVDESS calls it "disgust")
    8: 7,  # surprised
}

# Mapping from CREMA-D emotion abbreviation to label index
CREMAD_CODE_TO_IDX = {
    "NEU": 0,  # neutral
    "HAP": 2,  # happy
    "SAD": 3,  # sad
    "ANG": 4,  # angry
    "FEA": 5,  # fearful
    "DIS": 6,  # disgusted
}

# ---------------------------------------------------------------------------
# Model / scaler artifact paths (relative to api/ root)
# ---------------------------------------------------------------------------
_API_ROOT = Path(__file__).resolve().parents[2]  # …/api/

MODEL_DIR = _API_ROOT / "models"

# Embedding-based classifier artifacts (Wav2Vec2 pipeline)
EMBEDDING_CLASSIFIER_PATH = MODEL_DIR / "embedding_classifier.pt"
EMBEDDING_SCALER_PATH = MODEL_DIR / "embedding_scaler.joblib"

# Training data directories (user must download datasets here)
TRAINING_DATA_DIR = _API_ROOT / "training" / "data"
RAVDESS_DIR = TRAINING_DATA_DIR / "ravdess"
CREMAD_DIR = TRAINING_DATA_DIR / "crema-d"

# Cached extracted features
FEATURES_CACHE_DIR = _API_ROOT / "training" / "cache"
