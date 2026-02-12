"""
Centralized configuration for the emotion classification feature pipeline.

This module is the single source of truth for:
- Feature vector dimensionality
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
N_MFCC = 13  # Number of MFCC coefficients
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT
N_MELS = 128  # Number of mel bands
N_CONTRAST_BANDS = 6  # For spectral contrast (produces n_bands + 1 rows)

# ---------------------------------------------------------------------------
# Feature vector layout
# ---------------------------------------------------------------------------
# Each "row" feature is summarised via 8 functionals:
#   [mean, std, min, max, p10, p50, p90, slope]
FUNCTIONALS_PER_ROW = 8

# Counts of row-level features
MFCC_ROWS = N_MFCC  # 13
DELTA_ROWS = N_MFCC  # 13
DELTA2_ROWS = N_MFCC  # 13
RMS_ROWS = 1  # 1
F0_ROWS = 1  # 1
CENTROID_ROWS = 1  # 1
BANDWIDTH_ROWS = 1  # 1
ZCR_ROWS = 1  # 1  (NEW)
ROLLOFF_ROWS = 1  # 1  (NEW)
CONTRAST_ROWS = N_CONTRAST_BANDS + 1  # 7  (NEW)

# Scalar features (not summarised with functionals)
SCALAR_FEATURES = 3  # jitter, shimmer, HNR

FEATURE_DIM = (
    MFCC_ROWS
    + DELTA_ROWS
    + DELTA2_ROWS
    + RMS_ROWS
    + F0_ROWS
    + CENTROID_ROWS
    + BANDWIDTH_ROWS
    + ZCR_ROWS
    + ROLLOFF_ROWS
    + CONTRAST_ROWS
) * FUNCTIONALS_PER_ROW + SCALAR_FEATURES
# = (13+13+13 + 1+1+1+1 + 1+1+7) * 8 + 3
# = 52 * 8 + 3 = 416 + 3 = 419

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
MODEL_PATH = MODEL_DIR / "emotion_model.pt"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"

# Training data directories (user must download datasets here)
TRAINING_DATA_DIR = _API_ROOT / "training" / "data"
RAVDESS_DIR = TRAINING_DATA_DIR / "ravdess"
CREMAD_DIR = TRAINING_DATA_DIR / "crema-d"

# Cached extracted features
FEATURES_CACHE_DIR = _API_ROOT / "training" / "cache"
