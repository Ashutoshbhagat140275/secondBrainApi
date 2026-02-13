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
# Phase 1: Single shared classifier (legacy)
EMBEDDING_CLASSIFIER_PATH = MODEL_DIR / "embedding_classifier.pt"
EMBEDDING_SCALER_PATH = MODEL_DIR / "embedding_scaler.joblib"

# Phase 2: Dual-head emotion recognition system
# Global head: Shared classifier trained on public datasets (RAVDESS + CREMA-D)
# User heads: Per-user personalized classifiers trained on individual feedback
GLOBAL_HEAD_PATH = MODEL_DIR / "global_emotion_head.pt"
USER_HEADS_DIR = MODEL_DIR / "user_heads"
USER_HEAD_CACHE_SIZE = 100  # Max number of user models to keep in memory (LRU)

# Training data directories (user must download datasets here)
TRAINING_DATA_DIR = _API_ROOT / "training" / "data"
RAVDESS_DIR = TRAINING_DATA_DIR / "ravdess"
CREMAD_DIR = TRAINING_DATA_DIR / "crema-d"

# Cached extracted features
FEATURES_CACHE_DIR = _API_ROOT / "training" / "cache"

# ---------------------------------------------------------------------------
# Alpha Engine Configuration (Sigmoid-Based Blending)
# ---------------------------------------------------------------------------
"""
Alpha Engine: Sigmoid-based blending weight computation for dual-head system.

The alpha engine computes a blending weight that determines how to combine
predictions from the global head (trained on public data) and user head
(personalized on user feedback). The new sigmoid-based formula separates
data availability concerns from confidence concerns using multiplicative
combination.

Formula:
    alpha_data = 1 / (1 + N/K)
    alpha_conf = 1 / (1 + exp(-β(C_g - τ)))
    alpha = alpha_data × alpha_conf

Where:
    N = feedback_count (number of user feedback samples)
    C_g = global_confidence (global head prediction confidence)
    K = ALPHA_FEEDBACK_SCALE_K
    τ = ALPHA_CONFIDENCE_THRESHOLD_TAU
    β = ALPHA_SIGMOID_SHARPNESS_BETA
"""

USE_SIGMOID_ALPHA = False
"""
Enable sigmoid-based alpha formula (default: False for backward compatibility).

When False: Uses legacy linear formula for alpha computation.
When True: Uses new sigmoid-based formula with separate data and confidence components.

This flag allows safe deployment and instant rollback without code changes.
"""

ALPHA_FEEDBACK_SCALE_K = 50
"""
Feedback scaling constant K for alpha_data computation (default: 50).

Controls how quickly alpha_data decays as users provide more feedback.
At N=K feedback samples, alpha_data = 0.5 (equal weight between heads).

Valid range: K > 0
Typical values:
    - K=25: Fast decay (personalize quickly)
    - K=50: Medium decay (balanced) [DEFAULT]
    - K=100: Slow decay (trust global longer)

Formula: alpha_data = 1 / (1 + N/K)
"""

ALPHA_CONFIDENCE_THRESHOLD_TAU = 0.6
"""
Confidence threshold τ (tau) for sigmoid center point (default: 0.6).

Sets the global confidence level at which alpha_conf = 0.5 (equal weight).
Below this threshold, the system favors the user head; above it, favors global.

Valid range: 0 < τ < 1
Typical values:
    - τ=0.5: Lower threshold (trust global more easily)
    - τ=0.6: Medium threshold (balanced) [DEFAULT]
    - τ=0.7: Higher threshold (require high confidence)

Formula: alpha_conf = 1 / (1 + exp(-β(C_g - τ)))
"""

ALPHA_SIGMOID_SHARPNESS_BETA = 10
"""
Sigmoid sharpness parameter β (beta) for alpha_conf transition (default: 10).

Controls how steep the sigmoid transition is around the confidence threshold.
Higher values create sharper transitions (more decisive switching).

Valid range: β > 0
Typical values:
    - β=5: Gentle transition (wide confidence range)
    - β=10: Medium transition (balanced) [DEFAULT]
    - β=20: Sharp transition (narrow confidence range)

Formula: alpha_conf = 1 / (1 + exp(-β(C_g - τ)))
"""

# Validation of alpha engine constants
def _validate_alpha_config():
    """
    Validate alpha engine configuration constants.
    
    Raises
    ------
    ValueError
        If any constant is outside its valid range.
    """
    if ALPHA_FEEDBACK_SCALE_K <= 0:
        raise ValueError(
            f"ALPHA_FEEDBACK_SCALE_K must be > 0, got {ALPHA_FEEDBACK_SCALE_K}"
        )
    
    if not (0 < ALPHA_CONFIDENCE_THRESHOLD_TAU < 1):
        raise ValueError(
            f"ALPHA_CONFIDENCE_THRESHOLD_TAU must be in range (0, 1), "
            f"got {ALPHA_CONFIDENCE_THRESHOLD_TAU}"
        )
    
    if ALPHA_SIGMOID_SHARPNESS_BETA <= 0:
        raise ValueError(
            f"ALPHA_SIGMOID_SHARPNESS_BETA must be > 0, "
            f"got {ALPHA_SIGMOID_SHARPNESS_BETA}"
        )


# Run validation at module import time
_validate_alpha_config()

# ---------------------------------------------------------------------------
# Training Configuration (Feedback Loop & Personalization Engine)
# ---------------------------------------------------------------------------
"""
Training configuration for user-specific emotion head training.

These constants control when training is triggered and how models are trained
on user feedback data.
"""

MIN_FEEDBACK_FOR_TRAINING = 20
"""
Minimum number of feedback samples required to trigger initial training.

Users must provide at least this many corrections before their first
personalized model is trained.
"""

INCREMENTAL_TRAINING_INTERVAL = 10
"""
Interval for incremental training after initial training.

After the first training at MIN_FEEDBACK_FOR_TRAINING samples,
subsequent training occurs every INCREMENTAL_TRAINING_INTERVAL samples.

Example: With MIN=20 and INTERVAL=10, training occurs at 20, 30, 40, 50, etc.
"""

TRAINING_EPOCHS = 20
"""
Number of training epochs for user head training.

Optimized for small datasets (20-100 samples) to prevent overfitting.
"""

TRAINING_BATCH_SIZE = 16
"""
Batch size for user head training.

If the dataset has fewer than 16 samples, the full dataset is used as one batch.
"""

TRAINING_LEARNING_RATE = 1e-3
"""
Learning rate for Adam optimizer during user head training.
"""

TRAINING_WEIGHT_DECAY = 1e-4
"""
Weight decay (L2 regularization) for Adam optimizer.

Helps prevent overfitting on small user feedback datasets.
"""

# Performance targets (for monitoring and testing)
FEEDBACK_RESPONSE_TIMEOUT_MS = 100
"""Target response time for feedback submission endpoint (milliseconds)."""

TRAINING_TIMEOUT_MINUTES = 5
"""Maximum allowed time for a training job to complete (minutes)."""

MAX_CONCURRENT_TRAINING_JOBS = 10
"""Maximum number of training jobs that can run simultaneously."""
