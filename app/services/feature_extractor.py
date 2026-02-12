"""
Unified audio feature extraction pipeline.

This module extracts a fixed-length feature vector from a preprocessed audio
file.  The *exact same function* is used for both training and inference,
guaranteeing feature parity.

Feature vector layout (419 dims):
  - 13 MFCCs        × 8 functionals = 104
  - 13 Δ MFCCs      × 8 functionals = 104
  - 13 Δ² MFCCs     × 8 functionals = 104
  -  1 RMS           × 8 functionals =   8
  -  1 F0 (pitch)    × 8 functionals =   8
  -  1 sp. centroid  × 8 functionals =   8
  -  1 sp. bandwidth × 8 functionals =   8
  -  1 ZCR           × 8 functionals =   8   (NEW)
  -  1 sp. rolloff   × 8 functionals =   8   (NEW)
  -  7 sp. contrast  × 8 functionals =  56   (NEW)
  -  jitter, shimmer, HNR             =   3
                              TOTAL  = 419
"""

import numpy as np
import librosa
import parselmouth
import scipy.stats
import logging
from typing import List

from app.services.feature_config import (
    SAMPLE_RATE,
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
    N_CONTRAST_BANDS,
    FEATURE_DIM,
)

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────


def _compute_functionals(x: np.ndarray) -> List[float]:
    """
    Summarise a 1-D time-series into 8 statistical descriptors:
    [mean, std, min, max, p10, p50, p90, linear-slope].
    NaN values are dropped before computation.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return [0.0] * 8
    slope = scipy.stats.linregress(np.arange(len(x)), x).slope if len(x) > 1 else 0.0
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.percentile(x, 10)),
        float(np.percentile(x, 50)),
        float(np.percentile(x, 90)),
        float(slope),
    ]


def _extract_voice_quality(audio_path: str):
    """
    Extract jitter (local), shimmer (local), and HNR via Praat / parselmouth.
    Returns (jitter, shimmer, hnr) as floats.  Falls back to 0.0 on error.
    """
    try:
        snd = parselmouth.Sound(audio_path)
        if snd.duration < 0.3:
            logger.warning("Audio < 0.3 s — voice quality features default to 0.")
            return 0.0, 0.0, 0.0

        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

        jitter = parselmouth.praat.call(
            pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer = parselmouth.praat.call(
            [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        harmonicity = snd.to_harmonicity_cc(
            time_step=0.01,
            minimum_pitch=75,
            silence_threshold=0.1,
            periods_per_window=1.0,
        )
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

        return float(jitter), float(shimmer), float(hnr)
    except Exception as e:
        logger.error(f"Voice quality extraction failed: {e}")
        return 0.0, 0.0, 0.0


# ── main extraction function ──────────────────────────────────────────────


def extract_full_features(audio_path: str) -> List[float]:
    """
    Extract the complete 419-dim feature vector from a *preprocessed* audio
    file (16 kHz, VAD-trimmed, peak-normalised).

    Parameters
    ----------
    audio_path : str
        Path to the preprocessed .wav file on disk.

    Returns
    -------
    List[float]
        Feature vector of length ``FEATURE_DIM`` (419).
    """
    # Load at the canonical sample rate
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # ── spectral / temporal features ──────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )

    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # ── NEW features ──────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=0.85, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_bands=N_CONTRAST_BANDS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # contrast shape: (n_bands + 1, T) = (7, T)

    # ── voice quality ─────────────────────────────────────────────────
    jitter, shimmer, hnr = _extract_voice_quality(audio_path)

    # ── aggregate into fixed-length vector ────────────────────────────
    parts: List[float] = []

    # Multi-row features: apply functionals per row, then flatten
    for matrix in (mfcc, delta, delta2):
        for row in matrix:
            parts.extend(_compute_functionals(row))

    # Single-row features
    for single in (rms, np.atleast_2d(f0), centroid, bandwidth, zcr, rolloff):
        parts.extend(_compute_functionals(single.ravel()))

    # Contrast bands (7 rows)
    for row in contrast:
        parts.extend(_compute_functionals(row))

    # Scalar voice-quality features
    parts.extend([jitter, shimmer, hnr])

    # Ensure correct dimensionality
    assert (
        len(parts) == FEATURE_DIM
    ), f"Feature vector length mismatch: expected {FEATURE_DIM}, got {len(parts)}"

    return [float(x) for x in parts]
