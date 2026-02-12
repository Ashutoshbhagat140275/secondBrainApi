"""
Batch feature extraction for training.

Reads the manifest produced by ``prepare_dataset.py``, runs the same
``extract_full_features()`` pipeline used at inference time, and saves
the resulting arrays to disk:

  training/cache/features.npy   — (N, 419) float32
  training/cache/labels.npy     — (N,)     int64

It also fits a ``StandardScaler`` on the entire feature matrix and saves
it so that inference uses the same normalisation.

Usage
-----
    python -m training.extract_features                 # from api/
    python -m training.extract_features --manifest path/to/manifest.json
"""

import argparse
import json
import sys
import pathlib
import time
import numpy as np
import joblib
from typing import List, Dict

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.feature_extractor import extract_full_features
from app.services.feature_config import (
    FEATURE_DIM,
    FEATURES_CACHE_DIR,
    MODEL_DIR,
    SCALER_PATH,
    TRAINING_DATA_DIR,
)


def _preprocess_for_training(audio_path: str) -> str:
    """
    Lightweight in-place preprocessing that mirrors ``audio_processor.preprocess_audio``
    but works synchronously (no async) for batch usage.

    Writes a temp preprocessed file next to the original.
    Returns the path to the preprocessed file.
    """
    import librosa
    import soundfile as sf

    y, sr = librosa.load(audio_path, sr=16000)

    # VAD
    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals) > 0:
        y = np.concatenate([y[s:e] for s, e in intervals])

    # Peak normalise
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Write to a temporary file
    tmp_path = str(pathlib.Path(audio_path).with_suffix(".preprocessed.wav"))
    sf.write(tmp_path, y, 16000)
    return tmp_path


def extract_all(manifest: List[Dict], use_preprocessing: bool = True):
    """
    Extract features for every sample in *manifest*.

    Parameters
    ----------
    manifest : list of dict
        Each dict must have "path" (str) and "label_idx" (int).
    use_preprocessing : bool
        If True, apply VAD + normalisation (recommended for raw datasets).

    Returns
    -------
    features : np.ndarray, shape (N, FEATURE_DIM)
    labels   : np.ndarray, shape (N,)
    """
    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    skipped = 0
    t0 = time.time()

    for i, sample in enumerate(manifest):
        audio_path = sample["path"]
        label_idx = sample["label_idx"]

        try:
            if use_preprocessing:
                proc_path = _preprocess_for_training(audio_path)
            else:
                proc_path = audio_path

            feat = extract_full_features(proc_path)  # 419-dim
            features_list.append(feat)
            labels_list.append(label_idx)

            # Clean up temp file
            if use_preprocessing:
                tmp = pathlib.Path(proc_path)
                if tmp.exists() and tmp.name.endswith(".preprocessed.wav"):
                    tmp.unlink()

        except Exception as e:
            skipped += 1
            if skipped <= 20:  # Don't spam the console
                print(f"  [SKIP] {audio_path}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == len(manifest):
            elapsed = time.time() - t0
            per_sec = (i + 1 - skipped) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i+1}/{len(manifest)}]  ok={i+1-skipped}  "
                f"skip={skipped}  {per_sec:.1f} samples/s"
            )

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    print(
        f"\nExtraction complete: {features.shape[0]} samples, "
        f"{skipped} skipped, {time.time()-t0:.1f}s total"
    )
    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Extract features from audio dataset")
    parser.add_argument(
        "--manifest",
        "-m",
        type=str,
        default=str(TRAINING_DATA_DIR / "manifest.json"),
        help="Path to the JSON manifest from prepare_dataset.py",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip VAD/normalisation (if data is already preprocessed)",
    )
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        print("Run `python -m training.prepare_dataset` first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Loaded manifest with {len(manifest)} samples")

    features, labels = extract_all(manifest, use_preprocessing=not args.no_preprocess)

    # ── save features & labels ────────────────────────────────────────
    FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feat_path = FEATURES_CACHE_DIR / "features.npy"
    lbl_path = FEATURES_CACHE_DIR / "labels.npy"
    np.save(str(feat_path), features)
    np.save(str(lbl_path), labels)
    print(f"Saved features → {feat_path}  ({features.shape})")
    print(f"Saved labels   → {lbl_path}   ({labels.shape})")

    # ── fit & save scaler ─────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(features)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, str(SCALER_PATH))
    print(f"Saved scaler   → {SCALER_PATH}")


if __name__ == "__main__":
    main()
