"""
Dataset preparation for RAVDESS and CREMA-D emotion speech corpora.

This script scans the dataset directories, parses filenames to extract
emotion labels, and produces a unified manifest (list of sample dicts)
that the feature extraction script consumes.

Expected directory layout
-------------------------
api/training/data/
  ravdess/
    Actor_01/
      03-01-01-01-01-01-01.wav
      ...
    Actor_02/
      ...
  crema-d/
    1001_DFA_ANG_XX.wav
    1001_DFA_DIS_XX.wav
    ...

Usage
-----
    python -m training.prepare_dataset          # from api/ directory
    python -m training.prepare_dataset --help
"""

import argparse
import json
import sys
import pathlib
from typing import List, Dict

# Ensure api/ is on sys.path so feature_config imports work
_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.feature_config import (
    RAVDESS_DIR,
    CREMAD_DIR,
    RAVDESS_CODE_TO_IDX,
    CREMAD_CODE_TO_IDX,
    EMOTION_LABELS,
    TRAINING_DATA_DIR,
)


# ── RAVDESS parser ────────────────────────────────────────────────────────


def parse_ravdess(root: pathlib.Path) -> List[Dict]:
    """
    Parse RAVDESS directory.

    Filename format:
      {modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{rep}-{actor}.wav

    Emotion codes (1-indexed): 01=neutral … 08=surprised.
    We only keep speech files (modality == 03).
    """
    samples: List[Dict] = []
    if not root.exists():
        print(f"[WARN] RAVDESS directory not found: {root}")
        return samples

    for wav in sorted(root.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        modality = int(parts[0])
        if modality != 3:  # 03 = audio-only speech
            continue
        emotion_code = int(parts[2])
        if emotion_code not in RAVDESS_CODE_TO_IDX:
            continue
        label_idx = RAVDESS_CODE_TO_IDX[emotion_code]
        samples.append(
            {
                "path": str(wav),
                "label_idx": label_idx,
                "label": EMOTION_LABELS[label_idx],
                "dataset": "ravdess",
                "actor": parts[6],
            }
        )
    return samples


# ── CREMA-D parser ────────────────────────────────────────────────────────


def parse_cremad(root: pathlib.Path) -> List[Dict]:
    """
    Parse CREMA-D directory.

    Filename format:  {ActorID}_{Sentence}_{Emotion}_{Level}.wav
    Emotion codes: ANG, DIS, FEA, HAP, NEU, SAD.
    """
    samples: List[Dict] = []
    if not root.exists():
        print(f"[WARN] CREMA-D directory not found: {root}")
        return samples

    for wav in sorted(root.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        if emotion_code not in CREMAD_CODE_TO_IDX:
            continue
        label_idx = CREMAD_CODE_TO_IDX[emotion_code]
        samples.append(
            {
                "path": str(wav),
                "label_idx": label_idx,
                "label": EMOTION_LABELS[label_idx],
                "dataset": "crema-d",
                "actor": parts[0],
            }
        )
    return samples


# ── main ──────────────────────────────────────────────────────────────────


def prepare_manifest(
    ravdess_dir: pathlib.Path = RAVDESS_DIR,
    cremad_dir: pathlib.Path = CREMAD_DIR,
) -> List[Dict]:
    """Return a combined samples manifest from available datasets."""
    samples: List[Dict] = []
    samples.extend(parse_ravdess(ravdess_dir))
    samples.extend(parse_cremad(cremad_dir))

    # Summary
    from collections import Counter

    dist = Counter(s["label"] for s in samples)
    print(f"\n{'='*50}")
    print(f"Total samples: {len(samples)}")
    print(f"Label distribution:")
    for label in EMOTION_LABELS:
        print(f"  {label:12s}: {dist.get(label, 0):5d}")
    print(f"{'='*50}\n")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare emotion dataset manifest")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(TRAINING_DATA_DIR / "manifest.json"),
        help="Output path for the JSON manifest file",
    )
    args = parser.parse_args()

    samples = prepare_manifest()
    if not samples:
        print("ERROR: No samples found. Ensure datasets are in:")
        print(f"  RAVDESS: {RAVDESS_DIR}")
        print(f"  CREMA-D: {CREMAD_DIR}")
        sys.exit(1)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Manifest saved to {out_path} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
