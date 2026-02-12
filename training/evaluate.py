"""
Evaluation script for the trained EmotionMLP.

Loads the held-out test set (saved by ``train.py``), runs inference
with the best checkpoint, and prints accuracy, per-class metrics,
and a confusion matrix.

Usage
-----
    python -m training.evaluate               # from api/
"""

import sys
import pathlib
import numpy as np
import torch

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.feature_config import (
    FEATURE_DIM,
    NUM_CLASSES,
    EMOTION_LABELS,
    FEATURES_CACHE_DIR,
    MODEL_DIR,
    MODEL_PATH,
)
from training.model import EmotionMLP
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate():
    # ── load test data ────────────────────────────────────────────────
    X_test_path = FEATURES_CACHE_DIR / "X_test.npy"
    y_test_path = FEATURES_CACHE_DIR / "y_test.npy"

    if not X_test_path.exists() or not y_test_path.exists():
        print("ERROR: Test split not found. Run `python -m training.train` first.")
        sys.exit(1)

    X_test = np.load(str(X_test_path)).astype(np.float32)
    y_test = np.load(str(y_test_path)).astype(np.int64)
    print(f"Test set: {X_test.shape[0]} samples")

    # ── load model ────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    model = EmotionMLP(input_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
    )
    model.eval()

    # ── inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        preds = logits.argmax(dim=1).numpy()

    # ── metrics ───────────────────────────────────────────────────────
    # Filter to only labels that appear in test set
    present_labels = sorted(set(y_test.tolist()) | set(preds.tolist()))
    target_names = [EMOTION_LABELS[i] for i in present_labels]

    acc = accuracy_score(y_test, preds)
    print(f"\nOverall accuracy: {acc:.4f}  ({int(acc * len(y_test))}/{len(y_test)})\n")

    report = classification_report(
        y_test,
        preds,
        labels=present_labels,
        target_names=target_names,
        digits=3,
    )
    print("Classification Report")
    print("=" * 60)
    print(report)

    cm = confusion_matrix(y_test, preds, labels=present_labels)
    print("Confusion Matrix")
    print("=" * 60)
    # Header row
    header = "           " + "".join(f"{n[:6]:>8s}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:8d}" for v in row)
        print(f"{target_names[i]:>10s} {row_str}")
    print()

    # ── save report to file ───────────────────────────────────────────
    report_path = pathlib.Path(_api_root) / "training" / "evaluation_results.md"
    with open(report_path, "w") as f:
        f.write("# Emotion Classifier — Evaluation Results\n\n")
        f.write(f"**Test samples:** {len(y_test)}  \n")
        f.write(f"**Overall accuracy:** {acc:.4f}  \n\n")
        f.write("## Classification Report\n\n```\n")
        f.write(report)
        f.write("```\n\n## Confusion Matrix\n\n```\n")
        f.write(header + "\n")
        for i, row in enumerate(cm):
            row_str = "".join(f"{v:8d}" for v in row)
            f.write(f"{target_names[i]:>10s} {row_str}\n")
        f.write("```\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    evaluate()
