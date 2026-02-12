"""
Training script for the EmotionMLP classifier.

Loads cached features/labels produced by ``extract_features.py``,
performs a stratified train/val/test split, trains the MLP with
class-weighted cross-entropy and early stopping, and saves the best
checkpoint.

Usage
-----
    python -m training.train                     # from api/
    python -m training.train --epochs 100 --lr 1e-3 --batch 64
"""

import argparse
import sys
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

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
    SCALER_PATH,
)
from training.model import EmotionMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced datasets."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalise so mean ≈ 1
    return torch.from_numpy(weights)


def add_feature_noise(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Light Gaussian noise augmentation on the feature vector."""
    return x + sigma * torch.randn_like(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
    weight_decay: float = 1e-4,
    noise_sigma: float = 0.01,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── load data ─────────────────────────────────────────────────────
    feat_path = FEATURES_CACHE_DIR / "features.npy"
    lbl_path = FEATURES_CACHE_DIR / "labels.npy"
    if not feat_path.exists() or not lbl_path.exists():
        print(
            "ERROR: Cached features not found. Run `python -m training.extract_features` first."
        )
        sys.exit(1)

    X = np.load(str(feat_path)).astype(np.float32)  # (N, 419)
    y = np.load(str(lbl_path)).astype(np.int64)  # (N,)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {NUM_CLASSES} classes")

    assert (
        X.shape[1] == FEATURE_DIM
    ), f"Feature dim mismatch: {X.shape[1]} vs {FEATURE_DIM}"

    # ── normalise ─────────────────────────────────────────────────────
    scaler_path = SCALER_PATH
    if scaler_path.exists():
        scaler = joblib.load(str(scaler_path))
        print("Loaded existing scaler")
    else:
        scaler = StandardScaler()
        scaler.fit(X)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, str(scaler_path))
        print("Fitted and saved new scaler")
    X = scaler.transform(X).astype(np.float32)

    # ── stratified split ──────────────────────────────────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        stratify=y,
        random_state=seed,
    )
    val_fraction = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=val_fraction,
        stratify=y_tmp,
        random_state=seed,
    )
    print(f"Split — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    # Save test set for later evaluation
    np.save(str(FEATURES_CACHE_DIR / "X_test.npy"), X_test)
    np.save(str(FEATURES_CACHE_DIR / "y_test.npy"), y_test)

    # ── dataloaders ───────────────────────────────────────────────────
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── model / loss / optimiser ──────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionMLP(input_dim=FEATURE_DIM, num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # ── training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── train ─────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if noise_sigma > 0:
                xb = add_feature_noise(xb, noise_sigma)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── validate ──────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"({elapsed:.1f}s)"
        )

        # ── early stopping ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), str(MODEL_PATH))
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best val_acc={best_val_acc:.3f})"
                )
                break

    print(
        f"\nTraining complete. Best val_loss={best_val_loss:.4f}, "
        f"best val_acc={best_val_acc:.3f}"
    )
    print(f"Model saved to {MODEL_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train EmotionMLP classifier")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Gaussian noise σ for feature augmentation",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        patience=args.patience,
        noise_sigma=args.noise,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
