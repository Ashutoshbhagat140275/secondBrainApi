"""
Training script for the EmbeddingClassifier using Wav2Vec2 embeddings.

This script replaces the manual feature extraction pipeline with neural
embeddings from Facebook's pretrained Wav2Vec2-base model. It:
1. Loads RAVDESS and CREMA-D datasets
2. Extracts 768-dimensional embeddings (with caching to avoid repeated inference)
3. Trains a lightweight classifier head on the embeddings
4. Saves the trained model and scaler artifacts

Usage
-----
    python -m training.train_wav2vec2                # from api/
    python -m training.train_wav2vec2 --epochs 50 --lr 1e-3 --batch 64
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
from tqdm import tqdm

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.feature_config import (
    EMBEDDING_DIM,
    NUM_CLASSES,
    EMOTION_LABELS,
    FEATURES_CACHE_DIR,
    MODEL_DIR,
    EMBEDDING_CLASSIFIER_PATH,
    EMBEDDING_SCALER_PATH,
    GLOBAL_HEAD_PATH,
)
from app.services.emotion_classifier import EmbeddingClassifier
from app.services.global_emotion_head import GlobalEmotionHead
from app.services.wav2vec2_encoder import (
    load_wav2vec2_model,
    extract_wav2vec2_embedding,
)
from training.prepare_dataset import prepare_manifest


# ---------------------------------------------------------------------------
# Embedding extraction and caching
# ---------------------------------------------------------------------------


def extract_and_cache_embeddings(
    samples: list,
    cache_path: pathlib.Path,
    force_recompute: bool = False,
) -> tuple:
    """
    Extract Wav2Vec2 embeddings for all audio files and cache to disk.

    Parameters
    ----------
    samples : list
        List of sample dicts from prepare_manifest() with keys: path, label_idx
    cache_path : pathlib.Path
        Path to save/load cached embeddings (.npz file)
    force_recompute : bool
        If True, ignore existing cache and recompute embeddings

    Returns
    -------
    (embeddings, labels) : tuple
        embeddings: np.ndarray of shape (N, 768)
        labels: np.ndarray of shape (N,)
    """
    # Check if cache exists
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(str(cache_path))
        embeddings = data["embeddings"]
        labels = data["labels"]
        print(f"Loaded {len(embeddings)} cached embeddings")
        return embeddings, labels

    # Load Wav2Vec2 model
    print("Loading Wav2Vec2 model...")
    if not load_wav2vec2_model():
        raise RuntimeError("Failed to load Wav2Vec2 model. Cannot extract embeddings.")

    # Extract embeddings with progress bar
    print(f"Extracting embeddings for {len(samples)} audio files...")
    embeddings = []
    labels = []
    failed_count = 0

    for sample in tqdm(samples, desc="Extracting embeddings"):
        audio_path = sample["path"]
        label_idx = sample["label_idx"]

        try:
            embedding = extract_wav2vec2_embedding(audio_path)
            embeddings.append(embedding)
            labels.append(label_idx)
        except Exception as e:
            print(f"\nWARNING: Failed to extract embedding from {audio_path}: {e}")
            failed_count += 1
            continue

    if failed_count > 0:
        print(f"\nWarning: Failed to extract {failed_count} embeddings")

    if len(embeddings) == 0:
        raise RuntimeError("No embeddings were extracted successfully")

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Validate embeddings
    assert embeddings.shape[1] == EMBEDDING_DIM, (
        f"Expected embedding dim {EMBEDDING_DIM}, got {embeddings.shape[1]}"
    )
    if np.isnan(embeddings).any():
        print("WARNING: Some embeddings contain NaN values, replacing with zeros")
        embeddings = np.nan_to_num(embeddings, nan=0.0)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_path), embeddings=embeddings, labels=labels)
    print(f"Cached {len(embeddings)} embeddings to {cache_path}")

    return embeddings, labels


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced datasets."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalise so mean ≈ 1
    return torch.from_numpy(weights)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    force_recompute: bool = False,
    save_as_global_head: bool = False,
):
    """
    Train the EmbeddingClassifier on Wav2Vec2 embeddings.

    Parameters
    ----------
    epochs : int
        Maximum number of training epochs
    lr : float
        Learning rate for Adam optimizer
    batch_size : int
        Batch size for training
    patience : int
        Early stopping patience (epochs without improvement)
    weight_decay : float
        L2 regularization weight
    val_ratio : float
        Fraction of data for validation
    test_ratio : float
        Fraction of data for test set
    seed : int
        Random seed for reproducibility
    force_recompute : bool
        If True, recompute embeddings even if cache exists
    save_as_global_head : bool
        If True, train and save as global_emotion_head.pt with simplified architecture
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── load dataset manifest ─────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Loading dataset manifest")
    print("=" * 70)
    samples = prepare_manifest()
    if not samples:
        print("ERROR: No samples found. Ensure datasets are in:")
        print(f"  RAVDESS: {_api_root / 'training' / 'data' / 'ravdess'}")
        print(f"  CREMA-D: {_api_root / 'training' / 'data' / 'crema-d'}")
        sys.exit(1)

    # ── extract and cache embeddings ──────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Extracting Wav2Vec2 embeddings")
    print("=" * 70)
    cache_path = FEATURES_CACHE_DIR / "wav2vec2_embeddings.npz"
    X, y = extract_and_cache_embeddings(samples, cache_path, force_recompute)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} embedding dimensions, {NUM_CLASSES} classes")

    # ── normalize embeddings ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Normalizing embeddings")
    print("=" * 70)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    print(f"Fitted StandardScaler (mean={scaler.mean_[:5]}, std={scaler.scale_[:5]})")

    # ── stratified split ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Splitting dataset")
    print("=" * 70)
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
    FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(FEATURES_CACHE_DIR / "X_test_embeddings.npy"), X_test)
    np.save(str(FEATURES_CACHE_DIR / "y_test_embeddings.npy"), y_test)
    print(f"Saved test set to {FEATURES_CACHE_DIR}")

    # ── dataloaders ───────────────────────────────────────────────────
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── model / loss / optimizer ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Initializing model")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose model architecture based on flag
    if save_as_global_head:
        model = GlobalEmotionHead(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES).to(device)
        model_type = "GlobalEmotionHead (Linear 768→8)"
        save_path = GLOBAL_HEAD_PATH
        print("Training GLOBAL HEAD for dual-head emotion recognition system")
        print(f"Architecture: Simple linear classifier (768 → 8)")
    else:
        model = EmbeddingClassifier(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES).to(device)
        model_type = "EmbeddingClassifier"
        save_path = EMBEDDING_CLASSIFIER_PATH
        print("Training standard EmbeddingClassifier (backward compatibility mode)")
    
    print(f"Model type: {model_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Save path: {save_path}")

    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
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
    print("\n" + "=" * 70)
    print("STEP 6: Training")
    print("=" * 70)
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
            torch.save(model.state_dict(), str(save_path))
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best val_acc={best_val_acc:.3f})"
                )
                break

    # ── save artifacts ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7: Saving artifacts")
    print("=" * 70)
    joblib.dump(scaler, str(EMBEDDING_SCALER_PATH))
    print(f"Saved scaler to {EMBEDDING_SCALER_PATH}")
    print(f"Saved model to {save_path}")

    # ── final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model type: {model_type}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"\nModel artifacts:")
    print(f"  - Classifier: {save_path}")
    print(f"  - Scaler: {EMBEDDING_SCALER_PATH}")
    print(f"  - Test set: {FEATURES_CACHE_DIR / 'X_test_embeddings.npy'}")
    print(f"  - Embeddings cache: {cache_path}")
    if save_as_global_head:
        print(f"\n✓ Global head trained successfully for dual-head emotion recognition system")
        print(f"  This model will serve as the shared baseline classifier for all users.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train EmbeddingClassifier on Wav2Vec2 embeddings"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if cache exists",
    )
    parser.add_argument(
        "--save-as-global-head",
        action="store_true",
        help="Train and save as global_emotion_head.pt with simplified Linear(768→8) architecture for dual-head system",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        patience=args.patience,
        seed=args.seed,
        force_recompute=args.force_recompute,
        save_as_global_head=args.save_as_global_head,
    )


if __name__ == "__main__":
    main()
