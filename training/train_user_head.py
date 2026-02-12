"""
User Head Training Script for Personalized Emotion Recognition

This script trains or updates user-specific emotion heads on feedback data.
It provides both initial training (when user reaches 20 feedback samples)
and incremental training (fine-tuning on new feedback while preserving
previous learning).

Key responsibilities:
- Load user feedback data from MongoDB
- Train UserEmotionHead on user-specific corrections
- Save trained models to models/user_heads/{user_id}.pt
- Log training metrics (loss, accuracy)
- Support both fresh training and incremental updates

Usage:
    # Train a specific user's model
    python -m training.train_user_head --user-id <user_id>
    
    # Force retrain from scratch
    python -m training.train_user_head --user-id <user_id> --force-retrain
    
    # Incremental training (load existing weights)
    python -m training.train_user_head --user-id <user_id> --incremental

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.2, 7.3, 7.4, 7.5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add api directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.mongodb import get_database
from app.models.feedback import UserFeedback
from app.services.user_emotion_head import UserEmotionHead, create_fresh_user_head, USER_HEADS_DIR
from app.services.feature_config import EMBEDDING_DIM, NUM_CLASSES, EMOTION_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_user_feedback(user_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all feedback data for a user from MongoDB.
    
    Queries the UserFeedback collection and extracts embeddings and corrected
    emotion labels for training.
    
    Parameters:
        user_id: User identifier
    
    Returns:
        Tuple of (X, y) where:
            - X: Embeddings array of shape (n_samples, 768)
            - y: Label indices array of shape (n_samples,)
    
    Raises:
        ValueError: If user has no feedback data
    
    Requirements: 6.2
    """
    db = get_database()
    feedback_collection = UserFeedback.get_collection(db)
    
    # Query all feedback for this user, sorted by timestamp
    feedback_docs = list(
        feedback_collection.find({"user_id": user_id}).sort("timestamp", 1)
    )
    
    if not feedback_docs:
        raise ValueError(f"No feedback data found for user {user_id}")
    
    logger.info(f"Loaded {len(feedback_docs)} feedback samples for user {user_id}")
    
    # Extract embeddings and labels
    embeddings = []
    labels = []
    
    for doc in feedback_docs:
        embedding = doc["embedding"]
        corrected_emotion = doc["corrected_emotion"]
        
        # Validate embedding dimension
        if len(embedding) != EMBEDDING_DIM:
            logger.warning(
                f"Skipping feedback with invalid embedding dimension: "
                f"{len(embedding)} (expected {EMBEDDING_DIM})"
            )
            continue
        
        # Convert emotion label to index
        if corrected_emotion not in EMOTION_LABELS:
            logger.warning(
                f"Skipping feedback with invalid emotion label: {corrected_emotion}"
            )
            continue
        
        label_idx = EMOTION_LABELS.index(corrected_emotion)
        
        embeddings.append(embedding)
        labels.append(label_idx)
    
    if not embeddings:
        raise ValueError(f"No valid feedback data found for user {user_id}")
    
    # Convert to numpy arrays
    X = np.array(embeddings, dtype=np.float32)  # (n_samples, 768)
    y = np.array(labels, dtype=np.int64)  # (n_samples,)
    
    logger.info(
        f"Prepared training data: X.shape={X.shape}, y.shape={y.shape}, "
        f"unique_labels={np.unique(y).tolist()}"
    )
    
    return X, y


def train_user_head(user_id: str, force_retrain: bool = False) -> dict:
    """
    Train user-specific emotion head on feedback data.
    
    This function performs the complete training pipeline:
    1. Load user feedback data from MongoDB
    2. Validate minimum sample requirement (20 samples)
    3. Create or load existing UserEmotionHead
    4. Train for 20 epochs with Adam optimizer
    5. Save trained model to models/user_heads/{user_id}.pt
    6. Log training metrics
    
    Parameters:
        user_id: User identifier
        force_retrain: If True, train from scratch even if model exists.
                      If False, load existing weights for incremental training.
    
    Returns:
        Dictionary containing training metrics:
            - final_loss: Final training loss
            - final_accuracy: Final training accuracy
            - num_samples: Number of training samples
            - num_epochs: Number of training epochs
            - model_path: Path to saved model
    
    Raises:
        ValueError: If user has insufficient feedback data (< 20 samples)
    
    Training Details:
        - Loss: CrossEntropyLoss
        - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
        - Epochs: 20
        - Batch size: 16 (or full batch if < 16 samples)
        - No validation split (small dataset)
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
    """
    logger.info(f"Starting training for user {user_id} (force_retrain={force_retrain})")
    
    # Load feedback data
    X, y = load_user_feedback(user_id)
    num_samples = len(X)
    
    # Validate minimum sample requirement
    if num_samples < 20:
        raise ValueError(
            f"Insufficient feedback data for user {user_id}: "
            f"{num_samples} samples (minimum 20 required)"
        )
    
    logger.info(f"Training with {num_samples} feedback samples")
    
    # Create or load model
    USER_HEADS_DIR.mkdir(parents=True, exist_ok=True)
    user_model_path = USER_HEADS_DIR / f"{user_id}.pt"
    
    if force_retrain or not user_model_path.exists():
        logger.info("Creating fresh user emotion head")
        model = create_fresh_user_head()
    else:
        logger.info(f"Loading existing user emotion head from {user_model_path}")
        model = UserEmotionHead(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
        state = torch.load(str(user_model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    
    model.train()
    
    # Prepare data loader
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Use batch size of 16, or full batch if fewer samples
    batch_size = min(16, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 20
    
    logger.info(
        f"Training configuration: epochs={num_epochs}, batch_size={batch_size}, "
        f"lr=1e-3, weight_decay=1e-4"
    )
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            # Forward pass
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item() * len(batch_X)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += len(batch_X)
        
        # Compute epoch metrics
        avg_loss = epoch_loss / total
        accuracy = correct / total
        
        # Log every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"loss={avg_loss:.4f}, accuracy={accuracy:.4f}"
            )
    
    # Save trained model
    torch.save(model.state_dict(), str(user_model_path))
    logger.info(f"Model saved to {user_model_path}")
    
    # Return training metrics
    return {
        "final_loss": avg_loss,
        "final_accuracy": accuracy,
        "num_samples": num_samples,
        "num_epochs": num_epochs,
        "model_path": str(user_model_path)
    }


def incremental_train_user_head(user_id: str) -> dict:
    """
    Update existing user head with new feedback (incremental training).
    
    This function loads the existing user head weights and fine-tunes on all
    feedback data. Training on the full feedback history prevents catastrophic
    forgetting while adapting to new corrections.
    
    This is equivalent to calling train_user_head(user_id, force_retrain=False).
    
    Parameters:
        user_id: User identifier
    
    Returns:
        Dictionary containing training metrics (same as train_user_head)
    
    Raises:
        ValueError: If user has no existing model or insufficient feedback
    
    Requirements: 7.1, 7.2, 7.3, 7.4
    """
    logger.info(f"Starting incremental training for user {user_id}")
    
    # Verify existing model exists
    user_model_path = USER_HEADS_DIR / f"{user_id}.pt"
    if not user_model_path.exists():
        raise ValueError(
            f"No existing model found for user {user_id}. "
            f"Use train_user_head() for initial training."
        )
    
    # Train with existing weights (force_retrain=False)
    return train_user_head(user_id, force_retrain=False)


def main():
    """
    CLI interface for user head training.
    
    Examples:
        # Initial training
        python -m training.train_user_head --user-id user_123
        
        # Force retrain from scratch
        python -m training.train_user_head --user-id user_123 --force-retrain
        
        # Incremental training
        python -m training.train_user_head --user-id user_123 --incremental
    
    Requirements: 6.1, 7.1
    """
    parser = argparse.ArgumentParser(
        description="Train user-specific emotion head on feedback data"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        required=True,
        help="User identifier (MongoDB ObjectId as string)"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Train from scratch, ignoring existing model weights"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental training (load existing weights and fine-tune)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.incremental:
            metrics = incremental_train_user_head(args.user_id)
        else:
            metrics = train_user_head(args.user_id, force_retrain=args.force_retrain)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
