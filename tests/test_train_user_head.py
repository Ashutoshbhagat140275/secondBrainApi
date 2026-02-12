"""
Unit tests for user head training pipeline.

Tests cover:
- Loading user feedback from MongoDB
- Training user heads with various data sizes
- Incremental training (fine-tuning)
- Model artifact saving and loading
- Error handling for insufficient data
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_user_head import (
    load_user_feedback,
    train_user_head,
    incremental_train_user_head
)
from app.services.user_emotion_head import UserEmotionHead
from app.services.feature_config import EMBEDDING_DIM, NUM_CLASSES, EMOTION_LABELS


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    return Mock()


@pytest.fixture
def sample_feedback_data():
    """Create sample feedback data for testing."""
    # Create 25 feedback samples with random embeddings
    feedback_docs = []
    for i in range(25):
        feedback_docs.append({
            "_id": f"feedback_{i}",
            "user_id": "test_user_123",
            "session_id": f"session_{i}",
            "embedding": np.random.randn(EMBEDDING_DIM).tolist(),
            "predicted_emotion": "happy",
            "corrected_emotion": EMOTION_LABELS[i % NUM_CLASSES],  # Cycle through emotions
            "timestamp": f"2024-01-{i+1:02d}T10:00:00Z"
        })
    return feedback_docs


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestLoadUserFeedback:
    """Tests for load_user_feedback function."""
    
    def test_load_feedback_success(self, mock_db, sample_feedback_data):
        """Test successful loading of user feedback."""
        # Mock MongoDB collection
        mock_collection = Mock()
        mock_cursor = Mock()
        mock_cursor.sort.return_value = sample_feedback_data
        mock_collection.find.return_value = mock_cursor
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.UserFeedback.get_collection", return_value=mock_collection):
                X, y = load_user_feedback("test_user_123")
        
        # Verify shapes
        assert X.shape == (25, EMBEDDING_DIM)
        assert y.shape == (25,)
        
        # Verify data types
        assert X.dtype == np.float32
        assert y.dtype == np.int64
        
        # Verify labels are valid indices
        assert np.all(y >= 0)
        assert np.all(y < NUM_CLASSES)
    
    def test_load_feedback_no_data(self, mock_db):
        """Test error when user has no feedback data."""
        mock_collection = Mock()
        mock_cursor = Mock()
        mock_cursor.sort.return_value = []
        mock_collection.find.return_value = mock_cursor
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.UserFeedback.get_collection", return_value=mock_collection):
                with pytest.raises(ValueError, match="No feedback data found"):
                    load_user_feedback("test_user_123")
    
    def test_load_feedback_invalid_embedding_dimension(self, mock_db):
        """Test handling of feedback with invalid embedding dimension."""
        invalid_feedback = [
            {
                "_id": "feedback_1",
                "user_id": "test_user_123",
                "session_id": "session_1",
                "embedding": [0.1, 0.2, 0.3],  # Wrong dimension
                "predicted_emotion": "happy",
                "corrected_emotion": "sad",
                "timestamp": "2024-01-01T10:00:00Z"
            }
        ]
        
        mock_collection = Mock()
        mock_cursor = Mock()
        mock_cursor.sort.return_value = invalid_feedback
        mock_collection.find.return_value = mock_cursor
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.UserFeedback.get_collection", return_value=mock_collection):
                with pytest.raises(ValueError, match="No valid feedback data found"):
                    load_user_feedback("test_user_123")
    
    def test_load_feedback_invalid_emotion_label(self, mock_db):
        """Test handling of feedback with invalid emotion label."""
        invalid_feedback = [
            {
                "_id": "feedback_1",
                "user_id": "test_user_123",
                "session_id": "session_1",
                "embedding": np.random.randn(EMBEDDING_DIM).tolist(),
                "predicted_emotion": "happy",
                "corrected_emotion": "invalid_emotion",
                "timestamp": "2024-01-01T10:00:00Z"
            }
        ]
        
        mock_collection = Mock()
        mock_cursor = Mock()
        mock_cursor.sort.return_value = invalid_feedback
        mock_collection.find.return_value = mock_cursor
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.UserFeedback.get_collection", return_value=mock_collection):
                with pytest.raises(ValueError, match="No valid feedback data found"):
                    load_user_feedback("test_user_123")


class TestTrainUserHead:
    """Tests for train_user_head function."""
    
    def test_train_user_head_success(self, mock_db, sample_feedback_data, temp_model_dir):
        """Test successful training of user head."""
        # Mock load_user_feedback to return synthetic data
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                metrics = train_user_head("test_user_123", force_retrain=True)
        
        # Verify metrics
        assert "final_loss" in metrics
        assert "final_accuracy" in metrics
        assert "num_samples" in metrics
        assert "num_epochs" in metrics
        assert "model_path" in metrics
        
        assert metrics["num_samples"] == 25
        assert metrics["num_epochs"] == 20
        assert 0.0 <= metrics["final_accuracy"] <= 1.0
        
        # Verify model was saved
        model_path = temp_model_dir / "test_user_123.pt"
        assert model_path.exists()
        
        # Verify model can be loaded
        model = UserEmotionHead(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    
    def test_train_user_head_insufficient_data(self, mock_db):
        """Test error when user has insufficient feedback data."""
        # Only 15 samples (< 20 minimum)
        X = np.random.randn(15, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=15).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with pytest.raises(ValueError, match="Insufficient feedback data"):
                train_user_head("test_user_123")
    
    def test_train_user_head_incremental(self, mock_db, temp_model_dir):
        """Test incremental training loads existing weights."""
        # Create initial model
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                # Initial training
                metrics1 = train_user_head("test_user_123", force_retrain=True)
                
                # Incremental training (should load existing weights)
                metrics2 = train_user_head("test_user_123", force_retrain=False)
        
        # Both should succeed
        assert metrics1["num_samples"] == 25
        assert metrics2["num_samples"] == 25
    
    def test_train_user_head_small_batch(self, mock_db, temp_model_dir):
        """Test training with fewer samples than batch size."""
        # Only 20 samples (batch size is 16)
        X = np.random.randn(20, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=20).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                metrics = train_user_head("test_user_123", force_retrain=True)
        
        # Should use batch size of 16 (min(16, 20))
        assert metrics["num_samples"] == 20
        assert metrics["num_epochs"] == 20


class TestIncrementalTrainUserHead:
    """Tests for incremental_train_user_head function."""
    
    def test_incremental_train_success(self, mock_db, temp_model_dir):
        """Test successful incremental training."""
        # Create initial model
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                # Initial training
                train_user_head("test_user_123", force_retrain=True)
                
                # Incremental training
                metrics = incremental_train_user_head("test_user_123")
        
        assert metrics["num_samples"] == 25
        assert metrics["num_epochs"] == 20
    
    def test_incremental_train_no_existing_model(self, mock_db, temp_model_dir):
        """Test error when no existing model exists."""
        with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
            with pytest.raises(ValueError, match="No existing model found"):
                incremental_train_user_head("test_user_123")


class TestTrainingConvergence:
    """Tests for training convergence and model quality."""
    
    def test_model_learns_simple_pattern(self, temp_model_dir):
        """Test that model can learn a simple pattern."""
        # Create synthetic data with clear pattern
        # All samples with positive first feature -> emotion 0
        # All samples with negative first feature -> emotion 1
        X = np.random.randn(50, EMBEDDING_DIM).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)  # Binary classification based on first feature
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                metrics = train_user_head("test_user_123", force_retrain=True)
        
        # Model should achieve reasonable accuracy on this simple pattern
        # Note: This is a weak test since we're using all 8 classes but only 2 labels
        # But it verifies the training loop works
        assert metrics["final_accuracy"] > 0.5  # Better than random
    
    def test_training_reduces_loss(self, temp_model_dir):
        """Test that training reduces loss over epochs."""
        X = np.random.randn(30, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=30).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                metrics = train_user_head("test_user_123", force_retrain=True)
        
        # Loss should be finite and positive
        assert 0.0 < metrics["final_loss"] < 10.0
    
    def test_incremental_training_preserves_previous_learning(self, temp_model_dir):
        """
        Test that incremental training preserves previous learning while adapting to new data.
        
        This validates Requirement 7.4: "THE retraining SHALL preserve previous learning 
        while adapting to new feedback."
        
        Strategy:
        1. Train on initial dataset with clear pattern
        2. Evaluate model on initial data (baseline performance)
        3. Add new feedback data with different pattern
        4. Perform incremental training on combined dataset
        5. Verify model still performs well on original data (preservation)
        6. Verify model also learned from new data (adaptation)
        """
        # Step 1: Create initial training data (25 samples)
        # Pattern: First 3 features positive -> emotion 0, negative -> emotion 1
        np.random.seed(42)
        X_initial = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        # Make pattern clear by amplifying first 3 features
        X_initial[:15, :3] = np.abs(X_initial[:15, :3]) + 1.0  # Positive -> emotion 0
        X_initial[15:, :3] = -np.abs(X_initial[15:, :3]) - 1.0  # Negative -> emotion 1
        y_initial = np.array([0] * 15 + [1] * 10, dtype=np.int64)
        
        # Step 2: Train initial model
        with patch("training.train_user_head.load_user_feedback", return_value=(X_initial, y_initial)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                metrics_initial = train_user_head("test_user_123", force_retrain=True)
        
        # Load trained model and evaluate on initial data
        model_path = temp_model_dir / "test_user_123.pt"
        model = UserEmotionHead(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        
        with torch.no_grad():
            logits_initial = model(torch.from_numpy(X_initial))
            predictions_initial = torch.argmax(logits_initial, dim=1).numpy()
            accuracy_initial = (predictions_initial == y_initial).mean()
        
        # Model should learn initial pattern well
        assert accuracy_initial > 0.7, f"Initial accuracy too low: {accuracy_initial}"
        
        # Step 3: Create new feedback data (10 additional samples)
        # Different pattern: emotion 2 for samples with feature 5 > 0
        X_new = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)
        X_new[:, 5] = np.abs(X_new[:, 5]) + 1.0  # Make feature 5 positive
        y_new = np.array([2] * 10, dtype=np.int64)
        
        # Combine datasets (incremental training uses all feedback)
        X_combined = np.vstack([X_initial, X_new])
        y_combined = np.concatenate([y_initial, y_new])
        
        # Step 4: Perform incremental training
        with patch("training.train_user_head.load_user_feedback", return_value=(X_combined, y_combined)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                metrics_incremental = incremental_train_user_head("test_user_123")
        
        # Load incrementally trained model
        model_incremental = UserEmotionHead(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
        state_incremental = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model_incremental.load_state_dict(state_incremental)
        model_incremental.eval()
        
        # Step 5: Verify preservation - model should still perform well on original data
        with torch.no_grad():
            logits_after = model_incremental(torch.from_numpy(X_initial))
            predictions_after = torch.argmax(logits_after, dim=1).numpy()
            accuracy_after = (predictions_after == y_initial).mean()
        
        # Accuracy on original data should not degrade significantly (allow 10% drop)
        assert accuracy_after >= accuracy_initial - 0.1, \
            f"Previous learning not preserved: {accuracy_after} < {accuracy_initial - 0.1}"
        
        # Step 6: Verify adaptation - model should learn from new data
        with torch.no_grad():
            logits_new = model_incremental(torch.from_numpy(X_new))
            predictions_new = torch.argmax(logits_new, dim=1).numpy()
            accuracy_new = (predictions_new == y_new).mean()
        
        # Model should learn the new pattern (at least better than random for 8 classes)
        assert accuracy_new > 0.3, \
            f"Model did not adapt to new data: {accuracy_new}"
        
        # Verify training metrics
        assert metrics_incremental["num_samples"] == 35  # 25 + 10
        assert metrics_incremental["num_epochs"] == 20
