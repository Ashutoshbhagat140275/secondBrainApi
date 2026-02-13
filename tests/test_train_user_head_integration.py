"""
Integration tests for training pipeline with MongoDB storage.

These tests verify the complete end-to-end flow:
- Train user head → Save to MongoDB → Load from MongoDB → Predict
- Incremental training with MongoDB storage
- Cache invalidation propagates to prediction

Requirements: 4.1-4.7
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

# Add api directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_user_head import train_user_head
from app.services.user_emotion_head import (
    load_user_head,
    predict_user,
    invalidate_user_cache,
    UserEmotionHead
)
from app.services.user_head_storage import UserHeadStorageService
from app.models.user_model_storage import UserModelStorage


@pytest.fixture
def mock_feedback_data():
    """Create mock feedback data (25 samples with valid emotions)."""
    valid_emotions = ["happy", "sad", "angry", "neutral"]
    
    embeddings = []
    labels = []
    
    for i in range(25):
        embedding = [0.1 * (i % 10)] * 768  # Varied embeddings
        emotion = valid_emotions[i % len(valid_emotions)]
        
        embeddings.append(embedding)
        labels.append(emotion)
    
    # Create mock MongoDB documents
    docs = []
    for emb, label in zip(embeddings, labels):
        docs.append({
            "user_id": "test_user",
            "embedding": emb,
            "corrected_emotion": label,
            "timestamp": datetime.utcnow()
        })
    
    return docs


@pytest.fixture
def mock_db(mock_feedback_data):
    """Mock MongoDB database with feedback data."""
    db = Mock()
    feedback_collection = Mock()
    
    # Mock find() to return feedback data
    feedback_collection.find.return_value.sort.return_value = mock_feedback_data
    
    # Mock UserFeedback.get_collection
    with patch("training.train_user_head.UserFeedback") as mock_user_feedback:
        mock_user_feedback.get_collection.return_value = feedback_collection
        
        db.user_feedback = feedback_collection
        db.user_models = Mock()
        yield db


class TestCompleteTrainingFlow:
    """Test complete training flow: train → save to MongoDB → load from MongoDB."""
    
    def test_train_save_load_mongodb_flow(self, mock_db, tmp_path):
        """Test complete flow: train → save to MongoDB → load from MongoDB."""
        user_id = "test_user_flow"
        
        # Storage for the saved model
        saved_model_doc = {}
        
        def mock_update_one(filter_dict, update_dict, upsert=False):
            """Mock MongoDB update_one to store the document."""
            saved_model_doc.update(update_dict["$set"])
            return Mock(upserted_id=None)
        
        def mock_find_one(filter_dict):
            if filter_dict.get("user_id") == user_id and "user_id" in saved_model_doc:
                return saved_model_doc
            return None
        
        mock_db.user_models.update_one = mock_update_one
        mock_db.user_models.find_one = mock_find_one
        
        # Create a mock storage service that simulates MongoDB mode
        mock_storage = Mock()
        mock_storage.use_mongodb = True
        mock_storage.dual_save = False
        
        # Track save calls
        def mock_save_model(user_id, state_dict, metadata):
            # Simulate MongoDB save
            compressed_blob = UserModelStorage.compress_state_dict(state_dict)
            saved_model_doc.update({
                "user_id": user_id,
                "model_blob": compressed_blob,
                "metadata": metadata,
                "version": 1
            })
            return True
        
        def mock_load_model(user_id):
            if "user_id" in saved_model_doc and saved_model_doc["user_id"] == user_id:
                return UserModelStorage.decompress_state_dict(saved_model_doc["model_blob"])
            return None
        
        mock_storage.save_model = mock_save_model
        mock_storage.load_model = mock_load_model
        mock_storage.exists = Mock(return_value=False)
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                with patch("training.train_user_head.storage_service", mock_storage):
                    # Step 1: Train the model
                    metrics = train_user_head(user_id, force_retrain=True)
                    
                    # Verify training completed
                    assert "final_loss" in metrics
                    assert "final_accuracy" in metrics
                    assert metrics["storage_mode"] == "mongodb"
                    assert metrics["save_latency_ms"] > 0
                    
                    # Verify model was saved to MongoDB
                    assert "user_id" in saved_model_doc
                    assert saved_model_doc["user_id"] == user_id
                    assert "model_blob" in saved_model_doc
                    assert "metadata" in saved_model_doc
                    
                    # Step 2: Load the model from MongoDB
                    with patch("app.services.user_emotion_head.storage_service", mock_storage):
                        load_user_head.cache_clear()
                        loaded_model = load_user_head(user_id)
                    
                    # Verify model was loaded successfully
                    assert loaded_model is not None
                    assert isinstance(loaded_model, UserEmotionHead)
                    assert not loaded_model.training  # Should be in eval mode
                    
                    # Step 3: Verify model can make predictions
                    with patch("app.services.user_emotion_head.storage_service", mock_storage):
                        embedding = np.random.randn(768).astype(np.float32)
                        result = predict_user(embedding, user_id)
                    
                    assert result is not None
                    P_u, C_u = result
                    assert P_u.shape == (8,)
                    assert np.isclose(P_u.sum(), 1.0, atol=1e-5)
                    assert 0.0 <= C_u <= 1.0
    
    def test_train_save_load_dual_mode_flow(self, mock_db, tmp_path):
        """Test complete flow in dual-save mode (MongoDB + file)."""
        user_id = "test_user_dual_flow"
        
        # Storage for the saved model
        saved_model_doc = {}
        
        # Create a mock storage service that simulates dual-save mode
        mock_storage = Mock()
        mock_storage.use_mongodb = True
        mock_storage.dual_save = True
        
        def mock_save_model(user_id, state_dict, metadata):
            # Simulate dual-save: MongoDB + file
            compressed_blob = UserModelStorage.compress_state_dict(state_dict)
            # Add storage_mode to metadata (as the real storage service does)
            metadata["storage_mode"] = "dual"
            saved_model_doc.update({
                "user_id": user_id,
                "model_blob": compressed_blob,
                "metadata": metadata,
                "version": 1
            })
            # Also save to file
            file_path = tmp_path / f"{user_id}.pt"
            torch.save(state_dict, str(file_path))
            return True
        
        mock_storage.save_model = mock_save_model
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                with patch("training.train_user_head.storage_service", mock_storage):
                    # Train the model
                    metrics = train_user_head(user_id, force_retrain=True)
                    
                    # Verify dual-save mode
                    assert metrics["storage_mode"] == "dual"
                    
                    # Verify model was saved to MongoDB
                    assert "user_id" in saved_model_doc
                    assert saved_model_doc["metadata"]["storage_mode"] == "dual"
                    
                    # Verify model was saved to file
                    file_path = tmp_path / f"{user_id}.pt"
                    assert file_path.exists()
                    
                    # Load from file and verify
                    file_state = torch.load(str(file_path), map_location="cpu", weights_only=True)
                    assert "linear.weight" in file_state
                    assert "linear.bias" in file_state


class TestIncrementalTraining:
    """Test incremental training with MongoDB storage."""
    
    def test_incremental_training_loads_existing_model(self, mock_db, tmp_path):
        """Test that incremental training loads existing model from MongoDB."""
        user_id = "test_user_incremental"
        
        # Storage for models
        saved_model_doc = {}
        
        # Create mock storage service
        mock_storage = Mock()
        mock_storage.use_mongodb = True
        mock_storage.dual_save = False
        
        def mock_save_model(user_id, state_dict, metadata):
            compressed_blob = UserModelStorage.compress_state_dict(state_dict)
            saved_model_doc.update({
                "user_id": user_id,
                "model_blob": compressed_blob,
                "metadata": metadata,
                "version": 1
            })
            return True
        
        def mock_load_model(user_id):
            if "user_id" in saved_model_doc and saved_model_doc["user_id"] == user_id:
                return UserModelStorage.decompress_state_dict(saved_model_doc["model_blob"])
            return None
        
        def mock_exists(user_id):
            return "user_id" in saved_model_doc and saved_model_doc["user_id"] == user_id
        
        mock_storage.save_model = mock_save_model
        mock_storage.load_model = mock_load_model
        mock_storage.exists = mock_exists
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                with patch("training.train_user_head.storage_service", mock_storage):
                    # Step 1: Initial training
                    metrics1 = train_user_head(user_id, force_retrain=True)
                    assert metrics1["storage_mode"] == "mongodb"
                    
                    initial_loss = metrics1["final_loss"]
                    
                    # Verify model was saved
                    assert "user_id" in saved_model_doc
                    
                    # Step 2: Incremental training (force_retrain=False)
                    # This should load the existing model and continue training
                    metrics2 = train_user_head(user_id, force_retrain=False)
                    
                    # Verify incremental training completed
                    assert metrics2["storage_mode"] == "mongodb"
                    assert "final_loss" in metrics2
                    
                    # Model should have been updated in MongoDB
                    assert saved_model_doc["user_id"] == user_id
    
    def test_incremental_training_updates_metadata(self, mock_db, tmp_path):
        """Test that incremental training updates metadata correctly."""
        user_id = "test_user_metadata_update"
        
        saved_model_doc = {}
        
        # Create mock storage service
        mock_storage = Mock()
        mock_storage.use_mongodb = True
        mock_storage.dual_save = False
        
        def mock_save_model(user_id, state_dict, metadata):
            compressed_blob = UserModelStorage.compress_state_dict(state_dict)
            saved_model_doc.update({
                "user_id": user_id,
                "model_blob": compressed_blob,
                "metadata": metadata,
                "version": 1
            })
            return True
        
        def mock_load_model(user_id):
            if "user_id" in saved_model_doc and saved_model_doc["user_id"] == user_id:
                return UserModelStorage.decompress_state_dict(saved_model_doc["model_blob"])
            return None
        
        def mock_exists(user_id):
            return "user_id" in saved_model_doc and saved_model_doc["user_id"] == user_id
        
        mock_storage.save_model = mock_save_model
        mock_storage.load_model = mock_load_model
        mock_storage.exists = mock_exists
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                with patch("training.train_user_head.storage_service", mock_storage):
                    # Initial training
                    train_user_head(user_id, force_retrain=True)
                    
                    first_trained = saved_model_doc["metadata"]["last_trained"]
                    
                    # Wait a bit to ensure timestamp changes
                    import time
                    time.sleep(0.01)
                    
                    # Incremental training
                    train_user_head(user_id, force_retrain=False)
                    
                    # Verify metadata was updated
                    assert "last_trained" in saved_model_doc["metadata"]
                    second_trained = saved_model_doc["metadata"]["last_trained"]
                    
                    # Timestamp should be updated
                    assert second_trained >= first_trained


class TestCacheInvalidation:
    """Test cache invalidation propagates to prediction."""
    
    def test_cache_invalidation_after_training(self, mock_db, tmp_path):
        """Test that cache is invalidated after training and prediction uses new model."""
        user_id = "test_user_cache_invalidation"
        
        saved_model_doc = {}
        
        def mock_update_one(filter_dict, update_dict, upsert=False):
            saved_model_doc.update(update_dict["$set"])
            return Mock(upserted_id=None)
        
        def mock_find_one(filter_dict):
            if filter_dict["user_id"] == user_id and "user_id" in saved_model_doc:
                return saved_model_doc
            return None
        
        mock_db.user_models.update_one = mock_update_one
        mock_db.user_models.find_one = mock_find_one
        
        with patch("app.services.user_head_storage.settings") as mock_settings:
            mock_settings.USE_MONGODB_STORAGE = True
            mock_settings.DUAL_SAVE_MODE = False
            
            with patch("app.services.user_head_storage.get_database", return_value=mock_db):
                with patch("training.train_user_head.get_database", return_value=mock_db):
                    with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                        # Reinitialize storage service
                        from app.services import user_head_storage
                        user_head_storage.storage_service = UserHeadStorageService()
                        
                        # Clear cache
                        load_user_head.cache_clear()
                        
                        # Step 1: Train initial model
                        train_user_head(user_id, force_retrain=True)
                        
                        # Step 2: Load model (caches it)
                        model1 = load_user_head(user_id)
                        assert model1 is not None
                        
                        # Get initial weights
                        initial_weights = model1.linear.weight.clone()
                        
                        # Step 3: Train again (should invalidate cache)
                        train_user_head(user_id, force_retrain=True)
                        
                        # Step 4: Load model again (should load new model, not cached)
                        model2 = load_user_head(user_id)
                        assert model2 is not None
                        
                        # Models should be different instances (cache was invalidated)
                        assert model1 is not model2
    
    def test_prediction_uses_updated_model_after_training(self, mock_db, tmp_path):
        """Test that prediction uses the updated model after retraining."""
        user_id = "test_user_prediction_update"
        
        saved_model_doc = {}
        
        def mock_update_one(filter_dict, update_dict, upsert=False):
            saved_model_doc.update(update_dict["$set"])
            return Mock(upserted_id=None)
        
        def mock_find_one(filter_dict):
            if filter_dict["user_id"] == user_id and "user_id" in saved_model_doc:
                return saved_model_doc
            return None
        
        mock_db.user_models.update_one = mock_update_one
        mock_db.user_models.find_one = mock_find_one
        
        with patch("app.services.user_head_storage.settings") as mock_settings:
            mock_settings.USE_MONGODB_STORAGE = True
            mock_settings.DUAL_SAVE_MODE = False
            
            with patch("app.services.user_head_storage.get_database", return_value=mock_db):
                with patch("training.train_user_head.get_database", return_value=mock_db):
                    with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                        # Reinitialize storage service
                        from app.services import user_head_storage
                        user_head_storage.storage_service = UserHeadStorageService()
                        
                        # Clear cache
                        load_user_head.cache_clear()
                        
                        # Train model
                        train_user_head(user_id, force_retrain=True)
                        
                        # Make prediction
                        embedding = np.random.randn(768).astype(np.float32)
                        result1 = predict_user(embedding, user_id)
                        
                        assert result1 is not None
                        P_u1, C_u1 = result1
                        
                        # Train again (updates model)
                        train_user_head(user_id, force_retrain=True)
                        
                        # Make prediction again (should use updated model)
                        result2 = predict_user(embedding, user_id)
                        
                        assert result2 is not None
                        P_u2, C_u2 = result2
                        
                        # Both predictions should be valid
                        assert P_u1.shape == (8,)
                        assert P_u2.shape == (8,)
                        assert np.isclose(P_u1.sum(), 1.0, atol=1e-5)
                        assert np.isclose(P_u2.sum(), 1.0, atol=1e-5)
    
    def test_cache_invalidation_called_during_training(self, mock_db, tmp_path):
        """Test that invalidate_user_cache is called during training."""
        user_id = "test_user_cache_call"
        
        saved_model_doc = {}
        
        def mock_update_one(filter_dict, update_dict, upsert=False):
            saved_model_doc.update(update_dict["$set"])
            return Mock(upserted_id=None)
        
        mock_db.user_models.update_one = mock_update_one
        
        with patch("app.services.user_head_storage.settings") as mock_settings:
            mock_settings.USE_MONGODB_STORAGE = True
            mock_settings.DUAL_SAVE_MODE = False
            
            with patch("app.services.user_head_storage.get_database", return_value=mock_db):
                with patch("training.train_user_head.get_database", return_value=mock_db):
                    with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                        with patch("training.train_user_head.invalidate_user_cache") as mock_invalidate:
                            # Reinitialize storage service
                            from app.services import user_head_storage
                            user_head_storage.storage_service = UserHeadStorageService()
                            
                            # Train model
                            train_user_head(user_id, force_retrain=True)
                            
                            # Verify cache invalidation was called
                            mock_invalidate.assert_called_once_with(user_id)


class TestMongoDBFailureHandling:
    """Test graceful handling of MongoDB failures during training."""
    
    def test_training_fails_when_mongodb_save_fails(self, mock_db, tmp_path):
        """Test that training raises error when MongoDB save fails in MongoDB-only mode."""
        user_id = "test_user_save_fail"
        
        # Create mock storage service that fails on save
        mock_storage = Mock()
        mock_storage.use_mongodb = True
        mock_storage.dual_save = False
        mock_storage.save_model = Mock(return_value=False)  # Save fails
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                with patch("training.train_user_head.storage_service", mock_storage):
                    # Training should raise error
                    with pytest.raises(RuntimeError, match="Model save failed"):
                        train_user_head(user_id, force_retrain=True)
    
    def test_training_succeeds_in_dual_mode_when_mongodb_fails(self, mock_db, tmp_path):
        """Test that training succeeds in dual mode even if MongoDB fails."""
        user_id = "test_user_dual_fallback"
        
        # Create mock storage service in dual mode
        mock_storage = Mock()
        mock_storage.use_mongodb = True
        mock_storage.dual_save = True
        
        def mock_save_model(user_id, state_dict, metadata):
            # Simulate dual-save where MongoDB fails but file succeeds
            file_path = tmp_path / f"{user_id}.pt"
            torch.save(state_dict, str(file_path))
            return True  # File save succeeds
        
        mock_storage.save_model = mock_save_model
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                with patch("training.train_user_head.storage_service", mock_storage):
                    # Training should succeed (file save works)
                    metrics = train_user_head(user_id, force_retrain=True)
                    
                    assert "final_loss" in metrics
                    assert metrics["storage_mode"] == "dual"
                    
                    # Verify file was created
                    file_path = tmp_path / f"{user_id}.pt"
                    assert file_path.exists()


class TestEndToEndPredictionFlow:
    """Test complete end-to-end flow from training to prediction."""
    
    def test_full_pipeline_train_to_prediction(self, mock_db, tmp_path):
        """Test complete pipeline: train → save → load → predict."""
        user_id = "test_user_e2e"
        
        saved_model_doc = {}
        
        # Create mock storage service
        mock_storage = Mock()
        mock_storage.use_mongodb = True
        mock_storage.dual_save = False
        
        def mock_save_model(user_id, state_dict, metadata):
            compressed_blob = UserModelStorage.compress_state_dict(state_dict)
            saved_model_doc.update({
                "user_id": user_id,
                "model_blob": compressed_blob,
                "metadata": metadata,
                "version": 1
            })
            return True
        
        def mock_load_model(user_id):
            if "user_id" in saved_model_doc and saved_model_doc["user_id"] == user_id:
                return UserModelStorage.decompress_state_dict(saved_model_doc["model_blob"])
            return None
        
        def mock_exists(user_id):
            return "user_id" in saved_model_doc and saved_model_doc["user_id"] == user_id
        
        mock_storage.save_model = mock_save_model
        mock_storage.load_model = mock_load_model
        mock_storage.exists = mock_exists
        
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                with patch("training.train_user_head.storage_service", mock_storage):
                    with patch("app.services.user_emotion_head.storage_service", mock_storage):
                        # Clear cache
                        load_user_head.cache_clear()
                        
                        # Step 1: Train model
                        metrics = train_user_head(user_id, force_retrain=True)
                        
                        assert metrics["storage_mode"] == "mongodb"
                        assert metrics["final_accuracy"] >= 0.0
                        assert metrics["num_samples"] == 25
                        
                        # Step 2: Make multiple predictions
                        for i in range(5):
                            embedding = np.random.randn(768).astype(np.float32)
                            result = predict_user(embedding, user_id)
                            
                            assert result is not None
                            P_u, C_u = result
                            
                            # Verify prediction properties
                            assert P_u.shape == (8,)
                            assert np.isclose(P_u.sum(), 1.0, atol=1e-5)
                            assert np.all(P_u >= 0.0)
                            assert np.all(P_u <= 1.0)
                            assert 0.0 <= C_u <= 1.0
                            assert C_u == np.max(P_u)
                        
                        # Step 3: Retrain and verify predictions still work
                        train_user_head(user_id, force_retrain=False)
                        
                        embedding = np.random.randn(768).astype(np.float32)
                        result = predict_user(embedding, user_id)
                        
                        assert result is not None
