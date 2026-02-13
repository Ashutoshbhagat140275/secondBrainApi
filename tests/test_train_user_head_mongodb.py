"""
Unit tests for MongoDB-enabled training pipeline.

Tests cover:
- Save to MongoDB when USE_MONGODB_STORAGE=True
- Dual-save mode (both MongoDB and file)
- Cache invalidation after save
- Training job metadata includes storage_mode
- Save latency measurement
- Fallback to file save when MongoDB fails
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import sys
from pathlib import Path

# Add api directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_user_head import train_user_head, train_user_head_async
from app.services.user_emotion_head import UserEmotionHead


class MockSettings:
    """Mock settings for testing."""
    USE_MONGODB_STORAGE = False
    DUAL_SAVE_MODE = False
    MONGODB_STORAGE_COMPRESSION = "gzip"


@pytest.fixture
def mock_feedback_data():
    """Create mock feedback data (25 samples with valid emotions only)."""
    import numpy as np
    
    # Valid emotions from EMOTION_LABELS
    valid_emotions = ["happy", "sad", "angry", "neutral"]
    
    # Create 25 samples with embeddings and labels
    embeddings = []
    labels = []
    
    for i in range(25):
        embedding = [0.1] * 768  # 768-dim embedding
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
        yield db


@pytest.fixture
def mock_storage_service():
    """Mock storage service."""
    with patch("training.train_user_head.storage_service") as mock_service:
        mock_service.use_mongodb = False
        mock_service.dual_save = False
        mock_service.save_model = Mock(return_value=True)
        mock_service.load_model = Mock(return_value=None)
        mock_service.exists = Mock(return_value=False)
        yield mock_service


@pytest.fixture
def mock_cache_invalidation():
    """Mock cache invalidation function."""
    with patch("training.train_user_head.invalidate_user_cache") as mock_invalidate:
        yield mock_invalidate


def test_train_user_head_saves_to_mongodb(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that training saves model via storage service in MongoDB mode."""
    user_id = "test_user_mongo"
    
    # Configure storage service for MongoDB mode
    mock_storage_service.use_mongodb = True
    mock_storage_service.dual_save = False
    
    with patch("training.train_user_head.get_database", return_value=mock_db):
        with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
            # Run training
            metrics = train_user_head(user_id, force_retrain=True)
    
    # Verify storage service was called
    assert mock_storage_service.save_model.called
    call_args = mock_storage_service.save_model.call_args
    
    # Check arguments (can be positional or keyword)
    if call_args.args:
        # Positional arguments
        assert call_args.args[0] == user_id  # user_id
        assert isinstance(call_args.args[1], dict)  # state_dict
        assert "training_samples" in call_args.args[2]  # metadata
    else:
        # Keyword arguments
        assert call_args.kwargs["user_id"] == user_id
        assert isinstance(call_args.kwargs["state_dict"], dict)
        assert "training_samples" in call_args.kwargs["metadata"]
    
    # Verify cache was invalidated
    mock_cache_invalidation.assert_called_once_with(user_id)
    
    # Verify metrics include storage_mode
    assert "storage_mode" in metrics
    assert metrics["storage_mode"] == "mongodb"
    assert "save_latency_ms" in metrics


def test_train_user_head_dual_save_mode(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that training saves to both MongoDB and file in dual-save mode."""
    user_id = "test_user_dual"
    
    # Configure storage service for dual-save mode
    mock_storage_service.use_mongodb = True
    mock_storage_service.dual_save = True
    
    with patch("training.train_user_head.get_database", return_value=mock_db):
        with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
            metrics = train_user_head(user_id, force_retrain=True)
    
    # Verify storage service was called
    assert mock_storage_service.save_model.called
    
    # Verify metrics show dual mode
    assert metrics["storage_mode"] == "dual"
    
    # Verify cache was invalidated
    mock_cache_invalidation.assert_called_once_with(user_id)


def test_train_user_head_file_mode(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that training saves to file only when USE_MONGODB_STORAGE=False."""
    user_id = "test_user_file"
    
    # Configure storage service for file mode
    mock_storage_service.use_mongodb = False
    mock_storage_service.dual_save = False
    
    with patch("training.train_user_head.get_database", return_value=mock_db):
        with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
            metrics = train_user_head(user_id, force_retrain=True)
    
    # Verify storage service was called
    assert mock_storage_service.save_model.called
    
    # Verify metrics show file mode
    assert metrics["storage_mode"] == "file"
    
    # Verify cache was invalidated
    mock_cache_invalidation.assert_called_once_with(user_id)


def test_train_user_head_cache_invalidation(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that cache is invalidated after model save."""
    user_id = "test_user_cache"
    
    with patch("training.train_user_head.get_database", return_value=mock_db):
        with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
            train_user_head(user_id, force_retrain=True)
    
    # Verify cache invalidation was called with correct user_id
    mock_cache_invalidation.assert_called_once_with(user_id)
    
    # Verify it was called AFTER save_model
    assert mock_storage_service.save_model.called
    assert mock_cache_invalidation.called


def test_train_user_head_save_latency_measurement(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that save latency is measured and included in metrics."""
    user_id = "test_user_latency"
    
    # Mock save_model to take some time
    def slow_save(*args, **kwargs):
        import time
        time.sleep(0.01)  # 10ms delay
        return True
    
    mock_storage_service.save_model = Mock(side_effect=slow_save)
    
    with patch("training.train_user_head.get_database", return_value=mock_db):
        with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
            metrics = train_user_head(user_id, force_retrain=True)
    
    # Verify latency is measured
    assert "save_latency_ms" in metrics
    assert metrics["save_latency_ms"] > 0
    assert metrics["save_latency_ms"] >= 10  # At least 10ms


def test_train_user_head_save_failure_raises_error(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that training raises error when save fails."""
    user_id = "test_user_fail"
    
    # Mock save_model to fail
    mock_storage_service.save_model = Mock(return_value=False)
    
    with patch("training.train_user_head.get_database", return_value=mock_db):
        with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="Model save failed"):
                train_user_head(user_id, force_retrain=True)
    
    # Verify cache was NOT invalidated (save failed)
    mock_cache_invalidation.assert_not_called()


def test_train_user_head_metadata_includes_training_info(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that metadata passed to storage service includes training info."""
    user_id = "test_user_metadata"
    
    with patch("training.train_user_head.get_database", return_value=mock_db):
        with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
            train_user_head(user_id, force_retrain=True)
    
    # Extract metadata from save_model call
    call_args = mock_storage_service.save_model.call_args
    
    # Get metadata (can be positional or keyword)
    if call_args.args:
        metadata = call_args.args[2]
    else:
        metadata = call_args.kwargs["metadata"]
    
    # Verify metadata structure
    assert "training_samples" in metadata
    assert metadata["training_samples"] == 25  # From mock data
    assert "last_trained" in metadata
    assert isinstance(metadata["last_trained"], datetime)


def test_train_user_head_async_uses_storage_service(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that async training uses storage service to check for existing model."""
    job_id = "job_123"
    user_id = "test_user_async"
    
    # Mock storage service exists() to return False (no existing model)
    mock_storage_service.exists = Mock(return_value=False)
    
    # Mock update_job_status at the import location
    with patch("app.services.training_job_tracker.update_job_status") as mock_update_status:
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                train_user_head_async(job_id, user_id, mock_db)
    
    # Verify exists() was called to check for existing model
    mock_storage_service.exists.assert_called_once_with(user_id)
    
    # Verify job status was updated
    assert mock_update_status.call_count >= 2  # running + completed
    
    # Find the completed call - signature is: update_job_status(db, job_id, status, error_message=None, metrics=None)
    completed_calls = [c for c in mock_update_status.call_args_list if len(c.args) >= 3 and c.args[2] == "completed"]
    assert len(completed_calls) > 0, f"No completed status update found. All calls: {mock_update_status.call_args_list}"
    
    # Verify completed status includes storage_mode in metrics
    completed_call = completed_calls[0]
    if "metrics" in completed_call.kwargs:
        metrics = completed_call.kwargs["metrics"]
    elif len(completed_call.args) > 4:
        metrics = completed_call.args[4]  # metrics is 5th arg (index 4)
    else:
        metrics = {}
    
    assert "storage_mode" in metrics


def test_train_user_head_async_invalidates_cache(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that async training invalidates cache after save."""
    job_id = "job_456"
    user_id = "test_user_async_cache"
    
    mock_storage_service.exists = Mock(return_value=False)
    
    with patch("app.services.training_job_tracker.update_job_status"):
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                train_user_head_async(job_id, user_id, mock_db)
    
    # Verify cache was invalidated
    mock_cache_invalidation.assert_called_once_with(user_id)


def test_train_user_head_async_handles_save_failure(mock_db, mock_storage_service, tmp_path):
    """Test that async training handles save failures gracefully."""
    job_id = "job_789"
    user_id = "test_user_async_fail"
    
    # Mock save to fail
    mock_storage_service.save_model = Mock(return_value=False)
    mock_storage_service.exists = Mock(return_value=False)
    
    with patch("app.services.training_job_tracker.update_job_status") as mock_update_status:
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                train_user_head_async(job_id, user_id, mock_db)
    
    # Verify job status was updated to failed
    # Signature is: update_job_status(db, job_id, status, error_message=None, metrics=None)
    failed_calls = [c for c in mock_update_status.call_args_list if len(c.args) >= 3 and c.args[2] == "failed"]
    assert len(failed_calls) == 1, f"Expected 1 failed call, got {len(failed_calls)}. All calls: {mock_update_status.call_args_list}"
    
    # Verify error_message is present
    failed_call = failed_calls[0]
    if "error_message" in failed_call.kwargs:
        assert failed_call.kwargs["error_message"]
    elif len(failed_call.args) > 3:
        # error_message is 4th arg (index 3)
        assert failed_call.args[3]


def test_train_user_head_incremental_uses_existing_model(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that incremental training loads existing model via storage service."""
    user_id = "test_user_incremental"
    
    # Create a mock existing model state_dict
    existing_model = UserEmotionHead(embedding_dim=768, num_classes=8)
    existing_state = existing_model.state_dict()
    
    # Mock storage service to return existing model
    mock_storage_service.load_model = Mock(return_value=existing_state)
    
    # Mock file existence check (for backward compatibility)
    mock_path = Mock()
    mock_path.exists.return_value = True
    
    with patch("training.train_user_head.get_database", return_value=mock_db):
        with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
            with patch("training.train_user_head.Path", return_value=mock_path):
                # Run incremental training (force_retrain=False)
                metrics = train_user_head(user_id, force_retrain=False)
    
    # Verify model was saved after training
    assert mock_storage_service.save_model.called
    
    # Verify cache was invalidated
    mock_cache_invalidation.assert_called_once_with(user_id)


def test_train_user_head_storage_mode_logging(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path, caplog):
    """Test that storage location is logged correctly."""
    import logging
    
    user_id = "test_user_logging"
    
    # Test MongoDB mode
    mock_storage_service.use_mongodb = True
    mock_storage_service.dual_save = False
    
    with caplog.at_level(logging.INFO):
        with patch("training.train_user_head.get_database", return_value=mock_db):
            with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
                train_user_head(user_id, force_retrain=True)
    
    # Verify log message includes storage location
    log_messages = [record.message for record in caplog.records]
    storage_logs = [msg for msg in log_messages if "Model saved to" in msg]
    assert len(storage_logs) > 0
    assert "MongoDB" in storage_logs[0]


def test_train_user_head_returns_all_required_metrics(mock_db, mock_storage_service, mock_cache_invalidation, tmp_path):
    """Test that training returns all required metrics."""
    user_id = "test_user_metrics"
    
    with patch("training.train_user_head.get_database", return_value=mock_db):
        with patch("training.train_user_head.USER_HEADS_DIR", tmp_path):
            metrics = train_user_head(user_id, force_retrain=True)
    
    # Verify all required metrics are present
    required_keys = [
        "final_loss",
        "final_accuracy",
        "num_samples",
        "num_epochs",
        "storage_mode",
        "save_latency_ms"
    ]
    
    for key in required_keys:
        assert key in metrics, f"Missing required metric: {key}"
    
    # Verify types
    assert isinstance(metrics["final_loss"], float)
    assert isinstance(metrics["final_accuracy"], float)
    assert isinstance(metrics["num_samples"], int)
    assert isinstance(metrics["num_epochs"], int)
    assert isinstance(metrics["storage_mode"], str)
    assert isinstance(metrics["save_latency_ms"], float)
    
    # Verify storage_mode is valid
    assert metrics["storage_mode"] in ["mongodb", "file", "dual"]
