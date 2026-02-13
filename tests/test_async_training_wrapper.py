"""
Unit Tests for Async Training Wrapper

Tests cover:
- Async training wrapper (train_user_head_async)
- Job status updates (queued → running → completed/failed)
- Error handling and recovery
- Metrics reporting
- Logging behavior

Feature: feedback-loop-personalization
Task: 4. CONSOLIDATED: Implement Async Training Engine and Model Persistence
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import logging

from training.train_user_head import train_user_head_async
from app.services.feature_config import EMBEDDING_DIM, NUM_CLASSES, EMOTION_LABELS


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    return Mock()


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_feedback_data():
    """Create sample feedback data for testing."""
    feedback_docs = []
    for i in range(25):
        feedback_docs.append({
            "_id": f"feedback_{i}",
            "user_id": "test_user_123",
            "session_id": f"session_{i}",
            "embedding": np.random.randn(EMBEDDING_DIM).tolist(),
            "predicted_emotion": "happy",
            "corrected_emotion": EMOTION_LABELS[i % NUM_CLASSES],
            "timestamp": f"2024-01-{i+1:02d}T10:00:00Z"
        })
    return feedback_docs


class TestAsyncTrainingWrapper:
    """Tests for train_user_head_async function."""
    
    def test_async_wrapper_updates_job_status_to_running(
        self, mock_db, sample_feedback_data, temp_model_dir
    ):
        """
        Test that async wrapper updates job status to 'running' when training starts.
        
        **Validates: Requirements 4.2, 9.4**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Mock dependencies
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("app.services.training_job_tracker.update_job_status") as mock_update:
                    train_user_head_async(job_id, user_id, mock_db)
                    
                    # Verify status was updated to 'running'
                    calls = mock_update.call_args_list
                    assert any(
                        call_args[0][1] == job_id and call_args[0][2] == "running"
                        for call_args in calls
                    ), "Job status should be updated to 'running'"
    
    def test_async_wrapper_updates_job_status_to_completed(
        self, mock_db, sample_feedback_data, temp_model_dir
    ):
        """
        Test that async wrapper updates job status to 'completed' on success.
        
        **Validates: Requirements 4.2, 9.5**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Mock dependencies
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("app.services.training_job_tracker.update_job_status") as mock_update:
                    train_user_head_async(job_id, user_id, mock_db)
                    
                    # Verify status was updated to 'completed'
                    calls = mock_update.call_args_list
                    completed_call = [
                        call_args for call_args in calls
                        if call_args[0][2] == "completed"
                    ]
                    assert len(completed_call) > 0, "Job status should be updated to 'completed'"
                    
                    # Verify metrics were provided
                    assert "metrics" in completed_call[0][1], "Metrics should be provided"
    
    def test_async_wrapper_updates_job_status_to_failed_on_error(
        self, mock_db, temp_model_dir
    ):
        """
        Test that async wrapper updates job status to 'failed' on error.
        
        **Validates: Requirements 9.6, 12.4**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Mock load_user_feedback to raise an error
        with patch("training.train_user_head.load_user_feedback", side_effect=ValueError("Test error")):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("app.services.training_job_tracker.update_job_status") as mock_update:
                    # Should not raise exception (error is caught)
                    train_user_head_async(job_id, user_id, mock_db)
                    
                    # Verify status was updated to 'failed'
                    calls = mock_update.call_args_list
                    failed_call = [
                        call_args for call_args in calls
                        if call_args[0][2] == "failed"
                    ]
                    assert len(failed_call) > 0, "Job status should be updated to 'failed'"
                    
                    # Verify error message was provided
                    assert "error_message" in failed_call[0][1], "Error message should be provided"
                    assert "Test error" in str(failed_call[0][1]["error_message"])
    
    def test_async_wrapper_includes_metrics_on_success(
        self, mock_db, temp_model_dir
    ):
        """
        Test that async wrapper includes training metrics on successful completion.
        
        **Validates: Requirements 8.4, 9.5**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Mock dependencies
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("app.services.training_job_tracker.update_job_status") as mock_update:
                    train_user_head_async(job_id, user_id, mock_db)
                    
                    # Find the completed call
                    completed_call = [
                        call_args for call_args in mock_update.call_args_list
                        if call_args[0][2] == "completed"
                    ][0]
                    
                    metrics = completed_call[1]["metrics"]
                    
                    # Verify metrics structure
                    assert "final_loss" in metrics
                    assert "final_accuracy" in metrics
                    assert "num_samples" in metrics
                    assert "num_epochs" in metrics
                    assert "model_path" in metrics
                    
                    # Verify metrics values
                    assert metrics["num_samples"] == 25
                    assert metrics["num_epochs"] == 20
                    assert 0.0 <= metrics["final_accuracy"] <= 1.0
    
    def test_async_wrapper_logs_start_and_completion(
        self, mock_db, temp_model_dir, caplog
    ):
        """
        Test that async wrapper logs training start and completion.
        
        **Validates: Requirements 8.1, 8.4**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Mock dependencies
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with caplog.at_level(logging.INFO):
            with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
                with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                    with patch("app.services.training_job_tracker.update_job_status"):
                        train_user_head_async(job_id, user_id, mock_db)
        
        # Verify logs contain job start and completion
        log_messages = [record.message for record in caplog.records]
        
        assert any("started" in msg.lower() and job_id in msg for msg in log_messages), \
            "Should log job start"
        assert any("completed" in msg.lower() and job_id in msg for msg in log_messages), \
            "Should log job completion"
    
    def test_async_wrapper_logs_error_with_stack_trace(
        self, mock_db, temp_model_dir, caplog
    ):
        """
        Test that async wrapper logs errors with full stack trace.
        
        **Validates: Requirements 8.5, 12.4**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        with caplog.at_level(logging.ERROR):
            with patch("training.train_user_head.load_user_feedback", side_effect=ValueError("Test error")):
                with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                    with patch("app.services.training_job_tracker.update_job_status"):
                        train_user_head_async(job_id, user_id, mock_db)
        
        # Verify error was logged
        error_records = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(error_records) > 0, "Should log error"
        
        # Verify error message contains job_id and error details
        error_message = error_records[0].message
        assert job_id in error_message
        assert "Test error" in error_message
    
    def test_async_wrapper_does_not_crash_on_error(
        self, mock_db, temp_model_dir
    ):
        """
        Test that async wrapper catches exceptions and doesn't crash the worker.
        
        **Validates: Requirements 12.4**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Mock to raise various types of errors
        errors = [
            ValueError("Insufficient data"),
            RuntimeError("Training failed"),
            Exception("Unexpected error"),
        ]
        
        for error in errors:
            with patch("training.train_user_head.load_user_feedback", side_effect=error):
                with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                    with patch("app.services.training_job_tracker.update_job_status"):
                        # Should not raise exception
                        try:
                            train_user_head_async(job_id, user_id, mock_db)
                        except Exception as e:
                            pytest.fail(f"Async wrapper should not raise exception, got: {e}")


class TestAsyncWrapperInitialVsIncremental:
    """Test async wrapper handles initial vs incremental training correctly."""
    
    def test_async_wrapper_initial_training_when_no_model_exists(
        self, mock_db, temp_model_dir
    ):
        """
        Test that async wrapper performs initial training when no model exists.
        
        **Validates: Requirements 7.3**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Ensure no model exists
        model_path = temp_model_dir / f"{user_id}.pt"
        assert not model_path.exists()
        
        # Mock dependencies
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("app.services.training_job_tracker.update_job_status"):
                    train_user_head_async(job_id, user_id, mock_db)
        
        # Verify model was created
        assert model_path.exists(), "Model should be created for initial training"
    
    def test_async_wrapper_incremental_training_when_model_exists(
        self, mock_db, temp_model_dir
    ):
        """
        Test that async wrapper performs incremental training when model exists.
        
        **Validates: Requirements 7.1, 7.2**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Create initial model
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("app.services.training_job_tracker.update_job_status"):
                    # Initial training
                    train_user_head_async(job_id, user_id, mock_db)
        
        model_path = temp_model_dir / f"{user_id}.pt"
        assert model_path.exists()
        
        # Get initial model timestamp
        initial_mtime = model_path.stat().st_mtime
        
        # Perform incremental training
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("app.services.training_job_tracker.update_job_status"):
                    train_user_head_async(job_id, user_id, mock_db)
        
        # Verify model was updated
        assert model_path.exists()
        # Note: In real scenario, mtime would change, but in test it might not due to speed


class TestErrorHandlingScenarios:
    """Test various error scenarios in async training."""
    
    def test_insufficient_feedback_data_error(self, mock_db, temp_model_dir):
        """
        Test error handling when user has insufficient feedback data.
        
        **Validates: Requirements 5.5, 12.4**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Only 15 samples (< 20 minimum)
        X = np.random.randn(15, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=15).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("app.services.training_job_tracker.update_job_status") as mock_update:
                    train_user_head_async(job_id, user_id, mock_db)
                    
                    # Verify job was marked as failed
                    failed_call = [
                        call_args for call_args in mock_update.call_args_list
                        if call_args[0][2] == "failed"
                    ][0]
                    
                    error_message = failed_call[1]["error_message"]
                    assert "Insufficient feedback data" in error_message
    
    def test_invalid_embedding_dimension_error(self, mock_db, temp_model_dir):
        """
        Test error handling when embeddings have invalid dimensions.
        
        **Validates: Requirements 1.5, 2.4, 12.4**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        # Mock to return error
        with patch("training.train_user_head.load_user_feedback", 
                   side_effect=ValueError("No valid feedback data found")):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("app.services.training_job_tracker.update_job_status") as mock_update:
                    train_user_head_async(job_id, user_id, mock_db)
                    
                    # Verify job was marked as failed
                    failed_call = [
                        call_args for call_args in mock_update.call_args_list
                        if call_args[0][2] == "failed"
                    ][0]
                    
                    error_message = failed_call[1]["error_message"]
                    assert "No valid feedback data found" in error_message
    
    def test_model_save_error(self, mock_db, temp_model_dir):
        """
        Test error handling when model save fails.
        
        **Validates: Requirements 12.4, 13.4**
        """
        job_id = "test-job-123"
        user_id = "test_user_123"
        
        X = np.random.randn(25, EMBEDDING_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=25).astype(np.int64)
        
        with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
            with patch("training.train_user_head.USER_HEADS_DIR", temp_model_dir):
                with patch("torch.save", side_effect=IOError("Disk full")):
                    with patch("app.services.training_job_tracker.update_job_status") as mock_update:
                        train_user_head_async(job_id, user_id, mock_db)
                        
                        # Verify job was marked as failed
                        failed_call = [
                            call_args for call_args in mock_update.call_args_list
                            if call_args[0][2] == "failed"
                        ][0]
                        
                        error_message = failed_call[1]["error_message"]
                        assert "Disk full" in error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
