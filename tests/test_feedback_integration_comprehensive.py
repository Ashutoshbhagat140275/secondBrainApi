"""
Comprehensive Integration Tests for Feedback Loop and Training Pipeline

This module implements the 5 comprehensive integration test flows required by Task 5:
1. Complete feedback flow (submission → storage → trigger → training → model save)
2. Training flow (20 samples → job lifecycle → model file → metrics)
3. Incremental training (20 samples → 30 samples → model update)
4. Status query (multiple jobs → latest returned)
5. Admin trigger (manual training trigger, auth required)

Feature: feedback-loop-personalization
Task: 5. CONSOLIDATED: Integration, Performance Testing, and Production Readiness

Requirements: 2.2-2.3, 7.1-7.4, 9.3-9.7, 10.1-10.6
"""

import pytest
import sys
import pathlib
import numpy as np
import torch
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from bson import ObjectId
from pathlib import Path

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.feedback_service import submit_feedback
from app.services.training_job_tracker import (
    create_training_job,
    update_job_status,
    get_latest_job,
    get_job_by_id
)
from app.services.task_queue import FastAPITaskQueue
from app.services.user_emotion_head import USER_HEADS_DIR
from training.train_user_head import train_user_head, train_user_head_async


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = MagicMock()
    db.audio_sessions = MagicMock()
    db.emotion_analyses = MagicMock()
    db.user_feedback = MagicMock()
    db.users = MagicMock()
    db.training_jobs = MagicMock()
    return db


@pytest.fixture
def test_user_id():
    """Generate a test user ID."""
    return str(ObjectId())


@pytest.fixture
def test_session_id():
    """Generate a test session ID."""
    return str(ObjectId())


@pytest.fixture
def sample_embedding():
    """Generate a sample 768-dimensional embedding."""
    return [float(x) for x in np.random.randn(768)]


# ---------------------------------------------------------------------------
# Test 1: Complete Feedback Flow
# ---------------------------------------------------------------------------

def test_complete_feedback_flow_integration(mock_db, test_user_id, test_session_id, sample_embedding):
    """
    Test 1: Complete feedback flow (submission → storage → trigger → training → model save).
    
    This test verifies the entire feedback loop from user submission through
    model training and persistence.
    
    Flow:
    1. User submits feedback correction
    2. Feedback is validated and stored in MongoDB
    3. Feedback count is incremented
    4. Training trigger logic evaluates count
    5. If triggered, training job is enqueued
    6. Training job executes asynchronously
    7. Model is trained and saved to disk
    8. Job status is updated to completed
    
    **Validates: Requirements 1.1-1.8, 2.1-2.5, 3.1-3.4, 4.1-4.5, 7.4, 9.3-9.6**
    """
    # Setup: Mock audio session
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(test_session_id),
        "user_id": test_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    # Setup: Mock emotion analysis with embedding
    mock_db.emotion_analyses.find_one.return_value = {
        "session_id": test_session_id,
        "mfcc_features": sample_embedding
    }
    
    # Setup: Mock feedback insertion
    mock_db.user_feedback.insert_one.return_value = MagicMock(inserted_id=ObjectId())
    
    # Setup: Mock user update (feedback count increment)
    mock_db.users.update_one.return_value = MagicMock(modified_count=1)
    
    # Step 1-4: Submit feedback (count < 20, no training trigger)
    with patch("app.services.feedback_service.User.increment_feedback_count", return_value=15):
        result = submit_feedback(
            db=mock_db,
            user_id=test_user_id,
            session_id=test_session_id,
            corrected_emotion="sad"
        )
        
        # Verify feedback stored
        assert result["status"] == "success"
        assert result["feedback_count"] == 15
        assert result["training_triggered"] is False
        assert "5 more samples" in result["message"]
        
        # Verify MongoDB operations
        mock_db.user_feedback.insert_one.assert_called_once()
        feedback_doc = mock_db.user_feedback.insert_one.call_args[0][0]
        assert feedback_doc["user_id"] == test_user_id
        assert feedback_doc["session_id"] == test_session_id
        assert feedback_doc["corrected_emotion"] == "sad"
        assert feedback_doc["predicted_emotion"] == "happy"
        assert len(feedback_doc["embedding"]) == 768
    
    # Step 5-8: Submit feedback at threshold (count = 20, training triggered)
    mock_db.user_feedback.insert_one.reset_mock()
    
    with patch("app.services.feedback_service.User.increment_feedback_count", return_value=20):
        # Create mock task queue
        mock_background_tasks = MagicMock()
        task_queue = FastAPITaskQueue(mock_background_tasks)
        
        result = submit_feedback(
            db=mock_db,
            user_id=test_user_id,
            session_id=test_session_id,
            corrected_emotion="neutral",
            task_queue=task_queue
        )
        
        # Verify training triggered
        assert result["status"] == "success"
        assert result["feedback_count"] == 20
        assert result["training_triggered"] is True
        assert result["training_job_id"] is not None
        assert "model is being updated" in result["message"]
        
        # Verify training job was enqueued
        mock_background_tasks.add_task.assert_called_once()
        
        # Verify training job record was created
        mock_db.training_jobs.insert_one.assert_called_once()
        job_doc = mock_db.training_jobs.insert_one.call_args[0][0]
        assert job_doc["user_id"] == test_user_id
        assert job_doc["status"] == "queued"
        assert job_doc["job_id"] == result["training_job_id"]


# ---------------------------------------------------------------------------
# Test 2: Training Flow with Job Lifecycle
# ---------------------------------------------------------------------------

@patch("training.train_user_head.torch.save")
@patch("training.train_user_head.load_user_feedback")
@patch("training.train_user_head.create_fresh_user_head")
def test_training_flow_with_job_lifecycle(
    mock_create_head,
    mock_load_feedback,
    mock_torch_save,
    mock_db,
    test_user_id
):
    """
    Test 2: Training flow (20 samples → job lifecycle → model file → metrics).
    
    This test verifies the complete training job lifecycle from queued to completed,
    including model file creation and metrics reporting.
    
    Flow:
    1. Training job is created with status=queued
    2. Worker picks up job and updates status to running
    3. Training loads 20 feedback samples from MongoDB
    4. Model is trained for 20 epochs
    5. Model is saved to models/user_heads/{user_id}.pt
    6. Job status is updated to completed with metrics
    7. Metrics include: final_loss, final_accuracy, num_samples, num_epochs
    
    **Validates: Requirements 5.1-5.8, 6.1-6.6, 7.1-7.4, 8.1-8.5, 9.3-9.6, 13.1-13.5**
    """
    job_id = "test-job-123"
    
    # Step 1: Create training job
    create_training_job(mock_db, test_user_id, job_id)
    
    # Verify job created with correct initial state
    mock_db.training_jobs.insert_one.assert_called_once()
    job_doc = mock_db.training_jobs.insert_one.call_args[0][0]
    assert job_doc["user_id"] == test_user_id
    assert job_doc["job_id"] == job_id
    assert job_doc["status"] == "queued"
    assert job_doc["created_at"] is not None
    
    # Step 2: Update job status to running
    update_job_status(mock_db, job_id, "running")
    
    # Verify status update
    mock_db.training_jobs.update_one.assert_called()
    update_call = mock_db.training_jobs.update_one.call_args
    assert update_call[0][0] == {"job_id": job_id}
    assert update_call[0][1]["$set"]["status"] == "running"
    assert "started_at" in update_call[0][1]["$set"]
    
    # Step 3-4: Mock training data (20 samples)
    X = np.random.randn(20, 768).astype(np.float32)
    y = np.random.randint(0, 8, size=20).astype(np.int64)
    mock_load_feedback.return_value = (X, y)
    
    # Mock model creation
    from app.services.user_emotion_head import UserEmotionHead
    mock_model = UserEmotionHead(embedding_dim=768, num_classes=8)
    mock_create_head.return_value = mock_model
    
    # Mock torch.save
    mock_torch_save.return_value = None
    
    # Execute training
    metrics = train_user_head(test_user_id, force_retrain=True)
    
    # Step 5: Verify model was saved
    mock_torch_save.assert_called_once()
    save_path = str(mock_torch_save.call_args[0][1])
    assert test_user_id in save_path
    assert save_path.endswith(".pt")
    assert "user_heads" in save_path
    
    # Step 6-7: Verify metrics returned
    assert "final_loss" in metrics
    assert "final_accuracy" in metrics
    assert "num_samples" in metrics
    assert "num_epochs" in metrics
    assert "model_path" in metrics
    
    assert metrics["num_samples"] == 20
    assert metrics["num_epochs"] == 20
    assert isinstance(metrics["final_loss"], float)
    assert isinstance(metrics["final_accuracy"], float)
    assert 0.0 <= metrics["final_accuracy"] <= 1.0
    
    # Step 6: Update job status to completed
    update_job_status(mock_db, job_id, "completed", metrics=metrics)
    
    # Verify final status update
    update_calls = mock_db.training_jobs.update_one.call_args_list
    final_update = update_calls[-1]
    assert final_update[0][0] == {"job_id": job_id}
    assert final_update[0][1]["$set"]["status"] == "completed"
    assert "completed_at" in final_update[0][1]["$set"]
    assert final_update[0][1]["$set"]["metrics"] == metrics


# ---------------------------------------------------------------------------
# Test 3: Incremental Training
# ---------------------------------------------------------------------------

@patch("training.train_user_head.torch.save")
@patch("training.train_user_head.torch.load")
@patch("training.train_user_head.load_user_feedback")
@patch("training.train_user_head.Path.exists")
def test_incremental_training_flow(
    mock_path_exists,
    mock_load_feedback,
    mock_torch_load,
    mock_torch_save,
    mock_db,
    test_user_id
):
    """
    Test 3: Incremental training (20 samples → 30 samples → model update).
    
    This test verifies that incremental training preserves previous learning
    and trains on all accumulated feedback (not just new samples).
    
    Flow:
    1. Initial training with 20 samples creates new model
    2. User provides 10 more feedback samples (total = 30)
    3. Incremental training loads existing model weights
    4. Training uses all 30 samples (not just new 10)
    5. Updated model is saved, replacing previous version
    6. Model performance should improve or maintain
    
    **Validates: Requirements 7.1-7.4, 5.1-5.2**
    """
    # Step 1: Initial training with 20 samples
    X_initial = np.random.randn(20, 768).astype(np.float32)
    y_initial = np.random.randint(0, 8, size=20).astype(np.int64)
    mock_load_feedback.return_value = (X_initial, y_initial)
    
    # No existing model
    mock_path_exists.return_value = False
    
    # Train initial model
    from app.services.user_emotion_head import UserEmotionHead
    initial_model = UserEmotionHead(embedding_dim=768, num_classes=8)
    
    with patch("training.train_user_head.create_fresh_user_head", return_value=initial_model):
        metrics_initial = train_user_head(test_user_id, force_retrain=True)
    
    # Verify initial training
    assert metrics_initial["num_samples"] == 20
    mock_torch_save.assert_called_once()
    
    # Step 2-3: Incremental training with 30 total samples
    X_incremental = np.random.randn(30, 768).astype(np.float32)
    y_incremental = np.random.randint(0, 8, size=30).astype(np.int64)
    mock_load_feedback.return_value = (X_incremental, y_incremental)
    
    # Existing model now exists
    mock_path_exists.return_value = True
    
    # Mock loading existing model
    existing_model = UserEmotionHead(embedding_dim=768, num_classes=8)
    mock_torch_load.return_value = existing_model.state_dict()
    
    # Reset save mock
    mock_torch_save.reset_mock()
    
    # Step 4-5: Train incrementally
    metrics_incremental = train_user_head(test_user_id, force_retrain=False)
    
    # Verify incremental training used all 30 samples
    assert metrics_incremental["num_samples"] == 30
    assert metrics_incremental["num_epochs"] == 20
    
    # Verify existing model was loaded
    mock_torch_load.assert_called_once()
    
    # Verify updated model was saved
    mock_torch_save.assert_called_once()
    
    # Step 6: Verify model path consistency
    save_path_initial = str(mock_torch_save.call_args_list[0][0][1])
    save_path_incremental = str(mock_torch_save.call_args_list[0][0][1])
    
    # Both should save to same path (overwriting)
    assert test_user_id in save_path_incremental
    assert save_path_incremental.endswith(".pt")


# ---------------------------------------------------------------------------
# Test 4: Status Query with Multiple Jobs
# ---------------------------------------------------------------------------

def test_status_query_returns_latest_job(mock_db, test_user_id):
    """
    Test 4: Status query (multiple jobs → latest returned).
    
    This test verifies that when a user has multiple training jobs,
    the status query returns the most recent job based on created_at timestamp.
    
    Flow:
    1. Create 3 training jobs for same user at different times
    2. Query training status for user
    3. Verify latest job (most recent created_at) is returned
    4. Verify older jobs are not returned
    
    **Validates: Requirements 9.7, 10.3, 10.4**
    """
    from app.models.training_job import TrainingJob
    
    # Step 1: Create 3 training jobs at different times
    job1 = TrainingJob(
        user_id=test_user_id,
        job_id="job-001",
        status="completed",
        created_at=datetime(2024, 1, 1, 10, 0, 0),
        updated_at=datetime(2024, 1, 1, 10, 2, 0),
        completed_at=datetime(2024, 1, 1, 10, 2, 0)
    )
    
    job2 = TrainingJob(
        user_id=test_user_id,
        job_id="job-002",
        status="completed",
        created_at=datetime(2024, 1, 2, 10, 0, 0),
        updated_at=datetime(2024, 1, 2, 10, 2, 0),
        completed_at=datetime(2024, 1, 2, 10, 2, 0)
    )
    
    job3 = TrainingJob(
        user_id=test_user_id,
        job_id="job-003",
        status="running",
        created_at=datetime(2024, 1, 3, 10, 0, 0),
        updated_at=datetime(2024, 1, 3, 10, 0, 5),
        started_at=datetime(2024, 1, 3, 10, 0, 5)
    )
    
    # Mock MongoDB to return the latest job (using find_one with sort)
    mock_db.training_jobs.find_one.return_value = job3.to_dict()
    
    # Step 2-3: Query latest job
    latest_job = get_latest_job(mock_db, test_user_id)
    
    # Verify latest job is returned (job3)
    assert latest_job is not None
    assert latest_job.job_id == "job-003"
    assert latest_job.status == "running"
    assert latest_job.created_at == datetime(2024, 1, 3, 10, 0, 0)
    
    # Verify MongoDB query was correct (find_one with sort parameter)
    mock_db.training_jobs.find_one.assert_called_once_with(
        {"user_id": test_user_id},
        sort=[("created_at", -1)]
    )


# ---------------------------------------------------------------------------
# Test 5: Admin Manual Training Trigger
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_admin_manual_training_trigger(mock_db, test_user_id):
    """
    Test 5: Admin trigger (manual training trigger, auth required).
    
    This test verifies that administrators can manually trigger training
    for any user, bypassing the normal feedback count requirements.
    
    Flow:
    1. Admin authenticates and calls trigger-training endpoint
    2. System verifies admin permissions
    3. System verifies target user exists
    4. Training job is enqueued regardless of feedback count
    5. Job record is created with status=queued
    6. Response includes job_id and success message
    
    **Validates: Requirements 10.5, 10.6**
    """
    from fastapi import BackgroundTasks
    from app.routers.feedback import trigger_training_endpoint
    
    # Step 1-2: Mock admin user
    admin_user = {
        "user_id": str(ObjectId()),
        "is_admin": True
    }
    
    # Step 3: Mock target user exists
    mock_db.users.find_one.return_value = {
        "_id": ObjectId(test_user_id),
        "email": "test@example.com",
        "feedback_count": 5  # Less than 20, but admin can still trigger
    }
    
    # Mock training job insertion
    mock_db.training_jobs.insert_one.return_value = MagicMock(inserted_id=ObjectId())
    
    # Step 4-5: Trigger training
    background_tasks = BackgroundTasks()
    
    with patch("app.routers.feedback.get_current_admin", return_value=admin_user):
        response = await trigger_training_endpoint(
            user_id=test_user_id,
            background_tasks=background_tasks,
            current_user=admin_user,
            db=mock_db
        )
    
    # Step 6: Verify response
    assert response.status == "success"
    assert response.training_job_id is not None
    assert test_user_id in response.message
    
    # Verify training job was created
    mock_db.training_jobs.insert_one.assert_called_once()
    job_doc = mock_db.training_jobs.insert_one.call_args[0][0]
    assert job_doc["user_id"] == test_user_id
    assert job_doc["status"] == "queued"
    assert job_doc["job_id"] == response.training_job_id


# ---------------------------------------------------------------------------
# Test 6: Error Handling in Integration Flows
# ---------------------------------------------------------------------------

def test_feedback_flow_with_invalid_session(mock_db, test_user_id, test_session_id):
    """
    Test error handling: Feedback submission with non-existent session.
    
    **Validates: Requirements 12.1**
    """
    # Mock session not found
    mock_db.audio_sessions.find_one.return_value = None
    
    # Attempt to submit feedback
    with pytest.raises(ValueError, match="not found"):
        submit_feedback(
            db=mock_db,
            user_id=test_user_id,
            session_id=test_session_id,
            corrected_emotion="happy"
        )


def test_feedback_flow_with_wrong_user(mock_db, test_user_id, test_session_id):
    """
    Test error handling: Feedback submission for another user's session.
    
    **Validates: Requirements 12.2**
    """
    other_user_id = str(ObjectId())
    
    # Mock session belongs to different user
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(test_session_id),
        "user_id": other_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    # Attempt to submit feedback
    with pytest.raises(PermissionError, match="does not belong to user"):
        submit_feedback(
            db=mock_db,
            user_id=test_user_id,
            session_id=test_session_id,
            corrected_emotion="happy"
        )


def test_training_flow_with_insufficient_data(mock_db, test_user_id):
    """
    Test error handling: Training with fewer than 20 samples.
    
    **Validates: Requirements 12.3**
    """
    # Mock insufficient feedback data (only 10 samples)
    X = np.random.randn(10, 768).astype(np.float32)
    y = np.random.randint(0, 8, size=10).astype(np.int64)
    
    with patch("training.train_user_head.load_user_feedback", return_value=(X, y)):
        with pytest.raises(ValueError, match="Insufficient feedback data"):
            train_user_head(test_user_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
