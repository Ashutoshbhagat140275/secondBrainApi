"""
Unit tests for Training Job Tracker service.

Tests Requirements 9.3, 9.4, 9.5, 9.6, 9.7
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from bson import ObjectId

from app.services.training_job_tracker import (
    create_training_job,
    update_job_status,
    get_latest_job,
    get_job_by_id
)
from app.models.training_job import TrainingJob


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = Mock()
    db.training_jobs = Mock()
    return db


class TestCreateTrainingJob:
    """Test create_training_job function."""
    
    def test_create_training_job_with_valid_inputs(self, mock_db):
        """Test creating a training job with valid user_id and job_id."""
        user_id = str(ObjectId())
        job_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        
        result = create_training_job(mock_db, user_id, job_id)
        
        # Verify TrainingJob instance returned
        assert isinstance(result, TrainingJob)
        assert result.user_id == user_id
        assert result.job_id == job_id
        assert result.status == TrainingJob.STATUS_QUEUED
        assert isinstance(result.created_at, datetime)
        assert isinstance(result.updated_at, datetime)
        assert result.started_at is None
        assert result.completed_at is None
        assert result.error_message is None
        assert result.metrics is None
        
        # Verify MongoDB insert was called
        mock_db.training_jobs.insert_one.assert_called_once()
        inserted_doc = mock_db.training_jobs.insert_one.call_args[0][0]
        assert inserted_doc["user_id"] == user_id
        assert inserted_doc["job_id"] == job_id
        assert inserted_doc["status"] == "queued"
    
    def test_create_training_job_sets_timestamps(self, mock_db):
        """Test that create_training_job sets created_at and updated_at."""
        user_id = str(ObjectId())
        job_id = "test-job-123"
        
        result = create_training_job(mock_db, user_id, job_id)
        
        # Verify timestamps are set and equal
        assert result.created_at is not None
        assert result.updated_at is not None
        assert result.created_at == result.updated_at
    
    def test_create_training_job_generates_object_id(self, mock_db):
        """Test that create_training_job generates a MongoDB ObjectId."""
        user_id = str(ObjectId())
        job_id = "test-job-456"
        
        result = create_training_job(mock_db, user_id, job_id)
        
        assert isinstance(result._id, ObjectId)


class TestUpdateJobStatus:
    """Test update_job_status function."""
    
    def test_update_to_running_sets_started_at(self, mock_db):
        """Test updating status to 'running' sets started_at timestamp."""
        job_id = "test-job-789"
        mock_db.training_jobs.update_one.return_value = Mock(matched_count=1)
        
        update_job_status(mock_db, job_id, TrainingJob.STATUS_RUNNING)
        
        # Verify update_one was called
        mock_db.training_jobs.update_one.assert_called_once()
        call_args = mock_db.training_jobs.update_one.call_args
        
        # Verify filter
        assert call_args[0][0] == {"job_id": job_id}
        
        # Verify update document
        update_doc = call_args[0][1]["$set"]
        assert update_doc["status"] == "running"
        assert "started_at" in update_doc
        assert isinstance(update_doc["started_at"], datetime)
        assert "updated_at" in update_doc
        assert isinstance(update_doc["updated_at"], datetime)
    
    def test_update_to_completed_sets_completed_at(self, mock_db):
        """Test updating status to 'completed' sets completed_at timestamp."""
        job_id = "test-job-completed"
        metrics = {"final_loss": 0.234, "final_accuracy": 0.89}
        mock_db.training_jobs.update_one.return_value = Mock(matched_count=1)
        
        update_job_status(
            mock_db,
            job_id,
            TrainingJob.STATUS_COMPLETED,
            metrics=metrics
        )
        
        # Verify update document
        call_args = mock_db.training_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        
        assert update_doc["status"] == "completed"
        assert "completed_at" in update_doc
        assert isinstance(update_doc["completed_at"], datetime)
        assert update_doc["metrics"] == metrics
    
    def test_update_to_failed_sets_completed_at_and_error(self, mock_db):
        """Test updating status to 'failed' sets completed_at and error_message."""
        job_id = "test-job-failed"
        error_msg = "Training failed due to insufficient data"
        mock_db.training_jobs.update_one.return_value = Mock(matched_count=1)
        
        update_job_status(
            mock_db,
            job_id,
            TrainingJob.STATUS_FAILED,
            error_message=error_msg
        )
        
        # Verify update document
        call_args = mock_db.training_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        
        assert update_doc["status"] == "failed"
        assert "completed_at" in update_doc
        assert isinstance(update_doc["completed_at"], datetime)
        assert update_doc["error_message"] == error_msg
    
    def test_update_to_queued_no_special_timestamps(self, mock_db):
        """Test updating status to 'queued' doesn't set started_at or completed_at."""
        job_id = "test-job-queued"
        mock_db.training_jobs.update_one.return_value = Mock(matched_count=1)
        
        update_job_status(mock_db, job_id, TrainingJob.STATUS_QUEUED)
        
        # Verify update document
        call_args = mock_db.training_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        
        assert update_doc["status"] == "queued"
        assert "started_at" not in update_doc
        assert "completed_at" not in update_doc
        assert "updated_at" in update_doc
    
    def test_update_with_invalid_status_raises_error(self, mock_db):
        """Test updating with invalid status raises ValueError."""
        job_id = "test-job-invalid"
        
        with pytest.raises(ValueError, match="Invalid status: 'invalid_status'"):
            update_job_status(mock_db, job_id, "invalid_status")
        
        # Verify no database call was made
        mock_db.training_jobs.update_one.assert_not_called()
    
    def test_update_nonexistent_job_logs_warning(self, mock_db, caplog):
        """Test updating nonexistent job logs a warning."""
        job_id = "nonexistent-job"
        mock_db.training_jobs.update_one.return_value = Mock(matched_count=0)
        
        update_job_status(mock_db, job_id, TrainingJob.STATUS_RUNNING)
        
        # Verify warning was logged
        assert any("Training job not found" in record.message for record in caplog.records)
    
    def test_update_with_metrics_only(self, mock_db):
        """Test updating with metrics but no error message."""
        job_id = "test-job-metrics"
        metrics = {"final_loss": 0.15, "final_accuracy": 0.92, "num_samples": 50}
        mock_db.training_jobs.update_one.return_value = Mock(matched_count=1)
        
        update_job_status(
            mock_db,
            job_id,
            TrainingJob.STATUS_COMPLETED,
            metrics=metrics
        )
        
        # Verify metrics in update document
        call_args = mock_db.training_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        
        assert update_doc["metrics"] == metrics
        assert "error_message" not in update_doc
    
    def test_update_with_error_message_only(self, mock_db):
        """Test updating with error message but no metrics."""
        job_id = "test-job-error"
        error_msg = "Model training timeout"
        mock_db.training_jobs.update_one.return_value = Mock(matched_count=1)
        
        update_job_status(
            mock_db,
            job_id,
            TrainingJob.STATUS_FAILED,
            error_message=error_msg
        )
        
        # Verify error message in update document
        call_args = mock_db.training_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        
        assert update_doc["error_message"] == error_msg
        assert "metrics" not in update_doc


class TestGetLatestJob:
    """Test get_latest_job function."""
    
    def test_get_latest_job_returns_most_recent(self, mock_db):
        """Test get_latest_job returns the job with most recent created_at."""
        user_id = str(ObjectId())
        job_doc = {
            "_id": ObjectId(),
            "user_id": user_id,
            "job_id": "latest-job",
            "status": "completed",
            "created_at": datetime(2024, 1, 15, 10, 30, 0),
            "updated_at": datetime(2024, 1, 15, 10, 32, 0)
        }
        mock_db.training_jobs.find_one.return_value = job_doc
        
        result = get_latest_job(mock_db, user_id)
        
        # Verify find_one was called with correct filter and sort
        mock_db.training_jobs.find_one.assert_called_once_with(
            {"user_id": user_id},
            sort=[("created_at", -1)]
        )
        
        # Verify TrainingJob instance returned
        assert isinstance(result, TrainingJob)
        assert result.user_id == user_id
        assert result.job_id == "latest-job"
        assert result.status == "completed"
    
    def test_get_latest_job_returns_none_when_no_jobs(self, mock_db):
        """Test get_latest_job returns None when user has no jobs."""
        user_id = str(ObjectId())
        mock_db.training_jobs.find_one.return_value = None
        
        result = get_latest_job(mock_db, user_id)
        
        assert result is None
        mock_db.training_jobs.find_one.assert_called_once()
    
    def test_get_latest_job_with_multiple_jobs(self, mock_db):
        """Test get_latest_job correctly sorts by created_at descending."""
        user_id = str(ObjectId())
        # Simulate MongoDB returning the most recent job
        latest_job_doc = {
            "_id": ObjectId(),
            "user_id": user_id,
            "job_id": "job-3",
            "status": "running",
            "created_at": datetime(2024, 1, 15, 12, 0, 0),
            "updated_at": datetime(2024, 1, 15, 12, 0, 0)
        }
        mock_db.training_jobs.find_one.return_value = latest_job_doc
        
        result = get_latest_job(mock_db, user_id)
        
        # Verify correct sort order (descending)
        call_args = mock_db.training_jobs.find_one.call_args
        assert call_args[1]["sort"] == [("created_at", -1)]
        
        # Verify latest job returned
        assert result.job_id == "job-3"
        assert result.created_at == datetime(2024, 1, 15, 12, 0, 0)


class TestGetJobById:
    """Test get_job_by_id function."""
    
    def test_get_job_by_id_returns_job(self, mock_db):
        """Test get_job_by_id returns the correct job."""
        job_id = "specific-job-123"
        job_doc = {
            "_id": ObjectId(),
            "user_id": str(ObjectId()),
            "job_id": job_id,
            "status": "completed",
            "created_at": datetime(2024, 1, 15, 10, 30, 0),
            "updated_at": datetime(2024, 1, 15, 10, 32, 0),
            "metrics": {"final_loss": 0.234}
        }
        mock_db.training_jobs.find_one.return_value = job_doc
        
        result = get_job_by_id(mock_db, job_id)
        
        # Verify find_one was called with correct filter
        mock_db.training_jobs.find_one.assert_called_once_with({"job_id": job_id})
        
        # Verify TrainingJob instance returned
        assert isinstance(result, TrainingJob)
        assert result.job_id == job_id
        assert result.status == "completed"
        assert result.metrics == {"final_loss": 0.234}
    
    def test_get_job_by_id_returns_none_when_not_found(self, mock_db):
        """Test get_job_by_id returns None when job doesn't exist."""
        job_id = "nonexistent-job"
        mock_db.training_jobs.find_one.return_value = None
        
        result = get_job_by_id(mock_db, job_id)
        
        assert result is None
        mock_db.training_jobs.find_one.assert_called_once_with({"job_id": job_id})
    
    def test_get_job_by_id_with_all_fields(self, mock_db):
        """Test get_job_by_id correctly deserializes all fields."""
        job_id = "complete-job-456"
        job_doc = {
            "_id": ObjectId(),
            "user_id": str(ObjectId()),
            "job_id": job_id,
            "status": "failed",
            "created_at": datetime(2024, 1, 15, 10, 30, 0),
            "updated_at": datetime(2024, 1, 15, 10, 32, 0),
            "started_at": datetime(2024, 1, 15, 10, 30, 5),
            "completed_at": datetime(2024, 1, 15, 10, 32, 0),
            "error_message": "Training timeout",
            "metrics": None
        }
        mock_db.training_jobs.find_one.return_value = job_doc
        
        result = get_job_by_id(mock_db, job_id)
        
        assert result.job_id == job_id
        assert result.status == "failed"
        assert result.started_at == datetime(2024, 1, 15, 10, 30, 5)
        assert result.completed_at == datetime(2024, 1, 15, 10, 32, 0)
        assert result.error_message == "Training timeout"
        assert result.metrics is None


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_complete_job_lifecycle(self, mock_db):
        """Test complete job lifecycle: create -> running -> completed."""
        user_id = str(ObjectId())
        job_id = "lifecycle-job"
        
        # Create job
        mock_db.training_jobs.update_one.return_value = Mock(matched_count=1)
        created_job = create_training_job(mock_db, user_id, job_id)
        
        assert created_job.status == "queued"
        assert created_job.started_at is None
        assert created_job.completed_at is None
        
        # Update to running
        update_job_status(mock_db, job_id, TrainingJob.STATUS_RUNNING)
        
        # Verify started_at was set
        call_args = mock_db.training_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        assert update_doc["status"] == "running"
        assert "started_at" in update_doc
        
        # Update to completed with metrics
        metrics = {"final_loss": 0.15, "final_accuracy": 0.95}
        update_job_status(
            mock_db,
            job_id,
            TrainingJob.STATUS_COMPLETED,
            metrics=metrics
        )
        
        # Verify completed_at and metrics were set
        call_args = mock_db.training_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        assert update_doc["status"] == "completed"
        assert "completed_at" in update_doc
        assert update_doc["metrics"] == metrics
    
    def test_failed_job_lifecycle(self, mock_db):
        """Test failed job lifecycle: create -> running -> failed."""
        user_id = str(ObjectId())
        job_id = "failed-job"
        
        # Create job
        mock_db.training_jobs.update_one.return_value = Mock(matched_count=1)
        create_training_job(mock_db, user_id, job_id)
        
        # Update to running
        update_job_status(mock_db, job_id, TrainingJob.STATUS_RUNNING)
        
        # Update to failed with error
        error_msg = "Insufficient feedback data"
        update_job_status(
            mock_db,
            job_id,
            TrainingJob.STATUS_FAILED,
            error_message=error_msg
        )
        
        # Verify error message and completed_at were set
        call_args = mock_db.training_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        assert update_doc["status"] == "failed"
        assert "completed_at" in update_doc
        assert update_doc["error_message"] == error_msg
