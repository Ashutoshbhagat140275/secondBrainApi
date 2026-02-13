"""
Unit tests for TrainingJob MongoDB model.

Tests Requirements 9.1, 9.2
"""

import pytest
from datetime import datetime
from bson import ObjectId
from app.models.training_job import TrainingJob


class TestTrainingJobModel:
    """Test TrainingJob model initialization and methods."""
    
    def test_init_with_all_fields(self):
        """Test TrainingJob initialization with all fields provided."""
        job_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        user_id = str(ObjectId())
        status = TrainingJob.STATUS_COMPLETED
        created_at = datetime(2024, 1, 15, 10, 30, 0)
        updated_at = datetime(2024, 1, 15, 10, 32, 15)
        started_at = datetime(2024, 1, 15, 10, 30, 5)
        completed_at = datetime(2024, 1, 15, 10, 32, 15)
        metrics = {
            "final_loss": 0.234,
            "final_accuracy": 0.89,
            "num_samples": 25,
            "num_epochs": 20
        }
        
        training_job = TrainingJob(
            user_id=user_id,
            job_id=job_id,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            started_at=started_at,
            completed_at=completed_at,
            metrics=metrics
        )
        
        assert training_job.user_id == user_id
        assert training_job.job_id == job_id
        assert training_job.status == status
        assert training_job.created_at == created_at
        assert training_job.updated_at == updated_at
        assert training_job.started_at == started_at
        assert training_job.completed_at == completed_at
        assert training_job.error_message is None
        assert training_job.metrics == metrics
        assert isinstance(training_job._id, ObjectId)
    
    def test_init_with_defaults(self):
        """Test TrainingJob initialization with default timestamps."""
        job_id = "test-job-123"
        user_id = str(ObjectId())
        status = TrainingJob.STATUS_QUEUED
        
        training_job = TrainingJob(
            user_id=user_id,
            job_id=job_id,
            status=status
        )
        
        assert training_job.user_id == user_id
        assert training_job.job_id == job_id
        assert training_job.status == status
        assert isinstance(training_job.created_at, datetime)
        assert isinstance(training_job.updated_at, datetime)
        assert training_job.started_at is None
        assert training_job.completed_at is None
        assert training_job.error_message is None
        assert training_job.metrics is None
        assert isinstance(training_job._id, ObjectId)
    
    def test_init_with_queued_status(self):
        """Test TrainingJob initialization with queued status."""
        training_job = TrainingJob(
            user_id=str(ObjectId()),
            job_id="job-123",
            status=TrainingJob.STATUS_QUEUED
        )
        
        assert training_job.status == "queued"
    
    def test_init_with_running_status(self):
        """Test TrainingJob initialization with running status."""
        training_job = TrainingJob(
            user_id=str(ObjectId()),
            job_id="job-123",
            status=TrainingJob.STATUS_RUNNING
        )
        
        assert training_job.status == "running"
    
    def test_init_with_completed_status(self):
        """Test TrainingJob initialization with completed status."""
        training_job = TrainingJob(
            user_id=str(ObjectId()),
            job_id="job-123",
            status=TrainingJob.STATUS_COMPLETED
        )
        
        assert training_job.status == "completed"
    
    def test_init_with_failed_status(self):
        """Test TrainingJob initialization with failed status."""
        training_job = TrainingJob(
            user_id=str(ObjectId()),
            job_id="job-123",
            status=TrainingJob.STATUS_FAILED,
            error_message="Training failed due to insufficient data"
        )
        
        assert training_job.status == "failed"
        assert training_job.error_message == "Training failed due to insufficient data"
    
    def test_init_with_invalid_status_raises_error(self):
        """Test TrainingJob initialization with invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status: 'invalid_status'"):
            TrainingJob(
                user_id=str(ObjectId()),
                job_id="job-123",
                status="invalid_status"
            )
    
    def test_valid_statuses_constant(self):
        """Test that VALID_STATUSES contains all expected status values."""
        expected_statuses = {"queued", "running", "completed", "failed"}
        assert TrainingJob.VALID_STATUSES == expected_statuses
    
    def test_to_dict(self):
        """Test TrainingJob serialization to dictionary."""
        job_id = "test-job-456"
        user_id = str(ObjectId())
        status = TrainingJob.STATUS_COMPLETED
        created_at = datetime(2024, 1, 15, 10, 30, 0)
        updated_at = datetime(2024, 1, 15, 10, 32, 15)
        started_at = datetime(2024, 1, 15, 10, 30, 5)
        completed_at = datetime(2024, 1, 15, 10, 32, 15)
        metrics = {"final_loss": 0.234, "final_accuracy": 0.89}
        
        training_job = TrainingJob(
            user_id=user_id,
            job_id=job_id,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            started_at=started_at,
            completed_at=completed_at,
            metrics=metrics
        )
        
        job_dict = training_job.to_dict()
        
        assert job_dict["user_id"] == user_id
        assert job_dict["job_id"] == job_id
        assert job_dict["status"] == status
        assert job_dict["created_at"] == created_at
        assert job_dict["updated_at"] == updated_at
        assert job_dict["started_at"] == started_at
        assert job_dict["completed_at"] == completed_at
        assert job_dict["error_message"] is None
        assert job_dict["metrics"] == metrics
        assert isinstance(job_dict["_id"], ObjectId)
    
    def test_to_dict_with_error_message(self):
        """Test TrainingJob serialization includes error message."""
        error_msg = "Insufficient feedback data"
        training_job = TrainingJob(
            user_id=str(ObjectId()),
            job_id="job-789",
            status=TrainingJob.STATUS_FAILED,
            error_message=error_msg
        )
        
        job_dict = training_job.to_dict()
        
        assert job_dict["error_message"] == error_msg
        assert job_dict["status"] == "failed"
    
    def test_from_dict_with_all_fields(self):
        """Test TrainingJob deserialization from dictionary with all fields."""
        _id = ObjectId()
        job_id = "test-job-999"
        user_id = str(ObjectId())
        status = TrainingJob.STATUS_COMPLETED
        created_at = datetime(2024, 1, 15, 10, 30, 0)
        updated_at = datetime(2024, 1, 15, 10, 32, 15)
        started_at = datetime(2024, 1, 15, 10, 30, 5)
        completed_at = datetime(2024, 1, 15, 10, 32, 15)
        metrics = {"final_loss": 0.234, "final_accuracy": 0.89}
        
        data = {
            "_id": _id,
            "user_id": user_id,
            "job_id": job_id,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "error_message": None,
            "metrics": metrics
        }
        
        training_job = TrainingJob.from_dict(data)
        
        assert training_job._id == _id
        assert training_job.user_id == user_id
        assert training_job.job_id == job_id
        assert training_job.status == status
        assert training_job.created_at == created_at
        assert training_job.updated_at == updated_at
        assert training_job.started_at == started_at
        assert training_job.completed_at == completed_at
        assert training_job.error_message is None
        assert training_job.metrics == metrics
    
    def test_from_dict_with_optional_fields_missing(self):
        """Test TrainingJob deserialization with optional fields missing.
        
        Note: created_at and updated_at get default values from __init__
        when not present in the dictionary, which is expected behavior.
        """
        user_id = str(ObjectId())
        job_id = "job-minimal"
        status = TrainingJob.STATUS_QUEUED
        
        data = {
            "user_id": user_id,
            "job_id": job_id,
            "status": status
        }
        
        training_job = TrainingJob.from_dict(data)
        
        assert training_job.user_id == user_id
        assert training_job.job_id == job_id
        assert training_job.status == status
        # created_at and updated_at get default values from __init__
        assert isinstance(training_job.created_at, datetime)
        assert isinstance(training_job.updated_at, datetime)
        assert training_job.started_at is None
        assert training_job.completed_at is None
        assert training_job.error_message is None
        assert training_job.metrics is None
    
    def test_from_dict_with_failed_status_and_error(self):
        """Test TrainingJob deserialization with failed status and error message."""
        error_msg = "Model training timeout"
        data = {
            "user_id": str(ObjectId()),
            "job_id": "job-error",
            "status": TrainingJob.STATUS_FAILED,
            "error_message": error_msg
        }
        
        training_job = TrainingJob.from_dict(data)
        
        assert training_job.status == "failed"
        assert training_job.error_message == error_msg
    
    def test_get_collection(self):
        """Test get_collection returns correct collection name."""
        from unittest.mock import Mock
        
        mock_db = Mock()
        collection = TrainingJob.get_collection(mock_db)
        
        # Verify it accesses the training_jobs collection
        assert collection == mock_db.training_jobs
    
    def test_roundtrip_serialization(self):
        """Test that to_dict() and from_dict() are inverse operations."""
        original = TrainingJob(
            user_id=str(ObjectId()),
            job_id="roundtrip-test",
            status=TrainingJob.STATUS_RUNNING,
            started_at=datetime(2024, 1, 15, 10, 30, 5),
            metrics={"num_samples": 30}
        )
        
        # Serialize and deserialize
        job_dict = original.to_dict()
        restored = TrainingJob.from_dict(job_dict)
        
        # Verify all fields match
        assert restored.user_id == original.user_id
        assert restored.job_id == original.job_id
        assert restored.status == original.status
        assert restored.started_at == original.started_at
        assert restored.metrics == original.metrics
        assert restored._id == original._id


class TestTrainingJobStatusValidation:
    """Test status validation requirements."""
    
    def test_all_valid_statuses_accepted(self):
        """Test that all valid status values are accepted."""
        valid_statuses = ["queued", "running", "completed", "failed"]
        
        for status in valid_statuses:
            training_job = TrainingJob(
                user_id=str(ObjectId()),
                job_id=f"job-{status}",
                status=status
            )
            assert training_job.status == status
    
    def test_invalid_status_rejected(self):
        """Test that invalid status values are rejected."""
        invalid_statuses = [
            "pending",
            "cancelled",
            "QUEUED",  # Wrong case
            "Completed",  # Wrong case
            "",
            "unknown"
        ]
        
        for invalid_status in invalid_statuses:
            with pytest.raises(ValueError, match="Invalid status"):
                TrainingJob(
                    user_id=str(ObjectId()),
                    job_id="job-invalid",
                    status=invalid_status
                )
    
    def test_status_validation_error_message_format(self):
        """Test that status validation error includes valid options."""
        try:
            TrainingJob(
                user_id=str(ObjectId()),
                job_id="job-test",
                status="invalid"
            )
            pytest.fail("Expected ValueError to be raised")
        except ValueError as e:
            error_msg = str(e)
            assert "Invalid status: 'invalid'" in error_msg
            assert "queued" in error_msg
            assert "running" in error_msg
            assert "completed" in error_msg
            assert "failed" in error_msg
