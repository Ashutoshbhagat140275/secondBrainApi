"""
Unit Tests for Feedback Router

Tests the REST API endpoints for feedback submission and training status.

Feature: feedback-loop-personalization
Task: 3. CONSOLIDATED: Implement Core Feedback Loop Infrastructure
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from bson import ObjectId

from app.main import app
from app.models.user import User
from app.models.training_job import TrainingJob


client = TestClient(app)


class TestFeedbackEndpoint:
    """Test POST /api/feedback endpoint."""
    
    @patch('app.routers.feedback.submit_feedback')
    @patch('app.routers.feedback.get_current_user')
    @patch('app.routers.feedback.get_database')
    def test_submit_feedback_success_without_training(
        self, mock_db, mock_get_user, mock_submit
    ):
        """Test successful feedback submission without training trigger."""
        # Mock authenticated user
        mock_user = Mock()
        mock_user._id = ObjectId()
        mock_get_user.return_value = mock_user
        
        # Mock feedback service response
        mock_submit.return_value = {
            "status": "success",
            "feedback_count": 15,
            "training_triggered": False,
            "message": "Feedback recorded (15 total). 5 more samples until next model update."
        }
        
        # Make request
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "507f1f77bcf86cd799439011",
                "corrected_emotion": "happy"
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["feedback_count"] == 15
        assert data["training_triggered"] is False
        assert "training_job_id" not in data
        assert "5 more samples" in data["message"]
    
    @patch('app.routers.feedback.submit_feedback')
    @patch('app.routers.feedback.get_current_user')
    @patch('app.routers.feedback.get_database')
    def test_submit_feedback_success_with_training(
        self, mock_db, mock_get_user, mock_submit
    ):
        """Test successful feedback submission with training trigger."""
        # Mock authenticated user
        mock_user = Mock()
        mock_user._id = ObjectId()
        mock_get_user.return_value = mock_user
        
        # Mock feedback service response with training
        mock_submit.return_value = {
            "status": "success",
            "feedback_count": 20,
            "training_triggered": True,
            "training_job_id": "test-job-id-123",
            "message": "Feedback recorded (20 total). Your personalized model is being updated."
        }
        
        # Make request
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "507f1f77bcf86cd799439011",
                "corrected_emotion": "happy"
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["feedback_count"] == 20
        assert data["training_triggered"] is True
        assert data["training_job_id"] == "test-job-id-123"
        assert "model is being updated" in data["message"]
    
    @patch('app.routers.feedback.get_current_user')
    @patch('app.routers.feedback.get_database')
    def test_submit_feedback_invalid_emotion(self, mock_db, mock_get_user):
        """Test feedback submission with invalid emotion label."""
        # Mock authenticated user
        mock_user = Mock()
        mock_user._id = ObjectId()
        mock_get_user.return_value = mock_user
        
        # Make request with invalid emotion
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "507f1f77bcf86cd799439011",
                "corrected_emotion": "invalid_emotion"
            }
        )
        
        # Verify error response
        assert response.status_code == 400
        assert "Invalid emotion label" in response.json()["detail"]
    
    @patch('app.routers.feedback.submit_feedback')
    @patch('app.routers.feedback.get_current_user')
    @patch('app.routers.feedback.get_database')
    def test_submit_feedback_session_not_found(
        self, mock_db, mock_get_user, mock_submit
    ):
        """Test feedback submission for non-existent session."""
        # Mock authenticated user
        mock_user = Mock()
        mock_user._id = ObjectId()
        mock_get_user.return_value = mock_user
        
        # Mock feedback service raising ValueError
        mock_submit.side_effect = ValueError("Audio session not found")
        
        # Make request
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "507f1f77bcf86cd799439011",
                "corrected_emotion": "happy"
            }
        )
        
        # Verify error response
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]
    
    @patch('app.routers.feedback.submit_feedback')
    @patch('app.routers.feedback.get_current_user')
    @patch('app.routers.feedback.get_database')
    def test_submit_feedback_permission_denied(
        self, mock_db, mock_get_user, mock_submit
    ):
        """Test feedback submission for another user's session."""
        # Mock authenticated user
        mock_user = Mock()
        mock_user._id = ObjectId()
        mock_get_user.return_value = mock_user
        
        # Mock feedback service raising PermissionError
        mock_submit.side_effect = PermissionError("Session does not belong to user")
        
        # Make request
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "507f1f77bcf86cd799439011",
                "corrected_emotion": "happy"
            }
        )
        
        # Verify error response
        assert response.status_code == 403
        assert "does not belong to user" in response.json()["detail"]


class TestTrainingStatusEndpoint:
    """Test GET /api/training-status/{user_id} endpoint."""
    
    @patch('app.routers.feedback.get_latest_job')
    @patch('app.routers.feedback.get_current_user')
    @patch('app.routers.feedback.get_database')
    def test_get_training_status_with_completed_job(
        self, mock_db, mock_get_user, mock_get_job
    ):
        """Test querying training status with a completed job."""
        # Mock authenticated user
        user_id = str(ObjectId())
        mock_user = Mock()
        mock_user._id = ObjectId(user_id)
        mock_user.is_admin = False
        mock_get_user.return_value = mock_user
        
        # Mock training job
        mock_job = TrainingJob(
            user_id=user_id,
            job_id="test-job-id",
            status="completed",
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            updated_at=datetime(2024, 1, 15, 10, 32, 15),
            started_at=datetime(2024, 1, 15, 10, 30, 5),
            completed_at=datetime(2024, 1, 15, 10, 32, 15),
            metrics={
                "final_loss": 0.234,
                "final_accuracy": 0.89,
                "num_samples": 25,
                "num_epochs": 20
            }
        )
        mock_get_job.return_value = mock_job
        
        # Make request
        response = client.get(f"/api/training-status/{user_id}")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert data["status"] == "completed"
        assert data["metrics"]["final_accuracy"] == 0.89
    
    @patch('app.routers.feedback.get_latest_job')
    @patch('app.routers.feedback.get_current_user')
    @patch('app.routers.feedback.get_database')
    def test_get_training_status_no_jobs(
        self, mock_db, mock_get_user, mock_get_job
    ):
        """Test querying training status when no jobs exist."""
        # Mock authenticated user
        user_id = str(ObjectId())
        mock_user = Mock()
        mock_user._id = ObjectId(user_id)
        mock_user.is_admin = False
        mock_get_user.return_value = mock_user
        
        # Mock no training jobs
        mock_get_job.return_value = None
        
        # Make request
        response = client.get(f"/api/training-status/{user_id}")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] is None
        assert data["status"] is None
    
    @patch('app.routers.feedback.get_current_user')
    @patch('app.routers.feedback.get_database')
    def test_get_training_status_unauthorized(self, mock_db, mock_get_user):
        """Test querying another user's training status (non-admin)."""
        # Mock authenticated user (not admin)
        user_id = str(ObjectId())
        other_user_id = str(ObjectId())
        mock_user = Mock()
        mock_user._id = ObjectId(user_id)
        mock_user.is_admin = False
        mock_get_user.return_value = mock_user
        
        # Make request for different user
        response = client.get(f"/api/training-status/{other_user_id}")
        
        # Verify error response
        assert response.status_code == 403
        assert "your own training status" in response.json()["detail"]
    
    @patch('app.routers.feedback.get_latest_job')
    @patch('app.routers.feedback.get_current_user')
    @patch('app.routers.feedback.get_database')
    def test_get_training_status_admin_can_query_any_user(
        self, mock_db, mock_get_user, mock_get_job
    ):
        """Test that admin can query any user's training status."""
        # Mock authenticated admin user
        admin_id = str(ObjectId())
        other_user_id = str(ObjectId())
        mock_user = Mock()
        mock_user._id = ObjectId(admin_id)
        mock_user.is_admin = True
        mock_get_user.return_value = mock_user
        
        # Mock training job for other user
        mock_job = TrainingJob(
            user_id=other_user_id,
            job_id="test-job-id",
            status="running",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_get_job.return_value = mock_job
        
        # Make request for different user
        response = client.get(f"/api/training-status/{other_user_id}")
        
        # Verify success
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert data["status"] == "running"


class TestTriggerTrainingEndpoint:
    """Test POST /api/trigger-training/{user_id} endpoint."""
    
    @patch('app.routers.feedback.create_training_job')
    @patch('app.routers.feedback.require_admin')
    @patch('app.routers.feedback.get_database')
    def test_trigger_training_success(
        self, mock_db, mock_require_admin, mock_create_job
    ):
        """Test manually triggering training as admin."""
        # Mock admin user
        admin_user = Mock()
        admin_user._id = ObjectId()
        admin_user.is_admin = True
        mock_require_admin.return_value = admin_user
        
        # Mock database
        user_id = str(ObjectId())
        mock_db.return_value.users.find_one.return_value = {
            "_id": ObjectId(user_id),
            "email": "test@example.com"
        }
        
        # Make request
        response = client.post(f"/api/trigger-training/{user_id}")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "training_job_id" in data
        assert user_id in data["message"]
    
    @patch('app.routers.feedback.require_admin')
    @patch('app.routers.feedback.get_database')
    def test_trigger_training_user_not_found(
        self, mock_db, mock_require_admin
    ):
        """Test triggering training for non-existent user."""
        # Mock admin user
        admin_user = Mock()
        admin_user._id = ObjectId()
        admin_user.is_admin = True
        mock_require_admin.return_value = admin_user
        
        # Mock database - user not found
        user_id = str(ObjectId())
        mock_db.return_value.users.find_one.return_value = None
        
        # Make request
        response = client.post(f"/api/trigger-training/{user_id}")
        
        # Verify error response
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('app.routers.feedback.require_admin')
    @patch('app.routers.feedback.get_database')
    def test_trigger_training_invalid_user_id(
        self, mock_db, mock_require_admin
    ):
        """Test triggering training with invalid user ID format."""
        # Mock admin user
        admin_user = Mock()
        admin_user._id = ObjectId()
        admin_user.is_admin = True
        mock_require_admin.return_value = admin_user
        
        # Make request with invalid user ID
        response = client.post("/api/trigger-training/invalid-id")
        
        # Verify error response
        assert response.status_code == 400
        assert "Invalid user ID format" in response.json()["detail"]


class TestFeedbackRouterIntegration:
    """Integration tests for feedback router."""
    
    def test_feedback_endpoint_exists(self):
        """Test that feedback endpoint is registered."""
        # This will fail with 401 (unauthorized) but confirms endpoint exists
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "507f1f77bcf86cd799439011",
                "corrected_emotion": "happy"
            }
        )
        # Should get 401 (unauthorized) not 404 (not found)
        assert response.status_code in [401, 403, 422]  # Not 404
    
    def test_training_status_endpoint_exists(self):
        """Test that training status endpoint is registered."""
        user_id = str(ObjectId())
        response = client.get(f"/api/training-status/{user_id}")
        # Should get 401 (unauthorized) not 404 (not found)
        assert response.status_code in [401, 403]  # Not 404
    
    def test_trigger_training_endpoint_exists(self):
        """Test that trigger training endpoint is registered."""
        user_id = str(ObjectId())
        response = client.post(f"/api/trigger-training/{user_id}")
        # Should get 401 (unauthorized) not 404 (not found)
        assert response.status_code in [401, 403]  # Not 404
