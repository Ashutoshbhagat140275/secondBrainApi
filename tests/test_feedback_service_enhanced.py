"""
Unit Tests for Enhanced Feedback Service

Tests the feedback service with task queue integration and training triggers.

Feature: feedback-loop-personalization
Task: 3. CONSOLIDATED: Implement Core Feedback Loop Infrastructure
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from bson import ObjectId

from app.services.feedback_service import (
    submit_feedback,
    should_trigger_training,
    calculate_samples_until_training,
)
from app.services.feature_config import (
    MIN_FEEDBACK_FOR_TRAINING,
    INCREMENTAL_TRAINING_INTERVAL,
)


class TestTrainingTriggerLogic:
    """Test training trigger logic functions."""
    
    def test_should_trigger_training_at_minimum(self):
        """Test training triggers at MIN_FEEDBACK_FOR_TRAINING (20)."""
        assert should_trigger_training(20) is True
    
    def test_should_trigger_training_at_intervals(self):
        """Test training triggers at incremental intervals (30, 40, 50, etc.)."""
        assert should_trigger_training(30) is True
        assert should_trigger_training(40) is True
        assert should_trigger_training(50) is True
        assert should_trigger_training(100) is True
    
    def test_should_not_trigger_before_minimum(self):
        """Test training does not trigger before MIN_FEEDBACK_FOR_TRAINING."""
        assert should_trigger_training(0) is False
        assert should_trigger_training(10) is False
        assert should_trigger_training(19) is False
    
    def test_should_not_trigger_between_intervals(self):
        """Test training does not trigger between intervals."""
        assert should_trigger_training(21) is False
        assert should_trigger_training(25) is False
        assert should_trigger_training(29) is False
        assert should_trigger_training(35) is False
    
    def test_calculate_samples_until_training_before_minimum(self):
        """Test remaining samples calculation before first training."""
        assert calculate_samples_until_training(0) == 20
        assert calculate_samples_until_training(10) == 10
        assert calculate_samples_until_training(19) == 1
    
    def test_calculate_samples_until_training_after_minimum(self):
        """Test remaining samples calculation after first training."""
        assert calculate_samples_until_training(20) == 10
        assert calculate_samples_until_training(21) == 9
        assert calculate_samples_until_training(25) == 5
        assert calculate_samples_until_training(29) == 1
        assert calculate_samples_until_training(30) == 10
    
    def test_calculate_samples_always_positive(self):
        """Test that remaining samples is always positive."""
        for count in range(0, 100):
            remaining = calculate_samples_until_training(count)
            assert remaining > 0, f"Remaining should be > 0 for count={count}"


class TestFeedbackServiceWithTaskQueue:
    """Test feedback service with task queue integration."""
    
    @patch('app.services.feedback_service.AudioSession')
    @patch('app.services.feedback_service.EmotionAnalysis')
    @patch('app.services.feedback_service.UserFeedback')
    @patch('app.services.feedback_service.User')
    def test_submit_feedback_without_task_queue(
        self, mock_user, mock_feedback, mock_emotion, mock_audio
    ):
        """Test feedback submission without task queue (no training trigger)."""
        # Setup mocks
        mock_db = Mock()
        user_id = str(ObjectId())
        session_id = str(ObjectId())
        
        # Mock audio session
        mock_audio.get_collection.return_value.find_one.return_value = {
            "_id": ObjectId(session_id),
            "user_id": user_id,
            "emotion_data": {"emotion": "sad"}
        }
        
        # Mock emotion analysis
        mock_emotion.get_collection.return_value.find_one.return_value = {
            "session_id": session_id,
            "mfcc_features": [0.1] * 768
        }
        
        # Mock user feedback count (not at trigger threshold)
        mock_user.increment_feedback_count.return_value = 15
        
        # Submit feedback without task queue
        result = submit_feedback(
            db=mock_db,
            user_id=user_id,
            session_id=session_id,
            corrected_emotion="happy",
            task_queue=None
        )
        
        # Verify response
        assert result["status"] == "success"
        assert result["feedback_count"] == 15
        assert result["training_triggered"] is False
        assert "training_job_id" not in result
        assert "5 more samples" in result["message"]
    
    @patch('app.services.feedback_service.create_training_job')
    @patch('app.services.feedback_service.AudioSession')
    @patch('app.services.feedback_service.EmotionAnalysis')
    @patch('app.services.feedback_service.UserFeedback')
    @patch('app.services.feedback_service.User')
    def test_submit_feedback_with_task_queue_triggers_training(
        self, mock_user, mock_feedback, mock_emotion, mock_audio, mock_create_job
    ):
        """Test feedback submission with task queue triggers training at threshold."""
        # Setup mocks
        mock_db = Mock()
        mock_task_queue = Mock()
        mock_task_queue.enqueue.return_value = "test-job-id-123"
        
        user_id = str(ObjectId())
        session_id = str(ObjectId())
        
        # Mock audio session
        mock_audio.get_collection.return_value.find_one.return_value = {
            "_id": ObjectId(session_id),
            "user_id": user_id,
            "emotion_data": {"emotion": "sad"}
        }
        
        # Mock emotion analysis
        mock_emotion.get_collection.return_value.find_one.return_value = {
            "session_id": session_id,
            "mfcc_features": [0.1] * 768
        }
        
        # Mock user feedback count at trigger threshold
        mock_user.increment_feedback_count.return_value = 20
        
        # Submit feedback with task queue
        result = submit_feedback(
            db=mock_db,
            user_id=user_id,
            session_id=session_id,
            corrected_emotion="happy",
            task_queue=mock_task_queue
        )
        
        # Verify response
        assert result["status"] == "success"
        assert result["feedback_count"] == 20
        assert result["training_triggered"] is True
        assert result["training_job_id"] == "test-job-id-123"
        assert "model is being updated" in result["message"]
        
        # Verify task was enqueued
        mock_task_queue.enqueue.assert_called_once()
        
        # Verify training job was created
        mock_create_job.assert_called_once_with(mock_db, user_id, "test-job-id-123")
    
    @patch('app.services.feedback_service.AudioSession')
    def test_submit_feedback_invalid_emotion_raises_error(self, mock_audio):
        """Test that invalid emotion label raises ValueError."""
        mock_db = Mock()
        user_id = str(ObjectId())
        session_id = str(ObjectId())
        
        with pytest.raises(ValueError, match="Invalid emotion label"):
            submit_feedback(
                db=mock_db,
                user_id=user_id,
                session_id=session_id,
                corrected_emotion="invalid_emotion",
                task_queue=None
            )
    
    @patch('app.services.feedback_service.AudioSession')
    def test_submit_feedback_session_not_found_raises_error(self, mock_audio):
        """Test that non-existent session raises ValueError."""
        mock_db = Mock()
        mock_audio.get_collection.return_value.find_one.return_value = None
        
        user_id = str(ObjectId())
        session_id = str(ObjectId())
        
        with pytest.raises(ValueError, match="not found"):
            submit_feedback(
                db=mock_db,
                user_id=user_id,
                session_id=session_id,
                corrected_emotion="happy",
                task_queue=None
            )
    
    @patch('app.services.feedback_service.AudioSession')
    def test_submit_feedback_wrong_user_raises_permission_error(self, mock_audio):
        """Test that accessing another user's session raises PermissionError."""
        mock_db = Mock()
        user_id = str(ObjectId())
        other_user_id = str(ObjectId())
        session_id = str(ObjectId())
        
        # Mock audio session belonging to different user
        mock_audio.get_collection.return_value.find_one.return_value = {
            "_id": ObjectId(session_id),
            "user_id": other_user_id,  # Different user
            "emotion_data": {"emotion": "sad"}
        }
        
        with pytest.raises(PermissionError, match="does not belong to user"):
            submit_feedback(
                db=mock_db,
                user_id=user_id,
                session_id=session_id,
                corrected_emotion="happy",
                task_queue=None
            )


class TestFeedbackResponseFormatting:
    """Test feedback response message formatting."""
    
    def test_response_message_before_first_training(self):
        """Test response message format before first training."""
        # At 15 feedback samples, need 5 more for first training
        feedback_count = 15
        training_triggered = should_trigger_training(feedback_count)
        samples_until = calculate_samples_until_training(feedback_count)
        
        assert training_triggered is False
        assert samples_until == 5
    
    def test_response_message_at_first_training(self):
        """Test response message format at first training trigger."""
        feedback_count = 20
        training_triggered = should_trigger_training(feedback_count)
        
        assert training_triggered is True
    
    def test_response_message_between_trainings(self):
        """Test response message format between training triggers."""
        # At 25 feedback samples, need 5 more for next training
        feedback_count = 25
        training_triggered = should_trigger_training(feedback_count)
        samples_until = calculate_samples_until_training(feedback_count)
        
        assert training_triggered is False
        assert samples_until == 5
    
    def test_response_message_at_incremental_training(self):
        """Test response message format at incremental training trigger."""
        feedback_count = 30
        training_triggered = should_trigger_training(feedback_count)
        
        assert training_triggered is True
