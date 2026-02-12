"""
Unit tests for feedback API endpoint logic.

Tests the feedback endpoint handler logic including request validation,
service integration, and response format.
"""

import sys
import pathlib
import pytest
from unittest.mock import MagicMock, patch
from bson import ObjectId
from fastapi import BackgroundTasks

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.routers.audio import submit_feedback
from app.schemas.audio import FeedbackRequest


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = MagicMock()
    db.audio_sessions = MagicMock()
    db.emotion_analyses = MagicMock()
    db.user_feedback = MagicMock()
    db.users = MagicMock()
    return db


@pytest.fixture
def sample_user_id():
    """Generate a sample user ID."""
    return str(ObjectId())


@pytest.fixture
def sample_session_id():
    """Generate a sample session ID."""
    return str(ObjectId())


@pytest.fixture
def background_tasks():
    """Create a mock BackgroundTasks instance."""
    return MagicMock(spec=BackgroundTasks)


@pytest.mark.asyncio
async def test_submit_feedback_success(mock_db, sample_user_id, sample_session_id, background_tasks):
    """Test successful feedback submission."""
    request = FeedbackRequest(
        session_id=sample_session_id,
        corrected_emotion="sad"
    )
    
    with patch("app.routers.audio.get_database", return_value=mock_db):
        with patch("app.routers.audio.feedback_service.submit_feedback") as mock_submit:
            mock_submit.return_value = {
                "status": "success",
                "feedback_count": 5,
                "training_triggered": False,
                "message": "Feedback recorded (5 total). 15 more samples until next model update."
            }
            
            response = await submit_feedback(request, background_tasks, sample_user_id)
            
            assert response.status == "success"
            assert response.feedback_count == 5
            assert response.training_triggered is False
            assert "5 total" in response.message
            
            # Verify service was called correctly
            mock_submit.assert_called_once_with(
                db=mock_db,
                user_id=sample_user_id,
                session_id=sample_session_id,
                corrected_emotion="sad"
            )
            
            # Verify background task was NOT scheduled
            background_tasks.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_submit_feedback_training_triggered(mock_db, sample_user_id, sample_session_id, background_tasks):
    """Test feedback submission that triggers training."""
    request = FeedbackRequest(
        session_id=sample_session_id,
        corrected_emotion="happy"
    )
    
    with patch("app.routers.audio.get_database", return_value=mock_db):
        with patch("app.routers.audio.feedback_service.submit_feedback") as mock_submit:
            mock_submit.return_value = {
                "status": "success",
                "feedback_count": 20,
                "training_triggered": True,
                "message": "Feedback recorded (20 total). Your personalized model is being updated."
            }
            
            response = await submit_feedback(request, background_tasks, sample_user_id)
            
            assert response.status == "success"
            assert response.feedback_count == 20
            assert response.training_triggered is True
            assert "model is being updated" in response.message
            
            # Verify background task was scheduled
            background_tasks.add_task.assert_called_once()
            # Verify the task is the training function with correct user_id
            call_args = background_tasks.add_task.call_args
            assert call_args[0][1] == sample_user_id  # Second arg is user_id


@pytest.mark.asyncio
async def test_submit_feedback_invalid_emotion(mock_db, sample_user_id, sample_session_id, background_tasks):
    """Test feedback submission with invalid emotion label."""
    from fastapi import HTTPException
    
    request = FeedbackRequest(
        session_id=sample_session_id,
        corrected_emotion="happy"  # Valid for request, but service will reject
    )
    
    with patch("app.routers.audio.get_database", return_value=mock_db):
        with patch("app.routers.audio.feedback_service.submit_feedback") as mock_submit:
            mock_submit.side_effect = ValueError("Invalid emotion label: 'invalid_emotion'")
            
            with pytest.raises(HTTPException) as exc_info:
                await submit_feedback(request, background_tasks, sample_user_id)
            
            assert exc_info.value.status_code == 400
            assert "Invalid emotion label" in exc_info.value.detail


@pytest.mark.asyncio
async def test_submit_feedback_session_not_found(mock_db, sample_user_id, sample_session_id, background_tasks):
    """Test feedback submission when session doesn't exist."""
    from fastapi import HTTPException
    
    request = FeedbackRequest(
        session_id=sample_session_id,
        corrected_emotion="happy"
    )
    
    with patch("app.routers.audio.get_database", return_value=mock_db):
        with patch("app.routers.audio.feedback_service.submit_feedback") as mock_submit:
            mock_submit.side_effect = ValueError(f"Audio session {sample_session_id} not found")
            
            with pytest.raises(HTTPException) as exc_info:
                await submit_feedback(request, background_tasks, sample_user_id)
            
            assert exc_info.value.status_code == 400
            assert "not found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_submit_feedback_wrong_user(mock_db, sample_user_id, sample_session_id, background_tasks):
    """Test feedback submission for another user's session."""
    from fastapi import HTTPException
    
    request = FeedbackRequest(
        session_id=sample_session_id,
        corrected_emotion="sad"
    )
    
    with patch("app.routers.audio.get_database", return_value=mock_db):
        with patch("app.routers.audio.feedback_service.submit_feedback") as mock_submit:
            mock_submit.side_effect = PermissionError(
                f"Session {sample_session_id} does not belong to user {sample_user_id}"
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await submit_feedback(request, background_tasks, sample_user_id)
            
            assert exc_info.value.status_code == 403
            assert "does not belong to user" in exc_info.value.detail


@pytest.mark.asyncio
async def test_submit_feedback_response_format(mock_db, sample_user_id, sample_session_id, background_tasks):
    """Test that response matches the expected schema."""
    request = FeedbackRequest(
        session_id=sample_session_id,
        corrected_emotion="angry"
    )
    
    with patch("app.routers.audio.get_database", return_value=mock_db):
        with patch("app.routers.audio.feedback_service.submit_feedback") as mock_submit:
            mock_submit.return_value = {
                "status": "success",
                "feedback_count": 15,
                "training_triggered": False,
                "message": "Feedback recorded (15 total). 5 more samples until next model update."
            }
            
            response = await submit_feedback(request, background_tasks, sample_user_id)
            
            # Verify all required fields are present
            assert hasattr(response, "status")
            assert hasattr(response, "feedback_count")
            assert hasattr(response, "training_triggered")
            assert hasattr(response, "message")
            
            # Verify field types
            assert isinstance(response.status, str)
            assert isinstance(response.feedback_count, int)
            assert isinstance(response.training_triggered, bool)
            assert isinstance(response.message, str)


@pytest.mark.asyncio
async def test_submit_feedback_internal_error(mock_db, sample_user_id, sample_session_id, background_tasks):
    """Test handling of unexpected internal errors."""
    from fastapi import HTTPException
    
    request = FeedbackRequest(
        session_id=sample_session_id,
        corrected_emotion="happy"
    )
    
    with patch("app.routers.audio.get_database", return_value=mock_db):
        with patch("app.routers.audio.feedback_service.submit_feedback") as mock_submit:
            mock_submit.side_effect = Exception("Database connection failed")
            
            with pytest.raises(HTTPException) as exc_info:
                await submit_feedback(request, background_tasks, sample_user_id)
            
            assert exc_info.value.status_code == 500
            assert "Failed to submit feedback" in exc_info.value.detail


def test_feedback_request_schema_validation():
    """Test FeedbackRequest schema validation."""
    # Valid request
    request = FeedbackRequest(
        session_id="507f1f77bcf86cd799439011",
        corrected_emotion="happy"
    )
    assert request.session_id == "507f1f77bcf86cd799439011"
    assert request.corrected_emotion == "happy"
    
    # Missing fields should raise validation error
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FeedbackRequest(session_id="507f1f77bcf86cd799439011")
    
    with pytest.raises(ValidationError):
        FeedbackRequest(corrected_emotion="happy")


def test_feedback_response_schema():
    """Test FeedbackResponse schema."""
    from app.schemas.audio import FeedbackResponse
    
    response = FeedbackResponse(
        status="success",
        feedback_count=10,
        training_triggered=False,
        message="Feedback recorded"
    )
    
    assert response.status == "success"
    assert response.feedback_count == 10
    assert response.training_triggered is False
    assert response.message == "Feedback recorded"
