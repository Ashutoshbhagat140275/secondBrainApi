"""
Unit tests for feedback service.

Tests the feedback submission logic including validation, storage,
and training trigger determination.
"""

import pytest
from datetime import datetime
from bson import ObjectId
from unittest.mock import MagicMock, patch

from app.services.feedback_service import submit_feedback
from app.services.feature_config import EMOTION_LABELS


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = MagicMock()
    
    # Mock collections
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
def sample_embedding():
    """Generate a sample 768-dimensional embedding."""
    return [0.1] * 768


def test_submit_feedback_success(mock_db, sample_user_id, sample_session_id, sample_embedding):
    """Test successful feedback submission."""
    # Setup mock data
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(sample_session_id),
        "user_id": sample_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    mock_db.emotion_analyses.find_one.return_value = {
        "session_id": sample_session_id,
        "mfcc_features": sample_embedding
    }
    
    mock_db.user_feedback.insert_one.return_value = MagicMock()
    
    # Mock User.increment_feedback_count
    with patch("app.services.feedback_service.User.increment_feedback_count", return_value=5):
        result = submit_feedback(
            db=mock_db,
            user_id=sample_user_id,
            session_id=sample_session_id,
            corrected_emotion="sad"
        )
    
    # Verify result
    assert result["status"] == "success"
    assert result["feedback_count"] == 5
    assert result["training_triggered"] is False
    assert "5 total" in result["message"]
    
    # Verify feedback was stored
    mock_db.user_feedback.insert_one.assert_called_once()


def test_submit_feedback_invalid_emotion(mock_db, sample_user_id, sample_session_id):
    """Test feedback submission with invalid emotion label."""
    with pytest.raises(ValueError, match="Invalid emotion label"):
        submit_feedback(
            db=mock_db,
            user_id=sample_user_id,
            session_id=sample_session_id,
            corrected_emotion="invalid_emotion"
        )


def test_submit_feedback_session_not_found(mock_db, sample_user_id, sample_session_id):
    """Test feedback submission when session doesn't exist."""
    mock_db.audio_sessions.find_one.return_value = None
    
    with pytest.raises(ValueError, match="Audio session .* not found"):
        submit_feedback(
            db=mock_db,
            user_id=sample_user_id,
            session_id=sample_session_id,
            corrected_emotion="happy"
        )


def test_submit_feedback_wrong_user(mock_db, sample_user_id, sample_session_id):
    """Test feedback submission for another user's session."""
    different_user_id = str(ObjectId())
    
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(sample_session_id),
        "user_id": different_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    with pytest.raises(PermissionError, match="does not belong to user"):
        submit_feedback(
            db=mock_db,
            user_id=sample_user_id,
            session_id=sample_session_id,
            corrected_emotion="sad"
        )


def test_submit_feedback_no_emotion_analysis(mock_db, sample_user_id, sample_session_id):
    """Test feedback submission when emotion analysis is missing."""
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(sample_session_id),
        "user_id": sample_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    mock_db.emotion_analyses.find_one.return_value = None
    
    with pytest.raises(ValueError, match="No emotion analysis found"):
        submit_feedback(
            db=mock_db,
            user_id=sample_user_id,
            session_id=sample_session_id,
            corrected_emotion="sad"
        )


def test_submit_feedback_training_trigger_at_20(
    mock_db, sample_user_id, sample_session_id, sample_embedding
):
    """Test that training is triggered at 20 feedback samples."""
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(sample_session_id),
        "user_id": sample_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    mock_db.emotion_analyses.find_one.return_value = {
        "session_id": sample_session_id,
        "mfcc_features": sample_embedding
    }
    
    mock_db.user_feedback.insert_one.return_value = MagicMock()
    
    # Mock feedback count reaching 20
    with patch("app.services.feedback_service.User.increment_feedback_count", return_value=20):
        result = submit_feedback(
            db=mock_db,
            user_id=sample_user_id,
            session_id=sample_session_id,
            corrected_emotion="sad"
        )
    
    assert result["status"] == "success"
    assert result["feedback_count"] == 20
    assert result["training_triggered"] is True
    assert "model is being updated" in result["message"]


def test_submit_feedback_training_trigger_at_30(
    mock_db, sample_user_id, sample_session_id, sample_embedding
):
    """Test that training is triggered at 30 feedback samples (every 10)."""
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(sample_session_id),
        "user_id": sample_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    mock_db.emotion_analyses.find_one.return_value = {
        "session_id": sample_session_id,
        "mfcc_features": sample_embedding
    }
    
    mock_db.user_feedback.insert_one.return_value = MagicMock()
    
    # Mock feedback count reaching 30
    with patch("app.services.feedback_service.User.increment_feedback_count", return_value=30):
        result = submit_feedback(
            db=mock_db,
            user_id=sample_user_id,
            session_id=sample_session_id,
            corrected_emotion="sad"
        )
    
    assert result["status"] == "success"
    assert result["feedback_count"] == 30
    assert result["training_triggered"] is True


def test_submit_feedback_no_training_at_25(
    mock_db, sample_user_id, sample_session_id, sample_embedding
):
    """Test that training is NOT triggered at 25 feedback samples."""
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(sample_session_id),
        "user_id": sample_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    mock_db.emotion_analyses.find_one.return_value = {
        "session_id": sample_session_id,
        "mfcc_features": sample_embedding
    }
    
    mock_db.user_feedback.insert_one.return_value = MagicMock()
    
    # Mock feedback count at 25 (not a multiple of 10)
    with patch("app.services.feedback_service.User.increment_feedback_count", return_value=25):
        result = submit_feedback(
            db=mock_db,
            user_id=sample_user_id,
            session_id=sample_session_id,
            corrected_emotion="sad"
        )
    
    assert result["status"] == "success"
    assert result["feedback_count"] == 25
    assert result["training_triggered"] is False
    assert "5 more samples" in result["message"]


def test_submit_feedback_stores_correct_data(
    mock_db, sample_user_id, sample_session_id, sample_embedding
):
    """Test that feedback is stored with correct data structure."""
    predicted_emotion = "happy"
    corrected_emotion = "sad"
    
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(sample_session_id),
        "user_id": sample_user_id,
        "emotion_data": {"emotion": predicted_emotion}
    }
    
    mock_db.emotion_analyses.find_one.return_value = {
        "session_id": sample_session_id,
        "mfcc_features": sample_embedding
    }
    
    mock_db.user_feedback.insert_one.return_value = MagicMock()
    
    with patch("app.services.feedback_service.User.increment_feedback_count", return_value=1):
        submit_feedback(
            db=mock_db,
            user_id=sample_user_id,
            session_id=sample_session_id,
            corrected_emotion=corrected_emotion
        )
    
    # Verify the stored feedback structure
    call_args = mock_db.user_feedback.insert_one.call_args[0][0]
    assert call_args["user_id"] == sample_user_id
    assert call_args["session_id"] == sample_session_id
    assert call_args["predicted_emotion"] == predicted_emotion
    assert call_args["corrected_emotion"] == corrected_emotion
    assert call_args["embedding"] == sample_embedding
    assert "timestamp" in call_args


def test_submit_feedback_all_emotion_labels(
    mock_db, sample_user_id, sample_session_id, sample_embedding
):
    """Test that all valid emotion labels are accepted."""
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(sample_session_id),
        "user_id": sample_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    mock_db.emotion_analyses.find_one.return_value = {
        "session_id": sample_session_id,
        "mfcc_features": sample_embedding
    }
    
    mock_db.user_feedback.insert_one.return_value = MagicMock()
    
    # Test each emotion label
    for emotion in EMOTION_LABELS:
        with patch("app.services.feedback_service.User.increment_feedback_count", return_value=1):
            result = submit_feedback(
                db=mock_db,
                user_id=sample_user_id,
                session_id=sample_session_id,
                corrected_emotion=emotion
            )
            assert result["status"] == "success"
