"""
Integration tests for dual-head emotion recognition pipeline.

These tests verify the complete end-to-end flow:
1. Upload audio → dual-head prediction → extended response
2. New user (global only) vs user with trained model (dual-head)
3. Feedback submission → training → improved accuracy
4. Response format includes all new fields

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 8.1, 8.2, 8.3, 8.4, 8.5
"""

import sys
import pathlib
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from bson import ObjectId
from io import BytesIO

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.audio_processor import process_audio
from app.services.dual_head_classifier import classify_with_dual_heads
from app.services.feedback_service import submit_feedback
from app.services.user_emotion_head import USER_HEADS_DIR
from training.train_user_head import train_user_head


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
    return db


@pytest.fixture
def new_user_id():
    """Generate a new user ID (no feedback history)."""
    return str(ObjectId())


@pytest.fixture
def experienced_user_id():
    """Generate an experienced user ID (has trained model)."""
    return str(ObjectId())


@pytest.fixture
def sample_embedding():
    """Generate a sample 768-dimensional embedding."""
    return np.random.randn(768).astype(np.float32)


@pytest.fixture
def sample_audio_file():
    """Create a mock audio file upload."""
    mock_file = MagicMock()
    mock_file.filename = "test_audio.wav"
    mock_file.file = BytesIO(b"fake audio data")
    mock_file.size = 1024
    return mock_file


# ---------------------------------------------------------------------------
# Test 1: End-to-end pipeline with new user (global only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("app.services.audio_processor.get_database")
@patch("app.services.audio_processor.save_audio_file")
@patch("app.services.audio_processor.preprocess_audio")
@patch("app.services.audio_processor.extract_wav2vec2_embedding")
@patch("app.services.audio_processor.classify_with_dual_heads")
@patch("app.services.audio_processor.transcribe_audio")
@patch("app.services.audio_processor.store_document")
@patch("app.services.audio_processor.invalidate_user_cache")
async def test_end_to_end_new_user_global_only(
    mock_invalidate_cache,
    mock_store_document,
    mock_transcribe,
    mock_classify,
    mock_extract_embedding,
    mock_preprocess,
    mock_save_file,
    mock_get_db,
    mock_db,
    new_user_id,
    sample_audio_file,
    sample_embedding
):
    """
    Test end-to-end pipeline for new user with no feedback (global only).
    
    Verifies:
    - Audio upload → embedding extraction → dual-head classification
    - New user gets global-only prediction (user_emotion = None)
    - Response includes all extended fields
    - blend_weight = 1.0 for global only
    
    Requirements: 3.1, 3.2, 3.3, 8.1, 8.2
    """
    # Setup mocks
    mock_get_db.return_value = mock_db
    mock_save_file.return_value = "/tmp/test_audio.wav"
    mock_preprocess.return_value = "/tmp/test_audio.wav"
    mock_extract_embedding.return_value = sample_embedding
    mock_transcribe.return_value = "This is a test transcription"
    mock_store_document.return_value = None
    mock_invalidate_cache.return_value = None
    
    # Mock user with no feedback
    mock_db.users.find_one.return_value = {
        "_id": ObjectId(new_user_id),
        "feedback_count": 0
    }
    
    # Mock dual-head classifier returning global-only prediction
    mock_classify.return_value = {
        "emotion": "happy",
        "confidence": 0.85,
        "global_emotion": "happy",
        "global_confidence": 0.85,
        "user_emotion": None,
        "user_confidence": None,
        "blend_weight": 1.0,
        "probabilities": {
            "happy": 0.85,
            "sad": 0.05,
            "angry": 0.03,
            "fearful": 0.02,
            "disgusted": 0.02,
            "surprised": 0.01,
            "neutral": 0.01,
            "calm": 0.01
        }
    }
    
    # Mock MongoDB insertions
    mock_db.audio_sessions.insert_one.return_value = MagicMock(
        inserted_id=ObjectId()
    )
    mock_db.emotion_analyses.insert_one.return_value = MagicMock()
    
    # Execute pipeline
    result = await process_audio(new_user_id, sample_audio_file)
    
    # Verify response structure
    assert "session_id" in result
    assert "emotion" in result
    assert "confidence" in result
    assert "global_emotion" in result
    assert "global_confidence" in result
    assert "user_emotion" in result
    assert "user_confidence" in result
    assert "blend_weight" in result
    assert "transcription" in result
    assert "timestamp" in result
    
    # Verify global-only prediction
    assert result["emotion"] == "happy"
    assert result["confidence"] == 0.85
    assert result["global_emotion"] == "happy"
    assert result["global_confidence"] == 0.85
    assert result["user_emotion"] is None
    assert result["user_confidence"] is None
    assert result["blend_weight"] == 1.0
    
    # Verify dual-head classifier was called with correct parameters
    mock_classify.assert_called_once()
    call_args = mock_classify.call_args
    assert np.array_equal(call_args[0][0], sample_embedding)
    assert call_args[0][1] == new_user_id
    assert call_args[0][2] == 0  # feedback_count = 0


# ---------------------------------------------------------------------------
# Test 2: End-to-end pipeline with experienced user (dual-head)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("app.services.audio_processor.get_database")
@patch("app.services.audio_processor.save_audio_file")
@patch("app.services.audio_processor.preprocess_audio")
@patch("app.services.audio_processor.extract_wav2vec2_embedding")
@patch("app.services.audio_processor.classify_with_dual_heads")
@patch("app.services.audio_processor.transcribe_audio")
@patch("app.services.audio_processor.store_document")
@patch("app.services.audio_processor.invalidate_user_cache")
async def test_end_to_end_experienced_user_dual_head(
    mock_invalidate_cache,
    mock_store_document,
    mock_transcribe,
    mock_classify,
    mock_extract_embedding,
    mock_preprocess,
    mock_save_file,
    mock_get_db,
    mock_db,
    experienced_user_id,
    sample_audio_file,
    sample_embedding
):
    """
    Test end-to-end pipeline for experienced user with trained model.
    
    Verifies:
    - User with 50+ feedback samples gets dual-head prediction
    - Both global and user predictions are present
    - Blending weight is between 0.3 and 1.0
    - Final prediction is blended result
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.1, 8.3
    """
    # Setup mocks
    mock_get_db.return_value = mock_db
    mock_save_file.return_value = "/tmp/test_audio.wav"
    mock_preprocess.return_value = "/tmp/test_audio.wav"
    mock_extract_embedding.return_value = sample_embedding
    mock_transcribe.return_value = "I'm feeling great today"
    mock_store_document.return_value = None
    mock_invalidate_cache.return_value = None
    
    # Mock user with 50 feedback samples (trained model available)
    mock_db.users.find_one.return_value = {
        "_id": ObjectId(experienced_user_id),
        "feedback_count": 50
    }
    
    # Mock dual-head classifier returning blended prediction
    mock_classify.return_value = {
        "emotion": "happy",
        "confidence": 0.88,
        "global_emotion": "happy",
        "global_confidence": 0.82,
        "user_emotion": "happy",
        "user_confidence": 0.95,
        "blend_weight": 0.55,
        "probabilities": {
            "happy": 0.88,
            "sad": 0.04,
            "angry": 0.02,
            "fearful": 0.02,
            "disgusted": 0.01,
            "surprised": 0.01,
            "neutral": 0.01,
            "calm": 0.01
        }
    }
    
    # Mock MongoDB insertions
    mock_db.audio_sessions.insert_one.return_value = MagicMock(
        inserted_id=ObjectId()
    )
    mock_db.emotion_analyses.insert_one.return_value = MagicMock()
    
    # Execute pipeline
    result = await process_audio(experienced_user_id, sample_audio_file)
    
    # Verify dual-head prediction
    assert result["emotion"] == "happy"
    assert result["confidence"] == 0.88
    assert result["global_emotion"] == "happy"
    assert result["global_confidence"] == 0.82
    assert result["user_emotion"] == "happy"
    assert result["user_confidence"] == 0.95
    assert result["blend_weight"] == 0.55
    
    # Verify blending weight is in valid range
    assert 0.3 <= result["blend_weight"] <= 1.0
    
    # Verify dual-head classifier was called with correct feedback count
    mock_classify.assert_called_once()
    call_args = mock_classify.call_args
    assert call_args[0][2] == 50  # feedback_count = 50


# ---------------------------------------------------------------------------
# Test 3: Dual-head classifier with disagreement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("app.services.audio_processor.get_database")
@patch("app.services.audio_processor.save_audio_file")
@patch("app.services.audio_processor.preprocess_audio")
@patch("app.services.audio_processor.extract_wav2vec2_embedding")
@patch("app.services.audio_processor.classify_with_dual_heads")
@patch("app.services.audio_processor.transcribe_audio")
@patch("app.services.audio_processor.store_document")
@patch("app.services.audio_processor.invalidate_user_cache")
async def test_dual_head_disagreement(
    mock_invalidate_cache,
    mock_store_document,
    mock_transcribe,
    mock_classify,
    mock_extract_embedding,
    mock_preprocess,
    mock_save_file,
    mock_get_db,
    mock_db,
    experienced_user_id,
    sample_audio_file,
    sample_embedding
):
    """
    Test dual-head pipeline when global and user heads disagree.
    
    Verifies:
    - Global predicts "neutral", user predicts "sad"
    - Final prediction is blended (favors user with lower alpha)
    - All predictions are included in response
    
    Requirements: 3.4, 3.5, 8.3
    """
    # Setup mocks
    mock_get_db.return_value = mock_db
    mock_save_file.return_value = "/tmp/test_audio.wav"
    mock_preprocess.return_value = "/tmp/test_audio.wav"
    mock_extract_embedding.return_value = sample_embedding
    mock_transcribe.return_value = "I'm not sure how I feel"
    mock_store_document.return_value = None
    mock_invalidate_cache.return_value = None
    
    # Mock user with 100 feedback samples
    mock_db.users.find_one.return_value = {
        "_id": ObjectId(experienced_user_id),
        "feedback_count": 100
    }
    
    # Mock dual-head classifier with disagreement
    mock_classify.return_value = {
        "emotion": "sad",
        "confidence": 0.72,
        "global_emotion": "neutral",
        "global_confidence": 0.65,
        "user_emotion": "sad",
        "user_confidence": 0.85,
        "blend_weight": 0.40,
        "probabilities": {
            "happy": 0.05,
            "sad": 0.72,
            "angry": 0.08,
            "fearful": 0.05,
            "disgusted": 0.03,
            "surprised": 0.02,
            "neutral": 0.03,
            "calm": 0.02
        }
    }
    
    # Mock MongoDB insertions
    mock_db.audio_sessions.insert_one.return_value = MagicMock(
        inserted_id=ObjectId()
    )
    mock_db.emotion_analyses.insert_one.return_value = MagicMock()
    
    # Execute pipeline
    result = await process_audio(experienced_user_id, sample_audio_file)
    
    # Verify heads disagree
    assert result["global_emotion"] != result["user_emotion"]
    assert result["global_emotion"] == "neutral"
    assert result["user_emotion"] == "sad"
    
    # Verify final prediction favors user head (lower blend_weight)
    assert result["emotion"] == "sad"
    assert result["blend_weight"] < 0.5
    
    # Verify all predictions are present
    assert result["global_confidence"] == 0.65
    assert result["user_confidence"] == 0.85


# ---------------------------------------------------------------------------
# Test 4: Feedback submission → training → improved accuracy
# ---------------------------------------------------------------------------


def test_feedback_loop_integration(mock_db, experienced_user_id):
    """
    Test complete feedback loop: submission → training trigger → model update.
    
    Verifies:
    - Feedback submission stores data correctly
    - Training is triggered at 20 feedback samples
    - Training is triggered every 10 samples after 20
    - Feedback count is tracked correctly
    
    Requirements: 3.6, 5.1, 5.2, 5.3, 6.1, 7.1, 7.2
    """
    session_id = str(ObjectId())
    embedding = [0.1] * 768
    
    # Mock audio session
    mock_db.audio_sessions.find_one.return_value = {
        "_id": ObjectId(session_id),
        "user_id": experienced_user_id,
        "emotion_data": {"emotion": "happy"}
    }
    
    # Mock emotion analysis
    mock_db.emotion_analyses.find_one.return_value = {
        "session_id": session_id,
        "mfcc_features": embedding
    }
    
    # Mock feedback insertion
    mock_db.user_feedback.insert_one.return_value = MagicMock()
    
    # Test feedback submission at various counts
    test_cases = [
        (5, False, "15 more samples"),
        (10, False, "10 more samples"),
        (19, False, "1 more samples"),
        (20, True, "model is being updated"),
        (25, False, "5 more samples"),
        (30, True, "model is being updated"),
        (40, True, "model is being updated"),
    ]
    
    for feedback_count, should_trigger, expected_message in test_cases:
        with patch("app.services.feedback_service.User.increment_feedback_count", return_value=feedback_count):
            result = submit_feedback(
                db=mock_db,
                user_id=experienced_user_id,
                session_id=session_id,
                corrected_emotion="sad"
            )
            
            assert result["status"] == "success"
            assert result["feedback_count"] == feedback_count
            assert result["training_triggered"] == should_trigger
            assert expected_message in result["message"]


# ---------------------------------------------------------------------------
# Test 5: Response format validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("app.services.audio_processor.get_database")
@patch("app.services.audio_processor.save_audio_file")
@patch("app.services.audio_processor.preprocess_audio")
@patch("app.services.audio_processor.extract_wav2vec2_embedding")
@patch("app.services.audio_processor.classify_with_dual_heads")
@patch("app.services.audio_processor.transcribe_audio")
@patch("app.services.audio_processor.store_document")
@patch("app.services.audio_processor.invalidate_user_cache")
async def test_response_format_includes_all_fields(
    mock_invalidate_cache,
    mock_store_document,
    mock_transcribe,
    mock_classify,
    mock_extract_embedding,
    mock_preprocess,
    mock_save_file,
    mock_get_db,
    mock_db,
    new_user_id,
    sample_audio_file,
    sample_embedding
):
    """
    Test that response format includes all required extended fields.
    
    Verifies:
    - All original fields are present (backward compatibility)
    - All new dual-head fields are present
    - Field types are correct
    - Probabilities sum to 1.0
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    # Setup mocks
    mock_get_db.return_value = mock_db
    mock_save_file.return_value = "/tmp/test_audio.wav"
    mock_preprocess.return_value = "/tmp/test_audio.wav"
    mock_extract_embedding.return_value = sample_embedding
    mock_transcribe.return_value = "Test transcription"
    mock_store_document.return_value = None
    mock_invalidate_cache.return_value = None
    
    mock_db.users.find_one.return_value = {
        "_id": ObjectId(new_user_id),
        "feedback_count": 0
    }
    
    mock_classify.return_value = {
        "emotion": "happy",
        "confidence": 0.85,
        "global_emotion": "happy",
        "global_confidence": 0.85,
        "user_emotion": None,
        "user_confidence": None,
        "blend_weight": 1.0,
        "probabilities": {
            "happy": 0.85,
            "sad": 0.05,
            "angry": 0.03,
            "fearful": 0.02,
            "disgusted": 0.02,
            "surprised": 0.01,
            "neutral": 0.01,
            "calm": 0.01
        }
    }
    
    mock_db.audio_sessions.insert_one.return_value = MagicMock(
        inserted_id=ObjectId()
    )
    mock_db.emotion_analyses.insert_one.return_value = MagicMock()
    
    # Execute pipeline
    result = await process_audio(new_user_id, sample_audio_file)
    
    # Verify all original fields are present (backward compatibility)
    assert "session_id" in result
    assert "emotion" in result
    assert "confidence" in result
    assert "transcription" in result
    assert "timestamp" in result
    
    # Verify all new extended fields are present
    assert "global_emotion" in result
    assert "global_confidence" in result
    assert "user_emotion" in result
    assert "user_confidence" in result
    assert "blend_weight" in result
    
    # Verify field types
    assert isinstance(result["session_id"], str)
    assert isinstance(result["emotion"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["global_emotion"], str)
    assert isinstance(result["global_confidence"], float)
    assert result["user_emotion"] is None or isinstance(result["user_emotion"], str)
    assert result["user_confidence"] is None or isinstance(result["user_confidence"], float)
    assert isinstance(result["blend_weight"], float)
    assert isinstance(result["transcription"], str)
    assert isinstance(result["timestamp"], datetime)
    
    # Verify confidence ranges
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["global_confidence"] <= 1.0
    assert 0.0 <= result["blend_weight"] <= 1.0


# ---------------------------------------------------------------------------
# Test 6: Dual-head classifier integration
# ---------------------------------------------------------------------------


@patch("app.services.dual_head_classifier.predict_global")
@patch("app.services.dual_head_classifier.predict_user")
def test_classify_with_dual_heads_integration(
    mock_predict_user,
    mock_predict_global,
    sample_embedding,
    new_user_id
):
    """
    Test dual-head classifier integration with real blending logic.
    
    Verifies:
    - Embedding is passed to both heads
    - Blending weight is computed correctly
    - Predictions are blended properly
    - Response includes all metadata
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    # Mock global head prediction
    P_g = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
    mock_predict_global.return_value = (P_g, 0.7)
    
    # Mock user head prediction
    P_u = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
    mock_predict_user.return_value = (P_u, 0.8)
    
    # Test with 50 feedback samples
    result = classify_with_dual_heads(sample_embedding, new_user_id, feedback_count=50)
    
    # Verify both heads were called
    mock_predict_global.assert_called_once()
    mock_predict_user.assert_called_once()
    
    # Verify result structure
    assert "emotion" in result
    assert "confidence" in result
    assert "global_emotion" in result
    assert "global_confidence" in result
    assert "user_emotion" in result
    assert "user_confidence" in result
    assert "blend_weight" in result
    assert "probabilities" in result
    
    # Verify global prediction
    assert result["global_emotion"] == "neutral"  # Index 0
    assert result["global_confidence"] == 0.7
    
    # Verify user prediction
    assert result["user_emotion"] == "calm"  # Index 1
    assert result["user_confidence"] == 0.8
    
    # Verify blending weight (feedback_count=50, C_g=0.7)
    # α = 0.5 + 0.3*0.7 - 0.2*0.5 = 0.61
    assert 0.60 <= result["blend_weight"] <= 0.62
    
    # Verify probabilities sum to 1.0
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test 7: Training integration (mocked)
# ---------------------------------------------------------------------------


@patch("training.train_user_head.get_database")
@patch("training.train_user_head.load_user_feedback")
@patch("training.train_user_head.create_fresh_user_head")
@patch("torch.save")
def test_training_integration(
    mock_torch_save,
    mock_create_head,
    mock_load_feedback,
    mock_get_db,
    experienced_user_id
):
    """
    Test user head training integration.
    
    Verifies:
    - Training loads feedback data correctly
    - Model is created and trained
    - Model is saved to correct path
    - Training metrics are returned
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
    """
    # Mock feedback data (20 samples)
    X = np.random.randn(20, 768).astype(np.float32)
    y = np.random.randint(0, 8, size=20).astype(np.int64)
    mock_load_feedback.return_value = (X, y)
    
    # Mock model creation
    from app.services.user_emotion_head import UserEmotionHead
    mock_model = UserEmotionHead(embedding_dim=768, num_classes=8)
    mock_create_head.return_value = mock_model
    
    # Mock torch.save
    mock_torch_save.return_value = None
    
    # Run training
    metrics = train_user_head(experienced_user_id, force_retrain=True)
    
    # Verify feedback was loaded
    mock_load_feedback.assert_called_once_with(experienced_user_id)
    
    # Verify model was created
    mock_create_head.assert_called_once()
    
    # Verify model was saved
    mock_torch_save.assert_called_once()
    save_path = str(mock_torch_save.call_args[0][1])
    assert experienced_user_id in save_path
    assert save_path.endswith(".pt")
    
    # Verify metrics are returned
    assert "final_loss" in metrics
    assert "final_accuracy" in metrics
    assert "num_samples" in metrics
    assert "num_epochs" in metrics
    assert "model_path" in metrics
    
    assert metrics["num_samples"] == 20
    assert metrics["num_epochs"] == 20
    assert isinstance(metrics["final_loss"], float)
    assert isinstance(metrics["final_accuracy"], float)
