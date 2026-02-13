"""
Unit tests for audio upload API response format.

Tests that the AudioUploadResponse schema includes all required fields
for dual-head emotion classification system.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import sys
import pathlib
import pytest
from datetime import datetime
from pydantic import ValidationError

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.schemas.audio import AudioUploadResponse


def test_audio_upload_response_with_global_only():
    """
    Test response format when only global head is used (new user).
    
    Requirements: 8.1, 8.2
    """
    response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.85,
        global_emotion="happy",
        global_confidence=0.85,
        user_emotion=None,
        user_confidence=None,
        blend_weight=1.0,
        alpha_data=None,
        alpha_conf=None,
        alpha_formula="linear",
        transcription="I'm feeling great today!",
        timestamp=datetime.utcnow()
    )
    
    # Verify final prediction fields
    assert response.emotion == "happy"
    assert response.confidence == 0.85
    
    # Verify global head fields
    assert response.global_emotion == "happy"
    assert response.global_confidence == 0.85
    
    # Verify user head fields are None (no personalized model)
    assert response.user_emotion is None
    assert response.user_confidence is None
    
    # Verify blend weight is 1.0 (global only)
    assert response.blend_weight == 1.0
    
    # Verify alpha fields (linear formula)
    assert response.alpha_data is None
    assert response.alpha_conf is None
    assert response.alpha_formula == "linear"
    
    # Verify backward compatibility fields
    assert response.transcription == "I'm feeling great today!"
    assert isinstance(response.timestamp, datetime)


def test_audio_upload_response_with_dual_heads():
    """
    Test response format when both heads are used (user with trained model).
    
    Requirements: 8.1, 8.3
    """
    response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.88,
        global_emotion="happy",
        global_confidence=0.82,
        user_emotion="happy",
        user_confidence=0.95,
        blend_weight=0.55,
        alpha_data=None,
        alpha_conf=None,
        alpha_formula="linear",
        transcription="This is amazing!",
        timestamp=datetime.utcnow()
    )
    
    # Verify final blended prediction
    assert response.emotion == "happy"
    assert response.confidence == 0.88
    
    # Verify global head prediction
    assert response.global_emotion == "happy"
    assert response.global_confidence == 0.82
    
    # Verify user head prediction
    assert response.user_emotion == "happy"
    assert response.user_confidence == 0.95
    
    # Verify blend weight (partial blending)
    assert response.blend_weight == 0.55
    assert 0.3 <= response.blend_weight <= 1.0


def test_audio_upload_response_disagreement():
    """
    Test response when global and user heads disagree.
    
    Requirements: 8.1, 8.3
    """
    response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="sad",
        confidence=0.72,
        global_emotion="neutral",
        global_confidence=0.65,
        user_emotion="sad",
        user_confidence=0.85,
        blend_weight=0.40,
        alpha_formula="linear",
        transcription="I'm not sure how I feel",
        timestamp=datetime.utcnow()
    )
    
    # Verify heads disagree
    assert response.global_emotion != response.user_emotion
    
    # Verify final prediction favors user head (lower blend_weight)
    assert response.emotion == "sad"
    assert response.blend_weight < 0.5


def test_audio_upload_response_backward_compatibility():
    """
    Test that existing fields remain unchanged for backward compatibility.
    
    Requirements: 8.5
    """
    response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.85,
        global_emotion="happy",
        global_confidence=0.85,
        user_emotion=None,
        user_confidence=None,
        blend_weight=1.0,
        alpha_formula="linear",
        transcription="Test transcription",
        timestamp=datetime.utcnow()
    )
    
    # Verify all original fields are present
    assert hasattr(response, "session_id")
    assert hasattr(response, "emotion")
    assert hasattr(response, "confidence")
    assert hasattr(response, "transcription")
    assert hasattr(response, "timestamp")
    
    # Verify field types match original schema
    assert isinstance(response.session_id, str)
    assert isinstance(response.emotion, str)
    assert isinstance(response.confidence, float)
    assert isinstance(response.transcription, str)
    assert isinstance(response.timestamp, datetime)


def test_audio_upload_response_all_extended_fields():
    """
    Test that all extended fields are present and accessible.
    
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="angry",
        confidence=0.78,
        global_emotion="angry",
        global_confidence=0.75,
        user_emotion="angry",
        user_confidence=0.82,
        blend_weight=0.60,
        alpha_formula="linear",
        transcription="This is frustrating",
        timestamp=datetime.utcnow()
    )
    
    # Verify all extended fields are present
    assert hasattr(response, "global_emotion")
    assert hasattr(response, "global_confidence")
    assert hasattr(response, "user_emotion")
    assert hasattr(response, "user_confidence")
    assert hasattr(response, "blend_weight")
    
    # Verify field types
    assert isinstance(response.global_emotion, str)
    assert isinstance(response.global_confidence, float)
    assert isinstance(response.user_emotion, str)
    assert isinstance(response.user_confidence, float)
    assert isinstance(response.blend_weight, float)


def test_audio_upload_response_required_fields():
    """
    Test that required fields cannot be omitted.
    
    Requirements: 8.1, 8.4
    """
    # Missing global_emotion should raise validation error
    with pytest.raises(ValidationError):
        AudioUploadResponse(
            session_id="507f1f77bcf86cd799439011",
            emotion="happy",
            confidence=0.85,
            # global_emotion missing
            global_confidence=0.85,
            user_emotion=None,
            user_confidence=None,
            blend_weight=1.0,
            alpha_formula="linear",
            transcription="Test",
            timestamp=datetime.utcnow()
        )
    
    # Missing blend_weight should raise validation error
    with pytest.raises(ValidationError):
        AudioUploadResponse(
            session_id="507f1f77bcf86cd799439011",
            emotion="happy",
            confidence=0.85,
            global_emotion="happy",
            global_confidence=0.85,
            user_emotion=None,
            user_confidence=None,
            # blend_weight missing
            alpha_formula="linear",
            transcription="Test",
            timestamp=datetime.utcnow()
        )


def test_audio_upload_response_optional_user_fields():
    """
    Test that user_emotion and user_confidence are optional.
    
    Requirements: 8.2
    """
    # Should work without user fields (None values)
    response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.85,
        global_emotion="happy",
        global_confidence=0.85,
        user_emotion=None,
        user_confidence=None,
        blend_weight=1.0,
        alpha_formula="linear",
        transcription="Test",
        timestamp=datetime.utcnow()
    )
    
    assert response.user_emotion is None
    assert response.user_confidence is None
    
    # Should also work with user fields present
    response_with_user = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.88,
        global_emotion="happy",
        global_confidence=0.85,
        user_emotion="happy",
        user_confidence=0.92,
        blend_weight=0.50,
        alpha_formula="linear",
        transcription="Test",
        timestamp=datetime.utcnow()
    )
    
    assert response_with_user.user_emotion == "happy"
    assert response_with_user.user_confidence == 0.92


def test_audio_upload_response_serialization():
    """
    Test that response can be serialized to dict/JSON.
    
    Requirements: 8.4
    """
    response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.85,
        global_emotion="happy",
        global_confidence=0.82,
        user_emotion="happy",
        user_confidence=0.90,
        blend_weight=0.55,
        alpha_data=0.67,
        alpha_conf=0.82,
        alpha_formula="sigmoid",
        transcription="Great day!",
        timestamp=datetime.utcnow()
    )
    
    # Convert to dict
    response_dict = response.model_dump()
    
    # Verify all fields are in dict
    assert "session_id" in response_dict
    assert "emotion" in response_dict
    assert "confidence" in response_dict
    assert "global_emotion" in response_dict
    assert "global_confidence" in response_dict
    assert "user_emotion" in response_dict
    assert "user_confidence" in response_dict
    assert "blend_weight" in response_dict
    assert "alpha_data" in response_dict
    assert "alpha_conf" in response_dict
    assert "alpha_formula" in response_dict
    assert "transcription" in response_dict
    assert "timestamp" in response_dict
    
    # Verify values match
    assert response_dict["global_emotion"] == "happy"
    assert response_dict["user_emotion"] == "happy"
    assert response_dict["blend_weight"] == 0.55
    assert response_dict["alpha_data"] == 0.67
    assert response_dict["alpha_conf"] == 0.82
    assert response_dict["alpha_formula"] == "sigmoid"


def test_audio_upload_response_confidence_ranges():
    """
    Test that confidence values are within valid ranges.
    
    Requirements: 8.1, 8.3
    """
    response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.85,
        global_emotion="happy",
        global_confidence=0.82,
        user_emotion="happy",
        user_confidence=0.90,
        blend_weight=0.55,
        alpha_formula="linear",
        transcription="Test",
        timestamp=datetime.utcnow()
    )
    
    # Verify confidence values are in [0, 1] range
    assert 0.0 <= response.confidence <= 1.0
    assert 0.0 <= response.global_confidence <= 1.0
    assert 0.0 <= response.user_confidence <= 1.0
    
    # Verify blend weight is in [0, 1] range
    assert 0.0 <= response.blend_weight <= 1.0


def test_audio_upload_response_blend_weight_semantics():
    """
    Test blend weight semantic meaning.
    
    Requirements: 8.4
    """
    # Global only (blend_weight = 1.0)
    global_only = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.85,
        global_emotion="happy",
        global_confidence=0.85,
        user_emotion=None,
        user_confidence=None,
        blend_weight=1.0,
        alpha_formula="linear",
        transcription="Test",
        timestamp=datetime.utcnow()
    )
    assert global_only.blend_weight == 1.0
    assert global_only.user_emotion is None
    
    # Blended (0.3 <= blend_weight < 1.0)
    blended = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.88,
        global_emotion="happy",
        global_confidence=0.82,
        user_emotion="happy",
        user_confidence=0.95,
        blend_weight=0.55,
        alpha_formula="linear",
        transcription="Test",
        timestamp=datetime.utcnow()
    )
    assert 0.3 <= blended.blend_weight < 1.0
    assert blended.user_emotion is not None


def test_audio_upload_response_sigmoid_alpha_fields():
    """
    Test new alpha fields for sigmoid formula.
    
    Requirements: 5.5, 7.2
    """
    # Test with sigmoid formula (alpha components present)
    sigmoid_response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.88,
        global_emotion="happy",
        global_confidence=0.82,
        user_emotion="happy",
        user_confidence=0.95,
        blend_weight=0.55,
        alpha_data=0.67,
        alpha_conf=0.82,
        alpha_formula="sigmoid",
        transcription="Test",
        timestamp=datetime.utcnow()
    )
    
    # Verify sigmoid-specific fields
    assert sigmoid_response.alpha_formula == "sigmoid"
    assert sigmoid_response.alpha_data == 0.67
    assert sigmoid_response.alpha_conf == 0.82
    assert 0.0 <= sigmoid_response.alpha_data <= 1.0
    assert 0.0 <= sigmoid_response.alpha_conf <= 1.0
    
    # Test with linear formula (alpha components None)
    linear_response = AudioUploadResponse(
        session_id="507f1f77bcf86cd799439011",
        emotion="happy",
        confidence=0.85,
        global_emotion="happy",
        global_confidence=0.85,
        user_emotion=None,
        user_confidence=None,
        blend_weight=1.0,
        alpha_data=None,
        alpha_conf=None,
        alpha_formula="linear",
        transcription="Test",
        timestamp=datetime.utcnow()
    )
    
    # Verify linear formula has None alpha components
    assert linear_response.alpha_formula == "linear"
    assert linear_response.alpha_data is None
    assert linear_response.alpha_conf is None
    
    # Verify backward compatibility - alpha fields are optional
    assert hasattr(linear_response, "alpha_data")
    assert hasattr(linear_response, "alpha_conf")
    assert hasattr(linear_response, "alpha_formula")
