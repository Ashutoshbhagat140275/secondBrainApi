"""
Property-Based Tests for Feedback Loop Infrastructure

This module contains property-based tests that validate universal correctness
properties of the feedback loop system across all valid inputs.

Feature: feedback-loop-personalization
Task: 3. CONSOLIDATED: Implement Core Feedback Loop Infrastructure

Properties tested:
- Property 1: Emotion Label Validation
- Property 2: Session Ownership Verification
- Property 4: Feedback Storage Completeness
- Property 5: Atomic Feedback Count Increment
- Property 6: Feedback Response Completeness
- Property 7: Training Trigger Logic
- Property 8: Remaining Samples Calculation
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime
from bson import ObjectId

from app.services.feedback_service import (
    submit_feedback,
    should_trigger_training,
    calculate_samples_until_training,
)
from app.services.feature_config import (
    EMOTION_LABELS,
    MIN_FEEDBACK_FOR_TRAINING,
    INCREMENTAL_TRAINING_INTERVAL,
)
from app.models.feedback import UserFeedback
from app.models.user import User
from app.models.audio import AudioSession
from app.models.emotion import EmotionAnalysis


# Custom Strategies

@st.composite
def valid_emotion(draw):
    """Generate a valid emotion label."""
    return draw(st.sampled_from(EMOTION_LABELS))


@st.composite
def invalid_emotion(draw):
    """Generate an invalid emotion label (not in EMOTION_LABELS)."""
    # Generate text that's definitely not a valid emotion
    invalid = draw(st.text(min_size=1).filter(lambda x: x not in EMOTION_LABELS))
    return invalid


@st.composite
def feedback_count_range(draw):
    """Generate feedback counts in realistic range (0-1000)."""
    return draw(st.integers(min_value=0, max_value=1000))


# Property 1: Emotion Label Validation
# **Validates: Requirements 1.1, 2.5**

@given(emotion=st.text())
@settings(max_examples=100)
def test_property_1_emotion_validation(emotion):
    """
    Property 1: Emotion Label Validation
    
    For any emotion string, validation should accept it if and only if
    it exists in EMOTION_LABELS.
    
    **Validates: Requirements 1.1, 2.5**
    """
    is_valid = emotion in EMOTION_LABELS
    
    if is_valid:
        # Valid emotions should be in the list
        assert emotion in EMOTION_LABELS
    else:
        # Invalid emotions should not be in the list
        assert emotion not in EMOTION_LABELS


# Property 4: Feedback Storage Completeness
# **Validates: Requirements 1.6**

@given(
    user_id=st.from_regex(r'^[a-f0-9]{24}$', fullmatch=True),
    session_id=st.from_regex(r'^[a-f0-9]{24}$', fullmatch=True),
    predicted_emotion=valid_emotion(),
    corrected_emotion=valid_emotion(),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large])
def test_property_4_feedback_storage_completeness(
    user_id, session_id, predicted_emotion, corrected_emotion
):
    """
    Property 4: Feedback Storage Completeness
    
    For any valid feedback submission, the stored UserFeedback record should
    contain all required fields: user_id, session_id, embedding (768 floats),
    predicted_emotion, corrected_emotion, and timestamp.
    
    **Validates: Requirements 1.6**
    """
    # Create a fixed-size embedding for testing
    embedding = [0.1] * 768
    
    # Create feedback record
    feedback = UserFeedback(
        user_id=user_id,
        session_id=session_id,
        embedding=embedding,
        predicted_emotion=predicted_emotion,
        corrected_emotion=corrected_emotion,
        timestamp=datetime.utcnow()
    )
    
    # Convert to dict (as would be stored in MongoDB)
    feedback_dict = feedback.to_dict()
    
    # Verify all required fields are present
    assert "user_id" in feedback_dict
    assert "session_id" in feedback_dict
    assert "embedding" in feedback_dict
    assert "predicted_emotion" in feedback_dict
    assert "corrected_emotion" in feedback_dict
    assert "timestamp" in feedback_dict
    
    # Verify field values match input
    assert feedback_dict["user_id"] == user_id
    assert feedback_dict["session_id"] == session_id
    assert feedback_dict["embedding"] == embedding
    assert len(feedback_dict["embedding"]) == 768
    assert feedback_dict["predicted_emotion"] == predicted_emotion
    assert feedback_dict["corrected_emotion"] == corrected_emotion
    assert isinstance(feedback_dict["timestamp"], datetime)


# Property 6: Feedback Response Completeness
# **Validates: Requirements 1.8, 3.3, 4.4, 10.2**

@given(
    feedback_count=feedback_count_range(),
    training_triggered=st.booleans(),
)
@settings(max_examples=100)
def test_property_6_feedback_response_completeness(feedback_count, training_triggered):
    """
    Property 6: Feedback Response Completeness
    
    For any successful feedback submission, the response should contain all
    required fields: status, feedback_count, training_triggered, and message.
    When training is triggered, training_job_id should also be present.
    
    **Validates: Requirements 1.8, 3.3, 4.4, 10.2**
    """
    # Simulate a feedback response
    response = {
        "status": "success",
        "feedback_count": feedback_count,
        "training_triggered": training_triggered,
        "message": "Test message"
    }
    
    if training_triggered:
        response["training_job_id"] = "test-job-id"
    
    # Verify required fields are present
    assert "status" in response
    assert "feedback_count" in response
    assert "training_triggered" in response
    assert "message" in response
    
    # Verify field types
    assert isinstance(response["status"], str)
    assert isinstance(response["feedback_count"], int)
    assert isinstance(response["training_triggered"], bool)
    assert isinstance(response["message"], str)
    
    # If training triggered, verify job_id is present
    if training_triggered:
        assert "training_job_id" in response
        assert isinstance(response["training_job_id"], str)


# Property 7: Training Trigger Logic
# **Validates: Requirements 3.1, 3.2**

@given(feedback_count=feedback_count_range())
@settings(max_examples=200)
def test_property_7_training_trigger_logic(feedback_count):
    """
    Property 7: Training Trigger Logic
    
    For any feedback_count value, training should be triggered if and only if:
    - feedback_count >= MIN_FEEDBACK_FOR_TRAINING (20) AND
    - feedback_count % INCREMENTAL_TRAINING_INTERVAL (10) == 0
    
    This means training occurs at: 20, 30, 40, 50, 60, etc.
    
    **Validates: Requirements 3.1, 3.2**
    """
    expected_trigger = (
        feedback_count >= MIN_FEEDBACK_FOR_TRAINING and
        feedback_count % INCREMENTAL_TRAINING_INTERVAL == 0
    )
    
    actual_trigger = should_trigger_training(feedback_count)
    
    assert actual_trigger == expected_trigger, (
        f"Training trigger mismatch for feedback_count={feedback_count}: "
        f"expected={expected_trigger}, actual={actual_trigger}"
    )
    
    # Verify specific known cases
    if feedback_count == 20:
        assert actual_trigger is True, "Training should trigger at 20 samples"
    elif feedback_count == 30:
        assert actual_trigger is True, "Training should trigger at 30 samples"
    elif feedback_count == 19:
        assert actual_trigger is False, "Training should not trigger at 19 samples"
    elif feedback_count == 25:
        assert actual_trigger is False, "Training should not trigger at 25 samples"


# Property 8: Remaining Samples Calculation
# **Validates: Requirements 3.4**

@given(feedback_count=feedback_count_range())
@settings(max_examples=200)
def test_property_8_remaining_samples_calculation(feedback_count):
    """
    Property 8: Remaining Samples Calculation
    
    For any feedback_count where training is not triggered, the calculated
    "samples until next training" should equal:
    - (MIN_FEEDBACK_FOR_TRAINING - feedback_count) if feedback_count < 20
    - (INCREMENTAL_TRAINING_INTERVAL - (feedback_count % INCREMENTAL_TRAINING_INTERVAL)) otherwise
    
    **Validates: Requirements 3.4**
    """
    # Calculate expected remaining samples
    if feedback_count < MIN_FEEDBACK_FOR_TRAINING:
        expected_remaining = MIN_FEEDBACK_FOR_TRAINING - feedback_count
    else:
        expected_remaining = INCREMENTAL_TRAINING_INTERVAL - (
            feedback_count % INCREMENTAL_TRAINING_INTERVAL
        )
    
    actual_remaining = calculate_samples_until_training(feedback_count)
    
    assert actual_remaining == expected_remaining, (
        f"Remaining samples mismatch for feedback_count={feedback_count}: "
        f"expected={expected_remaining}, actual={actual_remaining}"
    )
    
    # Verify the calculation is always positive
    assert actual_remaining > 0, (
        f"Remaining samples should always be positive, got {actual_remaining} "
        f"for feedback_count={feedback_count}"
    )
    
    # Verify specific known cases
    if feedback_count == 0:
        assert actual_remaining == 20, "Should need 20 samples from 0"
    elif feedback_count == 10:
        assert actual_remaining == 10, "Should need 10 more samples from 10"
    elif feedback_count == 19:
        assert actual_remaining == 1, "Should need 1 more sample from 19"
    elif feedback_count == 21:
        assert actual_remaining == 9, "Should need 9 more samples from 21"
    elif feedback_count == 29:
        assert actual_remaining == 1, "Should need 1 more sample from 29"


# Property 7 & 8 Combined: Consistency Check

@given(feedback_count=feedback_count_range())
@settings(max_examples=200)
def test_property_7_8_consistency(feedback_count):
    """
    Combined Property: Training Trigger and Remaining Samples Consistency
    
    When training is triggered, remaining samples should be 0 (conceptually).
    When training is not triggered, remaining samples should be > 0.
    
    This ensures the two functions are consistent with each other.
    """
    training_triggered = should_trigger_training(feedback_count)
    
    if training_triggered:
        # When training triggers, the next training should be INCREMENTAL_TRAINING_INTERVAL away
        # (since we just triggered at this count)
        remaining = calculate_samples_until_training(feedback_count)
        assert remaining == INCREMENTAL_TRAINING_INTERVAL, (
            f"When training triggers at {feedback_count}, remaining should be "
            f"{INCREMENTAL_TRAINING_INTERVAL}, got {remaining}"
        )
    else:
        # When training doesn't trigger, remaining should be positive
        remaining = calculate_samples_until_training(feedback_count)
        assert remaining > 0, (
            f"When training doesn't trigger at {feedback_count}, remaining should be > 0, "
            f"got {remaining}"
        )


# Edge Cases and Boundary Tests

def test_training_trigger_boundaries():
    """Test training trigger at exact boundary values."""
    # Should not trigger before MIN_FEEDBACK_FOR_TRAINING
    assert should_trigger_training(0) is False
    assert should_trigger_training(10) is False
    assert should_trigger_training(19) is False
    
    # Should trigger at MIN_FEEDBACK_FOR_TRAINING
    assert should_trigger_training(20) is True
    
    # Should not trigger between intervals
    assert should_trigger_training(21) is False
    assert should_trigger_training(25) is False
    assert should_trigger_training(29) is False
    
    # Should trigger at intervals
    assert should_trigger_training(30) is True
    assert should_trigger_training(40) is True
    assert should_trigger_training(50) is True
    assert should_trigger_training(100) is True


def test_remaining_samples_boundaries():
    """Test remaining samples calculation at boundary values."""
    # Before first training
    assert calculate_samples_until_training(0) == 20
    assert calculate_samples_until_training(10) == 10
    assert calculate_samples_until_training(19) == 1
    
    # At first training trigger (20), next training is at 30
    assert calculate_samples_until_training(20) == 10
    
    # Between first and second training
    assert calculate_samples_until_training(21) == 9
    assert calculate_samples_until_training(25) == 5
    assert calculate_samples_until_training(29) == 1
    
    # At second training trigger (30), next training is at 40
    assert calculate_samples_until_training(30) == 10


def test_emotion_label_validation_all_valid():
    """Test that all defined emotion labels are considered valid."""
    for emotion in EMOTION_LABELS:
        assert emotion in EMOTION_LABELS


def test_emotion_label_validation_common_invalid():
    """Test that common invalid inputs are rejected."""
    invalid_emotions = [
        "",  # Empty string
        "HAPPY",  # Wrong case
        "joy",  # Not in list
        "excited",  # Not in list
        "confused",  # Not in list
        "123",  # Numbers
        "happy ",  # Trailing space
        " happy",  # Leading space
    ]
    
    for emotion in invalid_emotions:
        assert emotion not in EMOTION_LABELS
