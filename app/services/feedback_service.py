"""
Feedback Service for Personalized Emotion Recognition

This module handles user feedback submission for emotion corrections.
It stores feedback data and triggers user-specific model training when
sufficient feedback samples are collected.

Key responsibilities:
- Validate corrected emotion labels
- Retrieve embeddings and predictions from MongoDB
- Store feedback in UserFeedback collection
- Track user feedback counts
- Determine when to trigger user head training
"""

import logging
from typing import Dict, Any
from datetime import datetime
from bson import ObjectId

from app.models.feedback import UserFeedback
from app.models.user import User
from app.models.emotion import EmotionAnalysis
from app.models.audio import AudioSession
from app.services.feature_config import EMOTION_LABELS


logger = logging.getLogger(__name__)


def submit_feedback(
    db,
    user_id: str,
    session_id: str,
    corrected_emotion: str
) -> Dict[str, Any]:
    """
    Submit user feedback for emotion correction.
    
    This function stores user corrections of emotion predictions and determines
    whether to trigger personalized model training based on feedback count.
    
    Parameters:
        db: MongoDB database instance
        user_id: User identifier (ObjectId as string)
        session_id: Audio session identifier (ObjectId as string)
        corrected_emotion: User's correction of the emotion label
    
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - feedback_count: Updated feedback count for the user
            - training_triggered: Boolean indicating if training should start
            - message: Human-readable status message
    
    Raises:
        ValueError: If corrected_emotion is invalid or session not found
        PermissionError: If session doesn't belong to the user
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    # Validate corrected emotion label
    if corrected_emotion not in EMOTION_LABELS:
        raise ValueError(
            f"Invalid emotion label: '{corrected_emotion}'. "
            f"Must be one of: {', '.join(EMOTION_LABELS)}"
        )
    
    # Convert string IDs to ObjectId
    try:
        user_oid = ObjectId(user_id)
        session_oid = ObjectId(session_id)
    except Exception as e:
        raise ValueError(f"Invalid ID format: {e}")
    
    # Retrieve audio session to verify ownership and get predicted emotion
    audio_session = AudioSession.get_collection(db).find_one({"_id": session_oid})
    if audio_session is None:
        raise ValueError(f"Audio session {session_id} not found")
    
    # Verify session belongs to the user
    if audio_session["user_id"] != user_id:
        raise PermissionError(
            f"Session {session_id} does not belong to user {user_id}"
        )
    
    # Extract predicted emotion from emotion_data
    predicted_emotion = audio_session.get("emotion_data", {}).get("emotion")
    if predicted_emotion is None:
        raise ValueError(
            f"Session {session_id} has no emotion prediction data"
        )
    
    # Retrieve embedding from EmotionAnalysis collection
    emotion_analysis = EmotionAnalysis.get_collection(db).find_one({
        "session_id": str(session_oid)
    })
    
    if emotion_analysis is None:
        raise ValueError(
            f"No emotion analysis found for session {session_id}"
        )
    
    # Extract embedding (stored as mfcc_features in EmotionAnalysis)
    embedding = emotion_analysis.get("mfcc_features")
    if embedding is None or len(embedding) == 0:
        raise ValueError(
            f"No embedding found for session {session_id}"
        )
    
    # Create feedback record
    feedback = UserFeedback(
        user_id=user_id,
        session_id=str(session_oid),
        embedding=embedding,
        predicted_emotion=predicted_emotion,
        corrected_emotion=corrected_emotion,
        timestamp=datetime.utcnow()
    )
    
    # Store feedback in MongoDB
    UserFeedback.get_collection(db).insert_one(feedback.to_dict())
    
    logger.info(
        f"Feedback stored for user {user_id}, session {session_id}: "
        f"{predicted_emotion} -> {corrected_emotion}"
    )
    
    # Increment user feedback count atomically
    feedback_count = User.increment_feedback_count(db, user_oid)
    
    # Determine if training should be triggered
    # Training triggers at 20 feedback samples, then every 10 samples thereafter
    training_triggered = (
        feedback_count >= 20 and
        feedback_count % 10 == 0
    )
    
    if training_triggered:
        logger.info(
            f"Training threshold reached for user {user_id}: "
            f"{feedback_count} feedback samples"
        )
        message = (
            f"Feedback recorded ({feedback_count} total). "
            f"Your personalized model is being updated."
        )
    else:
        samples_until_training = 20 - feedback_count if feedback_count < 20 else (
            10 - (feedback_count % 10)
        )
        message = (
            f"Feedback recorded ({feedback_count} total). "
            f"{samples_until_training} more samples until next model update."
        )
    
    return {
        "status": "success",
        "feedback_count": feedback_count,
        "training_triggered": training_triggered,
        "message": message
    }
