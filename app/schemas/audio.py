from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class AudioUploadResponse(BaseModel):
    """
    Response for audio upload and processing.
    
    Includes both final blended prediction and individual head predictions
    from the dual-head emotion classification system.
    
    Requirements: 5.5, 7.2, 8.1, 8.2, 8.3, 8.4, 8.5
    """
    session_id: str
    emotion: str = Field(..., description="Final blended emotion prediction")
    confidence: float = Field(..., description="Final blended confidence score")
    global_emotion: str = Field(..., description="Global head emotion prediction")
    global_confidence: float = Field(..., description="Global head confidence score")
    user_emotion: Optional[str] = Field(None, description="User head emotion prediction (None if not available)")
    user_confidence: Optional[float] = Field(None, description="User head confidence score (None if not available)")
    blend_weight: float = Field(..., description="Blending weight alpha used (1.0 = global only, <1.0 = blended)")
    alpha_data: Optional[float] = Field(None, description="Data-driven alpha component based on feedback count (sigmoid formula only)")
    alpha_conf: Optional[float] = Field(None, description="Confidence-driven alpha component based on global head confidence (sigmoid formula only)")
    alpha_formula: str = Field(..., description="Alpha computation formula used: 'sigmoid' or 'linear'")
    transcription: str
    timestamp: datetime


class EmotionData(BaseModel):
    label: str
    confidence: float


class FeedbackRequest(BaseModel):
    """Request body for emotion feedback submission."""
    session_id: str = Field(..., description="Audio session ID to provide feedback for")
    corrected_emotion: str = Field(..., description="User's correction of the emotion label")


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    status: str = Field(..., description="Status of the feedback submission")
    feedback_count: int = Field(..., description="Total feedback count for the user")
    training_triggered: bool = Field(..., description="Whether model training was triggered")
    message: str = Field(..., description="Human-readable status message")

