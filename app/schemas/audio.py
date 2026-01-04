from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class AudioUploadResponse(BaseModel):
    session_id: str
    emotion: str
    confidence: float
    transcription: str
    timestamp: datetime


class EmotionData(BaseModel):
    label: str
    confidence: float

