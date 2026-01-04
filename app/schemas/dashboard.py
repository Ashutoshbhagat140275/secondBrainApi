from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime


class EmotionRecord(BaseModel):
    session_id: str
    emotion_label: str
    confidence: float
    timestamp: datetime


class EmotionsResponse(BaseModel):
    emotions: List[EmotionRecord]
    total: int


class StatsResponse(BaseModel):
    total_sessions: int
    emotion_distribution: Dict[str, int]
    avg_confidence: float

