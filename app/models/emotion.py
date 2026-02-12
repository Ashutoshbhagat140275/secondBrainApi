from pymongo.collection import Collection
from datetime import datetime
from typing import Optional, List
from bson import ObjectId


class EmotionAnalysis:
    def __init__(
        self,
        user_id: str,
        session_id: str,
        emotion_label: str,
        confidence: float,
        mfcc_features: List[float],
        timestamp: Optional[datetime] = None,
        _id: Optional[ObjectId] = None
    ):
        self._id = _id or ObjectId()
        self.user_id = user_id
        self.session_id = session_id
        self.emotion_label = emotion_label
        self.confidence = confidence
        self.mfcc_features = mfcc_features
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self):
        return {
            "_id": self._id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "emotion_label": self.emotion_label,
            "confidence": self.confidence,
            "mfcc_features": self.mfcc_features,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            _id=data.get("_id"),
            user_id=data["user_id"],
            session_id=data["session_id"],
            emotion_label=data["emotion_label"],
            confidence=data["confidence"],
            mfcc_features=data["mfcc_features"],
            timestamp=data.get("timestamp")
        )
    
    @staticmethod
    def get_collection(db) -> Collection:
        return db.emotion_analyses

