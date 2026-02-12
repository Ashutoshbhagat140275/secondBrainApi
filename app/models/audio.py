from pymongo.collection import Collection
from datetime import datetime
from typing import Optional, Dict, Any
from bson import ObjectId


class AudioSession:
    def __init__(
        self,
        user_id: str,
        audio_file_path: str,
        emotion_data: Dict[str, Any],
        transcription_text: str,
        qdrant_collection_id: str,
        timestamp: Optional[datetime] = None,
        _id: Optional[ObjectId] = None
    ):
        self._id = _id or ObjectId()
        self.user_id = user_id
        self.audio_file_path = audio_file_path
        self.timestamp = timestamp or datetime.utcnow()
        self.emotion_data = emotion_data
        self.transcription_text = transcription_text
        self.qdrant_collection_id = qdrant_collection_id
    
    def to_dict(self):
        return {
            "_id": self._id,
            "user_id": self.user_id,
            "audio_file_path": self.audio_file_path,
            "timestamp": self.timestamp,
            "emotion_data": self.emotion_data,
            "transcription_text": self.transcription_text,
            "qdrant_collection_id": self.qdrant_collection_id
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            _id=data.get("_id"),
            user_id=data["user_id"],
            audio_file_path=data["audio_file_path"],
            timestamp=data.get("timestamp"),
            emotion_data=data["emotion_data"],
            transcription_text=data["transcription_text"],
            qdrant_collection_id=data["qdrant_collection_id"]
        )
    
    @staticmethod
    def get_collection(db) -> Collection:
        return db.audio_sessions

