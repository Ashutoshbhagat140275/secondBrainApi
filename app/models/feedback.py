from pymongo.collection import Collection
from datetime import datetime
from typing import Optional, List
from bson import ObjectId


class UserFeedback:
    """
    MongoDB model for storing user feedback corrections.
    
    Used to train personalized emotion recognition models by storing
    user corrections of predicted emotions along with the audio embeddings.
    """
    
    def __init__(
        self,
        user_id: str,
        session_id: str,
        embedding: List[float],
        predicted_emotion: str,
        corrected_emotion: str,
        timestamp: Optional[datetime] = None,
        _id: Optional[ObjectId] = None
    ):
        """
        Initialize a UserFeedback instance.
        
        Parameters:
            user_id: User identifier
            session_id: Audio session identifier
            embedding: 768-dimensional Wav2Vec2 embedding
            predicted_emotion: Original emotion prediction from the model
            corrected_emotion: User's correction of the emotion
            timestamp: When the feedback was submitted (defaults to now)
            _id: MongoDB ObjectId (auto-generated if not provided)
        """
        self._id = _id or ObjectId()
        self.user_id = user_id
        self.session_id = session_id
        self.embedding = embedding
        self.predicted_emotion = predicted_emotion
        self.corrected_emotion = corrected_emotion
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self):
        """Convert UserFeedback instance to dictionary for MongoDB storage."""
        return {
            "_id": self._id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "embedding": self.embedding,
            "predicted_emotion": self.predicted_emotion,
            "corrected_emotion": self.corrected_emotion,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create UserFeedback instance from MongoDB document."""
        return cls(
            _id=data.get("_id"),
            user_id=data["user_id"],
            session_id=data["session_id"],
            embedding=data["embedding"],
            predicted_emotion=data["predicted_emotion"],
            corrected_emotion=data["corrected_emotion"],
            timestamp=data.get("timestamp")
        )
    
    @staticmethod
    def get_collection(db) -> Collection:
        """Get the MongoDB collection for user feedback."""
        return db.user_feedback
