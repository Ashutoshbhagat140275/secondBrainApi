"""
Unit tests for UserFeedback MongoDB model.

Tests Requirements 5.1, 5.2, 5.3, 5.5
"""

import pytest
from datetime import datetime
from bson import ObjectId
from app.models.feedback import UserFeedback


class TestUserFeedbackModel:
    """Test UserFeedback model initialization and methods."""
    
    def test_init_with_all_fields(self):
        """Test UserFeedback initialization with all fields provided."""
        user_id = "user123"
        session_id = "session456"
        embedding = [0.1] * 768  # 768-dimensional embedding
        predicted_emotion = "happy"
        corrected_emotion = "sad"
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        obj_id = ObjectId()
        
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            embedding=embedding,
            predicted_emotion=predicted_emotion,
            corrected_emotion=corrected_emotion,
            timestamp=timestamp,
            _id=obj_id
        )
        
        assert feedback.user_id == user_id
        assert feedback.session_id == session_id
        assert feedback.embedding == embedding
        assert len(feedback.embedding) == 768
        assert feedback.predicted_emotion == predicted_emotion
        assert feedback.corrected_emotion == corrected_emotion
        assert feedback.timestamp == timestamp
        assert feedback._id == obj_id
    
    def test_init_with_defaults(self):
        """Test UserFeedback initialization with default timestamp and _id."""
        user_id = "user123"
        session_id = "session456"
        embedding = [0.1] * 768
        predicted_emotion = "happy"
        corrected_emotion = "sad"
        
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            embedding=embedding,
            predicted_emotion=predicted_emotion,
            corrected_emotion=corrected_emotion
        )
        
        assert feedback.user_id == user_id
        assert feedback.session_id == session_id
        assert feedback.embedding == embedding
        assert feedback.predicted_emotion == predicted_emotion
        assert feedback.corrected_emotion == corrected_emotion
        assert isinstance(feedback.timestamp, datetime)
        assert isinstance(feedback._id, ObjectId)
    
    def test_to_dict(self):
        """Test conversion to dictionary for MongoDB storage."""
        user_id = "user123"
        session_id = "session456"
        embedding = [0.1] * 768
        predicted_emotion = "happy"
        corrected_emotion = "sad"
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        obj_id = ObjectId()
        
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            embedding=embedding,
            predicted_emotion=predicted_emotion,
            corrected_emotion=corrected_emotion,
            timestamp=timestamp,
            _id=obj_id
        )
        
        result = feedback.to_dict()
        
        assert result["_id"] == obj_id
        assert result["user_id"] == user_id
        assert result["session_id"] == session_id
        assert result["embedding"] == embedding
        assert result["predicted_emotion"] == predicted_emotion
        assert result["corrected_emotion"] == corrected_emotion
        assert result["timestamp"] == timestamp
        assert len(result) == 7  # All fields present
    
    def test_from_dict(self):
        """Test creation from MongoDB document."""
        obj_id = ObjectId()
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        embedding = [0.1] * 768
        
        data = {
            "_id": obj_id,
            "user_id": "user123",
            "session_id": "session456",
            "embedding": embedding,
            "predicted_emotion": "happy",
            "corrected_emotion": "sad",
            "timestamp": timestamp
        }
        
        feedback = UserFeedback.from_dict(data)
        
        assert feedback._id == obj_id
        assert feedback.user_id == "user123"
        assert feedback.session_id == "session456"
        assert feedback.embedding == embedding
        assert feedback.predicted_emotion == "happy"
        assert feedback.corrected_emotion == "sad"
        assert feedback.timestamp == timestamp
    
    def test_from_dict_without_optional_fields(self):
        """Test from_dict with missing optional fields (_id, timestamp)."""
        embedding = [0.1] * 768
        
        data = {
            "user_id": "user123",
            "session_id": "session456",
            "embedding": embedding,
            "predicted_emotion": "happy",
            "corrected_emotion": "sad"
        }
        
        feedback = UserFeedback.from_dict(data)
        
        assert feedback.user_id == "user123"
        assert feedback.session_id == "session456"
        assert feedback.embedding == embedding
        assert feedback.predicted_emotion == "happy"
        assert feedback.corrected_emotion == "sad"
        # _id and timestamp are auto-generated when not provided
        assert isinstance(feedback._id, ObjectId)
        assert isinstance(feedback.timestamp, datetime)
    
    def test_round_trip_conversion(self):
        """Test that to_dict() and from_dict() are inverse operations."""
        original = UserFeedback(
            user_id="user123",
            session_id="session456",
            embedding=[0.1] * 768,
            predicted_emotion="happy",
            corrected_emotion="sad",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            _id=ObjectId()
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = UserFeedback.from_dict(data)
        
        assert restored._id == original._id
        assert restored.user_id == original.user_id
        assert restored.session_id == original.session_id
        assert restored.embedding == original.embedding
        assert restored.predicted_emotion == original.predicted_emotion
        assert restored.corrected_emotion == original.corrected_emotion
        assert restored.timestamp == original.timestamp
    
    def test_get_collection(self):
        """Test get_collection returns correct collection name."""
        # Mock database object
        class MockDB:
            @property
            def user_feedback(self):
                return "user_feedback_collection"
        
        db = MockDB()
        collection = UserFeedback.get_collection(db)
        
        assert collection == "user_feedback_collection"
    
    def test_embedding_dimension(self):
        """Test that embedding field can store 768-dimensional vectors."""
        # Requirement 5.3: embedding should be 768-dim
        embedding_768 = [float(i) for i in range(768)]
        
        feedback = UserFeedback(
            user_id="user123",
            session_id="session456",
            embedding=embedding_768,
            predicted_emotion="happy",
            corrected_emotion="sad"
        )
        
        assert len(feedback.embedding) == 768
        assert feedback.embedding[0] == 0.0
        assert feedback.embedding[767] == 767.0
