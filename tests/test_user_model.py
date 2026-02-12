"""
Unit tests for User MongoDB model.

Tests Requirements 5.4
"""

import pytest
from datetime import datetime
from bson import ObjectId
from app.models.user import User


class TestUserModel:
    """Test User model initialization and methods."""
    
    def test_init_with_all_fields(self):
        """Test User initialization with all fields provided."""
        user_id = ObjectId()
        email = "test@example.com"
        password_hash = "hashed_password_123"
        feedback_count = 5
        created_at = datetime(2024, 1, 1, 12, 0, 0)
        
        user = User(
            _id=user_id,
            email=email,
            password_hash=password_hash,
            feedback_count=feedback_count,
            created_at=created_at
        )
        
        assert user._id == user_id
        assert user.email == email
        assert user.password_hash == password_hash
        assert user.feedback_count == feedback_count
        assert user.created_at == created_at
    
    def test_init_with_defaults(self):
        """Test User initialization with default feedback_count and created_at."""
        email = "test@example.com"
        password_hash = "hashed_password_123"
        
        user = User(email=email, password_hash=password_hash)
        
        assert user.email == email
        assert user.password_hash == password_hash
        assert user.feedback_count == 0  # Default value
        assert isinstance(user._id, ObjectId)
        assert isinstance(user.created_at, datetime)
    
    def test_to_dict(self):
        """Test User serialization to dictionary."""
        user_id = ObjectId()
        email = "test@example.com"
        password_hash = "hashed_password_123"
        feedback_count = 10
        created_at = datetime(2024, 1, 1, 12, 0, 0)
        
        user = User(
            _id=user_id,
            email=email,
            password_hash=password_hash,
            feedback_count=feedback_count,
            created_at=created_at
        )
        
        user_dict = user.to_dict()
        
        assert user_dict["_id"] == user_id
        assert user_dict["email"] == email
        assert user_dict["password_hash"] == password_hash
        assert user_dict["feedback_count"] == feedback_count
        assert user_dict["created_at"] == created_at
    
    def test_from_dict_with_all_fields(self):
        """Test User deserialization from dictionary with all fields."""
        user_id = ObjectId()
        email = "test@example.com"
        password_hash = "hashed_password_123"
        feedback_count = 15
        created_at = datetime(2024, 1, 1, 12, 0, 0)
        
        data = {
            "_id": user_id,
            "email": email,
            "password_hash": password_hash,
            "feedback_count": feedback_count,
            "created_at": created_at
        }
        
        user = User.from_dict(data)
        
        assert user._id == user_id
        assert user.email == email
        assert user.password_hash == password_hash
        assert user.feedback_count == feedback_count
        assert user.created_at == created_at
    
    def test_from_dict_with_missing_feedback_count(self):
        """Test User deserialization defaults feedback_count to 0 if missing."""
        email = "test@example.com"
        password_hash = "hashed_password_123"
        
        data = {
            "email": email,
            "password_hash": password_hash
        }
        
        user = User.from_dict(data)
        
        assert user.email == email
        assert user.password_hash == password_hash
        assert user.feedback_count == 0  # Default when missing


class TestIncrementFeedbackCount:
    """Test increment_feedback_count() helper function."""
    
    def test_increment_feedback_count_success(self):
        """Test successful feedback count increment."""
        from unittest.mock import Mock
        
        user_id = ObjectId()
        mock_db = Mock()
        mock_collection = Mock()
        
        # Mock the collection and update result
        mock_db.users = mock_collection
        mock_collection.find_one_and_update.return_value = {
            "_id": user_id,
            "email": "test@example.com",
            "feedback_count": 6
        }
        
        result = User.increment_feedback_count(mock_db, user_id)
        
        # Verify the update was called with correct parameters
        mock_collection.find_one_and_update.assert_called_once_with(
            {"_id": user_id},
            {"$inc": {"feedback_count": 1}},
            return_document=True
        )
        
        assert result == 6
    
    def test_increment_feedback_count_user_not_found(self):
        """Test increment raises ValueError when user not found."""
        from unittest.mock import Mock
        
        user_id = ObjectId()
        mock_db = Mock()
        mock_collection = Mock()
        
        # Mock the collection to return None (user not found)
        mock_db.users = mock_collection
        mock_collection.find_one_and_update.return_value = None
        
        with pytest.raises(ValueError, match=f"User with id {user_id} not found"):
            User.increment_feedback_count(mock_db, user_id)
    
    def test_increment_feedback_count_from_zero(self):
        """Test incrementing feedback count from 0 to 1."""
        from unittest.mock import Mock
        
        user_id = ObjectId()
        mock_db = Mock()
        mock_collection = Mock()
        
        mock_db.users = mock_collection
        mock_collection.find_one_and_update.return_value = {
            "_id": user_id,
            "email": "test@example.com",
            "feedback_count": 1
        }
        
        result = User.increment_feedback_count(mock_db, user_id)
        
        assert result == 1
