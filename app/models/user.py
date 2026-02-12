from pymongo.collection import Collection
from datetime import datetime
from typing import Optional
from bson import ObjectId


class User:
    def __init__(
        self,
        email: str,
        password_hash: str,
        feedback_count: int = 0,
        is_admin: bool = False,
        created_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None
    ):
        self._id = _id or ObjectId()
        self.email = email
        self.password_hash = password_hash
        self.feedback_count = feedback_count
        self.is_admin = is_admin
        self.created_at = created_at or datetime.utcnow()
    
    def to_dict(self):
        return {
            "_id": self._id,
            "email": self.email,
            "password_hash": self.password_hash,
            "feedback_count": self.feedback_count,
            "is_admin": self.is_admin,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            _id=data.get("_id"),
            email=data["email"],
            password_hash=data["password_hash"],
            feedback_count=data.get("feedback_count", 0),
            is_admin=data.get("is_admin", False),
            created_at=data.get("created_at")
        )
    
    @staticmethod
    def get_collection(db) -> Collection:
        return db.users
    
    @staticmethod
    def increment_feedback_count(db, user_id: ObjectId) -> int:
        """
        Atomically increment the feedback count for a user.
        
        Parameters:
            db: MongoDB database instance
            user_id: ObjectId of the user
        
        Returns:
            Updated feedback_count value
        
        Raises:
            ValueError: If user not found
        """
        result = User.get_collection(db).find_one_and_update(
            {"_id": user_id},
            {"$inc": {"feedback_count": 1}},
            return_document=True
        )
        
        if result is None:
            raise ValueError(f"User with id {user_id} not found")
        
        return result.get("feedback_count", 1)

