from pymongo.collection import Collection
from datetime import datetime
from typing import Optional
from bson import ObjectId


class User:
    def __init__(
        self,
        email: str,
        password_hash: str,
        created_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None
    ):
        self._id = _id or ObjectId()
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at or datetime.utcnow()
    
    def to_dict(self):
        return {
            "_id": self._id,
            "email": self.email,
            "password_hash": self.password_hash,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            _id=data.get("_id"),
            email=data["email"],
            password_hash=data["password_hash"],
            created_at=data.get("created_at")
        )
    
    @staticmethod
    def get_collection(db) -> Collection:
        return db.users

