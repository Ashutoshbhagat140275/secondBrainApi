from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class UserModelMetadata(BaseModel):
    """Metadata for a user-specific emotion model."""
    user_id: str = Field(..., description="User identifier")
    feedback_count: int = Field(..., description="Number of feedback samples")
    model_size_kb: float = Field(..., description="Size of model file in KB")
    last_trained: Optional[datetime] = Field(None, description="Timestamp of last training")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "feedback_count": 45,
                "model_size_kb": 24.5,
                "last_trained": "2024-02-12T10:30:00Z"
            }
        }


class UserModelListResponse(BaseModel):
    """Response containing list of all user models."""
    total_count: int = Field(..., description="Total number of user models")
    models: List[UserModelMetadata] = Field(..., description="List of user model metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_count": 2,
                "models": [
                    {
                        "user_id": "507f1f77bcf86cd799439011",
                        "feedback_count": 45,
                        "model_size_kb": 24.5,
                        "last_trained": "2024-02-12T10:30:00Z"
                    },
                    {
                        "user_id": "507f1f77bcf86cd799439012",
                        "feedback_count": 30,
                        "model_size_kb": 23.8,
                        "last_trained": "2024-02-11T15:20:00Z"
                    }
                ]
            }
        }


class ModelCleanupRequest(BaseModel):
    """Request to cleanup inactive user models."""
    min_days_inactive: Optional[int] = Field(
        None,
        description="Minimum days since last training to consider inactive",
        ge=1
    )
    max_feedback_count: Optional[int] = Field(
        None,
        description="Maximum feedback count to consider inactive",
        ge=0
    )
    user_ids: Optional[List[str]] = Field(
        None,
        description="Specific user IDs to delete (overrides other criteria)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "min_days_inactive": 90,
                "max_feedback_count": 20
            }
        }


class ModelCleanupResponse(BaseModel):
    """Response from model cleanup operation."""
    deleted_count: int = Field(..., description="Number of models deleted")
    deleted_user_ids: List[str] = Field(..., description="List of deleted user IDs")
    total_space_freed_kb: float = Field(..., description="Total disk space freed in KB")
    
    class Config:
        json_schema_extra = {
            "example": {
                "deleted_count": 3,
                "deleted_user_ids": ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"],
                "total_space_freed_kb": 72.3
            }
        }
