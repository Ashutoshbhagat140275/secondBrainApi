"""
Feedback Router

REST API endpoints for user feedback submission and training status queries.

Endpoints:
- POST /api/feedback: Submit emotion correction feedback
- GET /api/training-status/{user_id}: Query training job status
- POST /api/trigger-training/{user_id}: Manually trigger training (admin only)
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

from app.db.mongodb import get_database
from app.middleware.auth import get_current_user, get_current_admin
from app.models.user import User
from app.services.feedback_service import submit_feedback
from app.services.training_job_tracker import get_latest_job, create_training_job
from app.services.task_queue import FastAPITaskQueue
from app.services.feature_config import EMOTION_LABELS


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feedback"])


# Request/Response Models

class FeedbackRequest(BaseModel):
    """Request model for feedback submission."""
    session_id: str = Field(..., description="Audio session identifier")
    corrected_emotion: str = Field(..., description="User's corrected emotion label")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "507f1f77bcf86cd799439011",
                "corrected_emotion": "happy"
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    status: str = Field(..., description="Operation status")
    feedback_count: int = Field(..., description="Total feedback count for user")
    training_triggered: bool = Field(..., description="Whether training was triggered")
    training_job_id: Optional[str] = Field(None, description="Training job ID if triggered")
    message: str = Field(..., description="Human-readable status message")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "feedback_count": 25,
                "training_triggered": True,
                "training_job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "message": "Feedback recorded (25 total). Your personalized model is being updated."
            }
        }


class TrainingStatusResponse(BaseModel):
    """Response model for training status query."""
    job_id: Optional[str] = Field(None, description="Training job identifier")
    status: Optional[str] = Field(None, description="Job status (queued, running, completed, failed)")
    created_at: Optional[datetime] = Field(None, description="When job was created")
    started_at: Optional[datetime] = Field(None, description="When job started running")
    completed_at: Optional[datetime] = Field(None, description="When job completed")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics if completed")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "status": "completed",
                "created_at": "2024-01-15T10:30:00Z",
                "started_at": "2024-01-15T10:30:05Z",
                "completed_at": "2024-01-15T10:32:15Z",
                "error_message": None,
                "metrics": {
                    "final_loss": 0.234,
                    "final_accuracy": 0.89,
                    "num_samples": 25,
                    "num_epochs": 20,
                    "model_path": "models/user_heads/507f1f77bcf86cd799439011.pt"
                }
            }
        }


class TrainingTriggerResponse(BaseModel):
    """Response model for manual training trigger."""
    status: str = Field(..., description="Operation status")
    training_job_id: str = Field(..., description="Training job identifier")
    message: str = Field(..., description="Human-readable status message")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "training_job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "message": "Training job enqueued for user 507f1f77bcf86cd799439011"
            }
        }


# Endpoints

@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_200_OK)
async def submit_feedback_endpoint(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Submit feedback for emotion correction.
    
    This endpoint allows users to correct emotion predictions. When sufficient
    feedback is collected (20 samples initially, then every 10 samples), a
    background training job is automatically triggered to update the user's
    personalized emotion recognition model.
    
    **Requirements:**
    - User must be authenticated
    - Session must belong to the authenticated user
    - Corrected emotion must be a valid emotion label
    
    **Training Triggers:**
    - Initial training: 20 feedback samples
    - Incremental training: Every 10 samples after initial (30, 40, 50, etc.)
    
    **Performance:**
    - Target response time: <100ms
    - Training runs asynchronously (does not block response)
    
    Requirements: 1.1-1.8, 3.1-3.4, 4.2, 10.1, 10.2, 11.1
    """
    try:
        # Validate corrected emotion
        if request.corrected_emotion not in EMOTION_LABELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid emotion label: '{request.corrected_emotion}'. "
                       f"Must be one of: {', '.join(EMOTION_LABELS)}"
            )
        
        # Create task queue for background training
        task_queue = FastAPITaskQueue(background_tasks)
        
        # Submit feedback
        result = submit_feedback(
            db=db,
            user_id=current_user["user_id"],
            session_id=request.session_id,
            corrected_emotion=request.corrected_emotion,
            task_queue=task_queue
        )
        
        return FeedbackResponse(**result)
        
    except PermissionError as e:
        logger.warning(f"Permission denied: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback. Please try again."
        )


@router.get("/training-status/{user_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    user_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get the latest training job status for a user.
    
    Returns information about the most recent training job, including:
    - Job status (queued, running, completed, failed)
    - Timestamps (created, started, completed)
    - Training metrics (if completed)
    - Error message (if failed)
    
    **Requirements:**
    - User must be authenticated
    - Users can only query their own training status (unless admin)
    
    Requirements: 9.7, 10.3, 10.4
    """
    try:
        # Authorization: Users can only query their own status (unless admin)
        db_instance = db
        user_doc = db_instance.users.find_one({"_id": ObjectId(current_user["user_id"])})
        is_admin = user_doc.get("is_admin", False) if user_doc else False
        
        if current_user["user_id"] != user_id and not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only query your own training status"
            )
        
        # Get latest training job
        job = get_latest_job(db_instance, user_id)
        
        if job is None:
            # No training jobs found - return empty response
            return TrainingStatusResponse(
                job_id=None,
                status=None,
                created_at=None,
                started_at=None,
                completed_at=None,
                error_message=None,
                metrics=None
            )
        
        # Return job details
        return TrainingStatusResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            metrics=job.metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query training status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query training status. Please try again."
        )


@router.post("/trigger-training/{user_id}", response_model=TrainingTriggerResponse)
async def trigger_training_endpoint(
    user_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_admin),
    db = Depends(get_database)
):
    """
    Manually trigger training for a user (admin only).
    
    This endpoint allows administrators to manually trigger training for any user,
    regardless of their feedback count. Useful for:
    - Testing training pipeline
    - Retraining after model updates
    - Recovering from failed training jobs
    
    **Requirements:**
    - User must be authenticated as admin
    - Target user must exist
    
    **Note:** This bypasses the normal feedback count requirements and will
    attempt to train even if the user has fewer than 20 feedback samples
    (which may fail if insufficient data).
    
    Requirements: 10.5, 10.6
    """
    try:
        # Verify user exists
        from bson import ObjectId
        try:
            user_oid = ObjectId(user_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid user ID format: {user_id}"
            )
        
        user_doc = db.users.find_one({"_id": user_oid})
        if user_doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found"
            )
        
        # Create task queue and enqueue training job
        task_queue = FastAPITaskQueue(background_tasks)
        
        from training.train_user_head import train_user_head_async
        job_id = task_queue.enqueue(train_user_head_async, user_id=user_id, db=db)
        
        # Create training job record
        create_training_job(db, user_id, job_id)
        
        logger.info(
            f"Admin {current_user['user_id']} triggered training for user {user_id}: "
            f"job_id={job_id}"
        )
        
        return TrainingTriggerResponse(
            status="success",
            training_job_id=job_id,
            message=f"Training job enqueued for user {user_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger training: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger training. Please try again."
        )
