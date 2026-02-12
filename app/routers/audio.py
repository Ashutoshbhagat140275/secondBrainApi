from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends, BackgroundTasks
from app.schemas.audio import AudioUploadResponse, FeedbackRequest, FeedbackResponse
from app.services.audio_processor import process_audio
from app.services import feedback_service
from app.middleware.auth import get_current_user_id
from app.db.mongodb import get_database
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["Audio"])


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    """Upload and process audio file"""
    try:
        result = await process_audio(user_id, file)
        return AudioUploadResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process audio: {str(e)}"
        )


def _run_user_head_training(user_id: str):
    """
    Background task wrapper for user head training.
    
    This function wraps the training process with proper error handling
    and logging. Training failures are logged but don't affect the API response.
    
    Parameters:
        user_id: User identifier
    
    Requirements: 6.1, 7.1, 7.2, 7.3
    """
    try:
        from training.train_user_head import train_user_head
        
        logger.info(f"Starting user head training for user {user_id}")
        metrics = train_user_head(user_id, force_retrain=False)
        logger.info(
            f"User head training completed for user {user_id}: "
            f"loss={metrics['final_loss']:.4f}, "
            f"accuracy={metrics['final_accuracy']:.4f}, "
            f"samples={metrics['num_samples']}"
        )
    except Exception as e:
        logger.error(
            f"User head training failed for user {user_id}: {e}",
            exc_info=True
        )
        # Don't raise - training failures should not affect API response


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id)
):
    """
    Submit feedback for emotion prediction correction.
    
    This endpoint allows users to correct emotion predictions, which is used
    to train personalized emotion recognition models. When sufficient feedback
    is collected (20+ samples), training is triggered automatically.
    
    Requirements: 5.1, 5.2, 5.3, 6.1, 7.1, 7.2, 7.3
    """
    try:
        db = get_database()
        
        # Submit feedback and get result
        result = feedback_service.submit_feedback(
            db=db,
            user_id=user_id,
            session_id=request.session_id,
            corrected_emotion=request.corrected_emotion
        )
        
        # Trigger user head training asynchronously if threshold reached
        if result["training_triggered"]:
            background_tasks.add_task(_run_user_head_training, user_id)
            logger.info(f"Scheduled user head training for user {user_id}")
        
        return FeedbackResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


