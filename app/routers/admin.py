from fastapi import APIRouter, Depends, HTTPException, status
from app.middleware.auth import get_current_admin
from app.schemas.admin import UserModelListResponse, UserModelMetadata, ModelCleanupRequest, ModelCleanupResponse
from app.services.admin_service import get_all_user_models, cleanup_inactive_models
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/user-models", response_model=UserModelListResponse)
async def list_user_models(
    current_admin: dict = Depends(get_current_admin)
):
    """
    List all user models with metadata.
    
    Requires admin authentication.
    
    Returns:
        UserModelListResponse: List of user models with metadata including:
            - user_id: User identifier
            - feedback_count: Number of feedback samples
            - model_size_kb: Size of model file in KB
            - last_trained: Timestamp of last training
    """
    try:
        models = await get_all_user_models()
        return UserModelListResponse(
            total_count=len(models),
            models=models
        )
    except Exception as e:
        logger.error(f"Failed to list user models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user models: {str(e)}"
        )


@router.delete("/user-models/cleanup", response_model=ModelCleanupResponse)
async def cleanup_user_models(
    request: ModelCleanupRequest,
    current_admin: dict = Depends(get_current_admin)
):
    """
    Delete inactive user models based on criteria.
    
    Requires admin authentication.
    
    Parameters:
        request: Cleanup criteria including:
            - min_days_inactive: Minimum days since last training (optional)
            - max_feedback_count: Maximum feedback count to consider inactive (optional)
            - user_ids: Specific user IDs to delete (optional)
    
    Returns:
        ModelCleanupResponse: Summary of deleted models
    """
    try:
        result = await cleanup_inactive_models(
            min_days_inactive=request.min_days_inactive,
            max_feedback_count=request.max_feedback_count,
            user_ids=request.user_ids
        )
        return result
    except Exception as e:
        logger.error(f"Failed to cleanup user models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup user models: {str(e)}"
        )
