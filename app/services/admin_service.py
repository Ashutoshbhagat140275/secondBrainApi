from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
from bson import ObjectId
import logging

from app.services.feature_config import USER_HEADS_DIR
from app.schemas.admin import UserModelMetadata, ModelCleanupResponse
from app.db.mongodb import get_database
from app.models.user import User

logger = logging.getLogger(__name__)


async def get_all_user_models() -> List[UserModelMetadata]:
    """
    Retrieve metadata for all user models.
    
    Scans the user_heads directory and combines file metadata with
    database information about feedback counts.
    
    Returns:
        List[UserModelMetadata]: List of user model metadata
    """
    models = []
    
    # Ensure directory exists
    if not USER_HEADS_DIR.exists():
        logger.warning(f"User heads directory does not exist: {USER_HEADS_DIR}")
        return models
    
    # Get database connection
    db = get_database()
    user_collection = User.get_collection(db)
    
    # Scan all .pt files in user_heads directory
    for model_file in USER_HEADS_DIR.glob("*.pt"):
        try:
            # Extract user_id from filename (format: {user_id}.pt)
            user_id = model_file.stem
            
            # Validate user_id format (should be valid ObjectId)
            try:
                ObjectId(user_id)
            except Exception:
                logger.warning(f"Invalid user_id format in filename: {model_file.name}")
                continue
            
            # Get file metadata
            file_stat = model_file.stat()
            model_size_kb = file_stat.st_size / 1024.0
            last_modified = datetime.fromtimestamp(file_stat.st_mtime)
            
            # Get user feedback count from database
            user = user_collection.find_one({"_id": ObjectId(user_id)})
            feedback_count = user.get("feedback_count", 0) if user else 0
            
            models.append(UserModelMetadata(
                user_id=user_id,
                feedback_count=feedback_count,
                model_size_kb=round(model_size_kb, 2),
                last_trained=last_modified
            ))
            
        except Exception as e:
            logger.error(f"Error processing model file {model_file}: {e}")
            continue
    
    # Sort by user_id for consistent ordering
    models.sort(key=lambda m: m.user_id)
    
    return models


async def cleanup_inactive_models(
    min_days_inactive: Optional[int] = None,
    max_feedback_count: Optional[int] = None,
    user_ids: Optional[List[str]] = None
) -> ModelCleanupResponse:
    """
    Delete inactive user models based on criteria.
    
    Parameters:
        min_days_inactive: Minimum days since last training to consider inactive
        max_feedback_count: Maximum feedback count to consider inactive
        user_ids: Specific user IDs to delete (overrides other criteria)
    
    Returns:
        ModelCleanupResponse: Summary of deleted models
    """
    deleted_user_ids = []
    total_space_freed_kb = 0.0
    
    # Ensure directory exists
    if not USER_HEADS_DIR.exists():
        logger.warning(f"User heads directory does not exist: {USER_HEADS_DIR}")
        return ModelCleanupResponse(
            deleted_count=0,
            deleted_user_ids=[],
            total_space_freed_kb=0.0
        )
    
    # Get all models
    all_models = await get_all_user_models()
    
    # Filter models to delete
    models_to_delete = []
    
    if user_ids:
        # Delete specific user IDs
        models_to_delete = [m for m in all_models if m.user_id in user_ids]
    else:
        # Apply criteria filters
        cutoff_date = None
        if min_days_inactive:
            cutoff_date = datetime.utcnow() - timedelta(days=min_days_inactive)
        
        for model in all_models:
            should_delete = False
            
            # Check inactivity criteria
            if cutoff_date and model.last_trained:
                if model.last_trained < cutoff_date:
                    should_delete = True
            
            # Check feedback count criteria
            if max_feedback_count is not None:
                if model.feedback_count <= max_feedback_count:
                    should_delete = True
            
            # If both criteria specified, require both to be true (AND logic)
            if min_days_inactive and max_feedback_count:
                should_delete = (
                    model.last_trained and model.last_trained < cutoff_date
                    and model.feedback_count <= max_feedback_count
                )
            
            if should_delete:
                models_to_delete.append(model)
    
    # Delete models
    for model in models_to_delete:
        model_path = USER_HEADS_DIR / f"{model.user_id}.pt"
        
        try:
            if model_path.exists():
                # Track space freed
                total_space_freed_kb += model.model_size_kb
                
                # Delete file
                model_path.unlink()
                deleted_user_ids.append(model.user_id)
                
                logger.info(f"Deleted user model: {model.user_id} ({model.model_size_kb:.2f} KB)")
            else:
                logger.warning(f"Model file not found: {model_path}")
        except Exception as e:
            logger.error(f"Failed to delete model {model.user_id}: {e}")
            continue
    
    return ModelCleanupResponse(
        deleted_count=len(deleted_user_ids),
        deleted_user_ids=deleted_user_ids,
        total_space_freed_kb=round(total_space_freed_kb, 2)
    )
