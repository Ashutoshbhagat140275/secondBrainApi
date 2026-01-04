from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from app.schemas.audio import AudioUploadResponse
from app.services.audio_processor import process_audio
from app.middleware.auth import get_current_user_id
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

