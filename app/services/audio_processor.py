import os
import shutil
from pathlib import Path
from fastapi import UploadFile
from datetime import datetime
from app.config import settings
from app.services.emotion_analyzer import extract_mfcc_features, classify_emotion
from app.services.transcription import transcribe_audio
from app.services.vector_store import store_document
from app.db.mongodb import get_database
from app.models.audio import AudioSession
from app.models.emotion import EmotionAnalysis
import logging

logger = logging.getLogger(__name__)


def validate_audio_file(file: UploadFile) -> bool:
    """Validate audio file format and size"""
    # Check file extension
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise ValueError(f"Unsupported file format. Allowed: {allowed_extensions}")
    
    # Check file size (if available)
    if hasattr(file, "size") and file.size:
        max_size = settings.max_audio_size_mb * 1024 * 1024
        if file.size > max_size:
            raise ValueError(f"File too large. Maximum size: {settings.max_audio_size_mb}MB")
    
    return True


async def save_audio_file(user_id: str, file: UploadFile) -> str:
    """Save uploaded audio file to disk"""
    # Create user directory
    user_dir = Path(settings.audio_upload_dir) / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_ext = Path(file.filename).suffix
    filename = f"{timestamp}{file_ext}"
    file_path = user_dir / filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"Saved audio file: {file_path}")
    return str(file_path)


async def process_audio(
    user_id: str,
    audio_file: UploadFile
) -> dict:
    """
    Complete audio processing pipeline:
    1. Validate and save audio file
    2. Extract MFCC features
    3. Classify emotion
    4. Transcribe audio
    5. Store in vector database
    6. Store metadata in MongoDB
    
    Returns:
        Dictionary with session_id, emotion, confidence, transcription, timestamp
    """
    try:
        # Step 1: Validate file
        validate_audio_file(audio_file)
        print("iniside the processing audio function")
        # Step 2: Save file
        audio_path = await save_audio_file(user_id, audio_file)
        
        # Step 3: Extract MFCC features
        logger.info("Extracting MFCC features...")
        mfcc_features, aggregated_features = extract_mfcc_features(audio_path)
        
        # Step 4: Classify emotion
        logger.info("Classifying emotion...")
        emotion_label, confidence = classify_emotion(aggregated_features)
        
        # Step 5: Transcribe audio
        logger.info("Transcribing audio...")
        transcription = transcribe_audio(audio_path)
        
        # Step 6: Store metadata in MongoDB first to get session_id
        db = get_database()
        timestamp_str = datetime.utcnow().isoformat()
        collection_id = f"user_{user_id}_documents"
        
        session = AudioSession(
            user_id=user_id,
            audio_file_path=audio_path,
            emotion_data={
                "label": emotion_label,
                "confidence": confidence
            },
            transcription_text=transcription,
            qdrant_collection_id=collection_id
        )
        
        session_collection = AudioSession.get_collection(db)
        result = session_collection.insert_one(session.to_dict())
        session_id = str(result.inserted_id)
        
        # Step 7: Store in vector database with session_id
        logger.info("Storing in vector database...")
        await store_document(
            user_id=user_id,
            text=transcription,
            session_id=session_id,
            timestamp=timestamp_str,
            emotion_label=emotion_label
        )
        
        # Step 8: Store emotion analysis
        emotion_analysis = EmotionAnalysis(
            user_id=user_id,
            session_id=session_id,
            emotion_label=emotion_label,
            confidence=confidence,
            mfcc_features=aggregated_features
        )
        
        emotion_collection = EmotionAnalysis.get_collection(db)
        emotion_collection.insert_one(emotion_analysis.to_dict())
        
        logger.info(f"Audio processing completed. Session ID: {session_id}")
        
        return {
            "session_id": session_id,
            "emotion": emotion_label,
            "confidence": confidence,
            "transcription": transcription,
            "timestamp": session.timestamp
        }
    
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise
    finally:
        # Optional: Clean up temporary file after processing
        # os.remove(audio_path)  # Uncomment if you want to delete after processing
        pass

