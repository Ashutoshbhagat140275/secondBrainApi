import shutil
from pathlib import Path
from fastapi import UploadFile
from datetime import datetime

from app.config import settings
from app.services.wav2vec2_encoder import extract_wav2vec2_embedding
from app.services.emotion_classifier import classify_emotion_from_embedding
from app.services.dual_head_classifier import classify_with_dual_heads
from app.services.transcription import transcribe_audio
from app.services.vector_store import store_document
from app.services.query_cache import invalidate_user_cache
from app.db.mongodb import get_database
from app.models.audio import AudioSession
from app.models.emotion import EmotionAnalysis
from app.models.user import User
from bson import ObjectId
import logging
import librosa
import numpy as np
import soundfile as sf
import time


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_audio_file(file: UploadFile) -> bool:
    """Validate audio file format and size."""
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise ValueError(f"Unsupported file format. Allowed: {allowed_extensions}")

    if hasattr(file, "size") and file.size:
        max_size = settings.max_audio_size_mb * 1024 * 1024
        if file.size > max_size:
            raise ValueError(
                f"File too large. Maximum size: {settings.max_audio_size_mb}MB"
            )

    return True


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


async def save_audio_file(user_id: str, file: UploadFile) -> str:
    """Save uploaded audio file to disk."""
    user_dir = Path(settings.audio_upload_dir) / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_ext = Path(file.filename).suffix
    filename = f"{timestamp}{file_ext}"
    file_path = user_dir / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"Saved audio file: {file_path}")
    return str(file_path)


async def preprocess_audio(audio_path: str) -> str:
    """Resample to 16 kHz, apply VAD trimming, and peak-normalise."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        # Voice Activity Detection
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) > 0:
            y_voiced = np.concatenate([y[s:e] for s, e in intervals])
        else:
            y_voiced = y

        # Peak normalisation
        peak = np.max(np.abs(y_voiced))
        if peak > 0:
            y_voiced = y_voiced / peak

        sf.write(audio_path, y_voiced, 16000)
        logger.info("Preprocessing completed successfully")
        return audio_path
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return audio_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def process_audio(user_id: str, audio_file: UploadFile) -> dict:
    """
    Complete audio processing pipeline with dual-head emotion classification.

    Steps:
      1. Validate → save → preprocess (16 kHz, VAD, normalise)
      2. Extract 768-dim Wav2Vec2 embedding  (wav2vec2_encoder.py)
      3. Query user feedback count for dual-head classification
      4. Classify emotion via dual-head classifier (dual_head_classifier.py)
      5. Transcribe via Whisper              (transcription.py)
      6. Persist to MongoDB + Qdrant

    Returns
    -------
    dict with keys:
        - session_id: str
        - emotion: str (final blended prediction)
        - confidence: float (final confidence)
        - global_emotion: str (global head prediction)
        - global_confidence: float (global head confidence)
        - user_emotion: str | None (user head prediction, None if not available)
        - user_confidence: float | None (user head confidence, None if not available)
        - blend_weight: float (alpha value used for blending)
        - transcription: str
        - timestamp: datetime
    """
    try:
        # Step 1: Validate file
        validate_audio_file(audio_file)
        logger.info("Audio file validated")

        # Step 2: Save file
        audio_path = await save_audio_file(user_id, audio_file)

        # Step 3: Preprocess (resample, VAD, normalise)
        audio_path = await preprocess_audio(audio_path)

        # Step 4: Extract Wav2Vec2 embedding (768-dim vector)
        logger.info("Extracting Wav2Vec2 embedding...")
        start_time = time.time()
        embedding = extract_wav2vec2_embedding(audio_path)
        embedding_time = time.time() - start_time
        logger.info(f"Embedding extraction completed in {embedding_time:.2f}s (shape: {embedding.shape})")

        # Step 5: Query user feedback count for dual-head classification
        db = get_database()
        user_doc = User.get_collection(db).find_one({"_id": ObjectId(user_id)})
        feedback_count = user_doc.get("feedback_count", 0) if user_doc else 0
        
        # Step 6: Classify emotion using dual-head classifier
        logger.info("Classifying emotion with dual-head classifier...")
        start_time = time.time()
        prediction = classify_with_dual_heads(embedding, user_id, feedback_count)
        classification_time = time.time() - start_time
        
        emotion_label = prediction["emotion"]
        confidence = prediction["confidence"]
        logger.info(
            f"Emotion: {emotion_label} ({confidence:.2f}) - "
            f"Classification time: {classification_time:.2f}s - "
            f"Blend weight: {prediction['blend_weight']:.2f}"
        )

        # Step 7: Transcribe audio
        logger.info("Transcribing audio...")
        transcription = transcribe_audio(audio_path)

        # Step 8: Store metadata in MongoDB
        timestamp_str = datetime.utcnow().isoformat()
        collection_id = f"user_{user_id}_documents"

        session = AudioSession(
            user_id=user_id,
            audio_file_path=audio_path,
            emotion_data={"label": emotion_label, "confidence": confidence},
            transcription_text=transcription,
            qdrant_collection_id=collection_id,
        )

        session_collection = AudioSession.get_collection(db)
        result = session_collection.insert_one(session.to_dict())
        session_id = str(result.inserted_id)

        # Step 9: Store in Qdrant vector database
        logger.info("Storing in vector database...")
        await store_document(
            user_id=user_id,
            text=transcription,
            session_id=session_id,
            timestamp=timestamp_str,
            emotion_label=emotion_label,
        )

        # Invalidate query cache since new content is available
        logger.info("Invalidating query cache for user...")
        await invalidate_user_cache(user_id)

        # Step 10: Store emotion analysis (embedding stored in mfcc_features field)
        emotion_analysis = EmotionAnalysis(
            user_id=user_id,
            session_id=session_id,
            emotion_label=emotion_label,
            confidence=confidence,
            mfcc_features=embedding.tolist(),  # Store 768-dim embedding
        )
        emotion_collection = EmotionAnalysis.get_collection(db)
        emotion_collection.insert_one(emotion_analysis.to_dict())

        logger.info(f"Audio processing completed. Session ID: {session_id}")

        # Return extended response with dual-head prediction metadata
        return {
            "session_id": session_id,
            "emotion": emotion_label,
            "confidence": confidence,
            "global_emotion": prediction["global_emotion"],
            "global_confidence": prediction["global_confidence"],
            "user_emotion": prediction["user_emotion"],
            "user_confidence": prediction["user_confidence"],
            "blend_weight": prediction["blend_weight"],
            "transcription": transcription,
            "timestamp": session.timestamp,
        }

    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        raise
