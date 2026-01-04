from fastapi import APIRouter, HTTPException, status, Depends, Query
from app.schemas.dashboard import EmotionsResponse, StatsResponse, EmotionRecord
from app.db.mongodb import get_database
from app.models.emotion import EmotionAnalysis
from app.models.audio import AudioSession
from app.middleware.auth import get_current_user_id
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/emotions/{user_id}", response_model=EmotionsResponse)
async def get_emotions(
    user_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    current_user_id: str = Depends(get_current_user_id)
):
    """Get emotion analysis data for a user"""
    # Verify user can only access their own data
    if user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    try:
        db = get_database()
        emotion_collection = EmotionAnalysis.get_collection(db)
        
        # Build query
        query = {"user_id": user_id}
        
        if start_date or end_date:
            date_query = {}
            if start_date and start_date.lower() not in ['none', 'null', 'undefined', '']:
                try:
                    date_query["$gte"] = datetime.fromisoformat(start_date)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid start_date format: {start_date}. Expected ISO format (YYYY-MM-DDTHH:MM:SS)"
                    )
            if end_date and end_date.lower() not in ['none', 'null', 'undefined', '']:
                try:
                    date_query["$lte"] = datetime.fromisoformat(end_date)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid end_date format: {end_date}. Expected ISO format (YYYY-MM-DDTHH:MM:SS)"
                    )
            if date_query:
                query["timestamp"] = date_query
        
        # Fetch emotions
        cursor = emotion_collection.find(query).sort("timestamp", -1).limit(limit)
        emotions = []
        
        for doc in cursor:
            emotions.append(EmotionRecord(
                session_id=doc["session_id"],
                emotion_label=doc["emotion_label"],
                confidence=doc["confidence"],
                timestamp=doc["timestamp"]
            ))
        
        return EmotionsResponse(emotions=emotions, total=len(emotions))
    
    except Exception as e:
        logger.error(f"Error fetching emotions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch emotions: {str(e)}"
        )


@router.get("/stats/{user_id}", response_model=StatsResponse)
async def get_stats(
    user_id: str,
    current_user_id: str = Depends(get_current_user_id)
):
    """Get user statistics"""
    # Verify user can only access their own data
    if user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    try:
        db = get_database()
        session_collection = AudioSession.get_collection(db)
        emotion_collection = EmotionAnalysis.get_collection(db)
        
        # Total sessions
        total_sessions = session_collection.count_documents({"user_id": user_id})
        
        # Emotion distribution
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$emotion_label",
                "count": {"$sum": 1}
            }}
        ]
        
        emotion_dist = {}
        for doc in emotion_collection.aggregate(pipeline):
            emotion_dist[doc["_id"]] = doc["count"]
        
        # Average confidence
        pipeline_avg = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": None,
                "avg_confidence": {"$avg": "$confidence"}
            }}
        ]
        
        avg_result = list(emotion_collection.aggregate(pipeline_avg))
        avg_confidence = avg_result[0]["avg_confidence"] if avg_result else 0.0
        
        return StatsResponse(
            total_sessions=total_sessions,
            emotion_distribution=emotion_dist,
            avg_confidence=round(avg_confidence, 3)
        )
    
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stats: {str(e)}"
        )

