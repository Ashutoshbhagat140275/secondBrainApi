from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.rag import RAGQuery, RAGResponse
from app.services.rag_service import query_rag
from app.middleware.auth import get_current_user_id
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["RAG"])


@router.post("/query", response_model=RAGResponse)
async def query(query_data: RAGQuery, user_id: str = Depends(get_current_user_id)):
    """Query the RAG system with a natural language question"""
    try:
        result = await query_rag(user_id, query_data.query, query_data.top_k)
        print(f"RAG query result: {result}")
        return RAGResponse(**result)

    except Exception as e:
        error_str = str(e)
        logger.error(f"RAG query error: {e}")

        # Check for specific error types and provide user-friendly messages
        if (
            "doesn't exist" in error_str
            or "not found" in error_str.lower()
            or "404" in error_str
        ):
            # Collection doesn't exist - no documents uploaded yet
            return RAGResponse(
                answer="I don't have any audio transcriptions to search through yet. Please upload some audio files first, and then I'll be able to answer your questions based on your recordings.",
                sources=[],
                query=query_data.query,
            )
        elif "Failed to query Ollama" in error_str:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The AI service is temporarily unavailable. Please try again later.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process query: {str(e)}",
            )
