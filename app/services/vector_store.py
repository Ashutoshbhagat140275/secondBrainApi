from qdrant_client.models import PointStruct
from app.db.qdrant import get_qdrant_client, create_user_collection
from app.config import settings
from sentence_transformers import SentenceTransformer
import logging
from typing import List
import uuid

logger = logging.getLogger(__name__)

# Global embedding model (loaded once)
_embedding_model = None


def load_embedding_model():
    """Load sentence transformer model for embeddings"""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def get_embedding_dimension() -> int:
    """Get embedding dimension for the current model"""
    model = load_embedding_model()
    return model.get_sentence_embedding_dimension()


async def ensure_user_collection(user_id: str):
    """Ensure user's Qdrant collection exists"""
    vector_size = get_embedding_dimension()
    await create_user_collection(user_id, vector_size)


async def store_document(
    user_id: str,
    text: str,
    session_id: str,
    timestamp: str,
    emotion_label: str = None
) -> str:
    """
    Store a document in user's Qdrant collection
    
    Args:
        user_id: User ID
        text: Text to store
        session_id: Session ID
        timestamp: Timestamp string
        emotion_label: Optional emotion label
    
    Returns:
        Document ID
    """
    try:
        # Ensure collection exists
        await ensure_user_collection(user_id)
        
        # Generate embedding
        model = load_embedding_model()
        embedding = model.encode(text).tolist()
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Prepare payload
        payload = {
            "text": text,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp
        }
        if emotion_label:
            payload["emotion_label"] = emotion_label
        
        # Store in Qdrant
        client = get_qdrant_client()
        collection_name = f"user_{user_id}_documents"
        
        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload=payload
        )
        
        client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        logger.info(f"Stored document {doc_id} in collection {collection_name}")
        return doc_id
    
    except Exception as e:
        logger.error(f"Error storing document: {e}")
        raise


async def search_documents(
    user_id: str,
    query: str,
    top_k: int = 5
) -> List[dict]:
    """
    Search for similar documents in user's collection
    
    Args:
        user_id: User ID
        query: Search query
        top_k: Number of results to return
    
    Returns:
        List of similar documents with scores
    """
    try:
        # Generate query embedding
        model = load_embedding_model()
        query_embedding = model.encode(query).tolist()
        
        # Search in user's collection
        client = get_qdrant_client()
        collection_name = f"user_{user_id}_documents"
        
        # Check if collection exists first
        try:
            collections = client.get_collections()
            existing_names = [col.name for col in collections.collections]
            if collection_name not in existing_names:
                logger.info(f"Collection {collection_name} does not exist - no documents uploaded yet")
                return []  # Return empty list if collection doesn't exist
        except Exception as check_error:
            logger.warning(f"Could not check collection existence: {check_error}")
            # Continue to try search anyway
        
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Format results
        documents = []
        for result in results:
            documents.append({
                "text": result.payload.get("text", ""),
                "session_id": result.payload.get("session_id", ""),
                "timestamp": result.payload.get("timestamp", ""),
                "score": result.score
            })
        
        logger.info(f"Found {len(documents)} documents for query")
        return documents
    
    except Exception as e:
        error_str = str(e)
        # Check if it's a "collection doesn't exist" error
        if "doesn't exist" in error_str or "not found" in error_str.lower() or "404" in error_str:
            logger.info(f"Collection for user {user_id} does not exist - no documents uploaded yet")
            return []  # Return empty list instead of raising error
        logger.error(f"Error searching documents: {e}")
        raise

