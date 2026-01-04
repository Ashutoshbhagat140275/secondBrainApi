from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionStatus
from app.config import settings
import logging

logger = logging.getLogger(__name__)

client: QdrantClient = None


async def connect_to_qdrant():
    """Create Qdrant connection"""
    global client
    try:
        if settings.qdrant_api_key:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            client = QdrantClient(url=settings.qdrant_url)
        
        # Test connection
        client.get_collections()
        logger.info("Connected to Qdrant")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance"""
    return client


async def create_user_collection(user_id: str, vector_size: int = 384):
    """Create a user-specific collection in Qdrant"""
    collection_name = f"user_{user_id}_documents"
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        existing_names = [col.name for col in collections.collections]
        
        if collection_name not in existing_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.info(f"Collection already exists: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to create collection {collection_name}: {e}")
        raise

