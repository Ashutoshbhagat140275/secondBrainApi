import redis
from app.config import settings
import logging

logger = logging.getLogger(__name__)

redis_client: redis.Redis = None


async def connect_to_redis():
    """Create Redis connection"""
    global redis_client
    try:
        redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=False,  # Keep binary for storing numpy arrays
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # Test connection
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise


async def close_redis_connection():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        redis_client.close()
        logger.info("Disconnected from Redis")


def get_redis_client() -> redis.Redis:
    """Get Redis client instance"""
    return redis_client
