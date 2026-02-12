import json
import hashlib
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
from app.config import settings
from app.db.redis import get_redis_client
import logging

logger = logging.getLogger(__name__)


def _compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score (0 to 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def _generate_query_hash(query: str, top_k: int) -> str:
    """Generate a unique hash for query parameters"""
    # Normalize query: lowercase and strip whitespace
    normalized = query.lower().strip()
    hash_input = f"{normalized}:{top_k}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


async def check_cache(
    user_id: str, query: str, query_embedding: np.ndarray, top_k: int
) -> Optional[Dict]:
    """
    Check if a semantically similar query exists in cache

    Args:
        user_id: User ID
        query: Original query text
        query_embedding: Embedding vector of the query
        top_k: Number of documents requested

    Returns:
        Cached result dict with {answer, sources, query} if found, else None
    """
    try:
        redis_client = get_redis_client()
        if not redis_client:
            logger.warning("Redis client not available, skipping cache check")
            return None

        # Get all cache keys for this user
        user_keys_set = f"query_cache_keys:{user_id}"
        cached_keys = redis_client.zrange(user_keys_set, 0, -1)

        if not cached_keys:
            logger.debug(f"No cached queries found for user {user_id}")
            return None

        # Check each cached query for semantic similarity
        best_match = None
        best_similarity = 0.0

        for key in cached_keys:
            try:
                cached_data = redis_client.get(key)
                if not cached_data:
                    # Key expired or deleted, skip
                    continue

                cache_entry = json.loads(cached_data.decode("utf-8"))

                # Check if top_k matches
                if cache_entry.get("top_k") != top_k:
                    continue

                # Compute semantic similarity
                cached_embedding = np.array(cache_entry["query_embedding"])
                similarity = _compute_cosine_similarity(
                    query_embedding, cached_embedding
                )

                logger.debug(
                    f"Similarity between '{query}' and '{cache_entry['original_query']}': {similarity:.4f}"
                )

                # Track best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cache_entry

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse cache entry {key}: {e}")
                continue

        # Check if best match exceeds threshold
        if best_match and best_similarity >= settings.query_cache_similarity_threshold:
            logger.info(
                f"Cache HIT for user {user_id}: '{query}' matched '{best_match['original_query']}' "
                f"with similarity {best_similarity:.4f}"
            )
            return {
                "answer": best_match["answer"],
                "sources": best_match["sources"],
                "query": query,  # Return original query, not cached one
                "_cache_hit": True,
                "_similarity": best_similarity,
            }

        logger.debug(
            f"Cache MISS for user {user_id}: best similarity {best_similarity:.4f} "
            f"below threshold {settings.query_cache_similarity_threshold}"
        )
        return None

    except Exception as e:
        logger.error(f"Error checking cache: {e}")
        # Don't fail the request if cache check fails
        return None


async def store_in_cache(
    user_id: str,
    query: str,
    query_embedding: np.ndarray,
    top_k: int,
    answer: str,
    sources: List[Dict],
) -> None:
    """
    Store query result in cache with TTL

    Args:
        user_id: User ID
        query: Original query text
        query_embedding: Embedding vector of the query
        top_k: Number of documents requested
        answer: LLM-generated answer
        sources: Retrieved source documents
    """
    try:
        redis_client = get_redis_client()
        if not redis_client:
            logger.warning("Redis client not available, skipping cache storage")
            return

        # Generate cache key
        query_hash = _generate_query_hash(query, top_k)
        cache_key = f"query_cache:{user_id}:{query_hash}"
        user_keys_set = f"query_cache_keys:{user_id}"

        # Prepare cache entry
        cache_entry = {
            "original_query": query,
            "query_embedding": query_embedding.tolist(),
            "top_k": top_k,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store in Redis with TTL
        redis_client.setex(
            cache_key, settings.query_cache_ttl_seconds, json.dumps(cache_entry)
        )

        # Add to sorted set (for tracking and cleanup)
        current_timestamp = datetime.utcnow().timestamp()
        redis_client.zadd(user_keys_set, {cache_key: current_timestamp})

        # Enforce max cache size per user
        cache_size = redis_client.zcard(user_keys_set)
        if cache_size > settings.query_cache_max_per_user:
            # Remove oldest entries
            num_to_remove = cache_size - settings.query_cache_max_per_user
            oldest_keys = redis_client.zrange(user_keys_set, 0, num_to_remove - 1)

            if oldest_keys:
                # Delete the actual cache entries
                redis_client.delete(*oldest_keys)
                # Remove from sorted set
                redis_client.zrem(user_keys_set, *oldest_keys)
                logger.info(
                    f"Removed {len(oldest_keys)} old cache entries for user {user_id}"
                )

        logger.info(
            f"Stored query in cache for user {user_id}: '{query}' (key: {cache_key})"
        )

    except Exception as e:
        logger.error(f"Error storing in cache: {e}")
        # Don't fail the request if cache storage fails


async def invalidate_user_cache(user_id: str) -> None:
    """
    Clear all cached queries for a user (called when new documents are uploaded)

    Args:
        user_id: User ID
    """
    try:
        redis_client = get_redis_client()
        if not redis_client:
            logger.warning("Redis client not available, skipping cache invalidation")
            return

        user_keys_set = f"query_cache_keys:{user_id}"

        # Get all cache keys for this user
        cached_keys = redis_client.zrange(user_keys_set, 0, -1)

        if cached_keys:
            # Delete all cache entries
            redis_client.delete(*cached_keys)
            # Delete the sorted set itself
            redis_client.delete(user_keys_set)
            logger.info(
                f"Invalidated {len(cached_keys)} cache entries for user {user_id}"
            )
        else:
            logger.debug(f"No cache entries to invalidate for user {user_id}")

    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        # Don't fail the request if cache invalidation fails
