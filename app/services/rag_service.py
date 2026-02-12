from app.config import settings
from app.services.vector_store import search_documents, load_embedding_model
from app.services.query_cache import check_cache, store_in_cache
import logging
import requests
import numpy as np

logger = logging.getLogger(__name__)


async def query_rag(user_id: str, query: str, top_k: int = 5) -> dict:
    """
    Process RAG query: retrieve relevant documents and generate answer

    Args:
        user_id: User ID
        query: User query
        top_k: Number of documents to retrieve

    Returns:
        Dictionary with answer, sources, and query
    """
    try:
        # Step 0: Generate query embedding (needed for both cache check and vector search)
        logger.debug(f"Generating embedding for query: '{query}'")
        embedding_model = load_embedding_model()
        query_embedding = embedding_model.encode(query)

        # Step 1: Check cache for semantically similar queries
        cached_result = await check_cache(user_id, query, query_embedding, top_k)
        if cached_result:
            # Cache hit - return immediately without LLM call
            logger.info(f"Returning cached result for query: '{query}'")
            return cached_result

        # Cache miss - proceed with full RAG pipeline
        logger.debug(f"Cache miss - proceeding with full RAG pipeline for: '{query}'")

        # Step 2: Search for relevant documents (reuse pre-computed embedding)
        documents = await search_documents(
            user_id, query, top_k, query_embedding=query_embedding
        )

        if not documents:
            no_docs_response = {
                "answer": "I don't have any audio transcriptions to search through yet. Please upload some audio files first, and then I'll be able to answer your questions based on your recordings.",
                "sources": [],
                "query": query,
            }
            # Don't cache "no documents" responses
            return no_docs_response

        # Step 3: Build context from retrieved documents
        context = "\n\n".join(
            [
                f"Document {i+1} (from session {doc['session_id']}):\n{doc['text']}"
                for i, doc in enumerate(documents)
            ]
        )

        # Step 4: Build prompt for LLM
        prompt = f"""Based on the following context from audio transcriptions, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""

        # Step 5: Query Ollama
        logger.info(f"Querying Ollama model: {settings.ollama_model}")
        try:
            # Use Ollama REST API
            response = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,  # 2 minute timeout for LLM generation
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Failed to query Ollama: {str(e)}")

        # Step 6: Format sources
        sources = [
            {
                "text": doc["text"],
                "session_id": doc["session_id"],
                "timestamp": doc["timestamp"],
                "score": doc["score"],
            }
            for doc in documents
        ]

        # Step 7: Store in cache for future queries
        await store_in_cache(user_id, query, query_embedding, top_k, answer, sources)

        return {"answer": answer, "sources": sources, "query": query}

    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise
