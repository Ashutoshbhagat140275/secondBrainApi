from app.config import settings
from app.services.vector_store import search_documents
import logging
import requests

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
        # Step 1: Search for relevant documents
        documents = await search_documents(user_id, query, top_k)
        
        if not documents:
            return {
                "answer": "I don't have any audio transcriptions to search through yet. Please upload some audio files first, and then I'll be able to answer your questions based on your recordings.",
                "sources": [],
                "query": query
            }
        
        # Step 2: Build context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1} (from session {doc['session_id']}):\n{doc['text']}"
            for i, doc in enumerate(documents)
        ])
        
        # Step 3: Build prompt for LLM
        prompt = f"""Based on the following context from audio transcriptions, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Step 4: Query Ollama
        logger.info(f"Querying Ollama model: {settings.ollama_model}")
        try:
            # Use Ollama REST API
            response = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120  # 2 minute timeout for LLM generation
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Failed to query Ollama: {str(e)}")
        
        # Step 5: Format sources
        sources = [
            {
                "text": doc["text"],
                "session_id": doc["session_id"],
                "timestamp": doc["timestamp"],
                "score": doc["score"]
            }
            for doc in documents
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query
        }
    
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise

