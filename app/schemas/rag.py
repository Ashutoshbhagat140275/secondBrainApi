from pydantic import BaseModel
from typing import List, Optional


class RAGQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5


class RAGSource(BaseModel):
    text: str
    session_id: str
    timestamp: str
    score: float


class RAGResponse(BaseModel):
    answer: str
    sources: List[RAGSource]
    query: str

