from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "rag_audio_db"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # JWT
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Redis Query Cache
    redis_url: str = "redis://localhost:6379/0"
    query_cache_ttl_seconds: int = 3600  # 1 hour
    query_cache_max_per_user: int = 20
    query_cache_similarity_threshold: float = (
        0.95  # Minimum cosine similarity for cache hit
    )

    # Whisper Transcription
    whisper_model_size: str = "small"  # tiny | base | small | medium | large
    whisper_language: str = "en"  # ISO-639-1 code; set to None for auto-detect

    # Audio Processing
    audio_upload_dir: str = "./uploads"
    max_audio_size_mb: int = 50

    # Emotion Model
    emotion_model_path: str = "./models/emotion_model.pt"
    emotion_scaler_path: str = "./models/feature_scaler.joblib"

    # Application
    debug: bool = True
    log_level: str = "INFO"

    # User Head Storage (MongoDB Migration)
    USE_MONGODB_STORAGE: bool = False
    DUAL_SAVE_MODE: bool = False
    MONGODB_STORAGE_COMPRESSION: str = "gzip"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
