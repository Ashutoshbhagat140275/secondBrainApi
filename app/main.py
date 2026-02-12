from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.db.qdrant import connect_to_qdrant
from app.db.redis import connect_to_redis, close_redis_connection
from app.services.emotion_analyzer import load_emotion_model
from app.services.global_emotion_head import load_global_head
from app.services.feature_config import GLOBAL_HEAD_PATH, USER_HEADS_DIR
from app.routers import auth, audio, rag, dashboard, admin
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Backend with Audio Emotion Analysis",
    description="Multi-tenant RAG system with audio processing and emotion analysis",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(audio.router)
app.include_router(rag.router)
app.include_router(dashboard.router)
app.include_router(admin.router)


def detect_active_model_format() -> str:
    """
    Detect which emotion model format is currently active.
    
    Returns
    -------
    str
        One of:
        - "dual-head": Global head + at least one user head exists
        - "global-only": Only global head exists (no user heads yet)
        - "legacy": Only legacy embedding_classifier.pt exists (Phase 1)
        - "unavailable": No model artifacts found
    """
    global_head_exists = GLOBAL_HEAD_PATH.exists()
    
    # Check if any user heads exist
    user_heads_exist = False
    if USER_HEADS_DIR.exists():
        user_head_files = list(USER_HEADS_DIR.glob("*.pt"))
        user_heads_exist = len(user_head_files) > 0
    
    if global_head_exists and user_heads_exist:
        return "dual-head"
    elif global_head_exists:
        return "global-only"
    else:
        # Fall back to checking legacy model
        from app.services.feature_config import EMBEDDING_CLASSIFIER_PATH
        if EMBEDDING_CLASSIFIER_PATH.exists():
            return "legacy"
        return "unavailable"


@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup"""
    logger.info("Starting up application...")
    try:
        await connect_to_mongo()
        logger.info("MongoDB connected")
    except Exception as e:
        logger.warning(f"Failed to connect to MongoDB: {e}. Continuing anyway...")

    try:
        await connect_to_qdrant()
        logger.info("Qdrant connected")
    except Exception as e:
        logger.warning(f"Failed to connect to Qdrant: {e}. Some features may not work.")

    try:
        await connect_to_redis()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(
            f"Failed to connect to Redis: {e}. Query caching will be disabled."
        )

    # Detect and log active model format
    model_format = detect_active_model_format()
    logger.info(f"Emotion model format: {model_format}")
    
    # Pre-load emotion models based on detected format
    if model_format in ["dual-head", "global-only"]:
        # Phase 2/3: Dual-head system with global head
        try:
            global_loaded = load_global_head()
            if global_loaded:
                logger.info("Global emotion head preloaded successfully (eager loading)")
                if model_format == "dual-head":
                    # Count user heads for logging
                    user_head_count = len(list(USER_HEADS_DIR.glob("*.pt")))
                    logger.info(
                        f"Dual-head mode active: {user_head_count} user head(s) available "
                        "(lazy-loaded on first request per user)"
                    )
                else:
                    logger.info(
                        "Global-only mode active: User heads will be created as users provide feedback"
                    )
            else:
                logger.warning(
                    "Global emotion head not available — falling back to legacy model"
                )
                # Fall back to legacy model
                loaded = load_emotion_model()
                if loaded:
                    logger.info("Legacy emotion model loaded successfully")
                else:
                    logger.warning("No emotion model available — using fallback predictions")
        except Exception as e:
            logger.warning(f"Failed to load global emotion head: {e}")
            # Try legacy model as fallback
            try:
                loaded = load_emotion_model()
                if loaded:
                    logger.info("Legacy emotion model loaded successfully (fallback)")
                else:
                    logger.warning("No emotion model available — using fallback predictions")
            except Exception as e2:
                logger.warning(f"Failed to load any emotion model: {e2}")
    
    elif model_format == "legacy":
        # Phase 1: Legacy embedding classifier
        try:
            loaded = load_emotion_model()
            if loaded:
                logger.info("Legacy emotion model loaded successfully")
            else:
                logger.warning("Emotion model not available — using fallback predictions")
        except Exception as e:
            logger.warning(f"Failed to load emotion model: {e}")
    
    else:
        # No models available
        logger.warning(
            "No emotion model artifacts found. Classification will use fallback predictions. "
            "Train a model using train_wav2vec2.py or train_global_head.py"
        )

    logger.info("Application started (some services may be unavailable)")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    logger.info("Shutting down application...")
    await close_mongo_connection()
    await close_redis_connection()
    logger.info("Application shut down")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Backend with Audio Emotion Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
