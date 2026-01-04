from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.db.qdrant import connect_to_qdrant
from app.routers import auth, audio, rag, dashboard
import logging

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Backend with Audio Emotion Analysis",
    description="Multi-tenant RAG system with audio processing and emotion analysis",
    version="1.0.0"
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
    
    logger.info("Application started (some services may be unavailable)")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    logger.info("Shutting down application...")
    await close_mongo_connection()
    logger.info("Application shut down")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Backend with Audio Emotion Analysis API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

