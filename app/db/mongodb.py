from pymongo import MongoClient
from pymongo.database import Database
from app.config import settings
import logging

logger = logging.getLogger(__name__)

client: MongoClient = None
db: Database = None


async def connect_to_mongo():
    """Create database connection"""
    global client, db
    try:
        client = MongoClient(settings.mongodb_url)
        db = client[settings.mongodb_db_name]
        # Test connection
        client.admin.command('ping')
        logger.info("Connected to MongoDB")
        
        # Create indexes for feedback loop personalization
        await create_feedback_indexes()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def create_feedback_indexes():
    """Create database indexes for feedback loop and training jobs"""
    try:
        # UserFeedback indexes
        db.user_feedback.create_index("user_id")
        db.user_feedback.create_index("timestamp")
        db.user_feedback.create_index([("user_id", 1), ("timestamp", 1)])
        logger.info("Created UserFeedback indexes")
        
        # TrainingJob indexes
        db.training_jobs.create_index("user_id")
        db.training_jobs.create_index("job_id", unique=True)
        db.training_jobs.create_index([("user_id", 1), ("created_at", -1)])
        logger.info("Created TrainingJob indexes")
        
        # UserModelStorage indexes (MongoDB migration)
        db.user_models.create_index("user_id", unique=True)
        db.user_models.create_index("updated_at")
        logger.info("Created UserModelStorage indexes")
    except Exception as e:
        logger.warning(f"Failed to create indexes (may already exist): {e}")


async def close_mongo_connection():
    """Close database connection"""
    global client
    if client:
        client.close()
        logger.info("Disconnected from MongoDB")


def get_database() -> Database:
    """Get database instance"""
    return db

