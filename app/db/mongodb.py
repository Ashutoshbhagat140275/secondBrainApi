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
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close database connection"""
    global client
    if client:
        client.close()
        logger.info("Disconnected from MongoDB")


def get_database() -> Database:
    """Get database instance"""
    return db

