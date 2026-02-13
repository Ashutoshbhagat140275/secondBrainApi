"""
Database Migration Script: Feedback Loop Indexes

This script creates the required MongoDB indexes for the feedback loop feature:
- UserFeedback collection indexes
- TrainingJob collection indexes

The script supports:
- Forward migration (create indexes)
- Rollback (drop indexes)
- Dry-run mode (preview changes without applying)
- Idempotent execution (safe to run multiple times)

Usage:
    python scripts/migrate_feedback_indexes.py --action=migrate
    python scripts/migrate_feedback_indexes.py --action=rollback
    python scripts/migrate_feedback_indexes.py --action=migrate --dry-run

Requirements: 2.2, 2.3
"""

import sys
import pathlib
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.db.mongodb import get_database
from app.config import settings
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import OperationFailure


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index Definitions
# ---------------------------------------------------------------------------

USER_FEEDBACK_INDEXES = [
    {
        "name": "user_id_1",
        "keys": [("user_id", ASCENDING)],
        "description": "Index on user_id for fast training queries"
    },
    {
        "name": "timestamp_1",
        "keys": [("timestamp", ASCENDING)],
        "description": "Index on timestamp for chronological ordering"
    },
    {
        "name": "user_id_1_timestamp_1",
        "keys": [("user_id", ASCENDING), ("timestamp", ASCENDING)],
        "description": "Compound index for optimized training data loading"
    }
]

TRAINING_JOB_INDEXES = [
    {
        "name": "user_id_1",
        "keys": [("user_id", ASCENDING)],
        "description": "Index on user_id for user-specific queries"
    },
    {
        "name": "job_id_1",
        "keys": [("job_id", ASCENDING)],
        "unique": True,
        "description": "Unique index on job_id for job lookup"
    },
    {
        "name": "user_id_1_created_at_-1",
        "keys": [("user_id", ASCENDING), ("created_at", DESCENDING)],
        "description": "Compound index for latest job queries"
    }
]


# ---------------------------------------------------------------------------
# Migration Functions
# ---------------------------------------------------------------------------

def get_existing_indexes(collection) -> List[str]:
    """Get list of existing index names for a collection."""
    try:
        indexes = collection.list_indexes()
        return [idx["name"] for idx in indexes if idx["name"] != "_id_"]
    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")
        return []


def create_index(collection, index_def: Dict[str, Any], dry_run: bool = False) -> bool:
    """
    Create a single index on a collection.
    
    Args:
        collection: MongoDB collection
        index_def: Index definition dict
        dry_run: If True, only log what would be done
    
    Returns:
        True if index was created (or would be created in dry-run)
    """
    index_name = index_def["name"]
    keys = index_def["keys"]
    unique = index_def.get("unique", False)
    description = index_def.get("description", "")
    
    # Check if index already exists
    existing_indexes = get_existing_indexes(collection)
    
    if index_name in existing_indexes:
        logger.info(f"  ✓ Index '{index_name}' already exists - skipping")
        return False
    
    if dry_run:
        logger.info(f"  [DRY-RUN] Would create index '{index_name}': {description}")
        return True
    
    try:
        collection.create_index(keys, name=index_name, unique=unique)
        logger.info(f"  ✓ Created index '{index_name}': {description}")
        return True
    except OperationFailure as e:
        logger.error(f"  ✗ Failed to create index '{index_name}': {e}")
        return False


def drop_index(collection, index_name: str, dry_run: bool = False) -> bool:
    """
    Drop a single index from a collection.
    
    Args:
        collection: MongoDB collection
        index_name: Name of index to drop
        dry_run: If True, only log what would be done
    
    Returns:
        True if index was dropped (or would be dropped in dry-run)
    """
    # Check if index exists
    existing_indexes = get_existing_indexes(collection)
    
    if index_name not in existing_indexes:
        logger.info(f"  ✓ Index '{index_name}' does not exist - skipping")
        return False
    
    if dry_run:
        logger.info(f"  [DRY-RUN] Would drop index '{index_name}'")
        return True
    
    try:
        collection.drop_index(index_name)
        logger.info(f"  ✓ Dropped index '{index_name}'")
        return True
    except OperationFailure as e:
        logger.error(f"  ✗ Failed to drop index '{index_name}': {e}")
        return False


def migrate_forward(db, dry_run: bool = False) -> bool:
    """
    Create all required indexes for feedback loop feature.
    
    Args:
        db: MongoDB database instance
        dry_run: If True, only log what would be done
    
    Returns:
        True if migration succeeded
    """
    logger.info("="*70)
    logger.info("FORWARD MIGRATION: Creating Feedback Loop Indexes")
    logger.info("="*70)
    
    if dry_run:
        logger.info("[DRY-RUN MODE] No changes will be applied\n")
    
    success = True
    
    # Migrate UserFeedback indexes
    logger.info("\n1. UserFeedback Collection:")
    user_feedback = db.user_feedback
    
    for index_def in USER_FEEDBACK_INDEXES:
        if not create_index(user_feedback, index_def, dry_run):
            if not dry_run:
                success = False
    
    # Migrate TrainingJob indexes
    logger.info("\n2. TrainingJob Collection:")
    training_jobs = db.training_jobs
    
    for index_def in TRAINING_JOB_INDEXES:
        if not create_index(training_jobs, index_def, dry_run):
            if not dry_run:
                success = False
    
    # Summary
    logger.info("\n" + "="*70)
    if dry_run:
        logger.info("DRY-RUN COMPLETE: Review changes above")
    elif success:
        logger.info("MIGRATION COMPLETE: All indexes created successfully")
    else:
        logger.error("MIGRATION FAILED: Some indexes could not be created")
    logger.info("="*70 + "\n")
    
    return success


def migrate_rollback(db, dry_run: bool = False) -> bool:
    """
    Drop all feedback loop indexes (rollback migration).
    
    Args:
        db: MongoDB database instance
        dry_run: If True, only log what would be done
    
    Returns:
        True if rollback succeeded
    """
    logger.info("="*70)
    logger.info("ROLLBACK MIGRATION: Dropping Feedback Loop Indexes")
    logger.info("="*70)
    
    if dry_run:
        logger.info("[DRY-RUN MODE] No changes will be applied\n")
    
    success = True
    
    # Rollback UserFeedback indexes
    logger.info("\n1. UserFeedback Collection:")
    user_feedback = db.user_feedback
    
    for index_def in USER_FEEDBACK_INDEXES:
        if not drop_index(user_feedback, index_def["name"], dry_run):
            if not dry_run:
                success = False
    
    # Rollback TrainingJob indexes
    logger.info("\n2. TrainingJob Collection:")
    training_jobs = db.training_jobs
    
    for index_def in TRAINING_JOB_INDEXES:
        if not drop_index(training_jobs, index_def["name"], dry_run):
            if not dry_run:
                success = False
    
    # Summary
    logger.info("\n" + "="*70)
    if dry_run:
        logger.info("DRY-RUN COMPLETE: Review changes above")
    elif success:
        logger.info("ROLLBACK COMPLETE: All indexes dropped successfully")
    else:
        logger.error("ROLLBACK FAILED: Some indexes could not be dropped")
    logger.info("="*70 + "\n")
    
    return success


def show_status(db):
    """Show current index status for feedback loop collections."""
    logger.info("="*70)
    logger.info("CURRENT INDEX STATUS")
    logger.info("="*70)
    
    # UserFeedback indexes
    logger.info("\n1. UserFeedback Collection:")
    user_feedback = db.user_feedback
    existing = get_existing_indexes(user_feedback)
    
    for index_def in USER_FEEDBACK_INDEXES:
        status = "✓ EXISTS" if index_def["name"] in existing else "✗ MISSING"
        logger.info(f"  {status}: {index_def['name']}")
    
    # TrainingJob indexes
    logger.info("\n2. TrainingJob Collection:")
    training_jobs = db.training_jobs
    existing = get_existing_indexes(training_jobs)
    
    for index_def in TRAINING_JOB_INDEXES:
        status = "✓ EXISTS" if index_def["name"] in existing else "✗ MISSING"
        logger.info(f"  {status}: {index_def['name']}")
    
    logger.info("\n" + "="*70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate feedback loop database indexes"
    )
    parser.add_argument(
        "--action",
        choices=["migrate", "rollback", "status"],
        default="status",
        help="Migration action to perform"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    
    args = parser.parse_args()
    
    try:
        # Connect to database directly (not using the async global connection)
        logger.info("Connecting to MongoDB...")
        client = MongoClient(settings.mongodb_url)
        db = client[settings.mongodb_db_name]
        
        # Test connection
        client.admin.command('ping')
        logger.info("✓ Connected to MongoDB\n")
        
        # Execute action
        if args.action == "migrate":
            success = migrate_forward(db, dry_run=args.dry_run)
            sys.exit(0 if success else 1)
        
        elif args.action == "rollback":
            success = migrate_rollback(db, dry_run=args.dry_run)
            sys.exit(0 if success else 1)
        
        elif args.action == "status":
            show_status(db)
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Migration failed with error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Close connection
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    main()
