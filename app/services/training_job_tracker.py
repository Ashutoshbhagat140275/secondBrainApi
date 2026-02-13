"""
Training Job Tracker Service

This module manages training job records in MongoDB, tracking the lifecycle
of asynchronous user head training jobs.

Key responsibilities:
- Create training job records with initial status
- Update job status and metadata as jobs progress
- Query job status by user or job ID
- Track timestamps for job lifecycle events
"""

import logging
from typing import Optional, Dict
from datetime import datetime

from app.models.training_job import TrainingJob


logger = logging.getLogger(__name__)


def create_training_job(db, user_id: str, job_id: str) -> TrainingJob:
    """
    Create a new training job record with status=queued.
    
    Parameters:
        db: MongoDB database instance
        user_id: User identifier
        job_id: Unique job identifier (UUID string)
    
    Returns:
        TrainingJob instance with the created job data
    
    Requirements: 9.3
    """
    now = datetime.utcnow()
    
    training_job = TrainingJob(
        user_id=user_id,
        job_id=job_id,
        status=TrainingJob.STATUS_QUEUED,
        created_at=now,
        updated_at=now
    )
    
    # Insert into MongoDB
    TrainingJob.get_collection(db).insert_one(training_job.to_dict())
    
    logger.info(
        f"Training job created: job_id={job_id}, user_id={user_id}, status=queued"
    )
    
    return training_job


def update_job_status(
    db,
    job_id: str,
    status: str,
    error_message: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Update training job status and metadata.
    
    This function updates the job status and automatically sets appropriate
    timestamps based on the status transition:
    - "running": Sets started_at to current time
    - "completed" or "failed": Sets completed_at to current time
    
    Parameters:
        db: MongoDB database instance
        job_id: Unique job identifier
        status: New status (queued, running, completed, failed)
        error_message: Error message if job failed (optional)
        metrics: Training metrics if job completed (optional)
    
    Raises:
        ValueError: If status is not a valid status value
    
    Requirements: 9.4, 9.5, 9.6
    """
    # Validate status
    if status not in TrainingJob.VALID_STATUSES:
        raise ValueError(
            f"Invalid status: '{status}'. Must be one of: "
            f"{', '.join(sorted(TrainingJob.VALID_STATUSES))}"
        )
    
    now = datetime.utcnow()
    
    # Build update document
    update_doc = {
        "status": status,
        "updated_at": now
    }
    
    # Set timestamps based on status
    if status == TrainingJob.STATUS_RUNNING:
        update_doc["started_at"] = now
    elif status in (TrainingJob.STATUS_COMPLETED, TrainingJob.STATUS_FAILED):
        update_doc["completed_at"] = now
    
    # Add optional fields
    if error_message is not None:
        update_doc["error_message"] = error_message
    
    if metrics is not None:
        update_doc["metrics"] = metrics
    
    # Update in MongoDB
    result = TrainingJob.get_collection(db).update_one(
        {"job_id": job_id},
        {"$set": update_doc}
    )
    
    if result.matched_count == 0:
        logger.warning(f"Training job not found: job_id={job_id}")
    else:
        logger.info(
            f"Training job updated: job_id={job_id}, status={status}"
        )


def get_latest_job(db, user_id: str) -> Optional[TrainingJob]:
    """
    Get the most recent training job for a user.
    
    Returns the job with the most recent created_at timestamp.
    
    Parameters:
        db: MongoDB database instance
        user_id: User identifier
    
    Returns:
        TrainingJob instance if found, None otherwise
    
    Requirements: 9.7
    """
    job_doc = TrainingJob.get_collection(db).find_one(
        {"user_id": user_id},
        sort=[("created_at", -1)]  # Sort by created_at descending (most recent first)
    )
    
    if job_doc is None:
        logger.debug(f"No training jobs found for user {user_id}")
        return None
    
    return TrainingJob.from_dict(job_doc)


def get_job_by_id(db, job_id: str) -> Optional[TrainingJob]:
    """
    Get a specific training job by ID.
    
    Parameters:
        db: MongoDB database instance
        job_id: Unique job identifier
    
    Returns:
        TrainingJob instance if found, None otherwise
    
    Requirements: 9.7
    """
    job_doc = TrainingJob.get_collection(db).find_one({"job_id": job_id})
    
    if job_doc is None:
        logger.debug(f"Training job not found: job_id={job_id}")
        return None
    
    return TrainingJob.from_dict(job_doc)
