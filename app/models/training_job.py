from pymongo.collection import Collection
from datetime import datetime
from typing import Optional, Dict
from bson import ObjectId


class TrainingJob:
    """
    MongoDB model for tracking training job status.
    
    Tracks the lifecycle of asynchronous user head training jobs,
    including status transitions, error messages, and training metrics.
    """
    
    # Valid status values
    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    
    VALID_STATUSES = {STATUS_QUEUED, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED}
    
    def __init__(
        self,
        user_id: str,
        job_id: str,
        status: str,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        _id: Optional[ObjectId] = None
    ):
        """
        Initialize a TrainingJob instance.
        
        Parameters:
            user_id: User identifier
            job_id: Unique job identifier (UUID)
            status: Job status (queued, running, completed, failed)
            created_at: When the job was created (defaults to now)
            updated_at: When the job was last updated (defaults to now)
            started_at: When the job started running
            completed_at: When the job completed (success or failure)
            error_message: Error message if job failed
            metrics: Training metrics (final_loss, final_accuracy, num_samples, num_epochs)
            _id: MongoDB ObjectId (auto-generated if not provided)
        
        Raises:
            ValueError: If status is not a valid status value
        """
        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status: '{status}'. Must be one of: {', '.join(sorted(self.VALID_STATUSES))}"
            )
        
        self._id = _id or ObjectId()
        self.user_id = user_id
        self.job_id = job_id
        self.status = status
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.started_at = started_at
        self.completed_at = completed_at
        self.error_message = error_message
        self.metrics = metrics
    
    def to_dict(self):
        """Convert TrainingJob instance to dictionary for MongoDB storage."""
        return {
            "_id": self._id,
            "user_id": self.user_id,
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "metrics": self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create TrainingJob instance from MongoDB document."""
        return cls(
            _id=data.get("_id"),
            user_id=data["user_id"],
            job_id=data["job_id"],
            status=data["status"],
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
            metrics=data.get("metrics")
        )
    
    @staticmethod
    def get_collection(db) -> Collection:
        """Get the MongoDB collection for training jobs."""
        return db.training_jobs
