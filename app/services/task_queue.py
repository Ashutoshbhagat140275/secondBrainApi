"""
Task Queue Abstraction Layer

This module provides an abstract interface for background task queues,
allowing the system to start with FastAPI BackgroundTasks and migrate
to Celery + Redis for production without changing calling code.

Key responsibilities:
- Enqueue background tasks (training jobs)
- Generate unique job identifiers
- Support multiple queue implementations (FastAPI, Celery)
"""

import uuid
import logging
from abc import ABC, abstractmethod
from typing import Callable, Any


logger = logging.getLogger(__name__)


class TaskQueue(ABC):
    """Abstract interface for background task queue."""
    
    @abstractmethod
    def enqueue(self, func: Callable, *args, **kwargs) -> str:
        """
        Enqueue a background task.
        
        Parameters:
            func: The function to execute in the background
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        
        Returns:
            job_id: Unique identifier for the job (UUID string)
        """
        pass


class FastAPITaskQueue(TaskQueue):
    """
    FastAPI BackgroundTasks implementation.
    
    Suitable for MVP and development. Jobs are executed in the same
    process and are lost on server restart.
    
    Limitations:
    - No persistence (jobs lost on restart)
    - No distributed workers
    - Limited concurrency (single server)
    
    Migration path: Replace with CeleryTaskQueue for production.
    """
    
    def __init__(self, background_tasks):
        """
        Initialize with FastAPI BackgroundTasks instance.
        
        Parameters:
            background_tasks: FastAPI BackgroundTasks from request context
        """
        self.background_tasks = background_tasks
    
    def enqueue(self, func: Callable, *args, **kwargs) -> str:
        """
        Enqueue a background task using FastAPI BackgroundTasks.
        
        The job_id is generated here and passed to the function as the
        first argument, allowing the function to track its own job status.
        
        Parameters:
            func: Function to execute (must accept job_id as first arg)
            *args: Additional positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            job_id: UUID string identifying this job
        """
        job_id = str(uuid.uuid4())
        
        # Add task to FastAPI's background tasks
        # The function will receive job_id as the first argument
        self.background_tasks.add_task(func, job_id, *args, **kwargs)
        
        func_name = getattr(func, '__name__', repr(func))
        logger.info(f"Task enqueued: job_id={job_id}, func={func_name}")
        
        return job_id


class CeleryTaskQueue(TaskQueue):
    """
    Celery + Redis implementation (future).
    
    Suitable for production with >100 active users.
    
    Benefits:
    - Persistent queue (jobs survive restarts)
    - Distributed workers (horizontal scaling)
    - Retry logic and monitoring
    - Better observability
    
    Implementation note: This is a placeholder for future migration.
    Actual implementation requires Celery configuration and task decorators.
    """
    
    def __init__(self, celery_app):
        """
        Initialize with Celery app instance.
        
        Parameters:
            celery_app: Configured Celery application
        """
        self.celery_app = celery_app
    
    def enqueue(self, func: Callable, *args, **kwargs) -> str:
        """
        Enqueue a background task using Celery.
        
        Note: This requires the function to be decorated with @celery_app.task
        
        Parameters:
            func: Celery task function
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            job_id: Celery task ID
        """
        result = func.delay(*args, **kwargs)
        job_id = result.id
        
        logger.info(f"Celery task enqueued: job_id={job_id}, func={func.__name__}")
        
        return job_id
