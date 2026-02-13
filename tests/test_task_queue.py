"""
Unit Tests for Task Queue Abstraction Layer

Tests the task queue abstraction and FastAPI BackgroundTasks implementation.

Feature: feedback-loop-personalization
Task: 3. CONSOLIDATED: Implement Core Feedback Loop Infrastructure
"""

import pytest
import uuid
from unittest.mock import Mock, MagicMock
from fastapi import BackgroundTasks

from app.services.task_queue import TaskQueue, FastAPITaskQueue


class TestFastAPITaskQueue:
    """Test FastAPI BackgroundTasks implementation."""
    
    def test_enqueue_generates_job_id(self):
        """Test that enqueue generates a valid UUID job ID."""
        background_tasks = BackgroundTasks()
        queue = FastAPITaskQueue(background_tasks)
        
        def dummy_task(job_id, arg1, arg2):
            pass
        
        job_id = queue.enqueue(dummy_task, arg1="value1", arg2="value2")
        
        # Verify job_id is a valid UUID string
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID format: 8-4-4-4-12
        
        # Verify it can be parsed as UUID
        parsed_uuid = uuid.UUID(job_id)
        assert str(parsed_uuid) == job_id
    
    @pytest.mark.asyncio
    async def test_enqueue_adds_task_to_background_tasks(self):
        """Test that enqueue adds the task to FastAPI's background tasks."""
        background_tasks = BackgroundTasks()
        queue = FastAPITaskQueue(background_tasks)
        
        # Mock function to track if it was called
        mock_func = Mock()
        mock_func.__name__ = "mock_func"  # Add __name__ attribute
        
        job_id = queue.enqueue(mock_func, arg1="test")
        
        # Verify task was added to background_tasks
        assert len(background_tasks.tasks) == 1
        
        # Execute the background task
        task = background_tasks.tasks[0]
        await task()
        
        # Verify the function was called with job_id as first argument
        mock_func.assert_called_once()
        call_args = mock_func.call_args
        assert call_args[0][0] == job_id  # First positional arg is job_id
        assert call_args[1]["arg1"] == "test"  # Keyword args preserved
    
    @pytest.mark.asyncio
    async def test_enqueue_passes_arguments_correctly(self):
        """Test that enqueue passes positional and keyword arguments correctly."""
        background_tasks = BackgroundTasks()
        queue = FastAPITaskQueue(background_tasks)
        
        captured_args = {}
        
        def capture_task(job_id, pos1, pos2, kw1=None, kw2=None):
            captured_args["job_id"] = job_id
            captured_args["pos1"] = pos1
            captured_args["pos2"] = pos2
            captured_args["kw1"] = kw1
            captured_args["kw2"] = kw2
        
        job_id = queue.enqueue(capture_task, "arg1", "arg2", kw1="kwarg1", kw2="kwarg2")
        
        # Execute the background task
        task = background_tasks.tasks[0]
        await task()
        
        # Verify all arguments were passed correctly
        assert captured_args["job_id"] == job_id
        assert captured_args["pos1"] == "arg1"
        assert captured_args["pos2"] == "arg2"
        assert captured_args["kw1"] == "kwarg1"
        assert captured_args["kw2"] == "kwarg2"
    
    def test_enqueue_multiple_tasks(self):
        """Test that multiple tasks can be enqueued and get unique job IDs."""
        background_tasks = BackgroundTasks()
        queue = FastAPITaskQueue(background_tasks)
        
        def dummy_task(job_id):
            pass
        
        job_ids = []
        for i in range(5):
            job_id = queue.enqueue(dummy_task)
            job_ids.append(job_id)
        
        # Verify all job IDs are unique
        assert len(job_ids) == len(set(job_ids))
        
        # Verify all are valid UUIDs
        for job_id in job_ids:
            uuid.UUID(job_id)  # Should not raise
        
        # Verify all tasks were added
        assert len(background_tasks.tasks) == 5
    
    def test_task_queue_abstract_interface(self):
        """Test that TaskQueue is an abstract base class."""
        # Should not be able to instantiate TaskQueue directly
        with pytest.raises(TypeError):
            TaskQueue()
    
    def test_fastapi_task_queue_implements_interface(self):
        """Test that FastAPITaskQueue properly implements TaskQueue interface."""
        background_tasks = BackgroundTasks()
        queue = FastAPITaskQueue(background_tasks)
        
        # Should be an instance of TaskQueue
        assert isinstance(queue, TaskQueue)
        
        # Should have enqueue method
        assert hasattr(queue, "enqueue")
        assert callable(queue.enqueue)


class TestTaskQueueIntegration:
    """Integration tests for task queue with realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_training_job_enqueue_simulation(self):
        """Simulate enqueuing a training job."""
        background_tasks = BackgroundTasks()
        queue = FastAPITaskQueue(background_tasks)
        
        training_results = {}
        
        def mock_train_user_head(job_id, user_id, db):
            """Mock training function."""
            training_results["job_id"] = job_id
            training_results["user_id"] = user_id
            training_results["status"] = "completed"
        
        # Enqueue training job
        user_id = "507f1f77bcf86cd799439011"
        job_id = queue.enqueue(mock_train_user_head, user_id=user_id, db=None)
        
        # Execute background task
        task = background_tasks.tasks[0]
        await task()
        
        # Verify training was executed with correct parameters
        assert training_results["job_id"] == job_id
        assert training_results["user_id"] == user_id
        assert training_results["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_concurrent_job_enqueue(self):
        """Test enqueuing multiple jobs concurrently."""
        background_tasks = BackgroundTasks()
        queue = FastAPITaskQueue(background_tasks)
        
        executed_jobs = []
        
        def track_job(job_id, user_id):
            executed_jobs.append({"job_id": job_id, "user_id": user_id})
        
        # Enqueue multiple jobs
        user_ids = ["user1", "user2", "user3"]
        job_ids = []
        
        for user_id in user_ids:
            job_id = queue.enqueue(track_job, user_id=user_id)
            job_ids.append(job_id)
        
        # Execute all background tasks
        for task in background_tasks.tasks:
            await task()
        
        # Verify all jobs were executed
        assert len(executed_jobs) == 3
        
        # Verify job IDs match
        executed_job_ids = [job["job_id"] for job in executed_jobs]
        assert set(executed_job_ids) == set(job_ids)
        
        # Verify user IDs match
        executed_user_ids = [job["user_id"] for job in executed_jobs]
        assert set(executed_user_ids) == set(user_ids)
