"""
Unit tests for admin API endpoints.

Tests the admin endpoint handlers including authentication, authorization,
user model listing, and cleanup operations.
"""

import sys
import pathlib
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from bson import ObjectId
from datetime import datetime, timedelta
from fastapi import HTTPException

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.routers.admin import list_user_models, cleanup_user_models
from app.schemas.admin import ModelCleanupRequest, UserModelMetadata


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user."""
    return {
        "user_id": str(ObjectId()),
        "email": "admin@example.com"
    }


@pytest.fixture
def mock_regular_user():
    """Create a mock regular user."""
    return {
        "user_id": str(ObjectId()),
        "email": "user@example.com"
    }


@pytest.fixture
def sample_user_models():
    """Create sample user model metadata."""
    return [
        UserModelMetadata(
            user_id=str(ObjectId()),
            feedback_count=45,
            model_size_kb=24.5,
            last_trained=datetime.utcnow() - timedelta(days=5)
        ),
        UserModelMetadata(
            user_id=str(ObjectId()),
            feedback_count=30,
            model_size_kb=23.8,
            last_trained=datetime.utcnow() - timedelta(days=100)
        ),
        UserModelMetadata(
            user_id=str(ObjectId()),
            feedback_count=15,
            model_size_kb=22.1,
            last_trained=datetime.utcnow() - timedelta(days=200)
        )
    ]


@pytest.mark.asyncio
async def test_list_user_models_success(mock_admin_user, sample_user_models):
    """Test successful listing of user models."""
    with patch("app.routers.admin.get_all_user_models", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = sample_user_models
        
        response = await list_user_models(current_admin=mock_admin_user)
        
        assert response.total_count == 3
        assert len(response.models) == 3
        assert response.models[0].feedback_count == 45
        assert response.models[1].feedback_count == 30
        assert response.models[2].feedback_count == 15
        
        mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_list_user_models_empty(mock_admin_user):
    """Test listing when no user models exist."""
    with patch("app.routers.admin.get_all_user_models", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = []
        
        response = await list_user_models(current_admin=mock_admin_user)
        
        assert response.total_count == 0
        assert len(response.models) == 0


@pytest.mark.asyncio
async def test_list_user_models_service_error(mock_admin_user):
    """Test handling of service errors when listing models."""
    with patch("app.routers.admin.get_all_user_models", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Database connection failed")
        
        with pytest.raises(HTTPException) as exc_info:
            await list_user_models(current_admin=mock_admin_user)
        
        assert exc_info.value.status_code == 500
        assert "Failed to retrieve user models" in exc_info.value.detail


@pytest.mark.asyncio
async def test_cleanup_user_models_by_inactivity(mock_admin_user):
    """Test cleanup of inactive models based on days inactive."""
    request = ModelCleanupRequest(
        min_days_inactive=90,
        max_feedback_count=None,
        user_ids=None
    )
    
    expected_response = {
        "deleted_count": 2,
        "deleted_user_ids": ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"],
        "total_space_freed_kb": 45.9
    }
    
    with patch("app.routers.admin.cleanup_inactive_models", new_callable=AsyncMock) as mock_cleanup:
        mock_cleanup.return_value = expected_response
        
        response = await cleanup_user_models(request, current_admin=mock_admin_user)
        
        assert response["deleted_count"] == 2
        assert len(response["deleted_user_ids"]) == 2
        assert response["total_space_freed_kb"] == 45.9
        
        mock_cleanup.assert_called_once_with(
            min_days_inactive=90,
            max_feedback_count=None,
            user_ids=None
        )


@pytest.mark.asyncio
async def test_cleanup_user_models_by_feedback_count(mock_admin_user):
    """Test cleanup of models with low feedback count."""
    request = ModelCleanupRequest(
        min_days_inactive=None,
        max_feedback_count=20,
        user_ids=None
    )
    
    expected_response = {
        "deleted_count": 1,
        "deleted_user_ids": ["507f1f77bcf86cd799439013"],
        "total_space_freed_kb": 22.1
    }
    
    with patch("app.routers.admin.cleanup_inactive_models", new_callable=AsyncMock) as mock_cleanup:
        mock_cleanup.return_value = expected_response
        
        response = await cleanup_user_models(request, current_admin=mock_admin_user)
        
        assert response["deleted_count"] == 1
        assert response["total_space_freed_kb"] == 22.1


@pytest.mark.asyncio
async def test_cleanup_user_models_specific_users(mock_admin_user):
    """Test cleanup of specific user models by ID."""
    user_ids = ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"]
    request = ModelCleanupRequest(
        min_days_inactive=None,
        max_feedback_count=None,
        user_ids=user_ids
    )
    
    expected_response = {
        "deleted_count": 2,
        "deleted_user_ids": user_ids,
        "total_space_freed_kb": 48.3
    }
    
    with patch("app.routers.admin.cleanup_inactive_models", new_callable=AsyncMock) as mock_cleanup:
        mock_cleanup.return_value = expected_response
        
        response = await cleanup_user_models(request, current_admin=mock_admin_user)
        
        assert response["deleted_count"] == 2
        assert response["deleted_user_ids"] == user_ids


@pytest.mark.asyncio
async def test_cleanup_user_models_combined_criteria(mock_admin_user):
    """Test cleanup with both inactivity and feedback count criteria."""
    request = ModelCleanupRequest(
        min_days_inactive=90,
        max_feedback_count=20,
        user_ids=None
    )
    
    expected_response = {
        "deleted_count": 1,
        "deleted_user_ids": ["507f1f77bcf86cd799439013"],
        "total_space_freed_kb": 22.1
    }
    
    with patch("app.routers.admin.cleanup_inactive_models", new_callable=AsyncMock) as mock_cleanup:
        mock_cleanup.return_value = expected_response
        
        response = await cleanup_user_models(request, current_admin=mock_admin_user)
        
        assert response["deleted_count"] == 1


@pytest.mark.asyncio
async def test_cleanup_user_models_no_matches(mock_admin_user):
    """Test cleanup when no models match criteria."""
    request = ModelCleanupRequest(
        min_days_inactive=1000,
        max_feedback_count=None,
        user_ids=None
    )
    
    expected_response = {
        "deleted_count": 0,
        "deleted_user_ids": [],
        "total_space_freed_kb": 0.0
    }
    
    with patch("app.routers.admin.cleanup_inactive_models", new_callable=AsyncMock) as mock_cleanup:
        mock_cleanup.return_value = expected_response
        
        response = await cleanup_user_models(request, current_admin=mock_admin_user)
        
        assert response["deleted_count"] == 0
        assert len(response["deleted_user_ids"]) == 0
        assert response["total_space_freed_kb"] == 0.0


@pytest.mark.asyncio
async def test_cleanup_user_models_service_error(mock_admin_user):
    """Test handling of service errors during cleanup."""
    request = ModelCleanupRequest(
        min_days_inactive=90,
        max_feedback_count=None,
        user_ids=None
    )
    
    with patch("app.routers.admin.cleanup_inactive_models", new_callable=AsyncMock) as mock_cleanup:
        mock_cleanup.side_effect = Exception("File system error")
        
        with pytest.raises(HTTPException) as exc_info:
            await cleanup_user_models(request, current_admin=mock_admin_user)
        
        assert exc_info.value.status_code == 500
        assert "Failed to cleanup user models" in exc_info.value.detail


def test_model_cleanup_request_validation():
    """Test ModelCleanupRequest schema validation."""
    # Valid request with min_days_inactive
    request = ModelCleanupRequest(min_days_inactive=90)
    assert request.min_days_inactive == 90
    assert request.max_feedback_count is None
    assert request.user_ids is None
    
    # Valid request with max_feedback_count
    request = ModelCleanupRequest(max_feedback_count=20)
    assert request.max_feedback_count == 20
    
    # Valid request with user_ids
    request = ModelCleanupRequest(user_ids=["507f1f77bcf86cd799439011"])
    assert len(request.user_ids) == 1
    
    # Valid request with all criteria
    request = ModelCleanupRequest(
        min_days_inactive=90,
        max_feedback_count=20,
        user_ids=["507f1f77bcf86cd799439011"]
    )
    assert request.min_days_inactive == 90
    assert request.max_feedback_count == 20
    assert len(request.user_ids) == 1
    
    # Invalid: negative days
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ModelCleanupRequest(min_days_inactive=-1)
    
    # Invalid: negative feedback count
    with pytest.raises(ValidationError):
        ModelCleanupRequest(max_feedback_count=-1)


def test_user_model_metadata_schema():
    """Test UserModelMetadata schema."""
    metadata = UserModelMetadata(
        user_id="507f1f77bcf86cd799439011",
        feedback_count=45,
        model_size_kb=24.5,
        last_trained=datetime.utcnow()
    )
    
    assert metadata.user_id == "507f1f77bcf86cd799439011"
    assert metadata.feedback_count == 45
    assert metadata.model_size_kb == 24.5
    assert isinstance(metadata.last_trained, datetime)
    
    # Test with None last_trained
    metadata = UserModelMetadata(
        user_id="507f1f77bcf86cd799439011",
        feedback_count=45,
        model_size_kb=24.5,
        last_trained=None
    )
    assert metadata.last_trained is None
