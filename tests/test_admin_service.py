"""
Unit tests for admin service logic.

Tests the admin service functions for listing user models and cleanup operations.
"""

import sys
import pathlib
import pytest
from unittest.mock import MagicMock, patch
from bson import ObjectId
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.admin_service import get_all_user_models, cleanup_inactive_models
from app.models.user import User


@pytest.fixture
def temp_user_heads_dir():
    """Create a temporary directory for user head models."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = MagicMock()
    db.users = MagicMock()
    return db


def create_mock_model_file(directory: Path, user_id: str, size_kb: float, days_old: int):
    """Helper to create a mock model file."""
    file_path = directory / f"{user_id}.pt"
    
    # Create file with specified size
    content = b"0" * int(size_kb * 1024)
    file_path.write_bytes(content)
    
    # Set modification time
    timestamp = (datetime.utcnow() - timedelta(days=days_old)).timestamp()
    import os
    os.utime(file_path, (timestamp, timestamp))
    
    return file_path


@pytest.mark.asyncio
async def test_get_all_user_models_success(temp_user_heads_dir, mock_db):
    """Test successful retrieval of all user models."""
    # Create mock model files
    user_id_1 = str(ObjectId())
    user_id_2 = str(ObjectId())
    user_id_3 = str(ObjectId())
    
    create_mock_model_file(temp_user_heads_dir, user_id_1, 24.5, 5)
    create_mock_model_file(temp_user_heads_dir, user_id_2, 23.8, 100)
    create_mock_model_file(temp_user_heads_dir, user_id_3, 22.1, 200)
    
    # Mock database responses
    def mock_find_one(query):
        user_id = str(query["_id"])
        feedback_counts = {
            user_id_1: 45,
            user_id_2: 30,
            user_id_3: 15
        }
        return {"feedback_count": feedback_counts.get(user_id, 0)}
    
    mock_db.users.find_one = mock_find_one
    
    with patch("app.services.admin_service.USER_HEADS_DIR", temp_user_heads_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            models = await get_all_user_models()
    
    assert len(models) == 3
    
    # Verify models are sorted by user_id
    assert all(models[i].user_id <= models[i+1].user_id for i in range(len(models)-1))
    
    # Verify metadata
    for model in models:
        assert model.feedback_count in [45, 30, 15]
        assert 22.0 <= model.model_size_kb <= 25.0
        assert model.last_trained is not None


@pytest.mark.asyncio
async def test_get_all_user_models_empty_directory(temp_user_heads_dir, mock_db):
    """Test retrieval when no user models exist."""
    with patch("app.services.admin_service.USER_HEADS_DIR", temp_user_heads_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            models = await get_all_user_models()
    
    assert len(models) == 0


@pytest.mark.asyncio
async def test_get_all_user_models_directory_not_exists(mock_db):
    """Test retrieval when user_heads directory doesn't exist."""
    non_existent_dir = Path("/non/existent/directory")
    
    with patch("app.services.admin_service.USER_HEADS_DIR", non_existent_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            models = await get_all_user_models()
    
    assert len(models) == 0


@pytest.mark.asyncio
async def test_get_all_user_models_invalid_filename(temp_user_heads_dir, mock_db):
    """Test that invalid filenames are skipped."""
    # Create valid model
    user_id = str(ObjectId())
    create_mock_model_file(temp_user_heads_dir, user_id, 24.5, 5)
    
    # Create invalid filename (not a valid ObjectId)
    invalid_file = temp_user_heads_dir / "invalid_name.pt"
    invalid_file.write_bytes(b"test")
    
    mock_db.users.find_one = lambda query: {"feedback_count": 45}
    
    with patch("app.services.admin_service.USER_HEADS_DIR", temp_user_heads_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            models = await get_all_user_models()
    
    # Should only return the valid model
    assert len(models) == 1
    assert models[0].user_id == user_id


@pytest.mark.asyncio
async def test_get_all_user_models_user_not_in_db(temp_user_heads_dir, mock_db):
    """Test handling when user model exists but user not in database."""
    user_id = str(ObjectId())
    create_mock_model_file(temp_user_heads_dir, user_id, 24.5, 5)
    
    # Mock database to return None (user not found)
    mock_db.users.find_one = lambda query: None
    
    with patch("app.services.admin_service.USER_HEADS_DIR", temp_user_heads_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            models = await get_all_user_models()
    
    # Should still return model with feedback_count = 0
    assert len(models) == 1
    assert models[0].feedback_count == 0


@pytest.mark.asyncio
async def test_cleanup_inactive_models_by_days(temp_user_heads_dir, mock_db):
    """Test cleanup of models inactive for specified days."""
    # Create models with different ages
    user_id_recent = str(ObjectId())
    user_id_old_1 = str(ObjectId())
    user_id_old_2 = str(ObjectId())
    
    create_mock_model_file(temp_user_heads_dir, user_id_recent, 24.5, 5)
    create_mock_model_file(temp_user_heads_dir, user_id_old_1, 23.8, 100)
    create_mock_model_file(temp_user_heads_dir, user_id_old_2, 22.1, 200)
    
    def mock_find_one(query):
        return {"feedback_count": 30}
    
    mock_db.users.find_one = mock_find_one
    
    with patch("app.services.admin_service.USER_HEADS_DIR", temp_user_heads_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            result = await cleanup_inactive_models(min_days_inactive=90)
    
    assert result.deleted_count == 2
    assert len(result.deleted_user_ids) == 2
    assert user_id_old_1 in result.deleted_user_ids
    assert user_id_old_2 in result.deleted_user_ids
    assert result.total_space_freed_kb > 40.0
    
    # Verify files were deleted
    assert (temp_user_heads_dir / f"{user_id_recent}.pt").exists()
    assert not (temp_user_heads_dir / f"{user_id_old_1}.pt").exists()
    assert not (temp_user_heads_dir / f"{user_id_old_2}.pt").exists()


@pytest.mark.asyncio
async def test_cleanup_inactive_models_by_feedback_count(temp_user_heads_dir, mock_db):
    """Test cleanup of models with low feedback count."""
    user_id_high = str(ObjectId())
    user_id_low = str(ObjectId())
    
    create_mock_model_file(temp_user_heads_dir, user_id_high, 24.5, 5)
    create_mock_model_file(temp_user_heads_dir, user_id_low, 23.8, 5)
    
    def mock_find_one(query):
        user_id = str(query["_id"])
        feedback_counts = {
            user_id_high: 50,
            user_id_low: 15
        }
        return {"feedback_count": feedback_counts.get(user_id, 0)}
    
    mock_db.users.find_one = mock_find_one
    
    with patch("app.services.admin_service.USER_HEADS_DIR", temp_user_heads_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            result = await cleanup_inactive_models(max_feedback_count=20)
    
    assert result.deleted_count == 1
    assert user_id_low in result.deleted_user_ids
    assert user_id_high not in result.deleted_user_ids
    
    # Verify correct file was deleted
    assert (temp_user_heads_dir / f"{user_id_high}.pt").exists()
    assert not (temp_user_heads_dir / f"{user_id_low}.pt").exists()


@pytest.mark.asyncio
async def test_cleanup_inactive_models_combined_criteria(temp_user_heads_dir, mock_db):
    """Test cleanup with both inactivity and feedback count criteria (AND logic)."""
    user_id_1 = str(ObjectId())  # Old + low feedback -> DELETE
    user_id_2 = str(ObjectId())  # Old + high feedback -> KEEP
    user_id_3 = str(ObjectId())  # Recent + low feedback -> KEEP
    user_id_4 = str(ObjectId())  # Recent + high feedback -> KEEP
    
    create_mock_model_file(temp_user_heads_dir, user_id_1, 24.0, 100)
    create_mock_model_file(temp_user_heads_dir, user_id_2, 24.0, 100)
    create_mock_model_file(temp_user_heads_dir, user_id_3, 24.0, 5)
    create_mock_model_file(temp_user_heads_dir, user_id_4, 24.0, 5)
    
    def mock_find_one(query):
        user_id = str(query["_id"])
        feedback_counts = {
            user_id_1: 15,
            user_id_2: 50,
            user_id_3: 15,
            user_id_4: 50
        }
        return {"feedback_count": feedback_counts.get(user_id, 0)}
    
    mock_db.users.find_one = mock_find_one
    
    with patch("app.services.admin_service.USER_HEADS_DIR", temp_user_heads_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            result = await cleanup_inactive_models(
                min_days_inactive=90,
                max_feedback_count=20
            )
    
    # Only user_id_1 should be deleted (old AND low feedback)
    assert result.deleted_count == 1
    assert user_id_1 in result.deleted_user_ids


@pytest.mark.asyncio
async def test_cleanup_inactive_models_specific_users(temp_user_heads_dir, mock_db):
    """Test cleanup of specific user models by ID."""
    user_id_1 = str(ObjectId())
    user_id_2 = str(ObjectId())
    user_id_3 = str(ObjectId())
    
    create_mock_model_file(temp_user_heads_dir, user_id_1, 24.5, 5)
    create_mock_model_file(temp_user_heads_dir, user_id_2, 23.8, 5)
    create_mock_model_file(temp_user_heads_dir, user_id_3, 22.1, 5)
    
    mock_db.users.find_one = lambda query: {"feedback_count": 30}
    
    with patch("app.services.admin_service.USER_HEADS_DIR", temp_user_heads_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            result = await cleanup_inactive_models(
                user_ids=[user_id_1, user_id_3]
            )
    
    assert result.deleted_count == 2
    assert user_id_1 in result.deleted_user_ids
    assert user_id_3 in result.deleted_user_ids
    assert user_id_2 not in result.deleted_user_ids
    
    # Verify correct files were deleted
    assert not (temp_user_heads_dir / f"{user_id_1}.pt").exists()
    assert (temp_user_heads_dir / f"{user_id_2}.pt").exists()
    assert not (temp_user_heads_dir / f"{user_id_3}.pt").exists()


@pytest.mark.asyncio
async def test_cleanup_inactive_models_no_matches(temp_user_heads_dir, mock_db):
    """Test cleanup when no models match criteria."""
    user_id = str(ObjectId())
    create_mock_model_file(temp_user_heads_dir, user_id, 24.5, 5)
    
    mock_db.users.find_one = lambda query: {"feedback_count": 50}
    
    with patch("app.services.admin_service.USER_HEADS_DIR", temp_user_heads_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            result = await cleanup_inactive_models(min_days_inactive=1000)
    
    assert result.deleted_count == 0
    assert len(result.deleted_user_ids) == 0
    assert result.total_space_freed_kb == 0.0
    
    # Verify file still exists
    assert (temp_user_heads_dir / f"{user_id}.pt").exists()


@pytest.mark.asyncio
async def test_cleanup_inactive_models_directory_not_exists(mock_db):
    """Test cleanup when user_heads directory doesn't exist."""
    non_existent_dir = Path("/non/existent/directory")
    
    with patch("app.services.admin_service.USER_HEADS_DIR", non_existent_dir):
        with patch("app.services.admin_service.get_database", return_value=mock_db):
            result = await cleanup_inactive_models(min_days_inactive=90)
    
    assert result.deleted_count == 0
    assert len(result.deleted_user_ids) == 0
    assert result.total_space_freed_kb == 0.0
