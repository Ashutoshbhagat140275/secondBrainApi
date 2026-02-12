"""
Unit tests for admin authentication and authorization.

Tests the admin authentication middleware to ensure only admin users
can access admin endpoints.
"""

import sys
import pathlib
import pytest
from unittest.mock import MagicMock, patch
from bson import ObjectId
from fastapi import HTTPException

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.middleware.auth import get_current_admin


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = MagicMock()
    db.users = MagicMock()
    return db


@pytest.fixture
def admin_user_data():
    """Create admin user data."""
    return {
        "_id": ObjectId(),
        "email": "admin@example.com",
        "password_hash": "hashed_password",
        "is_admin": True,
        "feedback_count": 0
    }


@pytest.fixture
def regular_user_data():
    """Create regular user data."""
    return {
        "_id": ObjectId(),
        "email": "user@example.com",
        "password_hash": "hashed_password",
        "is_admin": False,
        "feedback_count": 10
    }


@pytest.mark.asyncio
async def test_get_current_admin_success(mock_db, admin_user_data):
    """Test successful admin authentication."""
    current_user = {
        "user_id": str(admin_user_data["_id"]),
        "email": admin_user_data["email"]
    }
    
    mock_db.users.find_one = MagicMock(return_value=admin_user_data)
    
    with patch("app.middleware.auth.get_database", return_value=mock_db):
        result = await get_current_admin(current_user)
    
    assert result == current_user
    mock_db.users.find_one.assert_called_once()


@pytest.mark.asyncio
async def test_get_current_admin_not_admin(mock_db, regular_user_data):
    """Test rejection of non-admin user."""
    current_user = {
        "user_id": str(regular_user_data["_id"]),
        "email": regular_user_data["email"]
    }
    
    mock_db.users.find_one = MagicMock(return_value=regular_user_data)
    
    with patch("app.middleware.auth.get_database", return_value=mock_db):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin(current_user)
    
    assert exc_info.value.status_code == 403
    assert "Admin access required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_admin_user_not_found(mock_db):
    """Test rejection when user not found in database."""
    current_user = {
        "user_id": str(ObjectId()),
        "email": "nonexistent@example.com"
    }
    
    mock_db.users.find_one = MagicMock(return_value=None)
    
    with patch("app.middleware.auth.get_database", return_value=mock_db):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin(current_user)
    
    assert exc_info.value.status_code == 403
    assert "Admin access required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_admin_missing_is_admin_field(mock_db):
    """Test rejection when is_admin field is missing (defaults to False)."""
    user_data = {
        "_id": ObjectId(),
        "email": "user@example.com",
        "password_hash": "hashed_password",
        "feedback_count": 0
        # is_admin field missing
    }
    
    current_user = {
        "user_id": str(user_data["_id"]),
        "email": user_data["email"]
    }
    
    mock_db.users.find_one = MagicMock(return_value=user_data)
    
    with patch("app.middleware.auth.get_database", return_value=mock_db):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin(current_user)
    
    assert exc_info.value.status_code == 403
    assert "Admin access required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_admin_invalid_user_id(mock_db):
    """Test handling of invalid user_id format."""
    current_user = {
        "user_id": "invalid_object_id",
        "email": "user@example.com"
    }
    
    mock_db.users.find_one = MagicMock(side_effect=Exception("Invalid ObjectId"))
    
    with patch("app.middleware.auth.get_database", return_value=mock_db):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin(current_user)
    
    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_admin_database_error(mock_db):
    """Test handling of database errors."""
    current_user = {
        "user_id": str(ObjectId()),
        "email": "admin@example.com"
    }
    
    mock_db.users.find_one = MagicMock(side_effect=Exception("Database connection failed"))
    
    with patch("app.middleware.auth.get_database", return_value=mock_db):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin(current_user)
    
    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_admin_is_admin_false_explicitly(mock_db):
    """Test rejection when is_admin is explicitly set to False."""
    user_data = {
        "_id": ObjectId(),
        "email": "user@example.com",
        "password_hash": "hashed_password",
        "is_admin": False,
        "feedback_count": 0
    }
    
    current_user = {
        "user_id": str(user_data["_id"]),
        "email": user_data["email"]
    }
    
    mock_db.users.find_one = MagicMock(return_value=user_data)
    
    with patch("app.middleware.auth.get_database", return_value=mock_db):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin(current_user)
    
    assert exc_info.value.status_code == 403
    assert "Admin access required" in exc_info.value.detail
