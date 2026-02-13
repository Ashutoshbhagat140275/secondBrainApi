"""
Unit tests for UserHeadStorageService.

Tests cover:
- save_model with MongoDB storage
- load_model from MongoDB
- exists() method
- delete_model method
- get_storage_stats method
- Compression/decompression round-trip
- Fallback when MongoDB unavailable
- File-based storage mode (USE_MONGODB_STORAGE=False)
- Dual-save mode
- Size validation
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.services.user_head_storage import (
    UserHeadStorageService,
    USER_HEADS_DIR,
    MAX_MODEL_SIZE_BYTES
)
from app.models.user_model_storage import UserModelStorage


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 8)


@pytest.fixture
def sample_state_dict():
    """Create a sample PyTorch state_dict."""
    model = SimpleModel()
    return model.state_dict()


@pytest.fixture
def mock_settings():
    """Mock settings object."""
    settings = Mock()
    settings.USE_MONGODB_STORAGE = False
    settings.DUAL_SAVE_MODE = False
    settings.MONGODB_STORAGE_COMPRESSION = "gzip"
    return settings


@pytest.fixture
def mock_db():
    """Mock MongoDB database."""
    db = Mock()
    db.user_models = Mock()
    return db


@pytest.fixture
def storage_service_file_mode(mock_settings, tmp_path):
    """Create storage service in file-only mode with temp directory."""
    with patch("app.services.user_head_storage.settings", mock_settings):
        with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
            with patch("app.services.feature_config.MODEL_DIR", tmp_path.parent):
                service = UserHeadStorageService()
                # Manually override the directory for this instance
                import app.services.user_head_storage
                app.services.user_head_storage.USER_HEADS_DIR = tmp_path
                return service


@pytest.fixture
def storage_service_mongodb_mode(mock_settings, mock_db, tmp_path):
    """Create storage service in MongoDB mode."""
    mock_settings.USE_MONGODB_STORAGE = True
    
    with patch("app.services.user_head_storage.settings", mock_settings):
        with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
            with patch("app.services.user_head_storage.get_database", return_value=mock_db):
                with patch("app.services.feature_config.MODEL_DIR", tmp_path.parent):
                    service = UserHeadStorageService()
                    service.db = mock_db
                    # Manually override the directory for this instance
                    import app.services.user_head_storage
                    app.services.user_head_storage.USER_HEADS_DIR = tmp_path
                    return service


@pytest.fixture
def storage_service_dual_mode(mock_settings, mock_db, tmp_path):
    """Create storage service in dual-save mode."""
    mock_settings.USE_MONGODB_STORAGE = True
    mock_settings.DUAL_SAVE_MODE = True
    
    with patch("app.services.user_head_storage.settings", mock_settings):
        with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
            with patch("app.services.user_head_storage.get_database", return_value=mock_db):
                with patch("app.services.feature_config.MODEL_DIR", tmp_path.parent):
                    service = UserHeadStorageService()
                    service.db = mock_db
                    # Manually override the directory for this instance
                    import app.services.user_head_storage
                    app.services.user_head_storage.USER_HEADS_DIR = tmp_path
                    return service


def test_service_initialization_file_mode(mock_settings):
    """Test service initializes correctly in file mode."""
    mock_settings.USE_MONGODB_STORAGE = False
    mock_settings.DUAL_SAVE_MODE = False
    
    with patch("app.services.user_head_storage.settings", mock_settings):
        service = UserHeadStorageService()
        
        assert service.use_mongodb is False
        assert service.dual_save is False
        assert service.compression == "gzip"
        assert service.db is None


def test_service_initialization_mongodb_mode(mock_settings):
    """Test service initializes correctly in MongoDB mode."""
    mock_settings.USE_MONGODB_STORAGE = True
    mock_settings.DUAL_SAVE_MODE = False
    
    with patch("app.services.user_head_storage.settings", mock_settings):
        service = UserHeadStorageService()
        
        assert service.use_mongodb is True
        assert service.dual_save is False


def test_save_model_file_mode(storage_service_file_mode, sample_state_dict, tmp_path):
    """Test saving model in file-only mode."""
    user_id = "test_user_123"
    metadata = {"training_samples": 50}
    
    # Patch the module-level constant during the save
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        result = storage_service_file_mode.save_model(user_id, sample_state_dict, metadata)
    
    assert result is True
    
    # Verify file was created
    file_path = tmp_path / f"{user_id}.pt"
    assert file_path.exists()
    
    # Verify file can be loaded
    loaded = torch.load(str(file_path), map_location="cpu", weights_only=True)
    assert set(loaded.keys()) == set(sample_state_dict.keys())


def test_save_model_mongodb_mode(storage_service_mongodb_mode, sample_state_dict, mock_db):
    """Test saving model in MongoDB mode."""
    user_id = "test_user_456"
    metadata = {"training_samples": 50}
    
    result = storage_service_mongodb_mode.save_model(user_id, sample_state_dict, metadata)
    
    assert result is True
    
    # Verify MongoDB update_one was called
    mock_db.user_models.update_one.assert_called_once()
    call_args = mock_db.user_models.update_one.call_args
    
    # Check filter
    assert call_args[0][0] == {"user_id": user_id}
    
    # Check upsert flag
    assert call_args[1]["upsert"] is True
    
    # Check document structure
    doc = call_args[0][1]["$set"]
    assert doc["user_id"] == user_id
    assert "model_blob" in doc
    assert doc["metadata"]["storage_mode"] == "mongodb"


def test_save_model_dual_mode(storage_service_dual_mode, sample_state_dict, mock_db, tmp_path):
    """Test saving model in dual-save mode."""
    user_id = "test_user_789"
    metadata = {"training_samples": 50}
    
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        result = storage_service_dual_mode.save_model(user_id, sample_state_dict, metadata)
    
    assert result is True
    
    # Verify MongoDB was called
    mock_db.user_models.update_one.assert_called_once()
    
    # Verify file was created
    file_path = tmp_path / f"{user_id}.pt"
    assert file_path.exists()
    
    # Check storage_mode in metadata
    call_args = mock_db.user_models.update_one.call_args
    doc = call_args[0][1]["$set"]
    assert doc["metadata"]["storage_mode"] == "dual"


def test_save_model_size_validation(storage_service_file_mode, tmp_path):
    """Test that oversized models are rejected."""
    user_id = "test_user_oversized"
    
    # Create a mock state_dict that will compress to > 1MB
    # We'll mock the compression to return a large blob
    large_blob = b"x" * (MAX_MODEL_SIZE_BYTES + 1000)
    
    with patch.object(UserModelStorage, "compress_state_dict", return_value=large_blob):
        result = storage_service_file_mode.save_model(user_id, {}, {})
        
        assert result is False
        
        # Verify file was NOT created
        file_path = tmp_path / f"{user_id}.pt"
        assert not file_path.exists()


def test_save_model_compression_metadata(storage_service_mongodb_mode, sample_state_dict, mock_db):
    """Test that compression metadata is added correctly."""
    user_id = "test_user_metadata"
    
    result = storage_service_mongodb_mode.save_model(user_id, sample_state_dict, {})
    
    assert result is True
    
    # Extract saved document
    call_args = mock_db.user_models.update_one.call_args
    doc = call_args[0][1]["$set"]
    metadata = doc["metadata"]
    
    # Verify compression metadata
    assert "model_size_bytes" in metadata
    assert "compressed_size_bytes" in metadata
    assert "checksum" in metadata
    assert "last_trained" in metadata
    assert metadata["model_size_bytes"] > metadata["compressed_size_bytes"]
    assert metadata["checksum"].startswith("sha256:")


def test_load_model_file_mode(storage_service_file_mode, sample_state_dict, tmp_path):
    """Test loading model from file."""
    user_id = "test_user_load_file"
    
    # Save model first
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        storage_service_file_mode.save_model(user_id, sample_state_dict, {})
    
    # Load model
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        loaded = storage_service_file_mode.load_model(user_id)
    
    assert loaded is not None
    assert set(loaded.keys()) == set(sample_state_dict.keys())
    
    # Verify weights match
    for key in sample_state_dict.keys():
        assert torch.allclose(loaded[key], sample_state_dict[key])


def test_load_model_mongodb_mode(storage_service_mongodb_mode, sample_state_dict, mock_db):
    """Test loading model from MongoDB."""
    user_id = "test_user_load_mongo"
    
    # Prepare mock MongoDB document
    compressed_blob = UserModelStorage.compress_state_dict(sample_state_dict)
    mock_doc = {
        "user_id": user_id,
        "model_blob": compressed_blob,
        "version": 1,
        "metadata": {}
    }
    mock_db.user_models.find_one.return_value = mock_doc
    
    # Load model
    loaded = storage_service_mongodb_mode.load_model(user_id)
    
    assert loaded is not None
    assert set(loaded.keys()) == set(sample_state_dict.keys())
    
    # Verify MongoDB was queried
    mock_db.user_models.find_one.assert_called_once_with({"user_id": user_id})


def test_load_model_fallback_to_file(storage_service_mongodb_mode, sample_state_dict, mock_db, tmp_path):
    """Test fallback to file when MongoDB fails."""
    user_id = "test_user_fallback"
    
    # MongoDB returns None (model not found)
    mock_db.user_models.find_one.return_value = None
    
    # Save to file directly
    file_path = tmp_path / f"{user_id}.pt"
    torch.save(sample_state_dict, str(file_path))
    
    # Load should fallback to file
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        loaded = storage_service_mongodb_mode.load_model(user_id)
    
    assert loaded is not None
    assert set(loaded.keys()) == set(sample_state_dict.keys())


def test_load_model_not_found(storage_service_file_mode):
    """Test loading non-existent model returns None."""
    user_id = "nonexistent_user"
    
    loaded = storage_service_file_mode.load_model(user_id)
    
    assert loaded is None


def test_load_model_mongodb_exception_fallback(storage_service_mongodb_mode, sample_state_dict, mock_db, tmp_path):
    """Test fallback to file when MongoDB raises exception."""
    user_id = "test_user_exception"
    
    # MongoDB raises exception
    mock_db.user_models.find_one.side_effect = Exception("MongoDB connection error")
    
    # Save to file
    file_path = tmp_path / f"{user_id}.pt"
    torch.save(sample_state_dict, str(file_path))
    
    # Load should fallback to file
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        loaded = storage_service_mongodb_mode.load_model(user_id)
    
    assert loaded is not None


def test_exists_file_mode(storage_service_file_mode, sample_state_dict, tmp_path):
    """Test exists() in file mode."""
    user_id = "test_user_exists_file"
    
    # Model doesn't exist yet
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        assert storage_service_file_mode.exists(user_id) is False
    
    # Save model
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        storage_service_file_mode.save_model(user_id, sample_state_dict, {})
    
    # Model exists now
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        assert storage_service_file_mode.exists(user_id) is True


def test_exists_mongodb_mode(storage_service_mongodb_mode, mock_db):
    """Test exists() in MongoDB mode."""
    user_id = "test_user_exists_mongo"
    
    # Model doesn't exist
    mock_db.user_models.count_documents.return_value = 0
    assert storage_service_mongodb_mode.exists(user_id) is False
    
    # Model exists
    mock_db.user_models.count_documents.return_value = 1
    assert storage_service_mongodb_mode.exists(user_id) is True
    
    # Verify query
    mock_db.user_models.count_documents.assert_called_with({"user_id": user_id}, limit=1)


def test_delete_model_file_mode(storage_service_file_mode, sample_state_dict, tmp_path):
    """Test deleting model in file mode."""
    user_id = "test_user_delete_file"
    
    # Save model
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        storage_service_file_mode.save_model(user_id, sample_state_dict, {})
    
    file_path = tmp_path / f"{user_id}.pt"
    assert file_path.exists()
    
    # Delete model
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        result = storage_service_file_mode.delete_model(user_id)
    
    assert result is True
    assert not file_path.exists()


def test_delete_model_mongodb_mode(storage_service_mongodb_mode, mock_db):
    """Test deleting model in MongoDB mode."""
    user_id = "test_user_delete_mongo"
    
    mock_result = Mock()
    mock_result.deleted_count = 1
    mock_db.user_models.delete_one.return_value = mock_result
    
    result = storage_service_mongodb_mode.delete_model(user_id)
    
    assert result is True
    mock_db.user_models.delete_one.assert_called_once_with({"user_id": user_id})


def test_delete_model_dual_mode(storage_service_dual_mode, sample_state_dict, mock_db, tmp_path):
    """Test deleting model in dual mode removes from both storages."""
    user_id = "test_user_delete_dual"
    
    # Save model (creates both MongoDB and file)
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        storage_service_dual_mode.save_model(user_id, sample_state_dict, {})
    
    mock_result = Mock()
    mock_result.deleted_count = 1
    mock_db.user_models.delete_one.return_value = mock_result
    
    # Delete model
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        result = storage_service_dual_mode.delete_model(user_id)
    
    assert result is True
    
    # Verify both were deleted
    mock_db.user_models.delete_one.assert_called_once()
    file_path = tmp_path / f"{user_id}.pt"
    assert not file_path.exists()


def test_get_storage_stats_file_mode(storage_service_file_mode, sample_state_dict, tmp_path):
    """Test storage stats in file mode."""
    # Save a few models
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        for i in range(3):
            storage_service_file_mode.save_model(f"user_{i}", sample_state_dict, {})
        
        stats = storage_service_file_mode.get_storage_stats()
    
    assert stats["file_models"] == 3
    assert stats["mongodb_models"] == 0
    assert stats["total_models"] == 3


def test_get_storage_stats_mongodb_mode(storage_service_mongodb_mode, mock_db):
    """Test storage stats in MongoDB mode."""
    # Mock MongoDB stats
    mock_db.user_models.count_documents.return_value = 5
    mock_db.user_models.aggregate.return_value = [{"total_size": 10240}]
    
    stats = storage_service_mongodb_mode.get_storage_stats()
    
    assert stats["mongodb_models"] == 5
    assert stats["total_size_bytes"] == 10240
    assert stats["avg_model_size_bytes"] == 10240 // 5


def test_compression_decompression_round_trip(storage_service_file_mode, sample_state_dict, tmp_path):
    """Test full save/load round trip preserves model weights."""
    user_id = "test_user_round_trip"
    metadata = {"training_samples": 50}
    
    # Save
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        save_result = storage_service_file_mode.save_model(user_id, sample_state_dict, metadata)
    assert save_result is True
    
    # Load
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        loaded = storage_service_file_mode.load_model(user_id)
    assert loaded is not None
    
    # Verify exact match
    assert set(loaded.keys()) == set(sample_state_dict.keys())
    for key in sample_state_dict.keys():
        assert torch.allclose(loaded[key], sample_state_dict[key])


def test_save_model_mongodb_failure_no_fallback(storage_service_mongodb_mode, sample_state_dict, mock_db):
    """Test that MongoDB-only mode fails when MongoDB is unavailable."""
    user_id = "test_user_mongo_fail"
    
    # MongoDB raises exception
    mock_db.user_models.update_one.side_effect = Exception("MongoDB error")
    
    result = storage_service_mongodb_mode.save_model(user_id, sample_state_dict, {})
    
    # Should fail in MongoDB-only mode
    assert result is False


def test_save_model_dual_mode_mongodb_failure(storage_service_dual_mode, sample_state_dict, mock_db, tmp_path):
    """Test that dual mode continues to file when MongoDB fails."""
    user_id = "test_user_dual_fail"
    
    # MongoDB raises exception
    mock_db.user_models.update_one.side_effect = Exception("MongoDB error")
    
    with patch("app.services.user_head_storage.USER_HEADS_DIR", tmp_path):
        result = storage_service_dual_mode.save_model(user_id, sample_state_dict, {})
    
    # Should still succeed (file save works)
    assert result is True
    
    # Verify file was created
    file_path = tmp_path / f"{user_id}.pt"
    assert file_path.exists()
