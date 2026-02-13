"""
Unit tests for UserModelStorage MongoDB model.

Tests cover:
- Model creation with all fields
- to_dict() and from_dict() serialization
- Blob compression and decompression
- Size validation
- Checksum computation
"""

import pytest
import torch
import torch.nn as nn
from datetime import datetime
from bson import ObjectId, Binary

from app.models.user_model_storage import UserModelStorage


class SimpleModel(nn.Module):
    """Simple test model for state_dict testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 8)


@pytest.fixture
def sample_state_dict():
    """Create a sample PyTorch state_dict."""
    model = SimpleModel()
    return model.state_dict()


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return {
        "training_samples": 50,
        "model_size_bytes": 5120,
        "compressed_size_bytes": 1024,
        "storage_mode": "mongodb",
        "checksum": "sha256:abc123"
    }


def test_model_creation_with_all_fields(sample_metadata):
    """Test creating UserModelStorage with all fields."""
    user_id = "test_user_123"
    model_blob = b"compressed_data"
    created_at = datetime(2026, 1, 1, 12, 0, 0)
    updated_at = datetime(2026, 1, 2, 12, 0, 0)
    
    storage = UserModelStorage(
        user_id=user_id,
        model_blob=model_blob,
        version=1,
        created_at=created_at,
        updated_at=updated_at,
        metadata=sample_metadata
    )
    
    assert storage.user_id == user_id
    assert storage.model_blob == model_blob
    assert storage.version == 1
    assert storage.created_at == created_at
    assert storage.updated_at == updated_at
    assert storage.metadata == sample_metadata
    assert isinstance(storage._id, ObjectId)


def test_model_creation_with_defaults():
    """Test creating UserModelStorage with default values."""
    user_id = "test_user_456"
    model_blob = b"compressed_data"
    
    storage = UserModelStorage(
        user_id=user_id,
        model_blob=model_blob
    )
    
    assert storage.user_id == user_id
    assert storage.model_blob == model_blob
    assert storage.version == 1
    assert isinstance(storage.created_at, datetime)
    assert isinstance(storage.updated_at, datetime)
    assert storage.metadata == {}
    assert isinstance(storage._id, ObjectId)


def test_compress_state_dict(sample_state_dict):
    """Test state_dict compression."""
    compressed = UserModelStorage.compress_state_dict(sample_state_dict)
    
    assert isinstance(compressed, bytes)
    assert len(compressed) > 0
    
    # Compressed should be smaller than uncompressed
    # (for typical model weights, compression ratio is ~80%)
    import io
    buffer = io.BytesIO()
    torch.save(sample_state_dict, buffer)
    uncompressed_size = len(buffer.getvalue())
    
    assert len(compressed) < uncompressed_size


def test_decompress_state_dict(sample_state_dict):
    """Test state_dict decompression."""
    # Compress then decompress
    compressed = UserModelStorage.compress_state_dict(sample_state_dict)
    decompressed = UserModelStorage.decompress_state_dict(compressed)
    
    assert isinstance(decompressed, dict)
    assert set(decompressed.keys()) == set(sample_state_dict.keys())
    
    # Verify weights match
    for key in sample_state_dict.keys():
        assert torch.allclose(decompressed[key], sample_state_dict[key])


def test_compression_decompression_round_trip(sample_state_dict):
    """Test full compression/decompression round trip."""
    # Original state_dict
    original = sample_state_dict
    
    # Compress
    compressed = UserModelStorage.compress_state_dict(original)
    
    # Decompress
    restored = UserModelStorage.decompress_state_dict(compressed)
    
    # Verify exact match
    assert set(restored.keys()) == set(original.keys())
    for key in original.keys():
        assert torch.allclose(restored[key], original[key])


def test_compute_checksum():
    """Test checksum computation."""
    data = b"test_data_123"
    checksum = UserModelStorage.compute_checksum(data)
    
    assert isinstance(checksum, str)
    assert checksum.startswith("sha256:")
    assert len(checksum) == 71  # "sha256:" (7) + 64 hex chars
    
    # Same data should produce same checksum
    checksum2 = UserModelStorage.compute_checksum(data)
    assert checksum == checksum2
    
    # Different data should produce different checksum
    checksum3 = UserModelStorage.compute_checksum(b"different_data")
    assert checksum != checksum3


def test_to_dict(sample_metadata):
    """Test conversion to MongoDB document."""
    user_id = "test_user_789"
    model_blob = b"compressed_data"
    created_at = datetime(2026, 1, 1, 12, 0, 0)
    updated_at = datetime(2026, 1, 2, 12, 0, 0)
    
    storage = UserModelStorage(
        user_id=user_id,
        model_blob=model_blob,
        version=1,
        created_at=created_at,
        updated_at=updated_at,
        metadata=sample_metadata
    )
    
    doc = storage.to_dict()
    
    assert isinstance(doc, dict)
    assert doc["user_id"] == user_id
    assert isinstance(doc["model_blob"], Binary)
    assert bytes(doc["model_blob"]) == model_blob
    assert doc["version"] == 1
    assert doc["created_at"] == created_at
    assert doc["updated_at"] == updated_at
    assert doc["metadata"] == sample_metadata
    assert doc["_id"] == storage._id


def test_from_dict(sample_metadata):
    """Test creation from MongoDB document."""
    user_id = "test_user_101"
    model_blob = b"compressed_data"
    created_at = datetime(2026, 1, 1, 12, 0, 0)
    updated_at = datetime(2026, 1, 2, 12, 0, 0)
    obj_id = ObjectId()
    
    doc = {
        "_id": obj_id,
        "user_id": user_id,
        "model_blob": Binary(model_blob),
        "version": 1,
        "created_at": created_at,
        "updated_at": updated_at,
        "metadata": sample_metadata
    }
    
    storage = UserModelStorage.from_dict(doc)
    
    assert storage._id == obj_id
    assert storage.user_id == user_id
    assert storage.model_blob == model_blob
    assert storage.version == 1
    assert storage.created_at == created_at
    assert storage.updated_at == updated_at
    assert storage.metadata == sample_metadata


def test_to_dict_from_dict_round_trip(sample_metadata):
    """Test full serialization round trip."""
    original = UserModelStorage(
        user_id="test_user_202",
        model_blob=b"compressed_data",
        version=1,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 2, 12, 0, 0),
        metadata=sample_metadata
    )
    
    # Convert to dict
    doc = original.to_dict()
    
    # Convert back
    restored = UserModelStorage.from_dict(doc)
    
    # Verify match
    assert restored._id == original._id
    assert restored.user_id == original.user_id
    assert restored.model_blob == original.model_blob
    assert restored.version == original.version
    assert restored.created_at == original.created_at
    assert restored.updated_at == original.updated_at
    assert restored.metadata == original.metadata


def test_size_validation_large_model():
    """Test that large models can be compressed and stored."""
    # Create a larger model (still under 1MB compressed)
    class LargerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(768, 256)
            self.linear2 = nn.Linear(256, 8)
    
    model = LargerModel()
    state_dict = model.state_dict()
    
    # Should compress successfully
    compressed = UserModelStorage.compress_state_dict(state_dict)
    
    assert isinstance(compressed, bytes)
    assert len(compressed) > 0
    
    # Should decompress successfully
    decompressed = UserModelStorage.decompress_state_dict(compressed)
    assert set(decompressed.keys()) == set(state_dict.keys())


def test_empty_metadata():
    """Test model with empty metadata."""
    storage = UserModelStorage(
        user_id="test_user_303",
        model_blob=b"data"
    )
    
    assert storage.metadata == {}
    
    doc = storage.to_dict()
    assert doc["metadata"] == {}
    
    restored = UserModelStorage.from_dict(doc)
    assert restored.metadata == {}


def test_from_dict_with_missing_optional_fields():
    """Test from_dict with missing optional fields uses defaults."""
    doc = {
        "user_id": "test_user_404",
        "model_blob": Binary(b"data")
    }
    
    storage = UserModelStorage.from_dict(doc)
    
    assert storage.user_id == "test_user_404"
    assert storage.model_blob == b"data"
    assert storage.version == 1  # Default
    # When created_at/updated_at are missing from doc, they get None from .get()
    # but __init__ applies datetime.utcnow() as default, so they won't be None
    assert storage.created_at is not None
    assert storage.updated_at is not None
    assert isinstance(storage.created_at, datetime)
    assert isinstance(storage.updated_at, datetime)
    assert storage.metadata == {}
    assert storage._id is not None  # ObjectId() is generated
