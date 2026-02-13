"""
Unit tests for user emotion head module.

Tests cover:
- UserEmotionHead model architecture and forward pass
- load_user_head() with existing and missing models
- predict_user() with valid embeddings and edge cases
- create_fresh_user_head() initialization
- LRU cache behavior
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from app.services.user_emotion_head import (
    UserEmotionHead,
    load_user_head,
    predict_user,
    create_fresh_user_head,
    USER_HEADS_DIR,
)
from app.services.feature_config import EMBEDDING_DIM, NUM_CLASSES


class TestUserEmotionHead:
    """Test UserEmotionHead model architecture."""
    
    def test_model_initialization(self):
        """Test that UserEmotionHead initializes with correct architecture."""
        model = UserEmotionHead(embedding_dim=768, num_classes=8)
        
        # Check linear layer dimensions
        assert model.linear.in_features == 768
        assert model.linear.out_features == 8
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = UserEmotionHead(embedding_dim=768, num_classes=8)
        
        # Create batch of embeddings
        batch_size = 4
        x = torch.randn(batch_size, 768)
        
        # Forward pass
        logits = model(x)
        
        # Check output shape
        assert logits.shape == (batch_size, 8)
    
    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample (batch_size=1)."""
        model = UserEmotionHead()
        x = torch.randn(1, 768)
        logits = model(x)
        
        assert logits.shape == (1, 8)


class TestLoadUserHead:
    """Test load_user_head() function."""

    
    def test_load_nonexistent_user_model(self):
        """Test loading a user model that doesn't exist returns None."""
        result = load_user_head("nonexistent_user_12345")
        assert result is None
    
    def test_load_existing_user_model(self, tmp_path):
        """Test loading an existing user model succeeds."""
        from unittest.mock import Mock
        
        user_id = "test_user_123"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service to return state_dict
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            # Clear LRU cache to force reload
            load_user_head.cache_clear()
            
            loaded_model = load_user_head(user_id)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, UserEmotionHead)
            assert not loaded_model.training  # Should be in eval mode
            
            # Verify storage service was called
            mock_storage.load_model.assert_called_once_with(user_id)
    
    def test_load_user_head_caching(self, tmp_path):
        """Test that load_user_head uses LRU cache."""
        from unittest.mock import Mock
        
        user_id = "test_user_cache"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            # First load
            model1 = load_user_head(user_id)
            assert mock_storage.load_model.call_count == 1
            
            # Second load (should use cache)
            model2 = load_user_head(user_id)
            assert mock_storage.load_model.call_count == 1  # No additional call
            
            # Should return the same cached instance
            assert model1 is model2


class TestPredictUser:
    """Test predict_user() function."""
    
    def test_predict_user_no_model(self):
        """Test prediction returns None when user has no trained model."""
        embedding = np.random.randn(768).astype(np.float32)
        result = predict_user(embedding, "nonexistent_user_99999")
        
        assert result is None

    
    def test_predict_user_with_model(self, tmp_path):
        """Test prediction returns valid probabilities when user has a model."""
        from unittest.mock import Mock
        
        user_id = "test_user_predict"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service to return state_dict
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            # Create embedding
            embedding = np.random.randn(768).astype(np.float32)
            
            # Predict
            result = predict_user(embedding, user_id)
            
            assert result is not None
            P_u, C_u = result
            
            # Check probability distribution properties
            assert P_u.shape == (8,)
            assert np.isclose(P_u.sum(), 1.0, atol=1e-5)
            assert np.all(P_u >= 0.0)
            assert np.all(P_u <= 1.0)
            
            # Check confidence
            assert 0.0 <= C_u <= 1.0
            assert C_u == np.max(P_u)
    
    def test_predict_user_invalid_embedding_shape(self, tmp_path):
        """Test prediction handles invalid embedding shape gracefully."""
        from unittest.mock import Mock
        
        user_id = "test_user_invalid"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service to return state_dict
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            # Invalid embedding shape
            embedding = np.random.randn(512).astype(np.float32)
            
            # Should return None on error
            result = predict_user(embedding, user_id)
            assert result is None


class TestCreateFreshUserHead:
    """Test create_fresh_user_head() function."""
    
    def test_create_fresh_head(self):
        """Test creating a fresh user head."""
        model = create_fresh_user_head()
        
        assert isinstance(model, UserEmotionHead)
        assert model.training  # Should be in training mode
        assert model.linear.in_features == EMBEDDING_DIM
        assert model.linear.out_features == NUM_CLASSES
    
    def test_fresh_head_forward_pass(self):
        """Test that fresh head can perform forward pass."""
        model = create_fresh_user_head()
        x = torch.randn(1, 768)
        logits = model(x)
        
        assert logits.shape == (1, 8)


class TestLoadUserHeadMongoDBSupport:
    """Test load_user_head() with MongoDB storage support."""
    
    def test_load_from_mongodb_when_enabled(self, tmp_path):
        """Test loading from MongoDB when USE_MONGODB_STORAGE=True."""
        from unittest.mock import Mock
        
        user_id = "test_user_mongodb"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service to return state_dict
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            loaded_model = load_user_head(user_id)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, UserEmotionHead)
            assert not loaded_model.training  # Should be in eval mode
            
            # Verify storage service was called
            mock_storage.load_model.assert_called_once_with(user_id)
    
    def test_load_fallback_to_file_when_mongodb_unavailable(self, tmp_path):
        """Test fallback to file when model not in MongoDB."""
        from unittest.mock import Mock
        
        user_id = "test_user_fallback"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service to simulate MongoDB → file fallback
        # (storage service returns state_dict from file)
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            loaded_model = load_user_head(user_id)
            
            assert loaded_model is not None
            mock_storage.load_model.assert_called_once_with(user_id)
    
    def test_load_returns_none_when_no_model_exists(self):
        """Test that load returns None when model doesn't exist in any storage."""
        from unittest.mock import Mock
        
        user_id = "nonexistent_user"
        
        # Mock storage service to return None (no model found)
        mock_storage = Mock()
        mock_storage.load_model.return_value = None
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            loaded_model = load_user_head(user_id)
            
            assert loaded_model is None
            mock_storage.load_model.assert_called_once_with(user_id)
    
    def test_cache_behavior_with_mongodb_storage(self):
        """Test that LRU cache works correctly with MongoDB storage."""
        from unittest.mock import Mock
        
        user_id = "test_user_cache_mongo"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            # First load - should call storage service
            model1 = load_user_head(user_id)
            assert mock_storage.load_model.call_count == 1
            
            # Second load - should use cache (no additional call)
            model2 = load_user_head(user_id)
            assert mock_storage.load_model.call_count == 1
            
            # Should return same cached instance
            assert model1 is model2
    
    def test_load_handles_storage_service_exception(self):
        """Test that load handles exceptions from storage service gracefully."""
        from unittest.mock import Mock
        
        user_id = "test_user_exception"
        
        # Mock storage service to raise exception
        mock_storage = Mock()
        mock_storage.load_model.side_effect = Exception("Storage error")
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            # Should return None on exception
            loaded_model = load_user_head(user_id)
            assert loaded_model is None


class TestInvalidateUserCache:
    """Test invalidate_user_cache() helper function."""
    
    def test_invalidate_cache_clears_lru_cache(self):
        """Test that invalidate_user_cache clears the LRU cache."""
        from unittest.mock import Mock
        from app.services.user_emotion_head import invalidate_user_cache
        
        user_id = "test_user_invalidate"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            # Load model (caches it)
            model1 = load_user_head(user_id)
            assert mock_storage.load_model.call_count == 1
            
            # Invalidate cache
            invalidate_user_cache(user_id)
            
            # Load again - should call storage service again (cache was cleared)
            model2 = load_user_head(user_id)
            assert mock_storage.load_model.call_count == 2
            
            # Models should be different instances (not cached)
            assert model1 is not model2
    
    def test_invalidate_cache_logs_correctly(self, caplog):
        """Test that invalidate_user_cache logs the invalidation."""
        from app.services.user_emotion_head import invalidate_user_cache
        import logging
        
        user_id = "test_user_log"
        
        with caplog.at_level(logging.INFO):
            invalidate_user_cache(user_id)
        
        # Check that log message was created
        assert any(user_id in record.message for record in caplog.records)
        assert any("Invalidated" in record.message for record in caplog.records)



class TestDualStorageLoadingIntegration:
    """Integration tests for dual-storage loading (MongoDB + file)."""
    
    def test_load_with_model_in_both_storages_mongodb_priority(self, tmp_path):
        """Test that MongoDB has priority when model exists in both storages."""
        from unittest.mock import Mock
        
        user_id = "test_user_both"
        
        # Create two different models to distinguish sources
        model_mongodb = UserEmotionHead()
        model_file = UserEmotionHead()
        
        # Make them different by modifying weights
        with torch.no_grad():
            model_mongodb.linear.weight.fill_(1.0)
            model_file.linear.weight.fill_(2.0)
        
        state_dict_mongodb = model_mongodb.state_dict()
        state_dict_file = model_file.state_dict()
        
        # Mock storage service to return MongoDB state_dict
        # (simulating MongoDB priority)
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict_mongodb
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            loaded_model = load_user_head(user_id)
            
            assert loaded_model is not None
            
            # Verify it loaded from MongoDB (weights should be 1.0, not 2.0)
            assert torch.allclose(
                loaded_model.linear.weight,
                torch.ones_like(loaded_model.linear.weight)
            )
    
    def test_load_with_model_only_in_file(self, tmp_path):
        """Test fallback to file when model only exists in file storage."""
        from unittest.mock import Mock
        
        user_id = "test_user_file_only"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service to simulate file-only scenario
        # (storage service returns state_dict from file)
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            loaded_model = load_user_head(user_id)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, UserEmotionHead)
            
            # Verify storage service was called
            mock_storage.load_model.assert_called_once_with(user_id)
    
    def test_load_with_model_only_in_mongodb(self, tmp_path):
        """Test loading when model only exists in MongoDB."""
        from unittest.mock import Mock
        
        user_id = "test_user_mongodb_only"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service to return MongoDB state_dict
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            loaded_model = load_user_head(user_id)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, UserEmotionHead)
            
            # Verify storage service was called
            mock_storage.load_model.assert_called_once_with(user_id)
    
    def test_cache_warming_from_mongodb(self):
        """Test that cache is populated correctly from MongoDB storage."""
        from unittest.mock import Mock
        
        user_id = "test_user_cache_warm"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            # First load - warms cache
            model1 = load_user_head(user_id)
            assert mock_storage.load_model.call_count == 1
            
            # Second load - uses cache
            model2 = load_user_head(user_id)
            assert mock_storage.load_model.call_count == 1  # No additional call
            
            # Third load - still uses cache
            model3 = load_user_head(user_id)
            assert mock_storage.load_model.call_count == 1
            
            # All should be the same cached instance
            assert model1 is model2 is model3
    
    def test_end_to_end_prediction_with_mongodb_storage(self):
        """Test complete prediction flow with MongoDB storage."""
        from unittest.mock import Mock
        
        user_id = "test_user_e2e"
        model = UserEmotionHead()
        state_dict = model.state_dict()
        
        # Mock storage service
        mock_storage = Mock()
        mock_storage.load_model.return_value = state_dict
        
        with patch("app.services.user_emotion_head.storage_service", mock_storage):
            load_user_head.cache_clear()
            
            # Create embedding
            embedding = np.random.randn(768).astype(np.float32)
            
            # Predict (internally calls load_user_head)
            result = predict_user(embedding, user_id)
            
            assert result is not None
            P_u, C_u = result
            
            # Verify prediction properties
            assert P_u.shape == (8,)
            assert np.isclose(P_u.sum(), 1.0, atol=1e-5)
            assert 0.0 <= C_u <= 1.0
            
            # Verify storage service was called
            mock_storage.load_model.assert_called_once_with(user_id)
            
            # Second prediction should use cache
            result2 = predict_user(embedding, user_id)
            assert result2 is not None
            assert mock_storage.load_model.call_count == 1  # Still only 1 call
