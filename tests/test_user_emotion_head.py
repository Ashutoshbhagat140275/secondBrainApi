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
        # Create a temporary user model
        user_id = "test_user_123"
        model = UserEmotionHead()
        
        # Save model to temporary directory
        temp_user_heads_dir = tmp_path / "user_heads"
        temp_user_heads_dir.mkdir()
        model_path = temp_user_heads_dir / f"{user_id}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Mock USER_HEADS_DIR to use temp directory
        with patch("app.services.user_emotion_head.USER_HEADS_DIR", temp_user_heads_dir):
            # Clear LRU cache to force reload
            load_user_head.cache_clear()
            
            loaded_model = load_user_head(user_id)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, UserEmotionHead)
            assert not loaded_model.training  # Should be in eval mode
    
    def test_load_user_head_caching(self, tmp_path):
        """Test that load_user_head uses LRU cache."""
        user_id = "test_user_cache"
        model = UserEmotionHead()
        
        # Save model
        temp_user_heads_dir = tmp_path / "user_heads"
        temp_user_heads_dir.mkdir()
        model_path = temp_user_heads_dir / f"{user_id}.pt"
        torch.save(model.state_dict(), model_path)
        
        with patch("app.services.user_emotion_head.USER_HEADS_DIR", temp_user_heads_dir):
            load_user_head.cache_clear()
            
            # First load
            model1 = load_user_head(user_id)
            # Second load (should use cache)
            model2 = load_user_head(user_id)
            
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
        user_id = "test_user_predict"
        model = UserEmotionHead()
        
        # Save model
        temp_user_heads_dir = tmp_path / "user_heads"
        temp_user_heads_dir.mkdir()
        model_path = temp_user_heads_dir / f"{user_id}.pt"
        torch.save(model.state_dict(), model_path)
        
        with patch("app.services.user_emotion_head.USER_HEADS_DIR", temp_user_heads_dir):
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
        user_id = "test_user_invalid"
        model = UserEmotionHead()
        
        # Save model
        temp_user_heads_dir = tmp_path / "user_heads"
        temp_user_heads_dir.mkdir()
        model_path = temp_user_heads_dir / f"{user_id}.pt"
        torch.save(model.state_dict(), model_path)
        
        with patch("app.services.user_emotion_head.USER_HEADS_DIR", temp_user_heads_dir):
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
