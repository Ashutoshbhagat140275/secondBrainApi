"""
Unit tests for the global emotion head module.

Tests cover:
- GlobalEmotionHead model architecture
- Model loading and caching
- Inference with valid embeddings
- Fallback behavior when model is missing
- Edge cases and error handling
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.global_emotion_head import (
    GlobalEmotionHead,
    load_global_head,
    predict_global,
    GLOBAL_HEAD_PATH,
)
from app.services.feature_config import EMBEDDING_DIM, NUM_CLASSES


class TestGlobalEmotionHead:
    """Test the GlobalEmotionHead neural network architecture."""
    
    def test_model_initialization(self):
        """Test that the model initializes with correct architecture."""
        model = GlobalEmotionHead(embedding_dim=768, num_classes=8)
        
        # Check that the model has a linear layer
        assert hasattr(model, 'linear')
        assert isinstance(model.linear, torch.nn.Linear)
        
        # Check input and output dimensions
        assert model.linear.in_features == 768
        assert model.linear.out_features == 8
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = GlobalEmotionHead(embedding_dim=768, num_classes=8)
        
        # Single sample
        x = torch.randn(1, 768)
        logits = model(x)
        assert logits.shape == (1, 8)
        
        # Batch of samples
        x_batch = torch.randn(16, 768)
        logits_batch = model(x_batch)
        assert logits_batch.shape == (16, 8)
    
    def test_forward_pass_dtype(self):
        """Test that forward pass preserves float32 dtype."""
        model = GlobalEmotionHead()
        x = torch.randn(1, 768, dtype=torch.float32)
        logits = model(x)
        assert logits.dtype == torch.float32


class TestLoadGlobalHead:
    """Test the load_global_head function."""
    
    def test_load_missing_model(self, caplog):
        """Test that loading fails gracefully when model file is missing."""
        # Reset the global model cache
        import app.services.global_emotion_head as module
        module._global_head_model = None
        
        # Mock the path to point to a non-existent file
        with patch('app.services.global_emotion_head.GLOBAL_HEAD_PATH', Path('/nonexistent/model.pt')):
            result = load_global_head()
            
            assert result is False
            assert module._global_head_model is None
            assert "Global emotion head not found" in caplog.text
    
    def test_load_valid_model(self):
        """Test that a valid model loads successfully."""
        # Reset the global model cache
        import app.services.global_emotion_head as module
        module._global_head_model = None
        
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model = GlobalEmotionHead()
            torch.save(model.state_dict(), tmp.name)
            tmp_path = Path(tmp.name)
        
        try:
            # Mock the path to point to our temporary file
            with patch('app.services.global_emotion_head.GLOBAL_HEAD_PATH', tmp_path):
                result = load_global_head()
                
                assert result is True
                assert module._global_head_model is not None
                assert isinstance(module._global_head_model, GlobalEmotionHead)
        finally:
            # Clean up
            tmp_path.unlink()
            module._global_head_model = None
    
    def test_load_corrupted_model(self, caplog):
        """Test that loading fails gracefully with corrupted model file."""
        # Reset the global model cache
        import app.services.global_emotion_head as module
        module._global_head_model = None
        
        # Create a corrupted model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp.write(b'corrupted data')
            tmp_path = Path(tmp.name)
        
        try:
            with patch('app.services.global_emotion_head.GLOBAL_HEAD_PATH', tmp_path):
                result = load_global_head()
                
                assert result is False
                assert module._global_head_model is None
                assert "Failed to load global emotion head" in caplog.text
        finally:
            tmp_path.unlink()
            module._global_head_model = None


class TestPredictGlobal:
    """Test the predict_global inference function."""
    
    def test_predict_with_valid_embedding(self):
        """Test prediction with a valid 768-dim embedding."""
        # Reset and load a fresh model
        import app.services.global_emotion_head as module
        module._global_head_model = GlobalEmotionHead()
        module._global_head_model.eval()
        
        # Create a random embedding
        embedding = np.random.randn(768).astype(np.float32)
        
        # Predict
        P_g, C_g = predict_global(embedding)
        
        # Check output shapes and types
        assert P_g.shape == (8,)
        assert isinstance(C_g, float)
        
        # Check probability distribution properties
        assert np.allclose(P_g.sum(), 1.0, atol=1e-5)
        assert np.all(P_g >= 0.0)
        assert np.all(P_g <= 1.0)
        
        # Check confidence is the max probability
        assert np.isclose(C_g, np.max(P_g))
        assert 0.0 <= C_g <= 1.0
        
        # Clean up
        module._global_head_model = None
    
    def test_predict_without_loaded_model(self, caplog):
        """Test that prediction falls back gracefully when model is not loaded."""
        # Reset the global model cache
        import app.services.global_emotion_head as module
        module._global_head_model = None
        
        # Mock load_global_head to fail
        with patch('app.services.global_emotion_head.load_global_head', return_value=False):
            embedding = np.random.randn(768).astype(np.float32)
            P_g, C_g = predict_global(embedding)
            
            # Should return uniform distribution
            expected_prob = 1.0 / NUM_CLASSES
            assert np.allclose(P_g, expected_prob)
            assert np.isclose(C_g, expected_prob)
            assert "Global head unavailable" in caplog.text
    
    def test_predict_with_wrong_shape(self, caplog):
        """Test that prediction handles incorrect embedding shape."""
        # Reset and load a fresh model
        import app.services.global_emotion_head as module
        module._global_head_model = GlobalEmotionHead()
        module._global_head_model.eval()
        
        # Create embedding with wrong shape
        wrong_embedding = np.random.randn(512).astype(np.float32)
        
        # Should fall back to uniform distribution
        P_g, C_g = predict_global(wrong_embedding)
        
        expected_prob = 1.0 / NUM_CLASSES
        assert np.allclose(P_g, expected_prob)
        assert np.isclose(C_g, expected_prob)
        assert "Global head prediction failed" in caplog.text
        
        # Clean up
        module._global_head_model = None
    
    def test_predict_deterministic(self):
        """Test that predictions are deterministic for the same input."""
        # Reset and load a fresh model
        import app.services.global_emotion_head as module
        module._global_head_model = GlobalEmotionHead()
        module._global_head_model.eval()
        
        # Create a fixed embedding
        embedding = np.random.RandomState(42).randn(768).astype(np.float32)
        
        # Predict twice
        P_g1, C_g1 = predict_global(embedding)
        P_g2, C_g2 = predict_global(embedding)
        
        # Results should be identical
        assert np.allclose(P_g1, P_g2)
        assert np.isclose(C_g1, C_g2)
        
        # Clean up
        module._global_head_model = None
    
    def test_predict_confidence_range(self):
        """Test that confidence values are always in valid range."""
        # Reset and load a fresh model
        import app.services.global_emotion_head as module
        module._global_head_model = GlobalEmotionHead()
        module._global_head_model.eval()
        
        # Test with multiple random embeddings
        for _ in range(10):
            embedding = np.random.randn(768).astype(np.float32)
            P_g, C_g = predict_global(embedding)
            
            assert 0.0 <= C_g <= 1.0
            assert C_g == np.max(P_g)
        
        # Clean up
        module._global_head_model = None


class TestModelCaching:
    """Test the singleton pattern for model caching."""
    
    def test_model_loaded_once(self):
        """Test that the model is loaded only once and cached."""
        import app.services.global_emotion_head as module
        module._global_head_model = None
        
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model = GlobalEmotionHead()
            torch.save(model.state_dict(), tmp.name)
            tmp_path = Path(tmp.name)
        
        try:
            with patch('app.services.global_emotion_head.GLOBAL_HEAD_PATH', tmp_path):
                # First load
                result1 = load_global_head()
                model1 = module._global_head_model
                
                # Second load (should use cached model)
                result2 = load_global_head()
                model2 = module._global_head_model
                
                assert result1 is True
                assert result2 is True
                # Should be the same object (cached)
                assert model1 is model2
        finally:
            tmp_path.unlink()
            module._global_head_model = None
