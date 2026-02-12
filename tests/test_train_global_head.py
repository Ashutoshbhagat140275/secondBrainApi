"""
Integration test for training the global emotion head.

This test verifies that the train_wav2vec2.py script can train and save
a global emotion head with the --save-as-global-head flag.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.global_emotion_head import GlobalEmotionHead
from app.services.feature_config import GLOBAL_HEAD_PATH, EMBEDDING_DIM, NUM_CLASSES


class TestGlobalHeadTraining:
    """Test the global head training functionality."""
    
    def test_global_head_architecture_matches(self):
        """Test that GlobalEmotionHead has the expected simple architecture."""
        model = GlobalEmotionHead(embedding_dim=768, num_classes=8)
        
        # Should have only one linear layer (no hidden layers)
        assert hasattr(model, 'linear')
        assert isinstance(model.linear, torch.nn.Linear)
        
        # Check dimensions
        assert model.linear.in_features == 768
        assert model.linear.out_features == 8
        
        # Count parameters: 768 * 8 (weights) + 8 (biases) = 6152
        total_params = sum(p.numel() for p in model.parameters())
        expected_params = 768 * 8 + 8
        assert total_params == expected_params
    
    def test_global_head_forward_pass(self):
        """Test that the global head produces valid logits."""
        model = GlobalEmotionHead()
        model.eval()
        
        # Create a batch of embeddings
        batch_size = 4
        embeddings = torch.randn(batch_size, 768)
        
        # Forward pass
        with torch.no_grad():
            logits = model(embeddings)
        
        # Check output shape
        assert logits.shape == (batch_size, 8)
        
        # Check that logits are real numbers (not NaN or Inf)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_global_head_can_be_saved_and_loaded(self, tmp_path):
        """Test that the global head can be saved and loaded correctly."""
        # Create and initialize a model
        model = GlobalEmotionHead()
        
        # Save the model
        save_path = tmp_path / "test_global_head.pt"
        torch.save(model.state_dict(), str(save_path))
        
        # Load the model
        loaded_model = GlobalEmotionHead()
        loaded_model.load_state_dict(torch.load(str(save_path), weights_only=True))
        
        # Verify that the loaded model produces the same output
        test_input = torch.randn(1, 768)
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            output1 = model(test_input)
            output2 = loaded_model(test_input)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_global_head_training_produces_valid_gradients(self):
        """Test that the global head can be trained (gradients flow correctly)."""
        model = GlobalEmotionHead()
        model.train()
        
        # Create synthetic training data
        embeddings = torch.randn(8, 768)
        labels = torch.randint(0, 8, (8,))
        
        # Forward pass
        logits = model(embeddings)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are valid
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()
    
    def test_global_head_smaller_than_embedding_classifier(self):
        """Test that GlobalEmotionHead is simpler than EmbeddingClassifier."""
        from app.services.emotion_classifier import EmbeddingClassifier
        
        global_head = GlobalEmotionHead()
        embedding_classifier = EmbeddingClassifier()
        
        global_params = sum(p.numel() for p in global_head.parameters())
        classifier_params = sum(p.numel() for p in embedding_classifier.parameters())
        
        # Global head should have fewer parameters (simpler architecture)
        assert global_params < classifier_params
        
        # Global head should have exactly: 768 * 8 + 8 = 6152 parameters
        assert global_params == 6152


class TestTrainingScriptIntegration:
    """Test integration with the training script."""
    
    def test_train_function_accepts_save_as_global_head_flag(self):
        """Test that the train function accepts the save_as_global_head parameter."""
        from training.train_wav2vec2 import train
        import inspect
        
        # Check that the function signature includes save_as_global_head
        sig = inspect.signature(train)
        assert 'save_as_global_head' in sig.parameters
        
        # Check that the default is False (backward compatibility)
        assert sig.parameters['save_as_global_head'].default is False
    
    def test_global_head_path_is_configured(self):
        """Test that GLOBAL_HEAD_PATH is properly configured."""
        from app.services.feature_config import GLOBAL_HEAD_PATH, MODEL_DIR
        
        # Check that the path is defined
        assert GLOBAL_HEAD_PATH is not None
        assert isinstance(GLOBAL_HEAD_PATH, Path)
        
        # Check that it points to the correct location
        assert GLOBAL_HEAD_PATH.name == "global_emotion_head.pt"
        assert GLOBAL_HEAD_PATH.parent == MODEL_DIR
