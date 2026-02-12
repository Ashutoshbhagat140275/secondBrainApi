"""
Tests for model format detection in emotion_analyzer.

Verifies that the system correctly detects which model artifacts are available
and routes to the appropriate pipeline (neural embeddings or manual features).

Run from the api/ directory:
    python -m pytest tests/test_model_format_detection.py -v
"""

import sys
import pathlib
import pytest
from unittest.mock import patch, MagicMock

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.emotion_analyzer import (
    detect_model_format,
    ModelFormat,
    get_active_model_format,
    load_emotion_model,
)


class TestModelFormatDetection:
    """Test model format detection logic."""
    
    def test_detect_neural_embedding_model(self):
        """Test detection when Wav2Vec2 model artifacts exist."""
        with patch('app.services.emotion_analyzer.Path') as mock_path:
            # Mock both embedding model and scaler exist
            mock_embedding_model = MagicMock()
            mock_embedding_model.exists.return_value = True
            mock_embedding_scaler = MagicMock()
            mock_embedding_scaler.exists.return_value = True
            
            # Mock legacy model doesn't exist (shouldn't matter due to priority)
            mock_legacy_model = MagicMock()
            mock_legacy_model.exists.return_value = False
            mock_legacy_scaler = MagicMock()
            mock_legacy_scaler.exists.return_value = False
            
            # Return mocks in order of Path() calls
            mock_path.side_effect = [
                mock_embedding_model,
                mock_embedding_scaler,
                mock_legacy_model,
                mock_legacy_scaler,
            ]
            
            result = detect_model_format()
            assert result == ModelFormat.NEURAL_EMBEDDING
    
    def test_detect_legacy_manual_model(self):
        """Test detection when only legacy model artifacts exist."""
        with patch('app.services.emotion_analyzer.Path') as mock_path:
            # Mock embedding model doesn't exist
            mock_embedding_model = MagicMock()
            mock_embedding_model.exists.return_value = False
            mock_embedding_scaler = MagicMock()
            mock_embedding_scaler.exists.return_value = False
            
            # Mock legacy model exists
            mock_legacy_model = MagicMock()
            mock_legacy_model.exists.return_value = True
            mock_legacy_scaler = MagicMock()
            mock_legacy_scaler.exists.return_value = True
            
            # Return mocks in order of Path() calls
            mock_path.side_effect = [
                mock_embedding_model,
                mock_embedding_scaler,
                mock_legacy_model,
                mock_legacy_scaler,
            ]
            
            result = detect_model_format()
            assert result == ModelFormat.LEGACY_MANUAL
    
    def test_detect_no_model_available(self):
        """Test detection when no model artifacts exist."""
        with patch('app.services.emotion_analyzer.Path') as mock_path:
            # Mock all models don't exist
            mock_model = MagicMock()
            mock_model.exists.return_value = False
            
            mock_path.return_value = mock_model
            
            result = detect_model_format()
            assert result == ModelFormat.UNAVAILABLE
    
    def test_neural_embedding_priority_over_legacy(self):
        """Test that neural embedding model takes priority when both exist."""
        with patch('app.services.emotion_analyzer.Path') as mock_path:
            # Mock both models exist
            mock_model = MagicMock()
            mock_model.exists.return_value = True
            
            mock_path.return_value = mock_model
            
            result = detect_model_format()
            # Should prefer neural embedding
            assert result == ModelFormat.NEURAL_EMBEDDING
    
    def test_incomplete_neural_embedding_artifacts(self):
        """Test detection when only one neural embedding artifact exists."""
        with patch('app.services.emotion_analyzer.Path') as mock_path:
            # Mock only embedding model exists, not scaler
            mock_embedding_model = MagicMock()
            mock_embedding_model.exists.return_value = True
            mock_embedding_scaler = MagicMock()
            mock_embedding_scaler.exists.return_value = False
            
            # Mock legacy model exists
            mock_legacy_model = MagicMock()
            mock_legacy_model.exists.return_value = True
            mock_legacy_scaler = MagicMock()
            mock_legacy_scaler.exists.return_value = True
            
            # Return mocks in order of Path() calls
            mock_path.side_effect = [
                mock_embedding_model,
                mock_embedding_scaler,
                mock_legacy_model,
                mock_legacy_scaler,
            ]
            
            result = detect_model_format()
            # Should fall back to legacy since neural embedding is incomplete
            assert result == ModelFormat.LEGACY_MANUAL
    
    def test_incomplete_legacy_artifacts(self):
        """Test detection when only one legacy artifact exists."""
        with patch('app.services.emotion_analyzer.Path') as mock_path:
            # Mock no embedding model
            mock_embedding_model = MagicMock()
            mock_embedding_model.exists.return_value = False
            mock_embedding_scaler = MagicMock()
            mock_embedding_scaler.exists.return_value = False
            
            # Mock only legacy model exists, not scaler
            mock_legacy_model = MagicMock()
            mock_legacy_model.exists.return_value = True
            mock_legacy_scaler = MagicMock()
            mock_legacy_scaler.exists.return_value = False
            
            # Return mocks in order of Path() calls
            mock_path.side_effect = [
                mock_embedding_model,
                mock_embedding_scaler,
                mock_legacy_model,
                mock_legacy_scaler,
            ]
            
            result = detect_model_format()
            # Should return unavailable since legacy is incomplete
            assert result == ModelFormat.UNAVAILABLE


class TestModelFormatLogging:
    """Test that appropriate log messages are generated."""
    
    def test_neural_embedding_detection_logs(self, caplog):
        """Test that neural embedding detection logs appropriate message."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch('app.services.emotion_analyzer.Path') as mock_path:
            mock_model = MagicMock()
            mock_model.exists.return_value = True
            mock_path.return_value = mock_model
            
            detect_model_format()
            
            assert "Wav2Vec2 neural embedding model (768-dim)" in caplog.text
            assert "Using neural embeddings" in caplog.text
    
    def test_legacy_detection_logs(self, caplog):
        """Test that legacy model detection logs appropriate message."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch('app.services.emotion_analyzer.Path') as mock_path:
            # Mock embedding doesn't exist, legacy does
            mock_embedding = MagicMock()
            mock_embedding.exists.return_value = False
            mock_legacy = MagicMock()
            mock_legacy.exists.return_value = True
            
            mock_path.side_effect = [
                mock_embedding,
                mock_embedding,
                mock_legacy,
                mock_legacy,
            ]
            
            detect_model_format()
            
            assert "legacy manual feature model (419-dim)" in caplog.text
            assert "Using manual feature extraction" in caplog.text
    
    def test_unavailable_detection_logs(self, caplog):
        """Test that unavailable model logs warning."""
        import logging
        caplog.set_level(logging.WARNING)
        
        with patch('app.services.emotion_analyzer.Path') as mock_path:
            mock_model = MagicMock()
            mock_model.exists.return_value = False
            mock_path.return_value = mock_model
            
            detect_model_format()
            
            assert "No emotion model artifacts found" in caplog.text
            assert "fallback predictions" in caplog.text
