"""
Tests for model preloading during application startup.

Verifies that the startup event correctly:
- Detects active model format (dual-head, global-only, legacy, unavailable)
- Preloads global head eagerly when available
- Logs appropriate messages for operators
- Falls back gracefully when models are missing

Run from the api/ directory:
    python -m pytest tests/test_startup_model_preloading.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestModelFormatDetection:
    """Test the detect_active_model_format() helper function."""
    
    def test_detect_dual_head_format(self):
        """Should detect dual-head when global head + user heads exist."""
        with patch("app.main.GLOBAL_HEAD_PATH") as mock_global, \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_global.exists.return_value = True
            mock_user_dir.exists.return_value = True
            mock_user_dir.glob.return_value = [Path("user1.pt"), Path("user2.pt")]
            
            from app.main import detect_active_model_format
            result = detect_active_model_format()
            
            assert result == "dual-head"
    
    def test_detect_global_only_format(self):
        """Should detect global-only when only global head exists."""
        with patch("app.main.GLOBAL_HEAD_PATH") as mock_global, \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_global.exists.return_value = True
            mock_user_dir.exists.return_value = True
            mock_user_dir.glob.return_value = []  # No user heads
            
            from app.main import detect_active_model_format
            result = detect_active_model_format()
            
            assert result == "global-only"
    
    def test_detect_legacy_format(self):
        """Should detect legacy when only embedding_classifier.pt exists."""
        with patch("app.main.GLOBAL_HEAD_PATH") as mock_global, \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir, \
             patch("app.services.feature_config.EMBEDDING_CLASSIFIER_PATH") as mock_legacy:
            
            mock_global.exists.return_value = False
            mock_user_dir.exists.return_value = False
            mock_legacy.exists.return_value = True
            
            from app.main import detect_active_model_format
            result = detect_active_model_format()
            
            assert result == "legacy"
    
    def test_detect_unavailable_format(self):
        """Should detect unavailable when no models exist."""
        with patch("app.main.GLOBAL_HEAD_PATH") as mock_global, \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir, \
             patch("app.services.feature_config.EMBEDDING_CLASSIFIER_PATH") as mock_legacy:
            
            mock_global.exists.return_value = False
            mock_user_dir.exists.return_value = False
            mock_legacy.exists.return_value = False
            
            from app.main import detect_active_model_format
            result = detect_active_model_format()
            
            assert result == "unavailable"
    
    def test_user_heads_dir_not_exists(self):
        """Should handle case where user_heads directory doesn't exist yet."""
        with patch("app.main.GLOBAL_HEAD_PATH") as mock_global, \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_global.exists.return_value = True
            mock_user_dir.exists.return_value = False  # Directory not created yet
            
            from app.main import detect_active_model_format
            result = detect_active_model_format()
            
            assert result == "global-only"


class TestStartupModelPreloading:
    """Test model preloading behavior during startup event."""
    
    @pytest.mark.asyncio
    async def test_preload_global_head_in_global_only_mode(self, caplog):
        """Should eagerly load global head when in global-only mode."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.load_global_head") as mock_load_global, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"), \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_detect.return_value = "global-only"
            mock_load_global.return_value = True
            mock_user_dir.glob.return_value = []
            
            from app.main import startup_event
            await startup_event()
            
            # Verify global head was loaded
            mock_load_global.assert_called_once()
            
            # Verify appropriate log messages
            assert "Emotion model format: global-only" in caplog.text
            assert "Global emotion head preloaded successfully" in caplog.text
            assert "Global-only mode active" in caplog.text
    
    @pytest.mark.asyncio
    async def test_preload_global_head_in_dual_head_mode(self, caplog):
        """Should eagerly load global head and log user head count in dual-head mode."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.load_global_head") as mock_load_global, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"), \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_detect.return_value = "dual-head"
            mock_load_global.return_value = True
            mock_user_dir.glob.return_value = [
                Path("user1.pt"),
                Path("user2.pt"),
                Path("user3.pt")
            ]
            
            from app.main import startup_event
            await startup_event()
            
            # Verify global head was loaded
            mock_load_global.assert_called_once()
            
            # Verify appropriate log messages
            assert "Emotion model format: dual-head" in caplog.text
            assert "Global emotion head preloaded successfully" in caplog.text
            assert "Dual-head mode active: 3 user head(s) available" in caplog.text
            assert "lazy-loaded on first request per user" in caplog.text
    
    @pytest.mark.asyncio
    async def test_fallback_to_legacy_when_global_head_missing(self, caplog):
        """Should fall back to legacy model when global head fails to load."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.load_global_head") as mock_load_global, \
             patch("app.main.load_emotion_model") as mock_load_legacy, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"), \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_detect.return_value = "global-only"
            mock_load_global.return_value = False  # Global head not available
            mock_load_legacy.return_value = True
            mock_user_dir.glob.return_value = []
            
            from app.main import startup_event
            await startup_event()
            
            # Verify fallback to legacy
            mock_load_global.assert_called_once()
            mock_load_legacy.assert_called_once()
            
            assert "Global emotion head not available" in caplog.text
            assert "Legacy emotion model loaded successfully" in caplog.text
    
    @pytest.mark.asyncio
    async def test_legacy_mode_loads_legacy_model(self, caplog):
        """Should load legacy model when in legacy mode."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.load_emotion_model") as mock_load_legacy, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"):
            
            mock_detect.return_value = "legacy"
            mock_load_legacy.return_value = True
            
            from app.main import startup_event
            await startup_event()
            
            # Verify legacy model was loaded
            mock_load_legacy.assert_called_once()
            
            assert "Emotion model format: legacy" in caplog.text
            assert "Legacy emotion model loaded successfully" in caplog.text
    
    @pytest.mark.asyncio
    async def test_unavailable_mode_logs_warning(self, caplog):
        """Should log warning when no models are available."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"):
            
            mock_detect.return_value = "unavailable"
            
            from app.main import startup_event
            await startup_event()
            
            assert "Emotion model format: unavailable" in caplog.text
            assert "No emotion model artifacts found" in caplog.text
            assert "Train a model using train_wav2vec2.py or train_global_head.py" in caplog.text
    
    @pytest.mark.asyncio
    async def test_handles_global_head_loading_exception(self, caplog):
        """Should handle exceptions during global head loading gracefully."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.load_global_head") as mock_load_global, \
             patch("app.main.load_emotion_model") as mock_load_legacy, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"), \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_detect.return_value = "global-only"
            mock_load_global.side_effect = Exception("Model file corrupted")
            mock_load_legacy.return_value = True
            mock_user_dir.glob.return_value = []
            
            from app.main import startup_event
            await startup_event()
            
            # Should not crash, should fall back to legacy
            assert "Failed to load global emotion head" in caplog.text
            assert "Legacy emotion model loaded successfully (fallback)" in caplog.text


class TestAlphaEngineStartupLogging:
    """Test alpha engine configuration logging during startup."""
    
    @pytest.mark.asyncio
    async def test_logs_sigmoid_alpha_with_parameters(self, caplog):
        """Should log sigmoid formula with K, tau, beta when USE_SIGMOID_ALPHA=True."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.load_global_head") as mock_load_global, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"), \
             patch("app.main.USE_SIGMOID_ALPHA", True), \
             patch("app.main.ALPHA_FEEDBACK_SCALE_K", 50), \
             patch("app.main.ALPHA_CONFIDENCE_THRESHOLD_TAU", 0.6), \
             patch("app.main.ALPHA_SIGMOID_SHARPNESS_BETA", 10), \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_detect.return_value = "global-only"
            mock_load_global.return_value = True
            mock_user_dir.glob.return_value = []
            
            from app.main import startup_event
            await startup_event()
            
            # Verify alpha engine configuration is logged
            assert "Alpha Engine: sigmoid (K=50, tau=0.6, beta=10)" in caplog.text
    
    @pytest.mark.asyncio
    async def test_logs_linear_alpha_when_disabled(self, caplog):
        """Should log linear formula when USE_SIGMOID_ALPHA=False."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.load_global_head") as mock_load_global, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"), \
             patch("app.main.USE_SIGMOID_ALPHA", False), \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_detect.return_value = "global-only"
            mock_load_global.return_value = True
            mock_user_dir.glob.return_value = []
            
            from app.main import startup_event
            await startup_event()
            
            # Verify alpha engine configuration is logged
            assert "Alpha Engine: linear" in caplog.text
    
    @pytest.mark.asyncio
    async def test_logs_custom_sigmoid_parameters(self, caplog):
        """Should log custom hyperparameters when configured."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.load_global_head") as mock_load_global, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"), \
             patch("app.main.USE_SIGMOID_ALPHA", True), \
             patch("app.main.ALPHA_FEEDBACK_SCALE_K", 100), \
             patch("app.main.ALPHA_CONFIDENCE_THRESHOLD_TAU", 0.7), \
             patch("app.main.ALPHA_SIGMOID_SHARPNESS_BETA", 20), \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_detect.return_value = "dual-head"
            mock_load_global.return_value = True
            mock_user_dir.glob.return_value = [Path("user1.pt")]
            
            from app.main import startup_event
            await startup_event()
            
            # Verify custom parameters are logged correctly
            assert "Alpha Engine: sigmoid (K=100, tau=0.7, beta=20)" in caplog.text
    
    @pytest.mark.asyncio
    async def test_alpha_logging_occurs_before_model_loading(self, caplog):
        """Should log alpha engine configuration before model loading messages."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch("app.main.detect_active_model_format") as mock_detect, \
             patch("app.main.load_global_head") as mock_load_global, \
             patch("app.main.connect_to_mongo"), \
             patch("app.main.connect_to_qdrant"), \
             patch("app.main.connect_to_redis"), \
             patch("app.main.USE_SIGMOID_ALPHA", True), \
             patch("app.main.ALPHA_FEEDBACK_SCALE_K", 50), \
             patch("app.main.ALPHA_CONFIDENCE_THRESHOLD_TAU", 0.6), \
             patch("app.main.ALPHA_SIGMOID_SHARPNESS_BETA", 10), \
             patch("app.main.USER_HEADS_DIR") as mock_user_dir:
            
            mock_detect.return_value = "global-only"
            mock_load_global.return_value = True
            mock_user_dir.glob.return_value = []
            
            from app.main import startup_event
            await startup_event()
            
            # Find positions of log messages
            log_text = caplog.text
            alpha_pos = log_text.find("Alpha Engine:")
            model_pos = log_text.find("Global emotion head preloaded")
            
            # Alpha logging should come before model loading
            assert alpha_pos > 0, "Alpha Engine log not found"
            assert model_pos > 0, "Model loading log not found"
            assert alpha_pos < model_pos, "Alpha Engine should be logged before model loading"
