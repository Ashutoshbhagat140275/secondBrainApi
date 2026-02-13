"""
Unit tests for feature_config.py alpha engine configuration.

Tests verify that:
- All alpha engine constants are defined
- Default values match specification (K=50, tau=0.6, beta=10)
- USE_SIGMOID_ALPHA defaults to False
- Validation logic correctly rejects invalid values
"""

import pytest
from unittest.mock import patch


class TestAlphaEngineConstants:
    """Test that all alpha engine constants are defined with correct defaults."""
    
    def test_use_sigmoid_alpha_defined(self):
        """Test that USE_SIGMOID_ALPHA constant is defined."""
        from app.services.feature_config import USE_SIGMOID_ALPHA
        assert USE_SIGMOID_ALPHA is not None
    
    def test_alpha_feedback_scale_k_defined(self):
        """Test that ALPHA_FEEDBACK_SCALE_K constant is defined."""
        from app.services.feature_config import ALPHA_FEEDBACK_SCALE_K
        assert ALPHA_FEEDBACK_SCALE_K is not None
    
    def test_alpha_confidence_threshold_tau_defined(self):
        """Test that ALPHA_CONFIDENCE_THRESHOLD_TAU constant is defined."""
        from app.services.feature_config import ALPHA_CONFIDENCE_THRESHOLD_TAU
        assert ALPHA_CONFIDENCE_THRESHOLD_TAU is not None
    
    def test_alpha_sigmoid_sharpness_beta_defined(self):
        """Test that ALPHA_SIGMOID_SHARPNESS_BETA constant is defined."""
        from app.services.feature_config import ALPHA_SIGMOID_SHARPNESS_BETA
        assert ALPHA_SIGMOID_SHARPNESS_BETA is not None


class TestAlphaEngineDefaults:
    """Test that alpha engine constants have correct default values."""
    
    def test_use_sigmoid_alpha_defaults_to_false(self):
        """Test that USE_SIGMOID_ALPHA defaults to False for backward compatibility."""
        from app.services.feature_config import USE_SIGMOID_ALPHA
        assert USE_SIGMOID_ALPHA is False
    
    def test_alpha_feedback_scale_k_default_value(self):
        """Test that ALPHA_FEEDBACK_SCALE_K defaults to 50."""
        from app.services.feature_config import ALPHA_FEEDBACK_SCALE_K
        assert ALPHA_FEEDBACK_SCALE_K == 50
    
    def test_alpha_confidence_threshold_tau_default_value(self):
        """Test that ALPHA_CONFIDENCE_THRESHOLD_TAU defaults to 0.6."""
        from app.services.feature_config import ALPHA_CONFIDENCE_THRESHOLD_TAU
        assert ALPHA_CONFIDENCE_THRESHOLD_TAU == 0.6
    
    def test_alpha_sigmoid_sharpness_beta_default_value(self):
        """Test that ALPHA_SIGMOID_SHARPNESS_BETA defaults to 10."""
        from app.services.feature_config import ALPHA_SIGMOID_SHARPNESS_BETA
        assert ALPHA_SIGMOID_SHARPNESS_BETA == 10


class TestAlphaEngineValidation:
    """Test validation logic for alpha engine configuration."""
    
    def test_validation_accepts_valid_defaults(self):
        """Test that validation passes with default values."""
        # If module imports successfully, validation passed
        from app.services import feature_config
        assert feature_config is not None
    
    def test_validation_rejects_negative_k(self):
        """Test that validation rejects K <= 0."""
        with patch('app.services.feature_config.ALPHA_FEEDBACK_SCALE_K', -10):
            with pytest.raises(ValueError, match="ALPHA_FEEDBACK_SCALE_K must be > 0"):
                from app.services.feature_config import _validate_alpha_config
                _validate_alpha_config()
    
    def test_validation_rejects_zero_k(self):
        """Test that validation rejects K = 0."""
        with patch('app.services.feature_config.ALPHA_FEEDBACK_SCALE_K', 0):
            with pytest.raises(ValueError, match="ALPHA_FEEDBACK_SCALE_K must be > 0"):
                from app.services.feature_config import _validate_alpha_config
                _validate_alpha_config()
    
    def test_validation_rejects_tau_below_zero(self):
        """Test that validation rejects tau <= 0."""
        with patch('app.services.feature_config.ALPHA_CONFIDENCE_THRESHOLD_TAU', 0):
            with pytest.raises(ValueError, match="ALPHA_CONFIDENCE_THRESHOLD_TAU must be in range"):
                from app.services.feature_config import _validate_alpha_config
                _validate_alpha_config()
    
    def test_validation_rejects_tau_above_one(self):
        """Test that validation rejects tau >= 1."""
        with patch('app.services.feature_config.ALPHA_CONFIDENCE_THRESHOLD_TAU', 1.0):
            with pytest.raises(ValueError, match="ALPHA_CONFIDENCE_THRESHOLD_TAU must be in range"):
                from app.services.feature_config import _validate_alpha_config
                _validate_alpha_config()
    
    def test_validation_rejects_negative_beta(self):
        """Test that validation rejects beta <= 0."""
        with patch('app.services.feature_config.ALPHA_SIGMOID_SHARPNESS_BETA', -5):
            with pytest.raises(ValueError, match="ALPHA_SIGMOID_SHARPNESS_BETA must be > 0"):
                from app.services.feature_config import _validate_alpha_config
                _validate_alpha_config()
    
    def test_validation_rejects_zero_beta(self):
        """Test that validation rejects beta = 0."""
        with patch('app.services.feature_config.ALPHA_SIGMOID_SHARPNESS_BETA', 0):
            with pytest.raises(ValueError, match="ALPHA_SIGMOID_SHARPNESS_BETA must be > 0"):
                from app.services.feature_config import _validate_alpha_config
                _validate_alpha_config()
    
    def test_validation_accepts_valid_k_values(self):
        """Test that validation accepts various valid K values."""
        from app.services.feature_config import _validate_alpha_config
        
        valid_k_values = [1, 25, 50, 100, 1000]
        for k in valid_k_values:
            with patch('app.services.feature_config.ALPHA_FEEDBACK_SCALE_K', k):
                # Should not raise
                _validate_alpha_config()
    
    def test_validation_accepts_valid_tau_values(self):
        """Test that validation accepts various valid tau values."""
        from app.services.feature_config import _validate_alpha_config
        
        valid_tau_values = [0.1, 0.5, 0.6, 0.7, 0.9]
        for tau in valid_tau_values:
            with patch('app.services.feature_config.ALPHA_CONFIDENCE_THRESHOLD_TAU', tau):
                # Should not raise
                _validate_alpha_config()
    
    def test_validation_accepts_valid_beta_values(self):
        """Test that validation accepts various valid beta values."""
        from app.services.feature_config import _validate_alpha_config
        
        valid_beta_values = [0.1, 5, 10, 20, 100]
        for beta in valid_beta_values:
            with patch('app.services.feature_config.ALPHA_SIGMOID_SHARPNESS_BETA', beta):
                # Should not raise
                _validate_alpha_config()


class TestAlphaEngineConstantTypes:
    """Test that alpha engine constants have correct types."""
    
    def test_use_sigmoid_alpha_is_boolean(self):
        """Test that USE_SIGMOID_ALPHA is a boolean."""
        from app.services.feature_config import USE_SIGMOID_ALPHA
        assert isinstance(USE_SIGMOID_ALPHA, bool)
    
    def test_alpha_feedback_scale_k_is_numeric(self):
        """Test that ALPHA_FEEDBACK_SCALE_K is numeric."""
        from app.services.feature_config import ALPHA_FEEDBACK_SCALE_K
        assert isinstance(ALPHA_FEEDBACK_SCALE_K, (int, float))
    
    def test_alpha_confidence_threshold_tau_is_numeric(self):
        """Test that ALPHA_CONFIDENCE_THRESHOLD_TAU is numeric."""
        from app.services.feature_config import ALPHA_CONFIDENCE_THRESHOLD_TAU
        assert isinstance(ALPHA_CONFIDENCE_THRESHOLD_TAU, (int, float))
    
    def test_alpha_sigmoid_sharpness_beta_is_numeric(self):
        """Test that ALPHA_SIGMOID_SHARPNESS_BETA is numeric."""
        from app.services.feature_config import ALPHA_SIGMOID_SHARPNESS_BETA
        assert isinstance(ALPHA_SIGMOID_SHARPNESS_BETA, (int, float))
