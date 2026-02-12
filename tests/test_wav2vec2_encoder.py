"""
Unit tests for the Wav2Vec2 encoder service.

Tests model loading, embedding extraction, edge case handling,
and validation of output dimensions.

Run from the api/ directory:
    python -m pytest tests/test_wav2vec2_encoder.py -v
"""

import sys
import pathlib
import numpy as np
import pytest
import tempfile
import soundfile as sf

_api_root = pathlib.Path(__file__).resolve().parents[1]
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from app.services.wav2vec2_encoder import (
    load_wav2vec2_model,
    extract_wav2vec2_embedding,
    EMBEDDING_DIM,
    MIN_AUDIO_LENGTH,
    TARGET_SAMPLE_RATE,
)


class TestLoadWav2Vec2Model:
    """Test Wav2Vec2 model loading functionality."""

    def test_load_returns_bool(self):
        """load_wav2vec2_model should return a boolean."""
        result = load_wav2vec2_model()
        assert isinstance(result, bool)

    def test_load_succeeds(self):
        """Model should load successfully from HuggingFace."""
        result = load_wav2vec2_model()
        assert result is True

    def test_load_is_idempotent(self):
        """Calling load multiple times should work without errors."""
        result1 = load_wav2vec2_model()
        result2 = load_wav2vec2_model()
        assert result1 is True
        assert result2 is True


class TestExtractWav2Vec2Embedding:
    """Test embedding extraction from audio files."""

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary audio file for testing."""
        # Generate 1 second of random audio at 16kHz
        duration = 1.0
        sample_rate = TARGET_SAMPLE_RATE
        samples = int(duration * sample_rate)
        waveform = np.random.randn(samples).astype(np.float32) * 0.1

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform, sample_rate)
            yield f.name

    @pytest.fixture
    def short_audio_file(self):
        """Create a very short audio file (< 0.3s) for edge case testing."""
        duration = 0.2  # 200ms
        sample_rate = TARGET_SAMPLE_RATE
        samples = int(duration * sample_rate)
        waveform = np.random.randn(samples).astype(np.float32) * 0.1

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform, sample_rate)
            yield f.name

    @pytest.fixture
    def silence_audio_file(self):
        """Create an audio file with only silence."""
        duration = 1.0
        sample_rate = TARGET_SAMPLE_RATE
        samples = int(duration * sample_rate)
        waveform = np.zeros(samples, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform, sample_rate)
            yield f.name

    def test_embedding_shape(self, temp_audio_file):
        """Embedding should have shape (768,)."""
        embedding = extract_wav2vec2_embedding(temp_audio_file)
        assert embedding.shape == (EMBEDDING_DIM,)

    def test_embedding_dtype(self, temp_audio_file):
        """Embedding should be numpy array with float dtype."""
        embedding = extract_wav2vec2_embedding(temp_audio_file)
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype in [np.float32, np.float64]

    def test_embedding_no_nan(self, temp_audio_file):
        """Embedding should not contain NaN values."""
        embedding = extract_wav2vec2_embedding(temp_audio_file)
        assert not np.isnan(embedding).any()

    def test_embedding_no_inf(self, temp_audio_file):
        """Embedding should not contain infinite values."""
        embedding = extract_wav2vec2_embedding(temp_audio_file)
        assert not np.isinf(embedding).any()

    def test_short_audio_handling(self, short_audio_file):
        """Short audio (< 0.3s) should be handled without errors."""
        embedding = extract_wav2vec2_embedding(short_audio_file)
        assert embedding.shape == (EMBEDDING_DIM,)
        assert not np.isnan(embedding).any()

    def test_silence_handling(self, silence_audio_file):
        """Silence should produce valid embedding without errors."""
        embedding = extract_wav2vec2_embedding(silence_audio_file)
        assert embedding.shape == (EMBEDDING_DIM,)
        assert not np.isnan(embedding).any()

    def test_deterministic_extraction(self, temp_audio_file):
        """Same audio should produce same embedding."""
        embedding1 = extract_wav2vec2_embedding(temp_audio_file)
        embedding2 = extract_wav2vec2_embedding(temp_audio_file)
        np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_invalid_audio_path(self):
        """Invalid audio path should raise appropriate error."""
        with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
            extract_wav2vec2_embedding("/nonexistent/path/to/audio.wav")

    def test_embedding_statistics(self, temp_audio_file):
        """Embedding should have reasonable statistics."""
        embedding = extract_wav2vec2_embedding(temp_audio_file)
        
        # Check that embedding has non-zero variance
        assert embedding.std() > 0
        
        # Check that values are in a reasonable range (not all zeros)
        assert np.abs(embedding).max() > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def nan_audio_file(self):
        """Create an audio file with NaN values."""
        duration = 1.0
        sample_rate = TARGET_SAMPLE_RATE
        samples = int(duration * sample_rate)
        waveform = np.random.randn(samples).astype(np.float32)
        # Inject some NaN values
        waveform[100:110] = np.nan

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform, sample_rate)
            yield f.name

    def test_nan_waveform_handling(self, nan_audio_file):
        """Audio with NaN values should be handled gracefully."""
        # Should not raise an exception
        embedding = extract_wav2vec2_embedding(nan_audio_file)
        
        # Embedding should be valid (no NaN)
        assert embedding.shape == (EMBEDDING_DIM,)
        assert not np.isnan(embedding).any()

    def test_model_lazy_loading(self):
        """Model should be loaded lazily on first extraction call."""
        # This test assumes model is already loaded from previous tests
        # Just verify that extraction works without explicit load call
        duration = 0.5
        sample_rate = TARGET_SAMPLE_RATE
        samples = int(duration * sample_rate)
        waveform = np.random.randn(samples).astype(np.float32) * 0.1

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform, sample_rate)
            embedding = extract_wav2vec2_embedding(f.name)
            assert embedding.shape == (EMBEDDING_DIM,)


class TestRequirementValidation:
    """
    Validate specific requirements from the spec.
    
    Requirements tested:
    - 1.1: Load facebook/wav2vec2-base model
    - 1.2: Produce embeddings of shape (T, 768) -> (768,) after pooling
    - 1.3: Apply mean pooling to produce fixed-size embedding
    - 1.5: Handle audio shorter than 0.3 seconds gracefully
    - 1.6: Cache model in memory after first load
    """

    def test_requirement_1_2_embedding_dimension(self):
        """Requirement 1.2: Embeddings should be 768-dimensional."""
        duration = 1.0
        sample_rate = TARGET_SAMPLE_RATE
        samples = int(duration * sample_rate)
        waveform = np.random.randn(samples).astype(np.float32) * 0.1

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform, sample_rate)
            embedding = extract_wav2vec2_embedding(f.name)
            
            # Validate 768-dimensional output
            assert embedding.shape == (768,), f"Expected (768,), got {embedding.shape}"

    def test_requirement_1_5_short_audio(self):
        """Requirement 1.5: Handle audio < 0.3s without errors."""
        duration = 0.2  # 200ms < 300ms
        sample_rate = TARGET_SAMPLE_RATE
        samples = int(duration * sample_rate)
        waveform = np.random.randn(samples).astype(np.float32) * 0.1

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform, sample_rate)
            
            # Should not raise exception
            embedding = extract_wav2vec2_embedding(f.name)
            assert embedding.shape == (768,)
            assert not np.isnan(embedding).any()

    def test_requirement_1_6_model_caching(self):
        """Requirement 1.6: Model should be cached after first load."""
        # Load model
        result1 = load_wav2vec2_model()
        assert result1 is True
        
        # Second load should return True immediately (cached)
        result2 = load_wav2vec2_model()
        assert result2 is True
        
        # Verify extraction works (model is cached)
        duration = 0.5
        sample_rate = TARGET_SAMPLE_RATE
        samples = int(duration * sample_rate)
        waveform = np.random.randn(samples).astype(np.float32) * 0.1

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, waveform, sample_rate)
            embedding = extract_wav2vec2_embedding(f.name)
            assert embedding.shape == (768,)
