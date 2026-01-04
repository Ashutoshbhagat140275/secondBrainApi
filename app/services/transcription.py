import whisper
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Global model instance (loaded once)
_whisper_model = None


def load_whisper_model(model_size: str = "base"):
    """Load Whisper model (lazy loading)"""
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {model_size}")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe audio file to text using Whisper
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        Transcribed text
    """
    try:
        model = load_whisper_model(model_size)
        logger.info(f"Transcribing audio: {audio_path}")
        
        result = model.transcribe(audio_path)
        text = result["text"].strip()
        
        logger.info(f"Transcription completed: {len(text)} characters")
        return text
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise

