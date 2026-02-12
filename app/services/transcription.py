import whisper
import logging
from pathlib import Path
from app.config import settings

logger = logging.getLogger(__name__)

# Global model instance (loaded once)
_whisper_model = None
_loaded_model_size = None


def load_whisper_model(model_size: str = None):
    """
    Load Whisper model (lazy singleton).

    Model sizes ranked by accuracy (and RAM cost):
      tiny (39M) < base (74M) < small (244M) < medium (769M) < large (1.5B)

    Default is taken from ``settings.whisper_model_size``.
    """
    global _whisper_model, _loaded_model_size
    if model_size is None:
        model_size = settings.whisper_model_size

    if _whisper_model is None or _loaded_model_size != model_size:
        logger.info(f"Loading Whisper model: {model_size}")
        _whisper_model = whisper.load_model(model_size)
        _loaded_model_size = model_size
    return _whisper_model


def transcribe_audio(
    audio_path: str,
    model_size: str = None,
    language: str = None,
) -> str:
    """
    Transcribe audio file to text using OpenAI Whisper.

    Parameters
    ----------
    audio_path : str
        Path to the audio file on disk.
    model_size : str, optional
        Override the default model size from config.
    language : str, optional
        ISO-639-1 code (e.g. ``"en"``).  Providing this skips Whisper's
        language-detection step and avoids misidentifying the language,
        which is a common cause of garbled output.

    Returns
    -------
    str
        Transcribed text.
    """
    try:
        model = load_whisper_model(model_size)
        logger.info(f"Transcribing audio: {audio_path}")

        # Language hint — prevents wrong-language detection
        lang = language or settings.whisper_language

        # Transcription options for better accuracy:
        #   beam_size=5        — explores 5 candidates per step (default is 1 / greedy)
        #   best_of=5          — sample 5 times and pick the best (only with temperature>0)
        #   temperature=0      — deterministic decoding (greedy within beam)
        #   condition_on_previous_text=True — helps with coherence
        transcribe_opts = {
            "language": lang,
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0,
            "condition_on_previous_text": True,
            "fp16": False,  # CPU-safe; set True if running on GPU
        }

        result = model.transcribe(audio_path, **transcribe_opts)
        text = result["text"].strip()

        detected = result.get("language", lang)
        logger.info(
            f"Transcription completed: {len(text)} chars, " f"detected_lang={detected}"
        )
        return text

    except Exception as e:
        logger.error(f"Error transcribing audio: {e}", exc_info=True)
        raise
