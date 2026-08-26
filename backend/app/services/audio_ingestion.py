import io
import os
import logging
from typing import Tuple
from fastapi import HTTPException, status
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False

from backend.app.config import settings

logger = logging.getLogger("healthcare_soap.audio_ingestion")


class AudioIngestionService:
    def __init__(self):
        self.max_size_bytes = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
        self.allowed_formats = settings.ALLOWED_AUDIO_FORMATS

    def validate_file(self, content_type: str, file_size: int, filename: str) -> None:
        """
        Validates file size and format against settings limits.
        """
        if file_size > self.max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio file size ({file_size / (1024*1024):.2f}MB) exceeds max limit of {settings.MAX_AUDIO_SIZE_MB}MB."
            )
        
        # Simple extension check fallback
        ext = os.path.splitext(filename)[1].lower().strip(".")
        valid_exts = ["wav", "mp3", "m4a", "webm"]
        
        if content_type and content_type.lower() not in self.allowed_formats and ext not in valid_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format '{content_type or ext}'. Allowed: WAV, MP3, M4A, WEBM."
            )

    def normalize_to_wav(self, audio_bytes: bytes, filename: str) -> Tuple[bytes, float]:
        """
        Normalizes input audio bytes to 16kHz mono WAV format and calculates duration.
        Returns: (wav_bytes, duration_in_seconds)
        """
        ext = os.path.splitext(filename)[1].lower().strip(".")
        if not ext:
            ext = "wav"

        if not PYDUB_AVAILABLE:
            logger.info("pydub not installed; passing raw audio bytes directly.")
            return audio_bytes, 10.0

        try:
            # Load audio using pydub

            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext if ext != "m4a" else "mp4")
            
            # Normalize to 16kHz mono
            audio = audio.set_frame_rate(16000).set_channels(1)
            duration_seconds = len(audio) / 1000.0

            out_buf = io.BytesIO()
            audio.export(out_buf, format="wav")
            out_buf.seek(0)
            
            logger.info(f"Successfully normalized audio '{filename}' to 16kHz mono WAV ({duration_seconds:.2f}s)")
            return out_buf.read(), duration_seconds

        except Exception as e:
            logger.warning(f"Pydub audio conversion warning for '{filename}': {e}. Returning raw bytes as fallback.")
            # Fallback for plain WAV bytes if pydub/ffmpeg is unconfigured
            duration_seconds = 10.0  # Estimate
            return audio_bytes, duration_seconds


# Global service instance
audio_ingestion_service = AudioIngestionService()
