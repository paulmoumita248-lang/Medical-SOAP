import io
import os
import wave
import logging
import subprocess
from typing import Tuple
from fastapi import HTTPException, status

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"

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
        
        # Extension check
        ext = os.path.splitext(filename)[1].lower().strip(".")
        valid_exts = ["wav", "mp3", "m4a", "webm", "aac", "ogg", "flac"]
        
        if content_type and content_type.lower() not in self.allowed_formats and ext not in valid_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format '{content_type or ext}'. Allowed: WAV, MP3, M4A, WEBM, AAC, OGG."
            )

    def normalize_to_wav(self, audio_bytes: bytes, filename: str) -> Tuple[bytes, float]:
        """
        Normalizes any input audio bytes (WAV, MP3, M4A, WEBM, AAC, OGG)
        to standardized 16kHz 16-bit mono PCM WAV format and calculates exact duration.
        Returns: (wav_bytes, duration_in_seconds)
        """
        ext = os.path.splitext(filename)[1].lower().strip(".")
        is_wav_header = audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"

        # 1. Primary Engine: Direct FFmpeg STDIN/STDOUT stream conversion
        ffmpeg_cmd = FFMPEG_PATH or "ffmpeg"
        try:
            cmd = [
                ffmpeg_cmd,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", "pipe:0",
                "-f", "wav",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "pipe:1"
            ]
            proc = subprocess.run(
                cmd,
                input=audio_bytes,
                capture_output=True,
                check=True
            )
            wav_bytes = proc.stdout
            if len(wav_bytes) > 44 and wav_bytes[:4] == b"RIFF":
                # 16kHz mono 16-bit PCM = 32,000 bytes per second
                duration_seconds = max(0.1, (len(wav_bytes) - 44) / 32000.0)
                logger.info(f"Successfully converted '{filename}' to 16kHz mono WAV via FFmpeg ({duration_seconds:.2f}s, {len(wav_bytes)} bytes)")
                return wav_bytes, duration_seconds
        except Exception as ffmpeg_err:
            logger.warning(f"Direct FFmpeg conversion failed for '{filename}': {ffmpeg_err}")

        # 2. Fallback: If file already has a valid WAV header
        if is_wav_header:
            duration_seconds = max(0.1, (len(audio_bytes) - 44) / 32000.0)
            logger.info(f"File '{filename}' already has valid WAV header; returning raw bytes.")
            return audio_bytes, duration_seconds

        # 3. Fallback Error: Unconvertible audio
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode audio file '{filename}' into valid WAV format. Ensure file is not corrupted."
        )


# Global service instance
audio_ingestion_service = AudioIngestionService()
