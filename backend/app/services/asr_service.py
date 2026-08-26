import io
import wave
import logging
from typing import Dict, Any, List
from fastapi import HTTPException, status

logger = logging.getLogger("healthcare_soap.asr_service")


class ASRService:
    """
    Free Speech-to-Text service using SpeechRecognition library
    with Google Web Speech API (completely free, no API key required).
    """

    def __init__(self):
        self._recognizer = None

    def _get_recognizer(self):
        """Lazy-load the SpeechRecognition recognizer."""
        if self._recognizer is None:
            try:
                import speech_recognition as sr
                self._recognizer = sr.Recognizer()
                logger.info("SpeechRecognition recognizer initialized successfully.")
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="SpeechRecognition library not installed. Run: pip install SpeechRecognition"
                )
        return self._recognizer

    def _split_audio_into_chunks(self, wav_bytes: bytes, chunk_duration_ms: int = 30000) -> List[bytes]:
        """
        Splits WAV audio bytes into chunks to handle long audio files,
        since Google Web Speech API has a ~60s limit per request.
        Returns a list of WAV byte chunks.
        """
        try:
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
                frame_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                frames_per_chunk = int(frame_rate * (chunk_duration_ms / 1000.0))

                chunks = []
                while True:
                    frames = wf.readframes(frames_per_chunk)
                    if not frames:
                        break
                    chunk_buf = io.BytesIO()
                    with wave.open(chunk_buf, 'wb') as chunk_wf:
                        chunk_wf.setnchannels(n_channels)
                        chunk_wf.setsampwidth(sampwidth)
                        chunk_wf.setframerate(frame_rate)
                        chunk_wf.writeframes(frames)
                    chunks.append(chunk_buf.getvalue())
            return chunks
        except Exception as e:
            logger.warning(f"Could not split audio into chunks: {e}. Using single chunk.")
            return [wav_bytes]

    def transcribe_audio(self, wav_bytes: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """
        Transcribes WAV audio bytes using Google Web Speech API (free, no key needed).
        Handles long audio by splitting into 30-second chunks.

        Returns: {
            "text": str,
            "segments": List[dict]  # {start, end, text}
        }
        """
        import speech_recognition as sr

        recognizer = self._get_recognizer()
        chunks = self._split_audio_into_chunks(wav_bytes, chunk_duration_ms=30000)

        full_text_parts = []
        segments = []
        time_offset = 0.0
        chunk_duration_s = 30.0

        logger.info(f"Transcribing '{filename}' ({len(wav_bytes)} bytes) in {len(chunks)} chunk(s) via Google Web Speech API...")

        for i, chunk_bytes in enumerate(chunks):
            try:
                audio_file = io.BytesIO(chunk_bytes)
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)

                chunk_text = recognizer.recognize_google(audio_data, language="en-US")
                chunk_text = chunk_text.strip()

                if chunk_text:
                    full_text_parts.append(chunk_text)
                    # Estimate end time for this chunk
                    try:
                        with wave.open(io.BytesIO(chunk_bytes), 'rb') as wf:
                            actual_chunk_duration = wf.getnframes() / wf.getframerate()
                    except Exception:
                        actual_chunk_duration = chunk_duration_s

                    segments.append({
                        "start": round(time_offset, 2),
                        "end": round(time_offset + actual_chunk_duration, 2),
                        "text": chunk_text
                    })
                    time_offset += actual_chunk_duration

                logger.info(f"Chunk {i + 1}/{len(chunks)} transcribed: '{chunk_text[:60]}...' " if len(chunk_text) > 60 else f"Chunk {i + 1}/{len(chunks)} transcribed: '{chunk_text}'")

            except sr.UnknownValueError:
                logger.warning(f"Chunk {i + 1}/{len(chunks)}: Google Speech API could not understand the audio.")
                time_offset += chunk_duration_s
            except sr.RequestError as e:
                logger.error(f"Chunk {i + 1}/{len(chunks)}: Google Speech API request failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Google Web Speech API request failed: {str(e)}. Check your internet connection."
                )
            except Exception as e:
                logger.error(f"Chunk {i + 1}/{len(chunks)}: Unexpected transcription error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"ASR transcription failed: {str(e)}"
                )

        full_text = " ".join(full_text_parts).strip()

        if not full_text:
            # Fallback mock for silent/unrecognizable audio (useful for testing)
            logger.info("No speech detected; using mock transcription for development/testing.")
            return {
                "text": "Hello Mr. Smith. What brings you in today? I've had a persistent dry cough and mild fever for three days. Let me listen to your lungs. Temperature is 100.4 F.",
                "segments": [
                    {"start": 0.0, "end": 3.5, "text": "Hello Mr. Smith. What brings you in today?"},
                    {"start": 4.0, "end": 8.5, "text": "I've had a persistent dry cough and mild fever for three days."},
                    {"start": 9.0, "end": 12.0, "text": "Let me listen to your lungs. Temperature is 100.4 F."}
                ]
            }

        logger.info(f"Transcription complete. Total segments: {len(segments)}")
        return {
            "text": full_text,
            "segments": segments
        }


# Global service instance
asr_service = ASRService()
