import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from backend.app.models.schemas import DiarizedTranscriptResponse
from backend.app.services.audio_ingestion import audio_ingestion_service
from backend.app.services.asr_service import asr_service
from backend.app.services.diarization_service import diarization_service

logger = logging.getLogger("healthcare_soap.audio_router")

router = APIRouter(prefix="/api/audio", tags=["Audio & ASR Pipeline"])


@router.post("/transcribe", response_model=DiarizedTranscriptResponse, status_code=status.HTTP_200_OK)
async def transcribe_audio_file(file: UploadFile = File(...)):
    """
    Accepts multi-format audio files (WAV, MP3, M4A, WEBM), normalizes format,
    transcribes speech via SR, and applies 2-speaker diarization (SPEAKER_00, SPEAKER_01).
    """
    filename = file.filename or "audio.wav"
    logger.info(f"Received audio upload request: filename='{filename}', content_type='{file.content_type}'")

    # Read raw bytes
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded audio payload: {str(e)}"
        )

    # 1. Format & Size Validation
    audio_ingestion_service.validate_file(
        content_type=file.content_type or "",
        file_size=len(content),
        filename=filename
    )

    # 2. Audio Normalization
    wav_bytes, duration_seconds = audio_ingestion_service.normalize_to_wav(content, filename)

    # 3. Speech-to-Text Transcription via Whisper
    asr_result = asr_service.transcribe_audio(wav_bytes, filename)
    raw_transcript = asr_result.get("text", "")
    segments = asr_result.get("segments", [])

    # 4. Speaker Diarization (SPEAKER_00, SPEAKER_01)
    utterances = diarization_service.diarize_segments(segments)

    session_id = str(uuid.uuid4())
    logger.info(f"Audio processing complete for session '{session_id}'. Duration: {duration_seconds:.2f}s, Utterances: {len(utterances)}")

    return DiarizedTranscriptResponse(
        session_id=session_id,
        audio_duration_seconds=round(duration_seconds, 2),
        raw_transcript=raw_transcript,
        utterances=utterances
    )
