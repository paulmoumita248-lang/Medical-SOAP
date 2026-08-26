import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify health check endpoint returns 200 OK and correct system metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["asr_provider"] == "Google Web Speech API (free)"


def test_audio_transcribe_invalid_format():
    """Verify upload router rejects invalid audio formats (e.g. text file)."""
    fake_file = io.BytesIO(b"Not an audio file")
    response = client.post(
        "/api/audio/transcribe",
        files={"file": ("test.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_audio_transcribe_pipeline_mock():
    """Verify audio upload pipeline returns valid DiarizedTranscriptResponse schema."""
    fake_wav_bytes = b"RIFF" + (36).to_bytes(4, 'little') + b"WAVEfmt " + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') + (1).to_bytes(2, 'little') + (16000).to_bytes(4, 'little') + (32000).to_bytes(4, 'little') + (2).to_bytes(2, 'little') + (16).to_bytes(2, 'little') + b"data" + (0).to_bytes(4, 'little')
    
    response = client.post(
        "/api/audio/transcribe",
        files={"file": ("test_consultation.wav", io.BytesIO(fake_wav_bytes), "audio/wav")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "session_id" in data
    assert "audio_duration_seconds" in data
    assert "raw_transcript" in data
    assert "utterances" in data
    assert isinstance(data["utterances"], list)
    
    if len(data["utterances"]) > 0:
        first_utt = data["utterances"][0]
        assert "speaker_id" in first_utt
        assert first_utt["speaker_id"] in ["SPEAKER_00", "SPEAKER_01"]
        assert "text" in first_utt
