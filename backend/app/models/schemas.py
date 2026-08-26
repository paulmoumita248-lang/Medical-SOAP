from typing import List
from pydantic import BaseModel, Field


# ==========================================
# Week 1: Audio & Diarized Transcript Schemas
# ==========================================

class Utterance(BaseModel):
    speaker_id: str = Field(..., description="Speaker identifier, e.g., SPEAKER_00 or SPEAKER_01")
    start_time: float = Field(0.0, description="Start timestamp in seconds")
    end_time: float = Field(0.0, description="End timestamp in seconds")
    text: str = Field(..., description="Transcribed spoken text")


class DiarizedTranscriptResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    audio_duration_seconds: float = Field(..., description="Total audio duration in seconds")
    raw_transcript: str = Field(..., description="Full raw transcript text")
    utterances: List[Utterance] = Field(default_factory=list, description="List of speaker-tagged utterances")


class TranscriptRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    utterances: List[Utterance] = Field(..., description="Diarized transcript utterances")
