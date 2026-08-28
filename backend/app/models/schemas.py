from typing import List
from pydantic import BaseModel, Field
from typing import Dict

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


# ==========================================
# Week 2: Clinical SOAP Note Schemas
# ==========================================


class Subjective(BaseModel):
    chief_complaint: str = Field(..., description="Primary reason for visit / main symptom reported by patient")
    history_of_present_illness: str = Field(..., description="Detailed timeline, onset, character, and severity of symptoms")
    review_of_systems: List[str] = Field(default_factory=list, description="Systemic symptoms reported or denied")


class Objective(BaseModel):
    vital_signs: Dict[str, str] = Field(default_factory=dict, description="Vitals like temperature, BP, heart rate if mentioned")
    physical_exam: str = Field("Not Stated", description="Physical exam findings or observations")


class Assessment(BaseModel):
    primary_diagnosis: str = Field(..., description="Main clinical diagnosis based on consultation")
    differential_diagnoses: List[str] = Field(default_factory=list, description="Possible alternative diagnoses")
    clinical_summary: str = Field(..., description="Synthesized clinical summary of case")


class Plan(BaseModel):
    medications: List[str] = Field(default_factory=list, description="Prescribed or recommended medications and dosages")
    diagnostic_tests: List[str] = Field(default_factory=list, description="Ordered labs, imaging, or diagnostic procedures")
    patient_education: str = Field("Not Stated", description="Instructions, lifestyle, self-care guidance given to patient")
    follow_up: str = Field("Not Stated", description="Follow-up timeframe or return instructions")


class SOAPNote(BaseModel):
    subjective: Subjective
    objective: Objective
    assessment: Assessment
    plan: Plan


class SOAPResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    soap_note: SOAPNote

