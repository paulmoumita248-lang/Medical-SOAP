import pytest
from backend.app.models.schemas import Utterance
from backend.app.services.prompt_templates import (
    SYSTEM_PROMPT,
    TARGET_JSON_SCHEMA_EXAMPLE,
    format_diarized_transcript,
    build_soap_prompt,
    build_retry_prompt,
)


def test_system_prompt_contains_guardrails():
    """Verify system prompt includes strict JSON instructions and anti-hallucination rules."""
    assert "STRICT JSON OUTPUT ONLY" in SYSTEM_PROMPT
    assert "ANTI-HALLUCINATION GUARDRAILS" in SYSTEM_PROMPT
    assert "Subjective" in SYSTEM_PROMPT
    assert "Objective" in SYSTEM_PROMPT
    assert "Assessment" in SYSTEM_PROMPT
    assert "Plan" in SYSTEM_PROMPT


def test_target_schema_example_structure():
    """Verify target JSON schema blueprint contains required top-level SOAP fields."""
    assert "subjective" in TARGET_JSON_SCHEMA_EXAMPLE
    assert "objective" in TARGET_JSON_SCHEMA_EXAMPLE
    assert "assessment" in TARGET_JSON_SCHEMA_EXAMPLE
    assert "plan" in TARGET_JSON_SCHEMA_EXAMPLE

    subj = TARGET_JSON_SCHEMA_EXAMPLE["subjective"]
    assert "chief_complaint" in subj
    assert "history_of_present_illness" in subj
    assert "review_of_systems" in subj


def test_format_diarized_transcript_dicts():
    """Verify formatting dictionary utterance lists into speaker-tagged text."""
    dict_utterances = [
        {"speaker_id": "SPEAKER_00", "text": "Hello Mr. Davis."},
        {"speaker_id": "SPEAKER_01", "text": "Hi Doctor, I have a fever."}
    ]
    formatted = format_diarized_transcript(dict_utterances)
    assert "SPEAKER_00: Hello Mr. Davis." in formatted
    assert "SPEAKER_01: Hi Doctor, I have a fever." in formatted


def test_format_diarized_transcript_pydantic():
    """Verify formatting Pydantic Utterance objects into speaker-tagged text."""
    utterances = [
        Utterance(speaker_id="SPEAKER_00", start_time=0.0, end_time=2.0, text="What brings you in today?"),
        Utterance(speaker_id="SPEAKER_01", start_time=2.5, end_time=5.0, text="I've been coughing for 3 days.")
    ]
    formatted = format_diarized_transcript(utterances)
    assert "SPEAKER_00: What brings you in today?" in formatted
    assert "SPEAKER_01: I've been coughing for 3 days." in formatted


def test_format_diarized_transcript_empty():
    """Verify formatting empty utterance list handles fallback gracefully."""
    assert format_diarized_transcript([]) == "[NO DIALOGUE PROVIDED]"


def test_build_soap_prompt():
    """Verify build_soap_prompt generates valid system and user prompt strings."""
    utterances = [
        Utterance(speaker_id="SPEAKER_00", start_time=0.0, end_time=2.0, text="How are you?"),
        Utterance(speaker_id="SPEAKER_01", start_time=2.5, end_time=5.0, text="I have a sore throat.")
    ]
    sys_p, user_p = build_soap_prompt(utterances)

    assert sys_p == SYSTEM_PROMPT
    assert "SPEAKER_00: How are you?" in user_p
    assert "SPEAKER_01: I have a sore throat." in user_p
    assert "Target JSON Schema Example:" in user_p


def test_build_retry_prompt():
    """Verify build_retry_prompt includes validation error details and previous response."""
    utterances = [
        Utterance(speaker_id="SPEAKER_00", start_time=0.0, end_time=2.0, text="How are you?")
    ]
    raw_response = "{'invalid_json': true}"
    error_msg = "JSONDecodeError: Expecting property name enclosed in double quotes"

    retry_sys, retry_user = build_retry_prompt(utterances, raw_response, error_msg)

    assert "IMPORTANT: Your previous output contained JSON syntax or schema validation errors" in retry_sys
    assert "{'invalid_json': true}" in retry_user
    assert "JSONDecodeError:" in retry_user
