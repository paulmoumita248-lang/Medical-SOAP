"""
Unit & Integration Tests for Clinical SOAP Note Synthesis Service (Week 2 Task W2.3).
"""

import json
import pytest
from backend.app.models.schemas import SOAPNote, Utterance
from backend.app.services.soap_synthesizer import (
    SOAPSynthesizer,
    clean_json_response,
    sanitize_soap_dict,
    synthesize_soap_note,
)

# Sample test utterances
SAMPLE_UTTERANCES = [
    Utterance(speaker_id="SPEAKER_00", start_time=0.0, end_time=3.5, text="Hello Mr. Smith, what brings you in today?"),
    Utterance(speaker_id="SPEAKER_01", start_time=4.0, end_time=8.5, text="I have had a bad cough and low grade fever for 3 days."),
    Utterance(speaker_id="SPEAKER_00", start_time=9.0, end_time=12.0, text="Let me listen to your lungs. Temperature is 100.4 F."),
    Utterance(speaker_id="SPEAKER_00", start_time=12.5, end_time=16.0, text="Lungs are clear. Sounds like an acute viral infection. Take acetaminophen for fever.")
]

VALID_MOCK_JSON = json.dumps({
    "subjective": {
        "chief_complaint": "Bad cough and low grade fever for 3 days",
        "history_of_present_illness": "Patient reports 3-day history of cough and low grade fever.",
        "review_of_systems": ["Fever (+)", "Cough (+)"]
    },
    "objective": {
        "vital_signs": {"temperature": "100.4 F"},
        "physical_exam": "Lungs clear to auscultation bilaterally"
    },
    "assessment": {
        "primary_diagnosis": "Acute viral respiratory infection",
        "differential_diagnoses": ["Acute Bronchitis", "Influenza"],
        "clinical_summary": "Patient presenting with 3 days of cough and fever. Vitals stable, lungs clear."
    },
    "plan": {
        "medications": ["Acetaminophen 500mg as needed"],
        "diagnostic_tests": [],
        "patient_education": "Rest, hydration, return if symptoms worsen",
        "follow_up": "1 week PRN"
    }
})


def test_clean_json_response():
    raw_markdown = "```json\n{\"subjective\": {}}\n```"
    cleaned = clean_json_response(raw_markdown)
    assert cleaned == '{"subjective": {}}'

    raw_plain = "  {\"subjective\": {}}  "
    cleaned_plain = clean_json_response(raw_plain)
    assert cleaned_plain == '{"subjective": {}}'


def test_sanitize_soap_dict():
    incomplete_dict = {
        "subjective": {
            "chief_complaint": "Cough"
        }
    }
    sanitized = sanitize_soap_dict(incomplete_dict)
    assert sanitized["subjective"]["chief_complaint"] == "Cough"
    assert sanitized["subjective"]["history_of_present_illness"] == "Not Stated"
    assert sanitized["subjective"]["review_of_systems"] == []
    assert sanitized["objective"]["vital_signs"] == {}
    assert sanitized["objective"]["physical_exam"] == "Not Stated"
    assert sanitized["assessment"]["primary_diagnosis"] == "Not Stated"
    assert sanitized["plan"]["patient_education"] == "Not Stated"


def test_synthesize_soap_note_with_mock_response():
    synthesizer = SOAPSynthesizer(api_key="mock_key")
    soap_note = synthesizer.synthesize_soap_note(
        SAMPLE_UTTERANCES,
        mock_response=VALID_MOCK_JSON
    )

    assert isinstance(soap_note, SOAPNote)
    assert soap_note.subjective.chief_complaint == "Bad cough and low grade fever for 3 days"
    assert soap_note.objective.vital_signs == {"temperature": "100.4 F"}
    assert soap_note.assessment.primary_diagnosis == "Acute viral respiratory infection"
    assert "Acetaminophen 500mg as needed" in soap_note.plan.medications


def test_synthesize_soap_note_empty_utterances():
    synthesizer = SOAPSynthesizer(api_key="mock_key")
    with pytest.raises(ValueError, match="Utterances list cannot be empty"):
        synthesizer.synthesize_soap_note([])


def test_synthesize_soap_note_missing_api_key():
    synthesizer = SOAPSynthesizer(api_key="")
    with pytest.raises(ValueError, match="MISTRAL_API_KEY is not configured"):
        synthesizer.synthesize_soap_note(SAMPLE_UTTERANCES)


def test_synthesize_soap_note_invalid_json_retry_failure():
    synthesizer = SOAPSynthesizer(api_key="mock_key", max_retries=1)
    with pytest.raises(ValueError, match="Failed to generate valid SOAP Note"):
        synthesizer.synthesize_soap_note(
            SAMPLE_UTTERANCES,
            mock_response="INVALID JSON STRING {{{{"
        )
