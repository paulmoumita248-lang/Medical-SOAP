import pytest
from backend.app.models.schemas import (
    Utterance,
    DiarizedTranscriptResponse,
    TranscriptRequest,
    Subjective,
    Objective,
    Assessment,
    Plan,
    SOAPNote,
    SOAPResponse,
)


def test_utterance_schema():
    u = Utterance(speaker_id="SPEAKER_00", start_time=0.0, end_time=3.5, text="Hello doctor.")
    assert u.speaker_id == "SPEAKER_00"
    assert u.text == "Hello doctor."


def test_soap_note_schema_valid():
    soap = SOAPNote(
        subjective=Subjective(
            chief_complaint="Persistent dry cough",
            history_of_present_illness="Patient reports 3 days of dry cough.",
            review_of_systems=["Fever", "Cough"]
        ),
        objective=Objective(
            vital_signs={"temperature": "100.4 F"},
            physical_exam="Clear lungs"
        ),
        assessment=Assessment(
            primary_diagnosis="Acute Upper Respiratory Infection",
            differential_diagnoses=["Bronchitis"],
            clinical_summary="Patient with acute URI symptoms."
        ),
        plan=Plan(
            medications=["Acetaminophen 500mg"],
            diagnostic_tests=["Rapid Flu Test"],
            patient_education="Rest and fluids",
            follow_up="1 week"
        )
    )
    assert soap.subjective.chief_complaint == "Persistent dry cough"
    assert soap.objective.vital_signs["temperature"] == "100.4 F"
    assert soap.assessment.primary_diagnosis == "Acute Upper Respiratory Infection"
    assert len(soap.plan.medications) == 1


def test_soap_note_schema_defaults():
    # Test optional default fields ("Not Stated", empty dicts/lists)
    subjective = Subjective(
        chief_complaint="Chest pain",
        history_of_present_illness="Sudden onset chest pain"
    )
    objective = Objective()
    assessment = Assessment(
        primary_diagnosis="Chest Pain Unspecified",
        clinical_summary="Rule out acute coronary syndrome."
    )
    plan = Plan()

    soap = SOAPNote(
        subjective=subjective,
        objective=objective,
        assessment=assessment,
        plan=plan
    )

    assert soap.subjective.review_of_systems == []
    assert soap.objective.vital_signs == {}
    assert soap.objective.physical_exam == "Not Stated"
    assert soap.assessment.differential_diagnoses == []
    assert soap.plan.patient_education == "Not Stated"
    assert soap.plan.follow_up == "Not Stated"


def test_soap_response_schema():
    soap = SOAPNote(
        subjective=Subjective(
            chief_complaint="Headache",
            history_of_present_illness="Mild headache for 2 days"
        ),
        objective=Objective(),
        assessment=Assessment(
            primary_diagnosis="Tension Headache",
            clinical_summary="Mild tension headache."
        ),
        plan=Plan()
    )
    resp = SOAPResponse(session_id="test-session-123", soap_note=soap)
    assert resp.session_id == "test-session-123"
    assert resp.soap_note.assessment.primary_diagnosis == "Tension Headache"
    
    # Test JSON schema generation
    json_data = resp.model_dump_json()
    assert "test-session-123" in json_data
    assert "Tension Headache" in json_data
