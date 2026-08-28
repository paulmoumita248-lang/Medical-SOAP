"""
Unit & Integration Tests for SOAP Router Endpoint (/api/soap/generate).
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "session_id": "test-session-456",
    "utterances": [
        {
            "speaker_id": "SPEAKER_00",
            "start_time": 0.0,
            "end_time": 3.0,
            "text": "Good morning, what brings you in today?"
        },
        {
            "speaker_id": "SPEAKER_01",
            "start_time": 3.5,
            "end_time": 7.0,
            "text": "I've had a sore throat and fever for 2 days."
        }
    ]
}


def test_generate_soap_note_endpoint_empty_utterances():
    """Verify router rejects empty utterances payload with 400 Bad Request."""
    response = client.post(
        "/api/soap/generate",
        json={"session_id": "empty-session", "utterances": []}
    )
    assert response.status_code == 400
    assert "Utterances list cannot be empty" in response.json()["detail"]


def test_generate_soap_note_endpoint_invalid_payload():
    """Verify router rejects payload with missing required fields with 422."""
    response = client.post(
        "/api/soap/generate",
        json={"invalid_field": "test"}
    )
    assert response.status_code == 422


def test_generate_soap_note_endpoint_success(monkeypatch):
    """Verify router endpoint generates valid SOAP note and returns 200 OK."""
    # Mock synthesize_soap_note to avoid network call in unit test
    from backend.app.models.schemas import SOAPNote, Subjective, Objective, Assessment, Plan
    
    mock_soap = SOAPNote(
        subjective=Subjective(
            chief_complaint="Sore throat and fever for 2 days",
            history_of_present_illness="Patient reports 2-day history of sore throat.",
            review_of_systems=["Fever (+)", "Sore throat (+)"]
        ),
        objective=Objective(
            vital_signs={"temperature": "100.1 F"},
            physical_exam="Erythematous pharynx"
        ),
        assessment=Assessment(
            primary_diagnosis="Acute Pharyngitis",
            differential_diagnoses=["Tonsillitis"],
            clinical_summary="Patient presenting with acute pharyngitis."
        ),
        plan=Plan(
            medications=["Acetaminophen 500mg"],
            diagnostic_tests=["Rapid Strep Swab"],
            patient_education="Rest and fluids",
            follow_up="As needed"
        )
    )

    monkeypatch.setattr(
        "backend.app.routers.soap_router.synthesize_soap_note",
        lambda utterances: mock_soap
    )

    response = client.post(
        "/api/soap/generate",
        json=SAMPLE_PAYLOAD
    )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-456"
    assert "soap_note" in data
    assert data["soap_note"]["subjective"]["chief_complaint"] == "Sore throat and fever for 2 days"
    assert data["soap_note"]["assessment"]["primary_diagnosis"] == "Acute Pharyngitis"
