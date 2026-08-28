"""
Prompt Templates Module for Clinical SOAP Note Synthesis (Mistral LLM Engine).

This module defines system prompts, strict JSON formatting requirements,
anti-hallucination safeguards, few-shot clinical examples, and helper functions
to build optimized prompts for generating structured SOAP notes from diarized transcripts.
"""

import json
from typing import List, Dict, Any, Union, Tuple
import logging

logger = logging.getLogger("healthcare_soap.prompt_templates")

# =====================================================================
# 1. System Prompt & Persona Definition
# =====================================================================

SYSTEM_PROMPT = """You are an expert board-certified Clinical Medical Scribe AI.
Your task is to analyze a speaker-diarized clinical conversation between a doctor and a patient (tagged with SPEAKER_00, SPEAKER_01, etc.) and synthesize a complete, professional, and accurate SOAP (Subjective, Objective, Assessment, Plan) Note in strict JSON format.

CRITICAL INSTRUCTIONS & CONSTRAINTS:
1. STRICT JSON OUTPUT ONLY: You MUST respond strictly with a single valid JSON object. Do NOT include markdown blocks (e.g. ```json), commentary, greetings, or conversational preambles.
2. ANTI-HALLUCINATION GUARDRAILS:
   - Extract ONLY facts explicitly stated in the transcript.
   - NEVER invent or assume vital signs (temperature, blood pressure, pulse, oxygen saturation), physical exam findings, lab results, or medications that were not explicitly mentioned.
   - If vital signs are NOT mentioned, return an empty dictionary `{}` for `"vital_signs"`.
   - If physical exam findings are NOT mentioned, return `"Not Stated"` for `"physical_exam"`.
   - If patient education or follow-up instructions are NOT mentioned, return `"Not Stated"`.
   - If review of systems, differential diagnoses, medications, or diagnostic tests are empty, return an empty list `[]`.
3. SPEAKER IDENTIFICATION HEURISTIC:
   - Understand that one speaker is the Clinician/Doctor (asking questions, giving recommendations) and the other is the Patient (describing symptoms, history).
4. COMPREHENSIVE CLINICAL MAP:
   - Subjective: Chief complaint, detailed timeline/onset (HPI), and positive/negative symptoms mentioned (ROS).
   - Objective: Explicitly stated vital signs (as key-value pairs) and observed/performed physical exam findings.
   - Assessment: Synthesize a primary diagnosis, realistic differential diagnoses, and a concise clinical summary.
   - Plan: Actionable medications with dosages if stated, ordered diagnostic tests, patient self-care instructions, and follow-up timeframe.
"""

# =====================================================================
# 2. Few-Shot Example & Target JSON Schema Blueprint
# =====================================================================

TARGET_JSON_SCHEMA_EXAMPLE = {
    "subjective": {
        "chief_complaint": "Persistent dry cough and fever for 3 days",
        "history_of_present_illness": "34-year-old patient reports 3-day history of non-productive cough, fatigue, and low-grade fever up to 100.4 F. Denies shortness of breath or chest pain.",
        "review_of_systems": [
            "Constitutional: Fever (+), Fatigue (+)",
            "Respiratory: Cough (+), Dyspnea (-), Chest Pain (-)"
        ]
    },
    "objective": {
        "vital_signs": {
            "temperature": "100.4 F",
            "blood_pressure": "120/80 mmHg",
            "heart_rate": "76 bpm"
        },
        "physical_exam": "Lungs clear to auscultation bilaterally. Oropharynx mildly erythematous without exudate."
    },
    "assessment": {
        "primary_diagnosis": "Acute viral upper respiratory infection",
        "differential_diagnoses": [
            "Acute Bronchitis",
            "Influenza A/B"
        ],
        "clinical_summary": "Young adult male with 3 days of low-grade fever and non-productive cough. Vitals stable, physical exam unremarkable except mild pharyngeal erythema, consistent with acute viral URI."
    },
    "plan": {
        "medications": [
            "Acetaminophen 500mg PO every 6 hours PRN fever",
            "Dextromethorphan syrup 10mL PO every 6 hours PRN cough"
        ],
        "diagnostic_tests": [
            "Rapid Influenza Swab"
        ],
        "patient_education": "Maintain hydration, rest, and isolate at home until fever-free for 24 hours.",
        "follow_up": "Return if high fever (>102 F), shortness of breath, or symptoms persist beyond 7 days."
    }
}

# =====================================================================
# 3. Helper Functions
# =====================================================================

def format_diarized_transcript(utterances: List[Union[Dict[str, Any], Any]]) -> str:
    """
    Formats a list of utterance objects or dictionaries into a clean, line-by-line speaker dialogue.

    Example Output:
      SPEAKER_00: Hello Mr. Smith, what brings you in today?
      SPEAKER_01: I've had a persistent dry cough for 3 days.
    """
    if not utterances:
        return "[NO DIALOGUE PROVIDED]"

    formatted_lines = []
    for u in utterances:
        if isinstance(u, dict):
            speaker = u.get("speaker_id", "SPEAKER_00")
            text = u.get("text", "").strip()
        else:
            speaker = getattr(u, "speaker_id", "SPEAKER_00")
            text = getattr(u, "text", "").strip()

        if text:
            formatted_lines.append(f"{speaker}: {text}")

    return "\n".join(formatted_lines)


def build_soap_prompt(utterances: List[Union[Dict[str, Any], Any]]) -> Tuple[str, str]:
    """
    Builds the complete System Prompt and User Prompt pair for LLM SOAP synthesis.

    Returns:
        Tuple[str, str]: (system_prompt, user_prompt)
    """
    formatted_dialogue = format_diarized_transcript(utterances)

    user_prompt = f"""Target JSON Schema Example:
{json.dumps(TARGET_JSON_SCHEMA_EXAMPLE, indent=2)}

Diarized Clinical Consultation Transcript:
----------------------------------------
{formatted_dialogue}
----------------------------------------

Instruction: Analyze the transcript above and generate the complete SOAP note strictly matching the JSON structure provided. Return ONLY the raw JSON object."""

    return SYSTEM_PROMPT, user_prompt


def build_retry_prompt(
    utterances: List[Union[Dict[str, Any], Any]],
    raw_llm_response: str,
    validation_error: str
) -> Tuple[str, str]:
    """
    Builds a retry prompt pair when previous LLM response failed Pydantic schema validation or JSON parsing.

    Returns:
        Tuple[str, str]: (system_prompt, retry_user_prompt)
    """
    formatted_dialogue = format_diarized_transcript(utterances)

    retry_system_prompt = SYSTEM_PROMPT + "\n\nIMPORTANT: Your previous output contained JSON syntax or schema validation errors. You MUST correct them and output strictly valid JSON matching the schema."

    retry_user_prompt = f"""Target JSON Schema Blueprint:
{json.dumps(TARGET_JSON_SCHEMA_EXAMPLE, indent=2)}

Original Clinical Transcript:
----------------------------------------
{formatted_dialogue}
----------------------------------------

Your Previous Attempt:
----------------------------------------
{raw_llm_response}
----------------------------------------

Validation Error Encountered:
{validation_error}

Instruction: Fix the errors reported above and return a valid, complete JSON SOAP Note object strictly matching the schema."""

    return retry_system_prompt, retry_user_prompt
