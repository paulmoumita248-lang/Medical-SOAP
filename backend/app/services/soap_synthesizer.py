"""
SOAP Synthesizer Service for Clinical SOAP Note Generation using Mistral LLM.

This module handles:
1. Ingestion of diarized transcript utterances.
2. Invocation of the Mistral LLM API using response_format={"type": "json_object"}.
3. Strict JSON parsing and Pydantic SOAPNote schema validation.
4. Validation retry loop with error feedback prompt if parsing/validation fails.
5. Anti-hallucination post-processing and guardrails.
"""

import json
import logging
import re
from typing import List, Dict, Any, Union, Optional
from pydantic import ValidationError

from backend.app.config import settings
from backend.app.models.schemas import SOAPNote, Utterance
from backend.app.services.prompt_templates import build_soap_prompt, build_retry_prompt

logger = logging.getLogger("healthcare_soap.soap_synthesizer")


def clean_json_response(raw_text: str) -> str:
    """
    Strips markdown fence blocks (```json ... ```) or leading/trailing whitespace
    from the LLM output string to extract clean raw JSON.
    """
    text = raw_text.strip()
    # Remove markdown code fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def sanitize_soap_dict(soap_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies anti-hallucination guardrails and standardizes default fields:
    - Normalizes missing, None, or empty fields to schema defaults ("Not Stated", empty dict/list).
    - Ensures key sections (subjective, objective, assessment, plan) exist with clean types.
    - Sanitizes vital signs to ensure strictly key-value string pairs without invented objects.
    """
    if not isinstance(soap_dict, dict):
        return {}

    # Ensure root sections exist
    sections = ["subjective", "objective", "assessment", "plan"]
    for sec in sections:
        if sec not in soap_dict or not isinstance(soap_dict[sec], dict):
            soap_dict[sec] = {}

    sub = soap_dict["subjective"]
    obj = soap_dict["objective"]
    ass = soap_dict["assessment"]
    pln = soap_dict["plan"]

    # Helper for string defaults
    def clean_str(val: Any, default: str = "Not Stated") -> str:
        if val is None or not isinstance(val, str) or not val.strip():
            return default
        return val.strip()

    # Helper for list defaults
    def clean_list(val: Any) -> List[str]:
        if not isinstance(val, list):
            return []
        return [str(item).strip() for item in val if item is not None and str(item).strip()]

    # Subjective defaults
    sub["chief_complaint"] = clean_str(sub.get("chief_complaint"))
    sub["history_of_present_illness"] = clean_str(sub.get("history_of_present_illness"))
    sub["review_of_systems"] = clean_list(sub.get("review_of_systems"))

    # Objective defaults
    vitals = obj.get("vital_signs")
    if not isinstance(vitals, dict):
        obj["vital_signs"] = {}
    else:
        obj["vital_signs"] = {
            str(k).strip(): str(v).strip()
            for k, v in vitals.items()
            if k is not None and v is not None and str(v).strip()
        }
    obj["physical_exam"] = clean_str(obj.get("physical_exam"))

    # Assessment defaults
    ass["primary_diagnosis"] = clean_str(ass.get("primary_diagnosis"))
    ass["differential_diagnoses"] = clean_list(ass.get("differential_diagnoses"))
    ass["clinical_summary"] = clean_str(ass.get("clinical_summary"))

    # Plan defaults
    pln["medications"] = clean_list(pln.get("medications"))
    pln["diagnostic_tests"] = clean_list(pln.get("diagnostic_tests"))
    pln["patient_education"] = clean_str(pln.get("patient_education"))
    pln["follow_up"] = clean_str(pln.get("follow_up"))

    return soap_dict


class SOAPSynthesizer:
    """
    Service class responsible for generating validated SOAP notes from clinical transcripts
    via Mistral LLM API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 2
    ):
        self.api_key = settings.MISTRAL_API_KEY if api_key is None else api_key
        self.model = model or settings.MISTRAL_MODEL
        self.max_retries = max_retries

    def _call_mistral_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Executes raw API call to Mistral LLM with JSON response format.
        Supports mistralai SDK v2.x, v1.x, and v0.x.
        """
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not configured. Please set MISTRAL_API_KEY in environment or .env file."
            )

        try:
            try:
                from mistralai import Mistral
            except ImportError:
                from mistralai.client import Mistral

            client = Mistral(api_key=self.api_key)
            response = client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Mistral API call failed: {e}")
            raise RuntimeError(f"Mistral API invocation error: {str(e)}") from e

    def synthesize_soap_note(
        self,
        utterances: List[Union[Utterance, Dict[str, Any]]],
        mock_response: Optional[str] = None
    ) -> SOAPNote:
        """
        Synthesizes a structured SOAP Note from a list of diarized utterances.

        Args:
            utterances: List of Utterance Pydantic models or dicts with 'speaker_id' and 'text'.
            mock_response: Optional pre-defined LLM raw output string for deterministic unit testing.

        Returns:
            SOAPNote: Validated Pydantic model.
        """
        if not utterances:
            raise ValueError("Utterances list cannot be empty.")

        system_prompt, user_prompt = build_soap_prompt(utterances)
        raw_response = ""
        last_error = ""

        for attempt in range(1, self.max_retries + 2):
            logger.info(f"SOAP Synthesis Attempt {attempt}/{self.max_retries + 1}")
            try:
                if mock_response:
                    raw_response = mock_response
                else:
                    raw_response = self._call_mistral_api(system_prompt, user_prompt)

                clean_text = clean_json_response(raw_response)
                json_dict = json.loads(clean_text)
                sanitized_dict = sanitize_soap_dict(json_dict)
                soap_note = SOAPNote.model_validate(sanitized_dict)
                logger.info("Successfully synthesized and validated SOAP note.")
                return soap_note

            except (json.JSONDecodeError, ValidationError, ValueError) as err:
                last_error = f"{type(err).__name__}: {str(err)}"
                logger.warning(
                    f"Attempt {attempt} failed schema/JSON validation: {last_error}"
                )

                # If mock_response is provided and failed, don't enter infinite loop
                if mock_response and attempt > 1:
                    break

                if attempt <= self.max_retries:
                    # Prepare retry prompt with error feedback
                    system_prompt, user_prompt = build_retry_prompt(
                        utterances, raw_response, last_error
                    )

        raise ValueError(
            f"Failed to generate valid SOAP Note after {self.max_retries + 1} attempts. Last error: {last_error}"
        )


def synthesize_soap_note(
    utterances: List[Union[Utterance, Dict[str, Any]]],
    api_key: Optional[str] = None,
    mock_response: Optional[str] = None
) -> SOAPNote:
    """
    Convenience function to synthesize a SOAP note.
    """
    synthesizer = SOAPSynthesizer(api_key=api_key)
    return synthesizer.synthesize_soap_note(utterances, mock_response=mock_response)
