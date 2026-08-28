"""
SOAP Router Module for Clinical SOAP Note Generation API Endpoint.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from backend.app.models.schemas import TranscriptRequest, SOAPResponse
from backend.app.services.soap_synthesizer import synthesize_soap_note

logger = logging.getLogger("healthcare_soap.soap_router")

router = APIRouter(prefix="/api/soap", tags=["Clinical SOAP Synthesis"])


@router.post("/generate", response_model=SOAPResponse, status_code=status.HTTP_200_OK)
async def generate_soap_note_endpoint(payload: TranscriptRequest):
    """
    Ingests speaker-diarized transcript utterances, invokes Mistral LLM engine,
    validates structured JSON output against SOAP Pydantic schema, and returns complete SOAP note.
    """
    logger.info(
        f"Received SOAP generation request for session '{payload.session_id}' with {len(payload.utterances)} utterances."
    )

    if not payload.utterances:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Utterances list cannot be empty."
        )

    try:
        soap_note = synthesize_soap_note(payload.utterances)
        logger.info(f"Successfully generated SOAP note for session '{payload.session_id}'.")
        return SOAPResponse(
            session_id=payload.session_id,
            soap_note=soap_note
        )
    except ValueError as ve:
        logger.error(f"SOAP generation validation/configuration error for session '{payload.session_id}': {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except RuntimeError as re:
        logger.error(f"SOAP generation LLM API failure for session '{payload.session_id}': {re}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(re)
        )
    except Exception as e:
        logger.error(f"Unexpected error during SOAP generation for session '{payload.session_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during SOAP synthesis: {str(e)}"
        )
