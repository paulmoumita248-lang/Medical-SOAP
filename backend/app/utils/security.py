import re
import logging

logger = logging.getLogger("healthcare_soap.security")

# Basic PHI patterns for logging safeguards
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


def sanitize_log_text(text: str) -> str:
    """
    Strips candidate PHI (SSN, Phone, Email) from strings before writing to server logs.
    """
    if not text:
        return text
    
    sanitized = SSN_PATTERN.sub("[REDACTED-SSN]", text)
    sanitized = PHONE_PATTERN.sub("[REDACTED-PHONE]", sanitized)
    sanitized = EMAIL_PATTERN.sub("[REDACTED-EMAIL]", sanitized)
    return sanitized


def validate_api_key_format(key: str, provider: str = "generic") -> bool:
    """
    Basic sanity check for non-empty API keys.
    """
    if not key or len(key.strip()) < 8 or "your_" in key.lower():
        logger.warning(f"Invalid or unconfigured API key for provider: {provider}")
        return False
    return True
