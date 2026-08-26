import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # LLM Settings (Mistral API)
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-small-latest"

    # ASR Settings (Google Web Speech API — free, no key required)
    STT_LANGUAGE: str = "en-US"

    # Vector Store Settings
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # Audio Settings
    MAX_AUDIO_SIZE_MB: int = 25
    ALLOWED_AUDIO_FORMATS: List[str] = [
        "audio/wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/m4a",
        "audio/x-m4a",
        "audio/webm"
    ]

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings singleton
settings = Settings()
