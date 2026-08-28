import logging
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.routers.audio_router import router as audio_router
from backend.app.routers.soap_router import router as soap_router

# Configure logging format
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("healthcare_soap.main")

# Initialize FastAPI App
app = FastAPI(
    title="Healthcare Ambient AI Clinical Scribe API",
    description="Automated clinical doctor-patient conversation transcription, speaker diarization, and SOAP note synthesis pipeline.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(audio_router)
app.include_router(soap_router)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System Infrastructure"])
async def health_check():
    """
    Basic health check endpoint returning system status.
    """
    return {
        "status": "healthy",
        "service": "Healthcare Ambient AI Scribe API",
        "environment": settings.ENVIRONMENT,
        "asr_provider": "Google Web Speech API (free)"
    }


@app.get("/", status_code=status.HTTP_200_OK, tags=["System Infrastructure"])
async def root():
    return {
        "message": "Welcome to Healthcare Ambient AI Clinical Scribe API (Week 1 Audio Pipeline). Access API docs at /docs."
    }
