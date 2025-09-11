"""
FastAPI application for Malayalam Voice Assistant
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import uuid
from datetime import datetime

from ..services import MalayalamVoiceAssistant
from ..utils.logger import logger
from ..utils.helpers import AudioUtils, rate_limiter, sanitize_filename
from ..config.settings import settings


# Initialize FastAPI app
app = FastAPI(
    title="Malayalam Voice Assistant API",
    description="Voice assistant for Malayalam-speaking farmers with agricultural RAG chatbot integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
assistant = MalayalamVoiceAssistant()
audio_utils = AudioUtils()


# Pydantic models
class TextQueryRequest(BaseModel):
    """Request model for text query"""
    text: str = Field(..., description="Malayalam text query")
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class VoiceQueryResponse(BaseModel):
    """Response model for voice queries"""
    success: bool
    malayalam_text: Optional[str] = None
    english_text: Optional[str] = None
    malayalam_response: Optional[str] = None
    english_response: Optional[str] = None
    audio_response_base64: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    services: Dict[str, bool]
    version: str


# Dependency for rate limiting
async def check_rate_limit(request: Request):
    """Check rate limit for requests"""
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return client_ip


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Malayalam Voice Assistant API")
    
    # Ensure temp directory exists
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    # Validate settings
    try:
        settings.validate()
    except ValueError as e:
        logger.warning(f"Configuration warning: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Malayalam Voice Assistant API")
    
    # Clean up services
    assistant.voice_service.cleanup()
    await assistant.rag_service.close()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Malayalam Voice Assistant API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Service health status
    """
    # Check individual service health
    rag_healthy = await assistant.rag_service.health_check()
    
    return HealthResponse(
        status="healthy" if rag_healthy else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        services={
            "rag_chatbot": rag_healthy,
            "voice_processing": True,  # Always available with fallbacks
            "translation": True,  # Always available with fallbacks
        },
        version="1.0.0"
    )


@app.post("/voice-query", response_model=VoiceQueryResponse, dependencies=[Depends(check_rate_limit)])
async def process_voice_query(
    audio_file: UploadFile = File(..., description="Audio file (wav, mp3, webm, ogg, m4a)"),
    user_id: Optional[str] = None
):
    """
    Process Malayalam voice query
    
    Workflow:
    1. Malayalam voice → Malayalam text (ASR)
    2. Malayalam text → English text (Translation)
    3. English text → English response (RAG Chatbot)
    4. English response → Malayalam response (Translation)
    5. Malayalam response → Malayalam voice (TTS)
    
    Args:
        audio_file: Audio file upload
        user_id: Optional user identifier
        
    Returns:
        Complete processing response with audio
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing voice query - Request ID: {request_id}")
    
    # Validate file format
    file_extension = audio_file.filename.split(".")[-1].lower()
    if file_extension not in settings.SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported: {', '.join(settings.SUPPORTED_AUDIO_FORMATS)}"
        )
    
    # Check file size
    file_size = 0
    audio_content = await audio_file.read()
    file_size = len(audio_content)
    
    if file_size > settings.MAX_AUDIO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file too large. Maximum size: {settings.MAX_AUDIO_SIZE_MB}MB"
        )
    
    try:
        # Save uploaded file
        safe_filename = sanitize_filename(audio_file.filename)
        audio_path = await audio_utils.save_uploaded_file(
            audio_content,
            safe_filename,
            settings.TEMP_DIR
        )
        
        # Process the audio query
        result = await assistant.process_audio_query(audio_path, user_id)
        
        # Clean up temporary file
        audio_utils.cleanup_temp_file(audio_path)
        
        # Prepare response
        if result["success"]:
            return VoiceQueryResponse(
                success=True,
                malayalam_text=result.get("malayalam_text"),
                english_text=result.get("english_text"),
                malayalam_response=result.get("malayalam_response"),
                english_response=result.get("english_response"),
                audio_response_base64=result.get("response_audio_b64"),
                processing_time=result.get("timings", {}).get("total"),
                request_id=request_id
            )
        else:
            return VoiceQueryResponse(
                success=False,
                error=result.get("error", "Processing failed"),
                request_id=request_id
            )
            
    except Exception as e:
        logger.error(f"Voice query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-query", response_model=VoiceQueryResponse, dependencies=[Depends(check_rate_limit)])
async def process_text_query(request: TextQueryRequest):
    """
    Process Malayalam text query
    
    Workflow:
    1. Malayalam text → English text (Translation)
    2. English text → English response (RAG Chatbot)
    3. English response → Malayalam response (Translation)
    4. Malayalam response → Malayalam voice (TTS)
    
    Args:
        request: Text query request
        
    Returns:
        Complete processing response with audio
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Processing text query - Request ID: {request_id}")
    
    try:
        # Process the text query
        result = await assistant.process_text_query(request.text, request.user_id)
        
        # Prepare response
        if result["success"]:
            return VoiceQueryResponse(
                success=True,
                malayalam_text=request.text,
                english_text=result.get("english_text"),
                malayalam_response=result.get("malayalam_response"),
                english_response=result.get("english_response"),
                audio_response_base64=result.get("response_audio_b64"),
                processing_time=result.get("timings", {}).get("total"),
                request_id=request_id
            )
        else:
            return VoiceQueryResponse(
                success=False,
                error=result.get("error", "Processing failed"),
                request_id=request_id
            )
            
    except Exception as e:
        logger.error(f"Text query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-context/{user_id}")
async def clear_user_context(user_id: str):
    """
    Clear conversation context for a user
    
    Args:
        user_id: User identifier
        
    Returns:
        Success status
    """
    assistant.rag_service.clear_context(user_id)
    return {"success": True, "message": f"Context cleared for user: {user_id}"}


@app.get("/supported-formats")
async def get_supported_formats():
    """
    Get supported audio formats and configuration
    
    Returns:
        Supported formats and limits
    """
    return {
        "audio_formats": settings.SUPPORTED_AUDIO_FORMATS,
        "max_file_size_mb": settings.MAX_AUDIO_SIZE_MB,
        "max_duration_seconds": settings.MAX_AUDIO_DURATION_SECONDS,
        "languages": {
            "source": "Malayalam (ml)",
            "target": "English (en)",
            "tts_voices": ["ml-IN-SobhanaNeural", "ml-IN-Wavenet-A"]
        }
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )
