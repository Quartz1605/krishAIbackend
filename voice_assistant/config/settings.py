"""
Configuration settings for Malayalam Voice Assistant
"""
import os
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Settings:
    """Application settings loaded from environment variables"""
    
    # API Configuration
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
    # Google Cloud Settings
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
    GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")
    
    # Voice Settings
    TTS_VOICE_NAME: str = os.getenv("TTS_VOICE_NAME", "ml-IN-Wavenet-A")
    TTS_SPEAKING_RATE: float = float(os.getenv("TTS_SPEAKING_RATE", "1.0"))
    TTS_PITCH: float = float(os.getenv("TTS_PITCH", "0.0"))
    
    # ASR Settings
    ASR_LANGUAGE_CODE: str = os.getenv("ASR_LANGUAGE_CODE", "ml-IN")
    ASR_MODEL: str = os.getenv("ASR_MODEL", "whisper-1")
    
    # Translation Settings
    TRANSLATION_SOURCE_LANG: str = os.getenv("TRANSLATION_SOURCE_LANG", "ml")
    TRANSLATION_TARGET_LANG: str = os.getenv("TRANSLATION_TARGET_LANG", "en")
    
    # Application Settings
    MAX_AUDIO_SIZE_MB: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "10"))
    TEMP_DIR: str = os.getenv("TEMP_DIR", "./temp")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # RAG Chatbot Settings
    RAG_API_URL: str = os.getenv("RAG_API_URL", "http://localhost:8000/chat")
    RAG_API_KEY: Optional[str] = os.getenv("RAG_API_KEY")
    RAG_TIMEOUT: int = int(os.getenv("RAG_TIMEOUT", "30"))
    
    # Cache Settings
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    # Audio Processing
    SUPPORTED_AUDIO_FORMATS: list = field(default_factory=lambda: ["wav", "mp3", "webm", "ogg", "m4a"])
    MAX_AUDIO_DURATION_SECONDS: int = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "60"))
    
    def validate(self):
        """Validate required settings"""
        errors = []
        
        # Check for required API keys based on service selection
        if not self.HUGGINGFACE_API_KEY and not self.OPENAI_API_KEY:
            errors.append("Either HUGGINGFACE_API_KEY or OPENAI_API_KEY must be set")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True

# Create global settings instance
settings = Settings()
