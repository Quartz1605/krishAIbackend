"""
Startup script for Malayalam Voice Assistant
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_assistant.config.settings import settings


def main():
    """Run the Malayalam Voice Assistant API"""
    
    print("="*60)
    print("Malayalam Voice Assistant for Farmers")
    print("Digital Krishi Officer System")
    print("="*60)
    
    # Check environment
    try:
        settings.validate()
        print("✓ Configuration validated")
    except ValueError as e:
        print(f"⚠ Configuration warning: {e}")
        print("Some features may not work without proper API keys")
    
    # Ensure temp directory exists
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    print(f"✓ Temp directory: {settings.TEMP_DIR}")
    
    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    
    print(f"\nStarting server on http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print("\nPress CTRL+C to stop the server")
    print("-"*60)
    
    # Run the FastAPI app
    uvicorn.run(
        "voice_assistant.api.main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
