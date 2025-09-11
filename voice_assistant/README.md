# Malayalam Voice Assistant for Farmers (Digital Krishi Officer)

A production-ready, cost-optimized Malayalam voice assistant that integrates with your existing English RAG chatbot. It converts Malayalam speech to text, translates to English, queries your RAG system, translates back to Malayalam, and returns natural Malayalam speech.

Highlights
- Free-first architecture using open/free APIs: Hugging Face Inference (free tier), Google Speech Recognition (free via SpeechRecognition), gTTS/Edge TTS (free), googletrans (unofficial free Translate)
- Async FastAPI microservice with robust error handling
- Caching, rate limiting, detailed logging, and Docker-ready structure
- Designed for poor audio quality and rural connectivity constraints


System Architecture
- Input: Malayalam voice audio (wav/mp3/webm/ogg/m4a)
- ASR: Malayalam speech → text (HuggingFace Whisper small → fallback to Google SpeechRecognition)
- Translate ML→EN: googletrans → fallbacks (HuggingFace Helsinki-NLP, MyMemory, LibreTranslate)
- RAG: Existing English chatbot HTTP API
- Translate EN→ML: same pipeline
- TTS: Edge TTS (ml-IN-SobhanaNeural) → fallback gTTS
- Output: Malayalam audio (base64 in JSON)

Components
- services/voice_processing.py: ASR + TTS with fallbacks
- services/translation.py: Malayalam↔English translations with caching
- services/rag_chatbot.py: HTTP integration to existing RAG
- services/assistant.py: End-to-end orchestrator
- api/main.py: FastAPI with endpoints
- utils/logger.py: Structured logging
- utils/helpers.py: Audio utils, caching, retry, rate limiting
- config/settings.py: Env-based configuration

API Endpoints
- POST /voice-query: multipart form-data with audio_file; returns JSON with ml/en texts and base64 audio
- POST /text-query: JSON body with Malayalam text; returns Malayalam response + base64 audio
- GET /health: Service health (RAG status included)
- GET /supported-formats
- POST /clear-context/{user_id}

Response JSON (example)
{
  "success": true,
  "malayalam_text": "...",
  "english_text": "...",
  "malayalam_response": "...",
  "english_response": "...",
  "audio_response_base64": "...",
  "processing_time": 1.23,
  "request_id": "uuid"
}


Setup and Installation
1. Prerequisites
- Python 3.10+
- FFmpeg installed and on PATH (required by pydub). On Windows: download ffmpeg and add bin to PATH.
- For Edge TTS voice output, internet connectivity is required.

2. Clone and branch
- Already prepared on branch: new

3. Environment
copy voice_assistant/.env.example voice_assistant/.env
Edit .env values as needed (HUGGINGFACE_API_KEY optional but recommended).

4. Install dependencies
python -m venv .venv
. .venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -r voice_assistant/requirements.txt

5. Run the API
uvicorn voice_assistant.api.main:app --host 0.0.0.0 --port 8080 --reload

6. Test quickly
- Open http://localhost:8080/docs
- Try POST /text-query
- Try POST /voice-query with an mp3/wav of Malayalam speech

Docker (optional)
Create a Dockerfile similar to the following:

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
COPY voice_assistant/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY voice_assistant /app/voice_assistant
ENV HOST=0.0.0.0 PORT=8080
EXPOSE 8080
CMD ["uvicorn", "voice_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

Build/Run:
- docker build -t ml-voice-assistant .
- docker run --env-file voice_assistant/.env -p 8080:8080 ml-voice-assistant


Integration Guide (RAG)
- Configure RAG_API_URL (and RAG_API_KEY if required) in .env
- Expected RAG /chat request body (example):
  { "query": "English question", "timestamp": "...", "language": "en", "user_id": "optional", "conversation_history": [] }
- Expected RAG /chat response body:
  { "response": "English answer", "confidence": 0.7, "sources": [], "metadata": {} }
- Health check: /health should return 200 for healthy (used by our /health)

If your RAG interface differs, adjust services/rag_chatbot.py accordingly.


Testing Examples
- Text query:
  curl -X POST http://localhost:8080/text-query -H "Content-Type: application/json" -d '{"text": "എന്റെ നെല്ലിൽ ഇലകൾ മഞ്ഞനിറമാകുന്നു. കാരണം?", "user_id": "farmer123"}'

- Voice query (PowerShell example):
  Invoke-WebRequest -Uri http://localhost:8080/voice-query -Method POST -Form @{ user_id = 'farmer123'; audio_file = Get-Item .\examples\query_ml.mp3 }

- Get supported formats:
  curl http://localhost:8080/supported-formats


Troubleshooting
- ASR returns None
  - Ensure FFmpeg is installed; ensure audio file is supported
  - Try shorter clips or clearer audio
  - Provide HUGGINGFACE_API_KEY to improve ASR (Whisper small via HF Inference)

- TTS fails
  - Ensure internet connectivity (Edge TTS, gTTS)
  - If edge-tts not installed, it falls back to gTTS

- Translation quality
  - Hugging Face Helsinki-NLP models are used if API key provided
  - googletrans is unofficial; availability may vary

- RAG errors/timeouts
  - Check RAG_API_URL and server availability
  - Our service provides generic fallback responses

- Large audio files rejected
  - Increase MAX_AUDIO_SIZE_MB in .env

- Rate limit 429
  - Adjust RATE_LIMIT_REQUESTS/RATE_LIMIT_WINDOW


Cost Analysis and Recommendations (Free-first)
- Hugging Face Inference: Free tier with rate limits; good accuracy with Whisper small
- Google SpeechRecognition (free endpoint): zero cost but quota/availability may vary
- gTTS and Edge TTS: free; Edge TTS quality is higher and supports ml-IN
- googletrans: free/unofficial; add Hugging Face (Helsinki-NLP) as paid-free hybrid for robustness
- MyMemory/LibreTranslate: free fallback; self-host LibreTranslate for local deployments

For production high volume:
- Self-host Whisper (small/base) with open-source models to avoid API limits
- Self-host OpenTTS/Mozilla TTS for Malayalam if quality is sufficient
- Use Redis for caching translations and results
- Add metrics (Prometheus) and tracing (OpenTelemetry)


Security and Privacy
- No secrets in code. Use environment variables
- Sanitize filenames and validate MIME types
- Do not log PII; logs are structured and minimal


Notes
- This project targets Malayalam farmers; agricultural terminology preservation is considered in translation postprocessing.
- The pipeline is designed to be resilient with multiple fallbacks to handle connectivity challenges.

