"""End-to-end Option A pipeline:
Malayalam audio -> transcription -> English translation -> RAG chatbot query.

Usage (PowerShell):
  $env:GEMINI_API_KEY="YOUR_KEY"
  # Start RAG API in another terminal: python flask_app.py
  python voice_assistant/run_voice_query.py voice_assistant/test_audio/sample.mp3

Output: JSON block with Malayalam transcription, English query, RAG answer, sources.
"""
from __future__ import annotations
import os
import sys
import json
from typing import Any, Dict

import requests

# Reuse functions from transcribe_translate
from transcribe_translate import (
    configure,
    read_audio_bytes,
    transcribe_malayalam,
    translate_malayalam_to_english,
    translate_english_to_malayalam,
    MODEL_NAME,
)
try:
    from tts import text_to_speech_malayalam
except Exception:  # pragma: no cover
    text_to_speech_malayalam = None
try:
    from text_clean import strip_markdown
except Exception:  # pragma: no cover
    def strip_markdown(x: str) -> str:
        return x

RAG_ENDPOINT = os.getenv("RAG_CHAT_ENDPOINT", "http://127.0.0.1:8000/chat")
TIMEOUT = 90  # seconds


def call_rag(query_english: str) -> Dict[str, Any]:
    try:
        resp = requests.post(
            RAG_ENDPOINT,
            json={"query": query_english},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"success": False, "error": f"RAG call failed: {e}"}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python voice_assistant/run_voice_query.py <audio_file>")
        return 1
    audio_path = argv[1]

    # 1. Configure Gemini
    try:
        configure()
    except SystemExit as e:  # missing key
        print(str(e))
        return 2

    # 2. Load audio
    try:
        audio_bytes, mime_type = read_audio_bytes(audio_path)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return 3

    # 3. Transcribe Malayalam
    try:
        print("[1/3] Transcribing Malayalam...")
        mal_text = transcribe_malayalam(audio_bytes, mime_type)
    except Exception as e:
        print(f"Transcription error: {e}")
        return 4

    # 4. Translate to English
    try:
        print("[2/3] Translating to English...")
        en_text = translate_malayalam_to_english(mal_text)
    except Exception as e:
        print(f"Translation error: {e}")
        return 5

    # 5. Query RAG chatbot
    try:
        print(f"[3/3] Querying RAG endpoint: {RAG_ENDPOINT}")
        rag_result = call_rag(en_text)
    except Exception as e:
        rag_result = {"success": False, "error": str(e)}

    # 6. Translate RAG answer back to Malayalam (if success)
    mal_answer = None
    tts_path = None
    if rag_result.get("success") and rag_result.get("answer"):
        try:
            raw_answer_en = rag_result["answer"]
            if not os.getenv("VOICE_KEEP_MARKDOWN"):
                raw_answer_en = strip_markdown(raw_answer_en)
            mal_answer = translate_english_to_malayalam(raw_answer_en)
        except Exception as e:
            mal_answer = f"<Malayalam translation failed: {e}>"
        # TTS (optional)
    if mal_answer and text_to_speech_malayalam and not os.getenv("VOICE_TTS_DISABLE"):
        try:
            # Final safety: remove any lingering * characters before TTS
            safe_text = mal_answer.replace('*', ' ')
            tts_path = text_to_speech_malayalam(
                safe_text,
                basename=os.path.splitext(os.path.basename(audio_path))[0]
            )
        except Exception as e:
            tts_path = f"<TTS failed: {e}>"

    # 7. Aggregate output
    output = {
        "audio_file": os.path.basename(audio_path),
        "model_used": MODEL_NAME,
        "malayalam_transcription": mal_text,
        "english_query": en_text,
        "rag": rag_result,
        "rag_answer_malayalam": mal_answer,
    "tts_audio_path": tts_path,
    }

    print("\n=== PIPELINE RESULT ===")
    print(json.dumps(output, ensure_ascii=False, indent=2))

    # Return non-zero if RAG failed (still useful partial result)
    if not rag_result.get("success", False):
        return 10
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
