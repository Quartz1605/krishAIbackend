"""Malayalam audio transcription then English translation using Google Generative AI.

Usage (PowerShell):
  $env:GEMINI_API_KEY="YOUR_KEY_HERE"
  python transcribe_translate.py path/to/audio.mp3

Outputs Malayalam transcription first, then English translation and JSON summary.

This is the first step toward a full voice assistant (speech->Malayalam->English->RAG->Malayalam->TTS).
"""
from __future__ import annotations
import os
import sys
import json
import mimetypes
from pathlib import Path
from typing import Optional

try:
    import google.generativeai as genai
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependency google-generativeai. Install with: pip install google-generativeai")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY_ENV = "GEMINI_API_KEY"

# Accept common audio extensions Gemini supports
_ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}


def configure() -> None:
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"Missing API key. Set environment variable {API_KEY_ENV}.")
    genai.configure(api_key=api_key)


def guess_mime(path: str) -> str:
    # Use extension guess fallback to audio/mp3
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        ext = Path(path).suffix.lower()
        if ext == ".mp3":
            return "audio/mp3"
        if ext == ".wav":
            return "audio/wav"
        if ext == ".m4a":
            return "audio/m4a"
        if ext == ".ogg":
            return "audio/ogg"
        if ext == ".webm":
            return "audio/webm"
        return "application/octet-stream"
    return mime


def read_audio_bytes(path: str) -> tuple[bytes, str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in _ALLOWED_EXT:
        raise ValueError(f"Unsupported audio extension '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXT))}")
    with open(path, "rb") as f:
        data = f.read()
    mime = guess_mime(path)
    return data, mime


def transcribe_malayalam(audio_bytes: bytes, mime_type: str, model: Optional[genai.GenerativeModel] = None) -> str:
    model = model or genai.GenerativeModel(MODEL_NAME)
    prompt = (
        "Transcribe the following audio into Malayalam text only. "
        "Do not translate, do not add commentary. Output only the Malayalam transcription."
    )
    response = model.generate_content([
        {"text": prompt},
        {"mime_type": mime_type, "data": audio_bytes},
    ])
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        # Some responses may put parts in candidates
        for cand in getattr(response, "candidates", []) or []:
            part_text = "".join(getattr(part, "text", "") for part in getattr(cand, "content", {}).parts or [])
            if part_text.strip():
                text = part_text.strip()
                break
    if not text:
        raise ValueError("Empty transcription returned.")
    return text


def translate_malayalam_to_english(malayalam_text: str, model: Optional[genai.GenerativeModel] = None) -> str:
    model = model or genai.GenerativeModel(MODEL_NAME)
    translate_prompt = (
        "Translate the following Malayalam text into natural English. "
        "Do not include the original Malayalam, only the English translation.\n\n" + malayalam_text
    )
    response = model.generate_content(translate_prompt)
    english = (getattr(response, "text", "") or "").strip()
    if not english:
        for cand in getattr(response, "candidates", []) or []:
            part_text = "".join(getattr(part, "text", "") for part in getattr(cand, "content", {}).parts or [])
            if part_text.strip():
                english = part_text.strip()
                break
    if not english:
        raise ValueError("Empty translation returned.")
    return english


def translate_english_to_malayalam(english_text: str, model: Optional[genai.GenerativeModel] = None) -> str:
    """Translate English text back into natural Malayalam (no English retained)."""
    model = model or genai.GenerativeModel(MODEL_NAME)
    prompt = (
        "Translate the following English text into natural, fluent Malayalam that is easy for Kerala farmers to understand. "
        "Do not include the original English. Only output Malayalam.\n\n" + english_text
    )
    response = model.generate_content(prompt)
    mal = (getattr(response, "text", "") or "").strip()
    if not mal:
        for cand in getattr(response, "candidates", []) or []:
            part_text = "".join(getattr(part, "text", "") for part in getattr(cand, "content", {}).parts or [])
            if part_text.strip():
                mal = part_text.strip()
                break
    if not mal:
        raise ValueError("Empty Malayalam translation returned.")
    return mal


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python transcribe_translate.py <audio_file.(mp3|wav|m4a|ogg|webm)>")
        return 1
    audio_path = argv[1]

    try:
        configure()
        audio_bytes, mime_type = read_audio_bytes(audio_path)

        print("[1/2] Transcribing to Malayalam...")
        malayalam = transcribe_malayalam(audio_bytes, mime_type)
        print("Malayalam transcription:\n" + malayalam + "\n")

        print("[2/2] Translating to English...")
        english = translate_malayalam_to_english(malayalam)
        print("English translation:\n" + english)

        summary = {"malayalam": malayalam, "english": english, "model": MODEL_NAME, "file": os.path.basename(audio_path)}
        print("\nJSON summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    except Exception as e:  # pragma: no cover
        print(f"Error: {e}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
