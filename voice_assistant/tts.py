"""Simple Malayalam Text-To-Speech helper using gTTS.

Generates an MP3 file from Malayalam text. gTTS uses Google translate TTS backend
(unofficial) – suitable for prototyping. For production / offline you may
replace with a local TTS engine.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from gtts import gTTS

OUTPUT_DIR = Path(__file__).parent / "output_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def text_to_speech_malayalam(text: str, basename: str = "answer", suffix: Optional[str] = None) -> str:
    """Convert Malayalam text to speech and save as MP3.

    Args:
        text: Malayalam text input.
        basename: base filename (without extension).
        suffix: optional extra tag appended before extension.

    Returns:
        str path to generated mp3 file.
    """
    if not text or not text.strip():  # pragma: no cover
        raise ValueError("Empty text for TTS")

    safe_base = "".join(c for c in basename if c.isalnum() or c in ("-","_")) or "answer"
    if suffix:
        safe_base += f"_{suffix}"

    out_path = OUTPUT_DIR / f"{safe_base}.mp3"

    # gTTS language code for Malayalam is 'ml'
    tts = gTTS(text=text, lang="ml")
    tts.save(str(out_path))

    return str(out_path)
