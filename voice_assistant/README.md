# Voice Assistant (Phase 1: Transcribe + Translate)

This module handles Malayalam speech transcription and English translation as the first step toward a full RAG-driven bilingual voice assistant.

## Features Implemented
- Malayalam audio transcription using Gemini (`gemini-2.5-flash` default)
- Malayalam → English translation
- JSON summary output
- Supports: mp3, wav, m4a, ogg, webm

## Folder Structure
```
voice_assistant/
  transcribe_translate.py
  requirements.txt
  test_audio/          # put sample input audio here
  output_audio/        # (reserved for future TTS output)
```

## Install Dependency (inside existing venv)
```powershell
pip install -r voice_assistant/requirements.txt
```

## Environment
Set Gemini API key:
```powershell
$env:GEMINI_API_KEY="YOUR_KEY_HERE"
```
(Optional) choose a different Gemini model:
```powershell
$env:GEMINI_MODEL="gemini-2.5-pro-exp"
```
Default model: `gemini-2.5-flash`.

## Run
```powershell
python voice_assistant/transcribe_translate.py voice_assistant/test_audio/sample.mp3
```
Output includes:
1. Malayalam transcription
2. English translation
3. JSON summary

### End-to-End Query (Option A Pipeline)
Start the RAG API in another terminal:
```powershell
python flask_app.py
```
Then run the voice→RAG pipeline:
```powershell
python voice_assistant/run_voice_query.py voice_assistant/test_audio/sample.mp3
```
It outputs JSON including:
- `malayalam_transcription`
- `english_query`
- `rag.answer` (English)
- `rag_answer_malayalam` ( Malayalam translation of the RAG answer )
- `rag.sources`
 - `tts_audio_path` (MP3 file path if TTS succeeded)

### Text To Speech (TTS)
The pipeline now auto-generates Malayalam speech (MP3) for the RAG answer using gTTS.

Disable TTS:
```powershell
$env:VOICE_TTS_DISABLE="1"
```

MP3 files are written to:
```
voice_assistant/output_audio/
```

If you see `<TTS failed: ...>` ensure:
- `pip install -r voice_assistant/requirements.txt` (gTTS installed)
- Internet access (gTTS requires network)
- Malayalam text not empty

### Markdown Stripping
The RAG answer may contain `*` / `**` emphasis which causes TTS to say "star". By default these markers are stripped before translation & speech.

Disable stripping (keep original formatting):
```powershell
$env:VOICE_KEEP_MARKDOWN="1"
```

If you need a different endpoint (e.g. deployed server):
```powershell
$env:RAG_CHAT_ENDPOINT="http://your-host:8000/chat"
python voice_assistant/run_voice_query.py voice_assistant/test_audio/sample.mp3
```

## Example JSON
```json
{
  "malayalam": "...",
  "english": "...",
  "model": "gemini-2.5-flash",
  "file": "sample.mp3"
}
```

## Next Planned Steps (Future Phases)
1. Feed English text into existing RAG chatbot
2. Translate RAG answer back to Malayalam
3. TTS synthesis → `output_audio/answer.wav`
4. REST endpoint wrapper
5. Streaming + latency optimization

Let me know when to proceed to the next phase.
